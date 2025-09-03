#!/usr/bin/env python3
import threading
from pathlib import Path
from typing import Optional

import time
import onnx
import onnxruntime
import numpy as np
import torch
from omegaconf import OmegaConf
#import wave

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, Bool, String

# Permite que audio y flag entren en paralelo
from rclpy.callback_groups import ReentrantCallbackGroup

from .llm_utils.config import SAMPLE_RATE_STT, CHANNELS_INPUT_STT, DEVICE_SELECTOR_STT, LANGUAGE
cb_group = ReentrantCallbackGroup()

class SileroSTTNode(Node):
    """
    Transcribe audio por segmentos controlados con /flag_wake_word.
    - Mientras flag=True => acumula audio en RAM
    - En el flanco de bajada (True→False) => ejecuta STT y publica /transcript
    """

    def __init__(self) -> None:
        super().__init__("silero_stt_node")

        # --- Parámetros ---
        self.declare_parameter("rate", SAMPLE_RATE_STT)               # Debe ser 16k para Silero
        self.declare_parameter("channels", CHANNELS_INPUT_STT)               # Silero espera mono
        self.declare_parameter("device", DEVICE_SELECTOR_STT)             # "cpu" o "cuda" si tienes GPU

        #Es inutil al momento, pero lo dejo por si acaso
        self.declare_parameter("language",LANGUAGE)            # p.ej. "en", "es" (según modelo)

        self.rate: int = int(self.get_parameter("rate").value)
        self.channels: int = int(self.get_parameter("channels").value)
        self.device: str = str(self.get_parameter("device").value).lower()
        self.language: str = str(self.get_parameter("language").value)

        if self.rate != 16000:
            self.get_logger().warn(f"Silero recomienda 16000 Hz; recibido {self.rate}. Re-samplea antes de publicar /audio.")
        if self.channels != 1:
            self.get_logger().warn(f"Silero espera audio mono; recibido {self.channels} canales. Publica mono en /audio.")


        # --- Carga del modelo (una sola vez) ---
        self.get_logger().info("Cargando Silero STT...")
        try:
            model, self.decoder, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_stt",          # stt
                language=self.language,      # depende del modelo disponible
                device="cpu" if self.device not in ("cuda", "cpu") else self.device,
            )
        except Exception as e:
            self.get_logger().error(f"No se pudo cargar Silero STT: {e}")
            raise

        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils

        base_dir = Path(__file__).resolve().parent
        models_yml = base_dir / "models.yml"
        if not models_yml.exists():
            torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', str(models_yml))
        models = OmegaConf.load(str(models_yml))
        lang = getattr(self, "language", "es")
        available_languages = list(models.stt_models.keys())
        if lang not in available_languages:
            self.get_logger().warn(f"[Silero] Idioma '{lang}' no disponible en models.yml; usando 'en'")
            lang = "es"

        onnx_url = models.stt_models[lang].latest.onnx
        onnx_model_path = base_dir / f"silero-stt-{lang}.onnx"
        if not onnx_model_path.exists():
            torch.hub.download_url_to_file(onnx_url, str(onnx_model_path), progress=True)
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

        self._ort_in_name = self.ort_session.get_inputs()[0].name 

        # --- Estado / buffers ---
        self._flag: bool = False
        self._prev_flag: bool = False
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.state_machine_flag = ""

        # Worker para transcribir sin bloquear callbacks
        self._work_queue: list[bytes] = []
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # --- ROS I/O ---
        self.pub_transcript = self.create_publisher(String, "/transcript", 10)

        self.audio_sub = self.create_subscription(
            Int16MultiArray, "/audio", self.audio_callback, 10, callback_group=cb_group
        )
        self.flag_sub = self.create_subscription(
            Bool, "/flag_wake_word", self.flag_callback, 10, callback_group=cb_group
        )
        self.state_machine_sub = self.create_subscription(
            String, "/state_machine_flag", self.state_machine_function, 10
        )

        self.state_machine_publisher = self.create_publisher(
            String, "/state_machine_flag", 10
        )
        self.get_logger().info(
            f"Silero listo 🔊 SR={self.rate}ch={self.channels} device={self.device} lang={self.language}\n"
            "Transcribe cuando /flag_wake_word cae de True a False."
        )
        


    # -------------------- Callbacks --------------------
    def state_machine_function(self, msg: String) -> None:
        self.state_machine_flag = msg.data

    def audio_callback(self, msg: Int16MultiArray) -> None:
        """Acumula frames Int16 mientras flag sea True."""
        if not self._flag:
            return
        frames = np.asarray(msg.data, dtype=np.int16)
        with self._lock:
            self._buffer.extend(frames.tobytes())

    def flag_callback(self, msg: Bool) -> None:
        """En el flanco de bajada, envía buffer a la cola de trabajo."""
        self._flag = bool(msg.data)
        if (not self._flag) and self._prev_flag:
            # Copiamos y vaciamos buffer de forma atómica
            with self._lock:
                if len(self._buffer) == 0:
                    self._prev_flag = self._flag
                    return
                chunk = bytes(self._buffer)
                self._buffer.clear()

            # Encolar para transcripción
            self._work_queue.append(chunk)
        self._prev_flag = self._flag

    # -------------------- Worker --------------------

    def _worker_loop(self) -> None:
        while True:
            if not self._work_queue:
                # Sleep muy corto para no hoggear CPU
                rclpy.spin_once(self, timeout_sec=0.01)  # mantiene callbacks responsivos
                continue
            chunk = self._work_queue.pop(0)
            try:
                text = self._stt_from_bytes(chunk)
                if text:
                    self.state_machine_publisher.publish(String(data="main_active"))
                    time.sleep(0.01)
                    self.pub_transcript.publish(String(data=text))
                    self.get_logger().info(f"📝 {text}")
                else:
                    self.state_machine_publisher.publish(String(data="wake_word"))
                    self.get_logger().info("📝 (vacío)")
                
            except Exception as e:
                self.state_machine_publisher.publish(String(data="wake_word"))
                self.get_logger().error(f"Error en STT: {e}")

    # -------------------- STT core --------------------

    def _stt_from_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """
        Convierte bytes Int16→tensor float32 normalizado y ejecuta Silero.
        """
        if not audio_bytes:
            return None

        # Int16 → float32 [-1, 1]
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return None
        
        #For Debugging purposes

        #with wave.open(f"/tmp/stt.wav", "wb") as w:
        #    w.setnchannels(1)
        #    w.setsampwidth(2)  # int16
        #    w.setframerate(self.rate)  # debería ser 16000
        #    w.writeframes(audio_bytes)
        #self.get_logger().info(f"💾 Guardé /tmp/stt.wav")

        x = pcm.astype(np.float32) / 32768.0

        onnx_in = x[np.newaxis, :]

        outs = self.ort_session.run(None, {self._ort_in_name: onnx_in})
        # Usa el decoder oficial de Silero sobre los logits
        text = self.decoder(torch.Tensor(outs[0])[0])
        
        #text = text.strip()
        return text or None

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SileroSTTNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
