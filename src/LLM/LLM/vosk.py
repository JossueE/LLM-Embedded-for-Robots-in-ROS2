#!/usr/bin/env python3
import threading
from pathlib import Path, PurePath
from typing import Optional

import wave

import onnx
import onnxruntime
import numpy as np
import torch
from omegaconf import OmegaConf

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, Bool, String

# Permite que audio y flag entren en paralelo
from rclpy.callback_groups import ReentrantCallbackGroup
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
        self.declare_parameter("rate", 16000)               # Debe ser 16k para Silero
        self.declare_parameter("channels", 1)               # Silero espera mono
        self.declare_parameter("device", "cpu")             # "cpu" o "cuda" si tienes GPU

        #Es inutil al momento, pero lo dejo por si acaso
        self.declare_parameter("language", "es")            # p.ej. "en", "es" (según modelo)

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
                device="cpu" if self.device not in ("gpu", "cpu") else self.device,
            )
        except Exception as e:
            self.get_logger().error(f"No se pudo cargar Silero STT: {e}")
            raise

        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils

         # see available models
        models_yml = Path("models.yml")
        if not models_yml.exists():
            torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
        models = OmegaConf.load('models.yml')
        available_languages = list(models.stt_models.keys())
        assert self.language in available_languages

        onnx_model_path = Path("model.onnx")
        if not onnx_model_path.exists():
            torch.hub.download_url_to_file(models.stt_models.es.latest.onnx, 'model.onnx', progress=True)
        onnx_model = onnx.load('model.onnx')
        onnx.checker.check_model(onnx_model)
        self.ort_session = onnxruntime.InferenceSession('model.onnx')

        self._ort_in_name = self.ort_session.get_inputs()[0].name 

        # --- Estado / buffers ---
        self._flag: bool = False
        self._prev_flag: bool = False
        self._buffer = bytearray()
        self._lock = threading.Lock()

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

        self.get_logger().info(
            f"Silero listo ✅ SR={self.rate}ch={self.channels} device={self.device} lang={self.language}\n"
            "Transcribe cuando /flag_wake_word cae de True a False."
        )

    # -------------------- Callbacks --------------------

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
                    self.pub_transcript.publish(String(data=text))
                    self.get_logger().info(f"📝 {text}")
                else:
                    self.get_logger().info("📝 (vacío)")
            except Exception as e:
                self.get_logger().error(f"Error en STT: {e}")

    # -------------------- STT core --------------------

    def _stt_from_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """
        Convierte bytes Int16→tensor float32 normalizado y ejecuta Silero.
        """
        if not audio_bytes:
            return None

        with wave.open(f"/tmp/stt.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16
            w.setframerate(self.rate)  # debería ser 16000
            w.writeframes(audio_bytes)
        self.get_logger().info(f"💾 Guardé /tmp/stt.wav")

        # Int16 → float32 [-1, 1]
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return None

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
