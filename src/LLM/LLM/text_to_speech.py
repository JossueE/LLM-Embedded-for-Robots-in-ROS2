#!/usr/bin/env python3
import threading
from typing import Optional

import numpy as np
import torch
import rclpy
from pathlib import Path
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, String
from rclpy.callback_groups import ReentrantCallbackGroup

cb_group = ReentrantCallbackGroup()

class SileroTTSNode(Node):
    def __init__(self) -> None:
        super().__init__("silero_tts_node")

        # --- Parámetros ---
        # 48k suena mejor; ajusta si tu reproductor solo soporta 16k/22.05k/24k/48k
        self.declare_parameter("rate", 24000)
        self.declare_parameter("device", "cpu")            # "cpu" o "cuda"
        self.declare_parameter("language", "es")           # "es"
        self.declare_parameter("speaker", "v3_es")    
        self.declare_parameter("voice", "es_2")            #voices are: es_0, es_1, es_2

        self.rate: int = int(self.get_parameter("rate").value)
        self.device: str = str(self.get_parameter("device").value).lower()
        self.language: str = str(self.get_parameter("language").value)
        self.speaker: str = str(self.get_parameter("speaker").value)
        self.voice: str = str(self.get_parameter("voice").value)

        self.local_bundle = ""

        # --- Carga del modelo TTS (PyTorch Hub) ---
        self.get_logger().info("Cargando Silero TTS...")
        try:
            self.model = self.load_silero_model()
            self.model.to(self.device)
        except Exception as e:
            self.get_logger().error(f"No se pudo cargar Silero TTS: {e}")
            raise

        # --- Estado / buffers ---
        self.state_machine_flag = ""

        # --- ROS I/O ---
        self.transcript_sub = self.create_subscription(
            String, "/answer", self.transcript_callback, 10, callback_group=cb_group
        )
        self.state_machine_sub = self.create_subscription(
            String, "/state_machine_flag", self.state_machine_cb, 10, callback_group=cb_group
        )
        self.state_machine_pub = self.create_publisher(String, "/state_machine_flag", 10)
        self.audio_pub = self.create_publisher(Int16MultiArray, "/tts_audio", 10)

        self.get_logger().info(
            f"Silero TTS listo 🔊 rate={self.rate} device={self.device} lang={self.language} speaker={self.speaker}"
        )

    def load_silero_model(self):
        if self.local_bundle:
            bundle_path = Path(self.local_bundle).expanduser().resolve()
            if not bundle_path.exists():
                raise FileNotFoundError(f"local_bundle no existe: {bundle_path}")
            # Carga desde .pt empaquetado (formato de Silero)
            self.get_logger().info(f"Cargando bundle local: {bundle_path}")
            importer = torch.package.PackageImporter(str(bundle_path))
            model = importer.load_pickle("tts_models", "model")
            return model

        # Fallback: torch.hub
        self.get_logger().info(
            "Cargando desde torch.hub snakers4/silero-models (usa TORCH_HOME para cachear en otra ruta)"
        )
        loaded = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.language,
            speaker=self.speaker,  # familia del modelo español
        )

        # Algunas versiones retornan solo el modelo; otras, tupla.
        if isinstance(loaded, (list, tuple)):
            model = loaded[0]
        else:
            model = loaded
        return model
    
    # -------------------- Callbacks --------------------
    def state_machine_cb(self, msg: String) -> None:
        self.state_machine_flag = msg.data

    def transcript_callback(self, msg: String) -> None:
        if self.state_machine_flag != "text_to_speech":
            return

        text = msg.data.strip()
        if not text:
            return

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.voice,
                sample_rate=self.rate,
                put_accent=True,
                put_yo=True
            )

            audio = np.asanyarray(audio,dtype=np.float32).ravel()
            # A int16 PCM para publicar por ROS
            audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

            msg_out = Int16MultiArray()
            # Layout opcional (una dimensión: frames)
            msg_out.layout.dim = [MultiArrayDimension(label="frames", size=len(audio_i16), stride=len(audio_i16))]
            msg_out.data = audio_i16.tolist()
            self.audio_pub.publish(msg_out)

            # Avanza la máquina de estados
            self.state_machine_pub.publish(String(data="wake_word"))

        except Exception as e:
            self.state_machine_pub.publish(String(data="wake_word"))
            self.get_logger().error(f"TTS falló: {e}")

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SileroTTSNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
