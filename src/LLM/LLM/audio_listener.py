#!/usr/bin/env python3
import atexit
import numpy as np
import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension

class AudioListenerNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("channels", 1),
                ("frames_per_buffer", 1000),  # ~62.5 ms a 16 kHz
                ("rate", 16000),
                ("device_index", 11),         # <- hazlo parámetro
            ],
        )

        self.channels_ = self.get_parameter("channels").value
        self.frames_per_buffer_ = self.get_parameter("frames_per_buffer").value
        self.rate_ = self.get_parameter("rate").value
        self.device_index_ = self.get_parameter("device_index").value

        self.pa = pyaudio.PyAudio()

        # Lista dispositivos para elegir (solo una vez al inicio)
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                self.get_logger().info(f"[{i}] {info['name']} (in={info['maxInputChannels']}, rate={int(info.get('defaultSampleRate',0))})")

        # Opcional: valida soporte
        try:
            self.pa.is_format_supported(self.rate_,
                                        input_device=self.device_index_ if self.device_index_ >= 0 else None,
                                        input_channels=self.channels_,
                                        input_format=pyaudio.paInt16)
        except ValueError as e:
            self.get_logger().warn(f"Formato no soportado por el dispositivo seleccionado: {e}")

        self.stream_ = self.pa.open(
            channels=self.channels_,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.frames_per_buffer_,
            rate=self.rate_,
            input_device_index=(self.device_index_ if self.device_index_ >= 0 else None),
        )

        # Usa QoS de sensor data para compatibilidad con tu suscriptor
        self.audio_publisher_ = self.create_publisher(
            Int16MultiArray, "/audio", 10
        )

        self.audio_publisher_timer_ = self.create_timer(
            float(self.frames_per_buffer_) / float(self.rate_),
            self.audio_publisher_timer_callback_,
        )

        atexit.register(self.cleanup_)

    def audio_publisher_timer_callback_(self) -> None:
        # Evita excepciones por overflow y rellenos silenciosos
        data = self.stream_.read(self.frames_per_buffer_, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)

        # Diagnóstico simple: RMS cada ~1s
        rms = float(np.sqrt(np.mean(audio.astype(np.float32)**2)) + 1e-9)
        if (self.get_clock().now().nanoseconds // 1_000_000_000) % 1 == 0:
            self.get_logger().debug(f"RMS={rms:.1f}")

        msg = Int16MultiArray()
        msg.data = audio.tolist()
        msg.layout.data_offset = 0
        msg.layout.dim.append(
            MultiArrayDimension(label="audio", size=self.frames_per_buffer_, stride=1)
        )
        self.audio_publisher_.publish(msg)

    def cleanup_(self):
        if self.stream_.is_active():
            self.stream_.stop_stream()
        self.stream_.close()
        self.pa.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = AudioListenerNode("audio_listener")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()



    """

    Verifica si el Micro jala Fuera de ROS2
    

    sudo apt update
    sudo apt install alsa-utils pavucontrol
    arecord -l                        # lista tarjetas/dispositivos ALSA
    arecord -f S16_LE -r 16000 -d 3 /tmp/test.wav
    aplay /tmp/test.wav               # ¿se escucha? entonces ALSA funciona

    Para el requirements
    pip install "empy==3.3.4"
    pip install lark

    Problema con el ALSA
    https://github.com/Uberi/speech_recognition/issues/526
"""