#!/usr/bin/env python3
import numpy as np
import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from threading import Lock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from .llm_utils.config import AUDIO_PUBLISHER_DEVICE_ID, AUDIO_PUBLISHER_CHANELS, AUDIO_PUBLISHER_FRAMES_PER_BUFFER, AUDIO_PUBLISHER_DEBUG, SAMPLE_RATE_TTS

class AudioSink(Node):
    def __init__(self) -> None:
        super().__init__("audio_sink")

        # --- Parámetros ---
        self.declare_parameter("rate", SAMPLE_RATE_TTS)       # Debe coincidir con tu TTS
        self.declare_parameter("channels", AUDIO_PUBLISHER_CHANELS)       # Mono int16
        self.declare_parameter("frames_per_buffer", AUDIO_PUBLISHER_FRAMES_PER_BUFFER)
        self.declare_parameter("device_index", AUDIO_PUBLISHER_DEVICE_ID)  # -1 = default output
        self.declare_parameter("debug", AUDIO_PUBLISHER_DEBUG)       # logs para ver tráfico

        self.rate = int(self.get_parameter("rate").value)
        self.channels = int(self.get_parameter("channels").value)
        self.frames_per_buffer = int(self.get_parameter("frames_per_buffer").value)
        self.debug = bool(self.get_parameter("debug").value)
        dev_idx = int(self.get_parameter("device_index").value)
        self.device_index = None if dev_idx < 0 else dev_idx

        # --- PyAudio ---
        self.pa = pyaudio.PyAudio()
        self.stream = self._open_stream()
        self._wlock = Lock()  # evitar writes simultáneos (no es cola)

        # --- QoS: que empareje con el default del publisher ---
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # --- ROS I/O ---
        self.sub = self.create_subscription(
            Int16MultiArray, "/tts_audio", self.cb_audio, qos
        )

        self.get_logger().info(
            f"AudioSink ▶️ rate={self.rate} Hz, ch={self.channels}, "
            f"fpb={self.frames_per_buffer}, device_index={self.device_index}"
        )

    def _open_stream(self):
        return self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
            start=True,
        )

    def _reopen_stream(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        self.stream = self._open_stream()

    def cb_audio(self, msg: Int16MultiArray) -> None:
        if not msg.data:
            return
        try:
            pcm = np.asarray(msg.data, dtype=np.int16)
            if pcm.ndim != 1:
                pcm = pcm.reshape(-1)
            data = pcm.tobytes(order="C")
            frame_bytes = self.frames_per_buffer * self.channels * 2  # int16 -> 2 bytes
            with self._wlock:
                for i in range(0, len(data), frame_bytes):
                    chunk = data[i:i + frame_bytes]
                    try:
                        self.stream.write(chunk, exception_on_underflow=False)
                    except Exception as e:
                        self.get_logger().warn(f"write() falló, reabriendo stream: {e}")
                        self._reopen_stream()

            if self.debug:
                dur_ms = (len(pcm) / max(self.rate, 1)) * 1000.0
                self.get_logger().info(f"/tts_audio: {len(pcm)} muestras (~{dur_ms:.1f} ms)")

        except Exception as e:
            self.get_logger().error(f"AudioSink error: {e}")

    def destroy_node(self) -> bool:
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.pa.terminate()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioSink()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
