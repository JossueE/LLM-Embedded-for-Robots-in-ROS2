import json
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, Bool, String
import webrtcvad
import vosk
from rclpy.parameter import Parameter


class Vosk(Node):
    """Transcrib phrases using Vosk and VAD"""

    def __init__(self) -> None:
        super().__init__("vosk_transcriber") 

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("vosk_model_path", "/home/snorlix/vosk-model-small-es-0.42")

        self.sample_rate = self.get_parameter("sample_rate").value
        model_path = self.get_parameter("vosk_model_path").value

        self.vad = webrtcvad.Vad(2)
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate) 

        # Estado
        self.listening: bool = False
        self.was_listening: bool = False  # para detectar flanco de bajada

        # ROS I/O
        self.pub_transcript = self.create_publisher(String, "/transcript", 10)

        self.audio_sub = self.create_subscription(
            Int16MultiArray, "/audio", self.audio_callback, 10
        )

        self.flag_sub = self.create_subscription(
            Bool, "/flag_wake_word", self.flag_callback, 10
        )   

        self.get_logger().info(
            f"Vosk listo ✅ SR={self.sample_rate}  model={model_path}\n"
            "Transcribe solo cuando /flag_wake_word = True"
        )     

    def audio_callback(self, msg: Int16MultiArray) -> None:

        if not self.listening:
            return
        
        buf = np.asarray(msg.data, dtype=np.int16).tobytes()

        try:
            if self.rec.AcceptWaveform(buf):
                # Final (phrase) result
                res = json.loads(self.rec.Result() or "{}")
                text = (res.get("text") or "").strip()
                if text:
                    self.pub_transcript.publish(String(data=text))
                    self.get_logger().info(f"📝 FINAL: {text}")
            else:
                # Low-latency partial
                part = json.loads(self.rec.PartialResult() or "{}").get("partial", "").strip()
                if part:
                    # self.pub_partial.publish(String(data=part))
                    self.get_logger().debug(f"… partial: {part}")
        except Exception as e:
            self.get_logger().warn(f"Vosk error: {e}")


        
    def flag_callback(self, msg: Bool) -> None:
        self.listening = bool(msg.data)
        if not self.listening and self.was_listening:
            try:
                final = json.loads(self.rec.FinalResult()).get("text", "").strip()
            except Exception:
                final = ""
            if final:
                self.pub_transcript.publish(String(data=final))
                self.get_logger().info(f"📝 (final) {final}")
            # Reinicia el reconocedor para la siguiente ventana
            self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.was_listening = self.listening


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Vosk()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()