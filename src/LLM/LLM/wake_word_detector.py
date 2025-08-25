import json
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, Bool
import webrtcvad
import vosk
from rclpy.parameter import Parameter


class WakeWordDetector(Node):
    """Detects a wake word using Vosk and VAD and toggles Whisper inference."""

    def __init__(self) -> None:
        super().__init__("wake_word_detector")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("vosk_model_path", "/home/snorlix/vosk-model-small-es-0.42")
        self.declare_parameter("wake_word", "ok robot")
        self.declare_parameter("inference_node", "inference")
        self.declare_parameter("listen_seconds", 3.0)

        self.sample_rate = self.get_parameter("sample_rate").value
        model_path = self.get_parameter("vosk_model_path").value
        wake_word = self.get_parameter("wake_word").value

        self.vad = webrtcvad.Vad(2)
        self.model = vosk.Model(model_path)
        # Restrict recognition to the wake word grammar
        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate, f'["{wake_word}"]')

        self.audio_sub = self.create_subscription(
            Int16MultiArray, "/audio", self.audio_callback, 10
        )

        self.flag_wake_word = self.create_publisher(
            Bool, "/flag_wake_word", 10
        )
        self.flag_wake_word.publish(Bool(data=False))

        self.param_client = self.get_parameter("inference_node").value
        

        self.listening = False
        self.listen_timer = None

    def audio_callback(self, msg: Int16MultiArray) -> None:

        audio_bytes = np.array(msg.data, dtype=np.int16).tobytes()
        frame_bytes = int(self.sample_rate / 1000 * 20) * 2  # 20ms
        for i in range(0, len(audio_bytes) - frame_bytes + 1, frame_bytes):
            frame = audio_bytes[i : i + frame_bytes]

            if not self.vad.is_speech(frame, self.sample_rate):
                continue
            
            if self.rec.AcceptWaveform(frame):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").lower()
                if self.get_parameter("wake_word").value in text:
                    self.get_logger().info("Wake word detected")
                    self.activate_whisper()

    def activate_whisper(self) -> None:
        if self.listening:
            return
        self.listening = True
        self.flag_wake_word.publish(Bool(data=True))
        duration = self.get_parameter("listen_seconds").value
        self.listen_timer = self.create_timer(duration, self.deactivate_whisper)

    def deactivate_whisper(self) -> None:  
        if self.listen_timer is not None:
            self.listen_timer.cancel()
            self.listen_timer = None  
        print("1 Wake word audio published")
        self.flag_wake_word.publish(Bool(data=False))
        self.listening = False

    def _set_inference_active(self, value: bool) -> None:
        param = Parameter("active", Parameter.Type.BOOL, value)
        future = self.param_client.set_parameters([param])
        # We don't need to wait for completion but log result
        future.add_done_callback(
            lambda fut: self.get_logger().info(
                f"Whisper active set to {value}: {fut.result().successful}"
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WakeWordDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()