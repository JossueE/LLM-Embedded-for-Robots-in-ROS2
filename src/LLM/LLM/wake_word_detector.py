import json
from typing import List

#This is Use for Donwload the model
import os
from pathlib import Path
import urllib.request
import zipfile

#To manage the audio stream
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
import webrtcvad
import vosk
from rclpy.parameter import Parameter
from .llm_utils.config import DEFAULT_MODEL_FILENAME_WAKE_WORD, DEFAULT_MODEL_URL_WAKE_WORD, ACTIVATION_PHRASE_WAKE_WORD, LISTEN_SECONDS_STT, AUDIO_LISTENER_SAMPLE_RATE, VARIANTS_WAKE_WORD

class WakeWordDetector(Node):
    """Detects a wake word using Vosk + VAD con baja latencia."""

    def __init__(self) -> None:
        super().__init__("wake_word_detector")

        self.declare_parameter("sample_rate", AUDIO_LISTENER_SAMPLE_RATE)
        self.declare_parameter("wake_word", ACTIVATION_PHRASE_WAKE_WORD)      # palabra “principal”
        self.declare_parameter("inference_node", "inference")
        self.declare_parameter("listen_seconds", LISTEN_SECONDS_STT)
        self.declare_parameter("variants", VARIANTS_WAKE_WORD)

        self.sample_rate = self.get_parameter("sample_rate").value
        self.wake_word = str(self.get_parameter("wake_word").value).lower()
        self.variants: List[str] = [str(v).lower() for v in self.get_parameter("variants").value]

        # VAD menos agresivo para no comerse el inicio (0–3; 1 es razonable)
        self.vad = webrtcvad.Vad(1)

        # Limita vocab a las variantes para que Vosk “tienda” a oírlas
        # Construimos gramática JSON con las variantes
        grammar = json.dumps(self.variants, ensure_ascii=False)

        model_path = self.ensure_vosk_model()
        self.model = vosk.Model(model_path)

        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate, grammar)

        self.audio_sub = self.create_subscription(
            Int16MultiArray, "/audio", self.audio_callback, 10
        )

        self.state_machine_sub = self.create_subscription(
            String, "/state_machine_flag", self.state_machine_function, 10
        )

        self.state_machine_publisher = self.create_publisher(
            String, "/state_machine_flag", 10
        )
        self.state_machine_publisher.publish(String(data="wake_word"))

        self.flag_wake_word = self.create_publisher(String, "/flag_wake_word", 10)
        self.flag_wake_word.publish(String(data="deactivate"))

        self.param_client = self.get_parameter("inference_node").value

        self.listening_confirm = False
        self.listening = False
        self.listen_timer = None
        self.state_machine_flag = "wake_word"

        # Debounce de parciales: p.ej. 2 aciertos seguidos
        self.partial_hits = 0
        self.required_hits = 15

        # 10 ms → menor latencia (160 muestras a 16 kHz)
        self.frame_ms = 10
        self.frame_bytes = int(self.sample_rate / 1000 * self.frame_ms) * 2  # int16 mono

    def state_machine_function(self, msg: String) -> None:
        self.state_machine_flag = msg.data

    def ensure_vosk_model(self) -> str:
        base_dir = Path(__file__).resolve().parent
        model_dir = base_dir / DEFAULT_MODEL_FILENAME_WAKE_WORD
        url = DEFAULT_MODEL_URL_WAKE_WORD 

        if not model_dir.exists():
            zip_path = base_dir / f"{DEFAULT_MODEL_FILENAME_WAKE_WORD}.zip"
            self.get_logger().info(f"[VOSK] Descargando modelo en {zip_path} ...")
            urllib.request.urlretrieve(url, zip_path)

            self.get_logger().info(f"[VOSK] Extrayendo en {base_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(base_dir)
            os.remove(zip_path)
        return str(model_dir)

    def norm(self, s: str) -> str:
        # Normaliza a lower y quita tildes simples
        s = s.lower()
        return (s.replace("á","a").replace("é","e").replace("í","i")
                 .replace("ó","o").replace("ú","u").replace("ü","u"))

    def matches_wake(self, text: str) -> bool:
        t = self.norm(text)
        for v in self.variants:
            if self.norm(v) in t:
                return True
        return False

    def audio_callback(self, msg: Int16MultiArray) -> None:
        if self.state_machine_flag != "wake_word":
            return
        
        audio_bytes = np.array(msg.data, dtype=np.int16).tobytes()

        # Procesa en frames cortos de 10 ms
        fb = self.frame_bytes
        for i in range(0, len(audio_bytes) - fb + 1, fb):
            frame = audio_bytes[i : i + fb]

            # Gate de VAD: si no parece voz, resetea hits y sigue
            if not self.vad.is_speech(frame, self.sample_rate):
                if self.partial_hits > -self.required_hits:
                    self.partial_hits -= 1
                self.rec.AcceptWaveform(frame)
                if (self.listening or self.listening_confirm) and self.partial_hits <= -self.required_hits:
                    self.deactivate_whisper()
                continue

            # Alimenta el recognizer
            # 1) Si Vosk cree que hay “cierre” de palabra/frase
            if self.rec.AcceptWaveform(frame):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").lower().strip()
                if text and self.matches_wake(text):
                    self.get_logger().info(f"[FULL] Wake word: {text!r}")
                    self.confirm_active_whisper()
                    self.partial_hits = 0
                    return
                # Si el full no trae, cae a parciales de nuevo
                self.partial_hits = 0
                if self.listening or self.listening_confirm:
                    self.deactivate_whisper()

            else:
                partial = json.loads(self.rec.PartialResult()).get("partial", "").lower().strip()
                if partial:
                    if self.matches_wake(partial):
                        self.partial_hits += 1
                        if self.partial_hits >= self.required_hits:
                            self.get_logger().info(f"[PARTIAL] Wake word: {partial!r}")
                            self.activate_whisper()
                            self.partial_hits = 0
                            return
                    else:
                        # si el parcial no contiene la clave, resetea el contador
                        self.partial_hits = 0

                            
    
    def confirm_active_whisper(self) -> None:
        if self.listening_confirm:
            return
        self.listening_confirm = True
        self.flag_wake_word.publish(String(data="confirmation"))
        duration = float(self.get_parameter("listen_seconds").value)
        self.listen_timer = self.create_timer(duration, self.deactivate_whisper)

    def activate_whisper(self) -> None:
        if self.listening:
            return
        self.listening = True
        self.flag_wake_word.publish(String(data="active"))
        

    def deactivate_whisper(self) -> None:
        if self.listen_timer is not None:
            self.listen_timer.cancel()
            self.listen_timer = None
        self.flag_wake_word.publish(String(data="deactivate"))
        if self.listening_confirm:
            self.state_machine_publisher.publish(String(data="speech_to_text"))
        self.listening = False
        self.listening_confirm = False

    def _set_inference_active(self, value: bool) -> None:
        param = Parameter("active", Parameter.Type.BOOL, value)
        future = self.param_client.set_parameters([param])
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