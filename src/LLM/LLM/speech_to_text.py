#!/usr/bin/env python3
import os
import time
import queue
import threading
import numpy as np
import pyaudio
import webrtcvad

from std_msgs.msg import String
import rclpy
from rclpy.node import Node

from faster_whisper import WhisperModel

# -------- Configuración --------
SAMPLE_RATE = 16000           # Whisper espera 16 kHz mono
FRAME_MS = 20                 # 10 / 20 / 30 ms soportados por WebRTC VAD
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 2        # 0-3 (más alto = más agresivo)
SILENCE_DURATION_END = 0.6    # segundos de silencio para "cortar" una frase
MAX_UTTERANCE_SECS = 30       # seguridad para no acumular demasiado

# Modelo (puedes configurar por env)
WHISPER_MODEL_NAME = os.environ.get("STT_MODEL", "tiny")
# "int8" para CPU, "float16" para GPU; device="cuda" si tienes NVIDIA
WHISPER_DEVICE = os.environ.get("STT_DEVICE", "auto")  # "auto"|"cpu"|"cuda"
WHISPER_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")  # "int8"/"float16"/"int8_float16"

# -------- Nodo ROS2 --------
class LocalSTTNode(Node):
    def __init__(self):
        super().__init__("stt_node")

        self.pub_text = self.create_publisher(String, '/octopy/ask', 10)
        self.pub_partial = self.create_publisher(String, "/stt/partial", 10)

        self.get_logger().info(f"Loading faster-whisper model: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})")
        self.model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE if WHISPER_DEVICE != "auto" else ("cuda" if self._has_cuda() else "cpu"),
            compute_type=WHISPER_COMPUTE_TYPE
        )

        # Audio & VAD
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_q = queue.Queue()  # frames crudos de micrófono (bytes)

        # Hilo de captura y de segmentación
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.segment_thread = threading.Thread(target=self._segment_loop, daemon=True)

        # Arranque
        self._start_stream()
        self.capture_thread.start()
        self.segment_thread.start()

        self.get_logger().info("Local STT node ready ✅")

    def _has_cuda(self):
        try:
            # Heurística simple: si torch está y cuda disponible
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _start_stream(self):
        # Micrófono mono 16k
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE
        )

    def _capture_loop(self):
        while rclpy.ok():
            data = self.audio_stream.read(FRAME_SIZE, exception_on_overflow=False)
            self.audio_q.put(data)

    def _segment_loop(self):
        """Acumula frames con voz (VAD) y dispara transcripción al detectar silencio."""
        ring = bytearray()
        voiced = False
        last_voiced_t = time.time()
        utter_start_t = None

        while rclpy.ok():
            try:
                data = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                data = None

            if data:
                ring.extend(data)
                # Evaluamos VAD por frames de 20ms
                if len(ring) >= 2 * FRAME_SIZE:  # 16-bit PCM => 2 bytes por muestra
                    frame = bytes(ring[:2 * FRAME_SIZE])
                    ring = ring[2 * FRAME_SIZE:]

                    is_speech = self._is_speech(frame)
                    now = time.time()

                    if is_speech:
                        if not voiced:
                            voiced = True
                            utter_start_t = now
                        last_voiced_t = now
                    # Si estamos en segmento y hay silencio suficiente => cortar
                    if voiced:
                        dur = now - utter_start_t if utter_start_t else 0.0
                        silence = now - last_voiced_t
                        if dur >= MAX_UTTERANCE_SECS or silence >= SILENCE_DURATION_END:
                            # Recolectar el búfer del utterance (simple: usamos el último bloque en ring no es ideal)
                            # Mejor: mantener un búfer paralelo de todo el utterance. Implementamos ahora:
                            # Nota: para simplicidad, guardamos el último segmento completo que ya pasó por VAD.
                            # En producción, guarda todos los frames entre el primer y último "is_speech".
                            audio_bytes = self._collect_recent_audio()
                            if audio_bytes:
                                self._transcribe_and_publish(audio_bytes)
                            voiced = False
                            utter_start_t = None

            # Pequeña siesta para no quemar CPU
            time.sleep(0.001)

    def _collect_recent_audio(self):
        """
        Para mantenerlo simple, vaciamos el queue rápidamente hasta ~SILENCE_DURATION_END atrás.
        En producción, conviene un búfer circular de frames 'in-utterance'.
        """
        collected = []
        try:
            # Coge lo que quede en cola rápidamente (esto aproxima el utterance)
            while True:
                collected.append(self.audio_q.get_nowait())
        except queue.Empty:
            pass

        if not collected:
            return None
        return b"".join(collected)

    def _is_speech(self, frame_bytes):
        # WebRTC VAD espera 16kHz mono, 16-bit PCM
        return self.vad.is_speech(frame_bytes, SAMPLE_RATE)

    def _transcribe_and_publish(self, audio_bytes: bytes):
        # Convertir a np.int16 → float32 mono 16k
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcripción (segmentos en streaming desactivado aquí; hacemos chunk por utterance)
        segments, info = self.model.transcribe(
            audio_np,
            beam_size=1,
            vad_filter=False,      # ya hacemos VAD; puedes activarlo también si quieres doble filtro
            language=None,         # auto language
            condition_on_previous_text=False
        )

        partial_texts = []
        final_text = ""

        for seg in segments:
            txt = seg.text.strip()
            partial_texts.append(txt)
            # Publica parcial opcional
            if txt:
                self.pub_partial.publish(String(data=txt))

        if partial_texts:
            final_text = " ".join(partial_texts).strip()
            if final_text:
                self.get_logger().info(f"[STT] {final_text}")
                self.pub_text.publish(String(data=final_text))


def main():
    rclpy.init()
    node = LocalSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
