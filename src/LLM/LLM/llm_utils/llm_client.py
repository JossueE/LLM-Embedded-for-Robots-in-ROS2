from __future__ import annotations
import threading
import os
from typing import Any, List

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError(
        f"No se pudo importar llama_cpp. Activa venv o instala llama-cpp-python. Detalle: {e}"
    )

class LLM:
    def __init__(self, system_prompt: str | None = None):
        self.system = system_prompt or (
            "Eres Octopy, asistente ROS2. Usa herramientas cuando apliquen.\n"
            "- Batería => 'Mi batería es: XX.X%'.\n"
            "- Pose actual => SOLO JSON {x,y,yaw_deg,frame}.\n"
            "- 'Ve a X' => nav_to_place simulate=false => 'Voy'.\n"
            "- 'Dónde queda X' => simulate=true => 'Por allá'.\n"
            "- Fuera del robot/KB => 'Solo puedo responder sobre el robot y mi base local.'\n"
            "Responde en español, conciso (≤120 palabras)."
        )
        self._llm = None
        self._lock = threading.Lock()

        # Defaults sensatos (CPU-only). Ajusta por env si quieres.
        self.model_path = os.path.expanduser(
            os.getenv("OCTOPY_MODEL", "~/llama.cpp/LLM/qwen2.5-3b-instruct-q4_k_m.gguf")
        )
        self.ctx = int(os.getenv("OCTOPY_CTX", "1024"))          # contexto razonable
        self.threads = int(os.getenv("OCTOPY_THREADS", str(os.cpu_count() or 4)))
        self.n_batch = int(os.getenv("OCTOPY_N_BATCH", "512"))   # 256–512 bien en CPU
        self.n_gpu_layers = int(os.getenv("OCTOPY_N_GPU_LAYERS", "0"))  # 0 si no hay CUDA
        self.chat_format = os.getenv("OCTOPY_CHAT_FORMAT", "").strip()

    def _ensure(self):
        if self._llm is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            kwargs = dict(
                model_path=self.model_path,
                n_ctx=self.ctx,
                n_threads=self.threads,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            self._llm = Llama(**kwargs)

    def answer_general(self, user_prompt: str) -> str:
        self._ensure()
        general_system = (
            "Eres un asistente útil y preciso. Responde en español y de forma concisa (≤120 palabras). "
            "Si la pregunta es ambigua, ofrece la aclaración mínima necesaria y una respuesta probable."
        )
        messages = [
            {"role": "system", "content": general_system},
            {"role": "user", "content": user_prompt},
        ]
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                top_p=0.9,
                max_tokens=100,
            )
        msg = out["choices"][0]["message"]
        return (msg.get("content") or "").strip() or "No tengo una respuesta."
