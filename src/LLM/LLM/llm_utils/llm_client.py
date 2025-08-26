from __future__ import annotations
import threading
import os
import json
from typing import Optional, Dict, Any
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
            #os.getenv("OCTOPY_MODEL", "~/llama.cpp/LLM/qwen2.5-3b-instruct-q4_k_m.gguf")
            os.getenv("OCTOPY_MODEL", "~/llama.cpp/LLM/qwen2.5-3b-instruct-q4_k_m.gguf")
        )
        self.ctx = int(os.getenv("OCTOPY_CTX", "1024"))          # contexto razonable
        self.threads = int(os.getenv("OCTOPY_THREADS", str(os.cpu_count() or 4)))
        self.n_batch = int(os.getenv("OCTOPY_N_BATCH", "512"))   # 256–512 bien en CPU
        self.n_gpu_layers = int(os.getenv("OCTOPY_N_GPU_LAYERS", "0"))  # 0 si no hay CUDA
        self.chat_format = os.getenv("OCTOPY_CHAT_FORMAT", "chatml-function-calling").strip()

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
    
    def plan_motion(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        self._ensure()
        system = (
            "Eres un planificador de movimiento para un robot. "
            "Interpreta instrucciones en español y SIEMPRE usa la función plan_motion "
            "para devolver yaw (radianes) y distance (metros). "
            "Convenciones: izquierda yaw negativa, derecha yaw positiva; avanzar → yaw=0; "
            "si el usuario dice 'avanza' sin distancia y no especifica un giro, usa 0.1m. Devuelve números simples."
            "Si el usuario dice 'gira' sin ángulo y no especifica una distancia, usa 90 grados (1.57 rad). "
        )
        messages = [
            {"role":"system","content": system},
            {"role":"user","content": user_prompt},
        ]
        tools = [{
            "type": "function",
            "function": {
                "name": "plan_motion",
                "description": "Devuelve yaw (rad) y distance (m) parseados de la orden.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "yaw":      {"type": "number", "description": "Rotación en rad, izquierda negativa."},
                        "distance": {"type": "number", "minimum": 0, "description": "Distancia en m."},
                        "reason":   {"type": "string"}
                    },
                    "required": ["yaw", "distance"]
                }
            }
        }]
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                top_p=0.8,
                max_tokens=64,
            )
        msg = out["choices"][0]["message"]

        # llama.cpp puede devolver tool_calls o function_call
        tc = msg.get("tool_calls") or []
        if tc:
            fn = (tc[0].get("function") or {})
            if fn.get("name") == "plan_motion":
                args = fn.get("arguments") or "{}"
                return json.loads(args) if isinstance(args, str) else (args or None)

        fc = msg.get("function_call")
        if fc and fc.get("name") == "plan_motion":
            args = fc.get("arguments") or "{}"
            return json.loads(args) if isinstance(args, str) else (args or None)

        return None
