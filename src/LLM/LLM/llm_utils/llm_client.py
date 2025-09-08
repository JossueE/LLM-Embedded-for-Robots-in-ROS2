from __future__ import annotations
import threading
import os
from pathlib import Path
import urllib.request
import json
from typing import Optional, Dict, Any
from typing import Any

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError(
        f"No se pudo importar llama_cpp. Activa venv o instala llama-cpp-python. Detalle: {e}")

from .config import CONTEXT_LLM,THREADS_LLM,N_BACH_LLM,GPU_LAYERS_LLM,CHAT_FORMAT_LLM

def ensure_stt_model(model_name:str , model_url:str) -> str:
    base_dir = Path(os.environ.get("OCTOPY_CACHE", os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))) / "octopy"
    model_dir = base_dir / model_name
    if not model_dir.exists():
        return f"[LLM_LOADER] Ruta directa no existe: {model_dir}\n"
    return str(model_dir)

class LLM:
    def __init__(self, model_path: str, system_prompt: str | None = None):
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
        self.model_path = os.path.expanduser(os.getenv("OCTOPY_MODEL", model_path))
        self.ctx = int(os.getenv("OCTOPY_CTX", str(CONTEXT_LLM)))         # contexto razonable
        self.threads = int(os.getenv("OCTOPY_THREADS",str(THREADS_LLM)))
        self.n_batch = int(os.getenv("OCTOPY_N_BATCH", str(N_BACH_LLM)))   # 256–512 bien en CPU
        self.n_gpu_layers = int(os.getenv("OCTOPY_N_GPU_LAYERS", str(GPU_LAYERS_LLM)))  # 0 si no hay CUDA
        self.chat_format = os.getenv("OCTOPY_CHAT_FORMAT", CHAT_FORMAT_LLM).strip()

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
            "Si la pregunta requiere números, responde escribiendo el número"
            "EJEMPLO: dos más dos es igual a cuatro"
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
                "Eres el planificador de movimiento. Tu ÚNICA salida es:"
                "plan_motion({yaw: <float>, distance: <float>})"
                "Sin texto extra. Usa punto decimal y ≤5 decimales."
                ""
                "Convenciones"
                "- yaw en rad (izq < 0, der > 0). Avanzar ⇒ yaw=0.0"
                "- distance en m (delante > 0, atrás < 0)"
                ""
                "Extracción (máx. 1 giro + 1 traslación, en el orden dicho)"
                "- Solo giro ⇒ distance=0.0"
                "- Solo traslación ⇒ yaw=0.0"
                "- Varias órdenes ⇒ toma la PRIMERA rotación y la PRIMERA traslación explícitas"
                "- No inventes ni sumes implícitamente"
                ""
                "Defaults (solo si el verbo lo implica)"
                "- “avanza/ve/camina” sin número ⇒ yaw=0.0, distance=0.1"
                "- “gira/voltea” sin ángulo ⇒ |yaw|=1.5708 (signo por dirección), distance=0.0"
                ""
                "Direcciones"
                "- “izquierda” ⇒ yaw<0 ; “derecha” ⇒ yaw>0 ; “retrocede/atrás” ⇒ distance<0"
                ""
                "Números y unidades"
                "- Si hay número (dígitos o palabras) + unidad (“m”, “metros”, “grado/s”), NUNCA uses defaults"
                "- Si faltan unidades: en giros asume GRADOS; en traslación METROS"
                "- Convierte cm/mm/km→m y grados→rad; acepta coma o punto decimal"
                "- Convierte palabras a número: “cuarenta y cinco”=45; “uno y medio”=1.5; “medio”=0.5; “veintidós”=22"
                "- MUY IMPORTANTE: no trates la “y” dentro de un cardinal (“treinta y cuatro”, “cuarenta y cinco”) como separador de órdenes"
                ""
                "Atajos de giro (grados→rad)"
                "- 45°=0.7854 ; 90°=1.5708 ; 180°=3.14159 ; 270°=4.71239 ; 360°=6.28318 ; “ligeramente/un poco”=10°=0.17453"
                ""
                "Ruido a ignorar"
                "- Nombres de lugares no aportan magnitud; aplica defaults solo si el verbo implica moverse"
                "- Si no hay intención de movimiento ⇒ yaw=0.0, distance=0.0"
                ""
                "SALIDA OBLIGATORIA (exacta)"
                "plan_motion({yaw: <float>, distance: <float>})"
                ""
                "Ejemplos (solo la llamada)"
                "- “avanza cuarenta y cinco metros” → plan_motion({yaw: 0.0, distance: 45.0})"
                "- “gira a la izquierda 45 grados y avanza 2 m” → plan_motion({yaw: -0.7854, distance: 2.0})"
                "- “retrocede 0,5 m por favor” → plan_motion({yaw: 0.0, distance: -0.5})"
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