from __future__ import annotations
import threading
import os
from pathlib import Path
import urllib.request
import json
from typing import Optional, Dict, Any
from typing import Any, List
from rclpy.logging import get_logger

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError(
        f"No se pudo importar llama_cpp. Activa venv o instala llama-cpp-python. Detalle: {e}")

def ensure_stt_model(model_name:str , model_url:str) -> str:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / model_name
    url = model_url
    if not model_dir.exists():
        print(f"[LLM_model] Descargando modelo en {model_dir} ...", flush=True)
        urllib.request.urlretrieve(url, model_dir)
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
        self.model_path = os.path.expanduser(
            os.getenv("OCTOPY_MODEL", model_path)
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
                "Eres el planificador de movimiento de un robot móvil."

                "Tu ÚNICA salida es invocar la herramienta `plan_motion` con los argumentos:"
                "- yaw (float, radianes; izquierda < 0, derecha > 0; avanzar → yaw=0.0)"
                "- distance (float, metros; hacia delante > 0, hacia atrás < 0)"

                "NUNCA respondas con texto libre, explicaciones, ni formato distinto a la invocación de la herramienta."

                "REGLAS DE INTERPRETACIÓN (ESPAÑOL):"
                "1) Extrae, si existen, UNA rotación (yaw) y UNA traslación (distance) de la instrucción."
                "- Si aparecen ambas, la rotación se ejecuta primero y luego la traslación."
                "- Si hay varias órdenes de movimiento, toma la **primera rotación** mencionada y la **primera traslación** mencionada, en ese orden de aparición."
                "- Ignora contenido no relacionado con movimiento (p. ej., “dime…”, “cuéntame…”, “por favor”, “gracias”)."

                "2) Defaults obligatorios:"
                "- “avanza/ve/camina” sin distancia (y sin giro) ⇒ yaw=0.0, distance=0.1"
                "- “gira/voltea” sin ángulo ⇒ |yaw| = 90° = 1.5708 rad (signo por dirección), distance=0.0"
                "- Una orden de solo giro ⇒ distance=0.0"
                "- Una orden de solo traslación ⇒ yaw=0.0"

                "3) Direcciones:"
                "- izquierda ⇒ yaw negativo"
                "- derecha ⇒ yaw positivo"
                "- “retrocede/atrás/hacia atrás” ⇒ distance negativa"

                "4) Unidades y números:"
                "- Distancia en m. Convierte: cm→m, mm→m, km→m. Acepta “m”, “metro(s)”, “centímetro(s)”, “milímetro(s)”, “kilómetro(s)”."
                "- Ángulos: por defecto interpreta “grados”; si se menciona “rad”/“radian(es)”, trata el número como radianes."
                "- Soporta decimales con punto o coma (1.5 = 1,5)."
                "- Convierte números en palabras a números: p. ej., “cuarenta y cinco” = 45; “uno/una”, “medio”=0.5, “un metro y medio”=1.5."
                "- No separes números compuestos por “y” dentro del número (p. ej., “cuarenta y cinco” ≠ 40 y 5; “uno y medio” = 1.5)."

                "5) Expresiones comunes de giro (en grados):"
                "- “media vuelta” ⇒ 180° = 3.14159 rad"
                "- “un cuarto de vuelta” ⇒ 90° = 1.5708 rad"
                "- “tres cuartos de vuelta” ⇒ 270° = 4.71239 rad"
                "- “vuelta completa” ⇒ 360° = 6.28318 rad"
                "- “ligeramente/un poco” ⇒ 10° = 0.17453 rad (si no se da un número)"

                "6) Formato de salida (obligatorio):"
                "- Invoca SIEMPRE la herramienta `plan_motion` con JSON simple y sin texto adicional."
                "- Redondea a 5 decimales como máximo, sin notación científica."

                "7) Seguridad semántica:"
                "- Si la instrucción no contiene ninguna intención de movimiento, aplica defaults solo si encaja (“avanza”/“gira” implícitos). Si no hay intención de movimiento, usa yaw=0.0, distance=0.0."

                "EJEMPLOS (NO los expliques, solo imítalos):"

                "Usuario: avanza"
                "→ plan_motion{ yaw: 0.0, distance: 0.1 }"

                "Usuario: gira a la izquierda"
                "→ plan_motion{ yaw: -1.5708, distance: 0.0 }"

                "Usuario: gira 45 grados a la derecha y avanza 2 metros"
                "→ plan_motion{ yaw: 0.7854, distance: 2.0 }"

                "Usuario: retrocede 30 cm"
                "→ plan_motion{ yaw: 0.0, distance: -0.3 }"

                "Usuario: da media vuelta y avanza un metro y medio"
                "→ plan_motion{ yaw: 3.14159, distance: 1.5 }"

                "Usuario: avanza y gira   (sin números)"
                "→ plan_motion{ yaw: 1.5708, distance: 0.1 }"

                "Usuario: camina 1.2 m"
                "→ plan_motion{ yaw: 0.0, distance: 1.2 }"

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