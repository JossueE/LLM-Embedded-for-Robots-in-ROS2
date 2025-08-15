#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OctopyAgent (versión saneada) — KB → Tools → Cultura General
- ROS2 + llama.cpp con tool-calling robusto y rutas de intención deterministas.
- Flujo solicitado:
    1) Compara la entrada con kb.json y responde si hay match (≥ umbral).
    2) Si no hay match, activa tools (forzando cuando aplique).
    3) Si tools no dan respuesta útil, envía a cultura general (LLM sin tools).

Tópicos:
  • Suscribe: /octopy/ask (std_msgs/String)
  • Publica:  /octopy/answer (std_msgs/String)
  • Lee:      /amcl_pose (geometry_msgs/PoseWithCovarianceStamped),
              /battery_state (sensor_msgs/BatteryState)
  • Publica comandos de navegación: /octopy/nav_cmd (std_msgs/String)

Parámetros (ROS2 o variables de entorno):
  - model (OCTOPY_MODEL): ruta al .gguf
  - threads (OCTOPY_THREADS)
  - ctx (OCTOPY_CTX)
  - chat_format (OCTOPY_CHAT_FORMAT): p.ej. "llama-3", "qwen2", "mistral-instruct",
    "chatml-function-calling" (solo si tu .gguf lo requiere). Si dudas, deja vacío.
  - kb_path (OCTOPY_KB)
  - poses_path (OCTOPY_POSES)
  - system_prompt (OCTOPY_SYSTEM_PROMPT)
"""

import os
import re
import json
import math
import threading
import unicodedata
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import PoseWithCovarianceStamped
from difflib import SequenceMatcher

# --- Llama.cpp (venv del proyecto) ---
import site, sys
site.addsitedir(os.path.expanduser('~/ROS2/Octopy/.venv/lib/python3.10/site-packages'))
print("USING PY:", sys.executable, flush=True)
try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError(f"No se pudo importar llama_cpp. Activa venv o instala llama-cpp-python. Detalle: {e}")

# ---------- Config ----------
KB_THRESHOLD = 0.75  # score mínimo para aceptar respuesta de KB


# -------------------- Utilidades --------------------
def quat_to_yaw_deg(q) -> float:
    """Convierte un cuaternión a yaw en grados en [-180, 180]."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    deg = math.degrees(yaw)
    if deg > 180.0:
        deg -= 360.0
    if deg <= -180.0:
        deg += 360.0
    return deg


def norm_text(s: str) -> str:
    """Normaliza texto: quita acentos, minúsculas, deja [a-z0-9 ] y colapsa espacios."""
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# -------------------- Nodo --------------------
class OctopyAgent(Node):
    def __init__(self):
        super().__init__('octopy_agent')

        # Parámetros
        self.declare_parameter('system_prompt', os.environ.get('OCTOPY_SYSTEM_PROMPT',
            "Eres Octopy, asistente para un robot ROS2. Usa herramientas siempre que apliquen.\n"
            "Reglas:\n"
            "- Batería: llama get_battery y responde 'Mi batería es: XX.X%'.\n"
            "- Pose actual: llama get_current_pose y devuelve SOLO JSON {x,y,yaw_deg,frame}.\n"
            "- Ir a lugar ('ve a X', 'dirígete a X'): llama nav_to_place simulate=false y responde 'Voy'.\n"
            "- Orientar ('dónde queda X', 'señala X'): nav_to_place simulate=true y responde 'Por allá'.\n"
            "- Preguntas fuera del robot/KB: responde 'Solo puedo responder sobre el robot y mi base local.'\n"
            "Responde en español, conciso (<=120 palabras)."
        ))

        self.declare_parameter('model', os.environ.get('OCTOPY_MODEL', '~/llama.cpp/LLM/Llama-3.2-3B-Instruct-Q4_0.gguf'))
        self.declare_parameter('threads', int(os.environ.get('OCTOPY_THREADS', str(os.cpu_count() or 4))))
        self.declare_parameter('ctx', int(os.environ.get('OCTOPY_CTX', '2048')))
        self.declare_parameter('kb_path', os.environ.get('OCTOPY_KB', '~/ROS2/Octopy/src/LLM/config/kb.json'))
        self.declare_parameter('poses_path', os.environ.get('OCTOPY_POSES', '~/ROS2/Octopy/src/LLM/config/poses.json'))
        self.declare_parameter('chat_format', os.environ.get('OCTOPY_CHAT_FORMAT', ''))  # vacío => auto si la build lo permite

        self._system_prompt: str = self.get_parameter('system_prompt').get_parameter_value().string_value
        model_path: str = os.path.expanduser(self.get_parameter('model').get_parameter_value().string_value)
        n_threads: int = int(self.get_parameter('threads').get_parameter_value().integer_value)
        ctx: int = int(self.get_parameter('ctx').get_parameter_value().integer_value)
        kb_path: str = os.path.expanduser(self.get_parameter('kb_path').get_parameter_value().string_value)
        poses_path: str = os.path.expanduser(self.get_parameter('poses_path').get_parameter_value().string_value)
        chat_format: str = self.get_parameter('chat_format').get_parameter_value().string_value.strip()

        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("Define 'model' (o OCTOPY_MODEL) con la ruta a tu .gguf")

        # LLM
        self.get_logger().info(f'Inicializando LLM: {model_path} (threads={n_threads}, ctx={ctx}, chat_format={chat_format or "auto"})')
        llm_kwargs = dict(model_path=model_path, n_ctx=ctx, n_threads=n_threads, n_gpu_layers=-1, n_batch = 1024)
        if chat_format:
            llm_kwargs['chat_format'] = chat_format
        self._llm = Llama(**llm_kwargs)
        self._llm_lock = threading.Lock()

        # ROS I/O
        self._pub = self.create_publisher(String, '/octopy/answer', 10)
        self._nav_pub = self.create_publisher(String, '/octopy/nav_cmd', 10)
        self._last_amcl: Optional[PoseWithCovarianceStamped] = None
        self._last_batt: Optional[BatteryState] = None

        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        self.create_subscription(BatteryState, '/battery_state', self._batt_cb, 10)
        self.create_subscription(String, '/octopy/ask', self._on_ask, 10)

        # KB / Poses
        self._kb_items: List[Dict[str, str]] = []
        self._poses: List[Dict[str, Any]] = []
        self._pose_by_key: Dict[str, Dict[str, Any]] = {}
        self._load_kb(kb_path)
        self._load_poses(poses_path)

        self.get_logger().info('Octopy listo ✅  Publica en /octopy/ask')

    # -------------------- Callbacks --------------------
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        self._last_amcl = msg

    def _batt_cb(self, msg: BatteryState):
        self._last_batt = msg
        self.get_logger().info(f"[battery] raw percentage={msg.percentage!r}")

    def _on_ask(self, msg: String):
        text = msg.data.strip()
        self.get_logger().info(f"ASK: {text}")
        try:
            answer = self._route_kb_tools_general(text)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            answer = {"error": type(e).__name__, "msg": str(e)}

        if answer is None:
            answer = ""
        if not isinstance(answer, str):
            try:
                answer = json.dumps(answer, ensure_ascii=False)
            except Exception:
                answer = str(answer)

        preview = answer[:200] + ("…" if len(answer) > 200 else "")
        self.get_logger().info(f"ANS: {preview}")
        self._pub.publish(String(data=answer))

    # -------------------- Tools (implementación) --------------------
    def _tool_get_battery(self) -> Dict[str, Any]:
        if self._last_batt is None or self._last_batt.percentage is None:
            return {"error": "sin_datos_bateria", "percentage": None}
        pct = float(self._last_batt.percentage)
        if pct > 1.5:
            pct /= 100.0
        return {"percentage": round(pct * 100.0, 1)}

    def _tool_get_current_pose(self) -> Dict[str, Any]:
        if self._last_amcl is None:
            return {"error": "sin_datos_amcl", "x": None, "y": None, "yaw_deg": None, "frame": "map"}
        p = self._last_amcl.pose.pose.position
        q = self._last_amcl.pose.pose.orientation
        yaw = quat_to_yaw_deg(q)
        return {"x": round(p.x, 3), "y": round(p.y, 3), "yaw_deg": round(yaw, 1), "frame": "map"}

    def _tool_kb_lookup(self, q: str) -> Dict[str, Any]:
        if not self._kb_items:
            return {"error": "kb_vacia", "answer": "", "score": 0.0}
        qn = norm_text(q)
        stops = set("el la los las un una unos unas de del al que cual cuales como donde cuando por para con segun sobre a en y o u es son eres".split())
        q_tokens = [t for t in qn.split() if t not in stops]
        best, best_s = None, 0.0
        for it in self._kb_items:
            tn = norm_text(it.get("q", ""))
            if tn in qn or qn in tn:
                s = 1.0
            else:
                t_tokens = [t for t in tn.split() if t not in stops]
                inter = len(set(q_tokens) & set(t_tokens))
                union = len(set(q_tokens) | set(t_tokens)) or 1
                jacc = inter / union
                fuzzy = SequenceMatcher(None, qn, tn).ratio()
                s = max(jacc, fuzzy)
            if s > best_s:
                best, best_s = it, s
        if best and best_s >= KB_THRESHOLD:
            return {"answer": best.get("a", ""), "score": round(best_s, 3)}
        return {"answer": "", "score": round(best_s, 3)}

    def _tool_lookup_named_pose(self, name: str) -> Dict[str, Any]:
        if not name:
            return {"error": "nombre_vacio"}
        key = norm_text(self._extract_place_query(name) or name)
        # 1) Exacto
        if key in self._pose_by_key:
            pose = self._pose_by_key[key]
            return {"x": pose.get("x"), "y": pose.get("y"), "yaw_deg": pose.get("yaw_deg", 0.0), "frame": pose.get("frame", "map"), "name": pose.get("name")}
        # 2) Contención (clave más larga)
        for k in sorted(self._pose_by_key.keys(), key=len, reverse=True):
            if k in key:
                pose = self._pose_by_key[k]
                return {"x": pose.get("x"), "y": pose.get("y"), "yaw_deg": pose.get("yaw_deg", 0.0), "frame": pose.get("frame", "map"), "name": pose.get("name")}
        # 3) Fuzzy >= 0.70
        best_k, best_s = None, 0.0
        for k in self._pose_by_key.keys():
            s = SequenceMatcher(None, key, k).ratio()
            if s > best_s:
                best_k, best_s = k, s
        if best_k and best_s >= 0.70:
            pose = self._pose_by_key[best_k]
            return {"x": pose.get("x"), "y": pose.get("y"), "yaw_deg": pose.get("yaw_deg", 0.0), "frame": pose.get("frame", "map"), "name": pose.get("name"), "note": "fuzzy"}
        return {"error": "no_encontrado"}

    def _publish_nav_cmd(self, pose: Dict[str, Any], simulate: bool):
        payload = {
            "type": "goto",
            "simulate": bool(simulate),
            "target": {
                "x": pose["x"],
                "y": pose["y"],
                "yaw_deg": pose.get("yaw_deg", 0.0),
                "frame": pose.get("frame", "map"),
                "name": pose.get("name", ""),
            },
        }
        self.get_logger().info(f"[nav_cmd] simulate={simulate} target={payload['target']}")
        self._nav_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _tool_nav_to_place(self, text: str, simulate: bool = False) -> Dict[str, Any]:
        t = norm_text(text)
        is_orient = re.search(r"\b(donde queda|donde esta|dónde queda|dónde está|senala|senalame|orienta|apunta|se[nñ]ala|se[nñ]alame)\b", t) is not None
        is_go = re.search(r"\b(ve a|vete a|dirigete a|dir[ií]gete a|ir a|camina a|lleva a|llevar a|hasta|hacia)\b", t) is not None
        if is_orient:
            simulate = True
        elif is_go:
            simulate = False
        place = self._extract_place_query(text)
        pose = self._tool_lookup_named_pose(place)
        if "error" in pose:
            return {"error": "destino_no_encontrado", "q": place}
        self._publish_nav_cmd(pose, simulate)
        return {"ok": True, "simulate": simulate, "name": pose.get("name"), "target": pose}

    # -------------------- Loaders --------------------
    def _load_kb(self, path: str):
        p = os.path.expanduser(path)
        if not os.path.exists(p):
            self.get_logger().warning(f"KB no encontrada: {p}")
            return
        try:
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
            try:
                obj = json.loads(txt)
                items: List[Dict[str, str]] = []
                if isinstance(obj, dict):
                    for _, lst in obj.items():
                        if isinstance(lst, list):
                            for it in lst:
                                ans = it.get("answer", "")
                                for trig in it.get("triggers", []):
                                    items.append({"q": trig, "a": ans})
                elif isinstance(obj, list):
                    items = obj
                self._kb_items = items
                self.get_logger().info(f"KB (JSON) cargada: {len(self._kb_items)} entradas desde {p}")
            except json.JSONDecodeError:
                items = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
                self._kb_items = items
                self.get_logger().info(f"KB (JSONL) cargada: {len(self._kb_items)} entradas desde {p}")
        except Exception as e:
            self.get_logger().error(f"Error leyendo KB: {e}")
            self._kb_items = []

    def _load_poses(self, path: str):
        p = os.path.expanduser(path)
        if not os.path.exists(p):
            self.get_logger().warning(f"Poses no encontrado: {p}")
            return
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._poses = data.get("poses", [])
            self._pose_by_key.clear()
            for pose in self._poses:
                keys = [pose.get("name", "")] + pose.get("aliases", [])
                for k in keys:
                    nk = norm_text(k)
                    if nk:
                        self._pose_by_key[nk] = pose
            self.get_logger().info(f"Poses cargados: {len(self._poses)} lugares, {len(self._pose_by_key)} claves desde {p}")
        except Exception as e:
            self.get_logger().error(f"Error leyendo poses: {e}")
            self._poses = []
            self._pose_by_key.clear()

    # -------------------- Tools spec & dispatch --------------------
    def _tools_spec(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "get_current_pose", "description": "Devuelve {x,y,yaw_deg,frame} desde /amcl_pose.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "lookup_named_pose", "description": "Busca lugar conocido y devuelve {x,y,yaw_deg,frame,name}.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
            {"type": "function", "function": {"name": "kb_lookup", "description": "Consulta la KB local y devuelve {'answer': str, 'score': float}.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
            {"type": "function", "function": {"name": "get_battery", "description": "Devuelve {'percentage': 0..100} desde /battery_state.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "nav_to_place", "description": "Resuelve destino en texto y publica comando de navegación.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "simulate": {"type": "boolean"}}, "required": ["text"]}}},
        ]

    def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_current_pose":
            return self._tool_get_current_pose()
        if name == "lookup_named_pose":
            return self._tool_lookup_named_pose(args.get("name", ""))
        if name == "kb_lookup":
            return self._tool_kb_lookup(args.get("q", ""))
        if name == "get_battery":
            return self._tool_get_battery()
        if name == "nav_to_place":
            return self._tool_nav_to_place(args.get("text", ""), bool(args.get("simulate", False)))
        return {"error": "tool_desconocida", "name": name}

    # -------------------- LLM helpers --------------------
    def _llm_chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, max_tokens=100, top_p=0.8):
        with self._llm_lock:
            kwargs = dict(messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            return self._llm.create_chat_completion(**kwargs)

    def _answer_general(self, user_prompt: str) -> str:
        """Respuestas de cultura general (sin tools)."""
        general_system = (
            "Eres un asistente útil y preciso. Responde en español y de forma concisa (≤120 palabras). "
            "Si la pregunta es ambigua, ofrece la aclaración mínima necesaria y una respuesta probable."
        )
        messages = [
            {"role": "system", "content": general_system},
            {"role": "user", "content": user_prompt},
        ]
        out = self._llm_chat(messages, tools=None, tool_choice="auto", temperature=0.2, top_p=0.9, max_tokens=100)
        msg = out["choices"][0]["message"]
        return (msg.get("content") or "").strip() or "No tengo una respuesta."

    # -------------------- Orquestador: KB → Tools → General --------------------
    def _route_kb_tools_general(self, user_prompt: str) -> str:
        """
        1) KB -> si score >= KB_THRESHOLD, devuelve KB.
        2) TOOLS -> intenta resolver con herramientas (detección por regex + LLM con tools).
        3) GENERAL -> si lo anterior falla, responde con cultura general (LLM sin tools).
        """
        # (1) KB primero
        kb_res = self._tool_kb_lookup(user_prompt)
        kb_ans = (kb_res or {}).get("answer", "").strip()
        kb_score = float((kb_res or {}).get("score", 0.0) or 0.0)
        self.get_logger().info(f"[route] KB score={kb_score:.3f}")
        if kb_ans and kb_score >= KB_THRESHOLD:
            return kb_ans

        # (2) Tools después
        tools = self._tools_spec()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Heurística de intención: fuerza tool específica cuando hay señales claras
        t = norm_text(user_prompt)
        is_batt = re.search(r"\b(bateria|bateri|battery|%)\b", t)
        is_pose = re.search(r"\b(pose|posicion|amcl|coordenadas|orientacion|heading|yaw)\b", t)
        is_nav  = re.search(r"\b(ve|vete|dirigete|dirigeme|camina|lleva|ir|hacia|hasta|a donde|adonde|donde|queda|ubicacion|orienta|apunta|senala|se[nñ]ala)\b", t)

        forced_tool_choice: Any = None
        if is_batt:
            forced_tool_choice = {"type": "function", "function": {"name": "get_battery"}}
        elif is_pose:
            forced_tool_choice = {"type": "function", "function": {"name": "get_current_pose"}}
        elif is_nav:
            forced_tool_choice = {"type": "function", "function": {"name": "nav_to_place"}}

        out = self._llm_chat(messages, tools=tools, tool_choice=(forced_tool_choice if forced_tool_choice else "auto"),
                             temperature=0.0, top_p=0.8, max_tokens=100)
        msg = out["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")
        function_call = msg.get("function_call")
        content = (msg.get("content") or "").strip()

        # Si forzamos tool y no hubo tool_call, damos respuesta segura del dominio robot o pasamos a general
        if forced_tool_choice and not (tool_calls or function_call):
            self.get_logger().warning("LLM no emitió tool_call pese a forzarlo (ruta tools fallida).")
            if is_batt:
                return "Aún no tengo lectura de batería."
            if is_pose:
                return "Aún no tengo pose de AMCL."
            if is_nav:
                return "No encuentro ese destino."
            return self._answer_general(user_prompt)

        # Si hay tool_call, ejecuta UNA tool y compone determinísticamente
        if tool_calls or function_call:
            if tool_calls:
                tc = tool_calls[0]
                name = (tc.get("function") or {}).get("name", "")
                args_raw = (tc.get("function") or {}).get("arguments", "{}")
            else:
                fc = function_call or {}
                name = fc.get("name", "")
                args_raw = fc.get("arguments", "{}")
            try:
                fn_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if not isinstance(fn_args, dict):
                    fn_args = {}
            except Exception:
                fn_args = {}

            result = self._dispatch_tool(name, fn_args)

            # Composición determinista
            if name == "nav_to_place":
                if (result or {}).get("ok"):
                    return "Por allá" if (result or {}).get("simulate") else "Voy"
                return "No encuentro ese destino."

            if name == "get_battery":
                pct = (result or {}).get("percentage")
                if isinstance(pct, (int, float)):
                    return f"Mi batería es: {pct:.1f}%"
                return "Aún no tengo lectura de batería."

            if name == "get_current_pose":
                if any((result or {}).get(k) is None for k in ("x", "y", "yaw_deg")):
                    return "Aún no tengo pose de AMCL."
                payload = {
                    "x": (result or {}).get("x"),
                    "y": (result or {}).get("y"),
                    "yaw_deg": (result or {}).get("yaw_deg"),
                    "frame": (result or {}).get("frame", "map"),
                }
                return json.dumps(payload, ensure_ascii=False)

            if name == "lookup_named_pose":
                if "error" in (result or {}):
                    return "No encuentro ese destino."
                payload = {
                    "x": result.get("x"),
                    "y": result.get("y"),
                    "yaw_deg": result.get("yaw_deg"),
                    "frame": result.get("frame", "map"),
                    "name": result.get("name", ""),
                }
                return json.dumps(payload, ensure_ascii=False)

            if name == "kb_lookup":
                ans = (result or {}).get("answer", "")
                return ans.strip() if ans else "No tengo esa información en mi base local."

            return "He llamado una herramienta, pero no pude componer la respuesta."

        # Si el LLM devolvió texto sin tools: según el requerimiento, pasa a cultura general
        self.get_logger().info("[route] Tools no aplican o no dieron respuesta -> cultura general")
        if content:
            # Si el texto parece fuera de dominio del robot, preferimos cultura general igualmente
            return self._answer_general(user_prompt)
        return self._answer_general(user_prompt)

    # -------------------- Extracción de lugar --------------------
    def _extract_place_query(self, text: str) -> str:
        t = norm_text(text)
        t = re.sub(r'^(donde queda|donde esta|a donde|adonde|ve a|vete a|dirigete a|dirigete|dir[ií]gete a|ir a|llevar a|lleva a)\s+', '', t)
        m = re.search(r'(?:a|al|a la|en|en la|hacia|hasta)\s+(.+)', t)
        if m:
            return m.group(1).strip()
        return t


# -------------------- main --------------------
def main():
    rclpy.init()
    node = OctopyAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Saliendo por Ctrl+C')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
