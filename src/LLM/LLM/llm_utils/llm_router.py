from __future__ import annotations
import json
import math
from typing import Any
from rclpy.logging import get_logger
from .llm_intentions import is_battery, is_pose, is_nav

class Router:
    def __init__(self, kb, poses, llm, tool_get_batt, tool_get_pose, tool_nav):
        self.kb = kb
        self.poses = poses
        self.llm = llm
        self.tool_get_batt = tool_get_batt
        self.tool_get_pose = tool_get_pose
        self.tool_nav = tool_nav

    def handle(self, user_prompt: str) -> str:

        k = self.kb.loockup(user_prompt)

        # 1) KB
        if k.get('answer') and k.get('score',0.0) >= 0.75:
            print("[llm_router] hola mundo", flush=True)
            return k['answer'].strip()
        # 2) INTENTS (sin LLM)
        if is_battery(user_prompt):
            r = self.tool_get_batt()
            pct = r.get('percentage')
            return f"Mi batería es: {pct:.1f}%" if isinstance(pct,(int,float)) else "Aún no tengo lectura de batería."
        if is_pose(user_prompt):
            r = self.tool_get_pose()
            if any(r.get(k) is None for k in ('x','y','yaw_deg')):
                return "Aún no tengo pose de AMCL."
            return json.dumps({k:r.get(k) for k in ('x','y','yaw_deg','frame')}, ensure_ascii=False)
        if is_nav(user_prompt):
            r = self.tool_nav(user_prompt)
            if r.get("ok"):
                return "Por allá" if r.get("simulate") else "Voy"
            else:
                plan = self.llm.plan_motion(user_prompt)
                if plan:
                    yaw, dist = _clamp_motion(plan.get("yaw", 0.0), plan.get("distance", 0.0))
                    return json.dumps({"yaw": yaw, "distance": dist}, ensure_ascii=False)
                return "No encontré ese destino ni entiendo la orden."
            
        # 3) General (entra al modelo UNA sola vez)
        return self.llm.answer_general(user_prompt)

def _clamp_motion(yaw: float, dist: float,
    max_abs_yaw: float = 2*math.pi,
    max_dist_m: float = 10.0) -> tuple[float, float]:
    # límites duros para evitar órdenes absurdas
    y = max(-max_abs_yaw, min(max_abs_yaw, float(yaw)))
    d = max(0.0, min(max_dist_m, float(dist)))
    return y, d