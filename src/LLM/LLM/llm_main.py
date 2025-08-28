from __future__ import annotations
from typing import Optional, Dict, Any
import json
import rclpy
import os
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import PoseWithCovarianceStamped


from .llm_utils.llm_tools import KB, PosesIndex, quat_to_yaw_deg
from .llm_utils.llm_client import LLM
from .llm_utils.llm_router import Router
from .llm_utils.llm_intentions import extract_place_query, norm_text, split_and_prioritize

class OctopyAgent(Node):
    def __init__(self):
        super().__init__('octopy_agent')
        # ROS IO
        self.pub = self.create_publisher(String, '/octopy/answer', 10)
        self.nav_pub = self.create_publisher(String, '/octopy/nav_cmd', 10)
        self.last_amcl: Optional[PoseWithCovarianceStamped] = None
        self.last_batt: Optional[BatteryState] = None
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        self.create_subscription(BatteryState, '/battery_state', self._batt_cb, 10)
        self.create_subscription(String, '/transcript', self._on_ask, 10)
        # Data
        self.kb = KB( os.path.expanduser(os.getenv("OCTOPY_KB", "~/ROS2/Octopy/src/LLM/config/kb.json")))
        self.poses = PosesIndex(os.path.expanduser(os.getenv("OCTOPY_POSES", "~/ROS2/Octopy/src/LLM/config/poses.json")))
        self.llm = LLM()
        self.router = Router(self.kb, self.poses, self.llm, self._tool_get_battery, self._tool_get_current_pose, self._tool_nav_to_place)
        self.get_logger().info('Octopy listo ✅  Publica en /octopy/ask')

    # ---- callbacks ----
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        self.last_amcl = msg

    def _batt_cb(self, msg: BatteryState):
        self.last_batt = msg

    def _on_ask(self, msg: String):      
        text = msg.data.strip()
        try:
            actions = split_and_prioritize(text, self.router.kb)
            for action in actions:
                print(action, flush=True) 
                data = action.get("params", {}).get("data")
                kind = action.get("kind")
                ans = self.router.handle(data,kind)
                if not isinstance(ans, str):
                    ans = json.dumps(ans, ensure_ascii=False)
                self.pub.publish(String(data=ans))        # ← publica ya
                self.get_logger().info(ans)
            return

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            ans = json.dumps({"error": type(e).__name__, "msg": str(e)})
        self.pub.publish(String(data=ans))
        self.get_logger().info(ans)

    # ---- tools ----
    def _tool_get_battery(self) -> Dict[str, Any]:
        if self.last_batt is None or self.last_batt.percentage is None:
            return {"error":"sin_datos_bateria","percentage": None}
        pct = float(self.last_batt.percentage)
        if pct > 1.5: pct /= 100.0
        return {"percentage": round(pct*100.0,1)}

    def _tool_get_current_pose(self) -> Dict[str, Any]:
        if self.last_amcl is None:
            return {"error":"sin_datos_amcl","x":None,"y":None,"yaw_deg":None,"frame":"map"}
        p = self.last_amcl.pose.pose.position
        q = self.last_amcl.pose.pose.orientation
        yaw = quat_to_yaw_deg(q)
        return {"x": round(p.x,3), "y": round(p.y,3), "yaw_deg": round(yaw,1), "frame": "map"}

    def _publish_nav_cmd(self, pose: Dict[str,Any], simulate: bool):
        payload = {"type":"goto","simulate": bool(simulate), "target": {k: pose.get(k) for k in ("x","y","yaw_deg","frame","name")}}
        self.get_logger().info(f"[nav_cmd] simulate={simulate} target={payload['target']}")
        self.nav_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _tool_nav_to_place(self, text: str, simulate: bool=False) -> Dict[str,Any]:
        key = extract_place_query(text)
        pose = self.poses.loockup(key)
        if 'error' in pose: return {"error":"destino_no_encontrado","q": key}
        # auto simulate por intención
        t = norm_text(text)
        is_orient = any(w in t for w in ["donde queda","donde esta","orienta","apunta","senala","senalame","se\u00f1ala","se\u00f1alame"])  # simple
        simulate = True if is_orient else False
        self._publish_nav_cmd(pose, simulate)
        return {"ok": True, "simulate": simulate, "name": pose.get("name"), "target": pose}

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


if __name__ == '__main__':
    main()