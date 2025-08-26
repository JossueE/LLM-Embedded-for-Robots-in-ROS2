from __future__ import annotations
import json
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
import math 
from difflib import SequenceMatcher
try:
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

from .llm_intentions import norm_text, extract_place_query

@dataclass
class Pose: 
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"
    name: str = ""

class KB:
    def __init__(self, path: str):
        self.items: List[Dict[str,str]] = []
        self.load(path)
    
    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()

            try:
                obj = json.loads(txt)
                items: List[Dict[str,str]] = []
                if isinstance(obj, dict):
                    for _, lst in obj.items():
                        if isinstance(lst, list):
                            for it in lst:
                                ans = it.get('answer','')
                                for trig in it.get('triggers',[]):
                                    items.append({'q': trig, 'a': ans})
                elif isinstance(obj, list):
                    items = obj
                self.items = items
            except json.JSONDecodeError:
                self.items = [json.loads(line) for line in txt.splitlines() if line.strip()]
        except Exception as e:
            self.items = []
    
    def loockup(self, query: str) -> Dict[str]:
        if not self.items:
            return {"error":"kb_vacia","answer":"","score":0.0}
        query = norm_text(query)
        best, best_s = None, 0.0

        for item in self.items:
            q = norm_text(item.get('q',''))
            if q in query or query in q:
                s = 1.0
            else:
                fuzzy = (rf_fuzz.ratio(query, q)/100.0) if _HAS_RF else SequenceMatcher(None, query, q).ratio()
                s = fuzzy
            if s > best_s:
                best, best_s = item, s
        if best and best_s >= 0.75:
            return {"answer": best.get('a',''), "score": round(best_s,3)}
        return {"answer":"","score": round(best_s,3)}




class PosesIndex:
    def __init__(self, path: str):
        self.by_key: Dict[str,Pose] = {}
        self.load(path)

    def load(self, path: str):
        print("[llm_tools] Cargando Poses", flush=True)
        try:
            with open(path,'r',encoding='utf-8') as f:
                data = json.load(f)
            for p in data.get('poses', []):
                pose = Pose(x=p.get('x'), y=p.get('y'), yaw=p.get('yaw_deg',0.0), frame_id=p.get('frame','map'), name=p.get('name',''))
                keys = [p.get('name','')] + p.get('aliases',[])
                for k in keys:
                    nk = norm_text(k)
                    if nk:
                        self.by_key[nk] = pose
        except Exception:
            self.by_key = {}
            print("[llm_tools] Pobemas", flush=True)

    def loockup(self, name: str) -> Dict[str,Any]:
        key = norm_text(extract_place_query(name) or name)
        #print(f"[llm_tools] {self.by_key}", flush=True)
        if key in self.by_key:
            p = self.by_key[key]
            #print(f"[llm_tools] {p}", flush=True)
            return p.__dict__
        # fuzzy simple
        best_k, best_s = None, 0.0
        for k in self.by_key.keys():
            s = (rf_fuzz.ratio(key,k)/100.0) if _HAS_RF else SequenceMatcher(None, key, k).ratio()
        if s > best_s:
            best_k, best_s = k, s
        if best_k and best_s >= 0.70:
            return {**self.by_key[best_k].__dict__, "note":"fuzzy"}
        return {"error":"no_encontrado"}

def quat_to_yaw_deg(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    deg = math.degrees(yaw)
    if deg > 180.0: deg -= 360.0
    if deg <= -180.0: deg += 360.0
    return deg