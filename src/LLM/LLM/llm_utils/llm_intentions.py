import re
import unicodedata
from typing import List, Dict, Any

def norm_text(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("ascii")
    s = re.sub(r'[^a-z0-9 ]+',' ', s.lower())
    return re.sub(r'\s+',' ', s).strip()

def is_battery(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(bateria|bateri|battery|%)\b", t))

def is_pose(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(pos[eé]|posicion|amcl|coordenadas|orientacion|heading|yaw|pose[eé])\b", t))

def is_nav(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(|ve a|ve|gira|giera|ir|orientate|vete|avanza|dirigete|dir[ii]gete|camina|lleva|ir|hacia|hasta|a donde|adonde|vea|donde|queda|ubicacion|orienta|apunta|se[nn]ala)\b", t))

def extract_place_query(text: str) -> str:
    t = norm_text(text)
    t = re.sub(r'^(donde queda|donde esta|a donde|adonde|ve a|vea|vete a|dirigete a|llevame a|dirigete|dir[ii]gete a|ir a|llevar a|lleva a)\s+', '', t)
    m = re.search(r'(?:a|al|a la|en|en la|hacia|hasta)\s+(.+)', t)
    return m.group(1).strip() if m else t

def _best_hit(res) -> Dict[str, Any]:
    if isinstance(res, list) and res:
        return max((x for x in res if isinstance(x, dict)), key=lambda x: x.get('score', 0.0), default={})
    return res if isinstance(res, dict) else {}

def split_and_prioritize(text: str, kb) -> List[Dict[str, Any]]:
    """
    kb: instancia con método loockup(str) -> dict o list[dict]
    """
    t = norm_text(text)
    parts = re.split(r"\b(y|luego|despues|después|entonces)\b", t)
    clauses = [p.strip() for p in parts if p and p.strip() not in {"y","luego","despues","después","entonces"}]

    accions = []
    for c in clauses:
        # 1) Respuestas cortas por KB si hay alta confianza
        var = _best_hit(kb.loockup(c))
        if var.get('answer') and var.get('score', 0.0) >= 0.75:
            accions.append(("corto", "rag", {"answer": var['answer'].strip()}))
            continue

        # 2) Clasificación básica
        if is_battery(c):
            accions.append(("corto","battery",{}))
        elif is_pose(c):
            accions.append(("corto","pose",{}))
        elif is_nav(c):
            accions.append(("largo","navigate",{"data": c}))
        else:
            accions.append(("largo","general",{"data": c}))

    # Primero cortas, luego largas (orden estable preservado)
    accions.sort(key=lambda x: 0 if x[0] == "corto" else 1)
    return [{"kind": k, "params": p} for _, k, p in accions]
