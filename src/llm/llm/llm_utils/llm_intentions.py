import re
import unicodedata
from typing import List, Dict, Any

COURTESY_RE = re.compile(
    r"\b(?:"
    r"por\s+favor|porfa(?:vor|s)?|porfis|"
    r"gracias(?:\s+de\s+antemano)?|muchas\s+gracias|please|"
    r"disculp(?:a|ame)|perdon(?:ame)?|"
    r"hola|buen(?:os|as)\s+(?:dias|tardes|noches)|"
    r"me\s+puedes\s+decir|me\s+podrias\s+decir|puedes\s+decirme|podrias\s+decirme|"
    r"puedes|podrias|"
    r"dime|cuentame|indica(?:me)?"
    r")\b"
)

MOV_VERB_RE = re.compile(r"""
(?ix)                          # i: ignorecase, x: verbose
\b(?: 
    # ir / ve
    ve(?:\s+a)? | vete | vayan | vamos | vamonos | ir(?:se)? |
    # desplazamiento basico
    anda(?:r)? | camina(?:r)? | marcha(?:r)? | pasa(?:r)? | cruza(?:r)? |
    # avanzar / retroceder
    a[bv]anza(?:r)? | avanz[ao]? | abanza | retrocede(?:r)? | regresa(?:r)? | vuelve(?:r)? |
    # mover / desplazar
    mueve(?:te)? | mover(?:se)? | desplaza(?:te|r)? |
    # dirigir / orientar / apuntar
    dirigete | dirijete | dirigir(?:se)? | encaminate | encaminar(?:se)? |
    orienta(?:r|rte|rse)? | apunta(?:r)? |
    # giro
    gira(?:r)? | gier[a]? | gir\b | voltea(?:r)? | dobla(?:r)? |
    # ubicar / senalar
    se+n+ala(?:r)? | senala(?:r)? | ubica(?:te|r)? | localiza(?:r)? |
    # destino / rumbo / preps direccionales
    rumbo(?:\s+a)? | hasta | hacia | a\s*donde | adonde | donde(?:\s+queda)? | dondede
)\b
""")

_SPLIT_RE = re.compile(r"""
(?xi)
\b(?: 
    luego | despues | entonces
  | y(?=\s+(?:por\s+favor\s+)?(?: 
        ve(?:\s+a)? | vete | vayan | vamos | vamonos | ir(?:se)?
      | anda(?:r)? | camina(?:r)? | marcha(?:r)? | pasa(?:r)? | cruza(?:r)?
      | a[bv]anza(?:r)? | avanz[ao]? | abanza
      | retrocede(?:r)? | regresa(?:r)? | vuelve(?:r)?
      | mueve(?:te)? | mover(?:se)? | desplaza(?:te|r)?
      | dirigete | dirijete | dirigir(?:se)? | encaminate | encaminar(?:se)?
      | orienta(?:r|rte|rse)? | apunta(?:r)?
      | gira(?:r)? | gier[a]? | gir\b | voltea(?:r)? | dobla(?:r)?
      | se+n+ala(?:r)? | senala(?:r)? | ubica(?:te|r)? | localiza(?:r)?
      | rumbo(?:\s+a)? | hasta | hacia | a\s*donde | adonde | donde(?:\s+queda)? | dondede
    )\b)
)\b
""")

def norm_text(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("ascii")
    s = re.sub(r'[^a-z0-9 ]+',' ', s.lower())
    s = COURTESY_RE.sub(' ', s)
    return re.sub(r'\s+',' ', s).strip()

def is_battery(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(bateria|bateri|battery|%)\b", t))

def is_pose(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(pos[eé]|posicion|amcl|coordenadas|orientacion|heading|yaw|pose[eé])\b", t))

def is_nav(t: str) -> bool:
    t = norm_text(t)
    return bool(MOV_VERB_RE.search(t))

def extract_place_query(t: str) -> str:
    t = norm_text(t) 
    t = re.sub(
        r'^(?:d[oó]nde\s+(?:queda|est[aá])|a\s*d[oó]nde|ad[oó]nde|ve(?:te)?\s+a|'
        r'dir[ií]gete\s+a|ll[ée]vame\s+a|lleva\s+a|ir\s+a|llevar\s+a|camina)(?:\s+a)?\s+',
        '',t,flags=re.I
    )
    
    m = re.search(
        r'\b(?:a|al|a la|en|en la|hacia|hasta)\s+(?:el|la|los|las)?\s*(.+?)\s*$',t,flags=re.I
    )
    place = m.group(1).strip() if m else t.strip()
    
    place = re.compile(r'^(?:el|la|los|las)\s+', flags=re.I).sub('', place).strip(" .,:;!?\"'")
    return place

def _best_hit(res) -> Dict[str, Any]:
    if isinstance(res, list) and res:
        return max((x for x in res if isinstance(x, dict)), key=lambda x: x.get('score', 0.0), default={})
    return res if isinstance(res, dict) else {}

def split_and_prioritize(text: str, kb) -> List[Dict[str, Any]]:
    """
    kb: instancia con método loockup(str) -> dict o list[dict]
    """
    t = norm_text(text)

    parts = _SPLIT_RE.split(t)
    clauses = [p.strip() for p in parts if p and p.strip() not in {"y","luego","despues","entonces"}]

    accions = []
    for c in clauses:
        # 1) Respuestas cortas por KB si hay alta confianza
        var = _best_hit(kb.loockup(c))
        if var.get('answer') and var.get('score', 0.0) >= 0.75:
            accions.append(("corto", "rag", {"data": var['answer'].strip()}))
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
