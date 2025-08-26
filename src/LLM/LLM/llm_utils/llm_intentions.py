import re
import unicodedata

stop_word = set("el la los las un una unos unas de del al que cual cuales como donde cuando por para con segun sobre a en y o u es son eres".split())

def norm_text(s: str) -> str:
    # Normaliza a lower, quita tildes y signos de puntuación
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("ascii")
    s = re.sub(r'[^a-z0-9 ]+',' ', s.lower())  # quita puntuación
    return re.sub(r'\s+',' ', s).strip()

def is_battery(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(bateria|bateri|battery|%)\b", t))

def is_pose(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(pos[eé]|posicion|amcl|coordenadas|orientacion|heading|yaw|pose[eé])\b", t))

def is_nav(t: str) -> bool:
    t = norm_text(t)
    return bool(re.search(r"\b(ve|vete|dirigete|dir[ii]gete|camina|lleva|ir|hacia|hasta|a donde|adonde|donde|queda|ubicacion|orienta|apunta|se[nn]ala)\b", t))

def extract_place_query(text: str) -> str:
    t = norm_text(text)
    t = re.sub(r'^(donde queda|donde esta|a donde|adonde|ve a|vete a|dirigete a|dirigete|dir[ii]gete a|ir a|llevar a|lleva a)\s+', '', t)
    m = re.search(r'(?:a|al|a la|en|en la|hacia|hasta)\s+(.+)', t)
    return m.group(1).strip() if m else t