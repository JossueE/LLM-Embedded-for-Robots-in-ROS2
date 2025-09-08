#!/usr/bin/env bash
set -euo pipefail

# ====== rutas fijas (relativas al propio script) ======
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODELS_FILE="$SCRIPT_DIR/../config/models.yml"   # => src/LLM/config/models.yml
CACHE_DIR="${OCTOPY_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/octopy}"
LANG="${OCTOPY_LANG:-es}"   # es|en|ru...

# ====== helpers ======
have_cmd(){ command -v "$1" >/dev/null 2>&1; }
die(){ echo "ERROR: $*" >&2; exit 1; }
fetch(){
  local url="$1" out="$2"
  if have_cmd curl; then curl -L --fail --retry 3 -o "$out" "$url"
  else have_cmd wget || die "Necesitas curl o wget"; wget -O "$out" "$url"; fi
}

download_file_or_zip(){
  local url="$1"; local out_dir="$2"; local name_hint="${3:-}"
  local fname ext tmp out dname
  fname="$(basename "${url%%\?*}")"     # nombre real de la URL (sin query)
  ext="${fname##*.}"
  mkdir -p "$out_dir"

  if [[ "$ext" == "zip" ]]; then
    # Si ya existe la carpeta descomprimida, no re-descargar
    dname="${name_hint:-${fname%.zip}}"
    if [[ -d "$out_dir/$dname" ]]; then
      echo "  - ya existe: $out_dir/$dname"
      return 0
    fi
    tmp="$out_dir/${name_hint:-pkg}.zip"
    echo "  - bajando ZIP: $url"
    fetch "$url" "$tmp"
    have_cmd unzip || die "Necesitas 'unzip' para extraer ZIP (sudo apt-get install -y unzip)"
    echo "  - descomprimiendo en $out_dir"
    unzip -q -o "$tmp" -d "$out_dir"
    rm -f "$tmp"
    return 0
  fi

  # Archivos normales: por defecto usa el nombre real (con extensión)
  out="$out_dir/$fname"

  # Si el name_hint trae extensión conocida, respétala
  case "${name_hint##*.}" in
    pt|onnx|gguf) out="$out_dir/$name_hint" ;;
  esac

  if [[ -f "$out" ]]; then
    echo "  - ya existe: $out"
  else
    echo "  - bajando: $url -> $out"
    fetch "$url" "$out"
  fi
}


# ====== prerequisitos ======
[[ -f "$MODELS_FILE" ]] || die "No se encontró $MODELS_FILE (esperado en src/LLM/config/models.yml)."
have_cmd yq || die "Falta 'yq'. Instálalo con: sudo apt-get install -y yq"
mkdir -p "$CACHE_DIR"
echo "[*] Usando catálogo: $MODELS_FILE"
echo "[*] Caché: $CACHE_DIR"

# ====== STT (Silero) ======
STT_LEN="$(yq -r '(.stt  // []) | length'  "$MODELS_FILE" 2>/dev/null || echo 0)"
if [[ -n "$STT_LEN" && "$STT_LEN" != "0" ]]; then
  echo "[STT] Descargando modelos STT..."
  for i in $(seq 0 $((STT_LEN-1))); do
    NAME="$(yq -r ".stt[$i].name // \"\"" "$MODELS_FILE")"
    URL="$(yq -r  ".stt[$i].url  // \"\"" "$MODELS_FILE")"
    [[ -n "$URL" && "$URL" != "null" ]] || continue
    download_file_or_zip "$URL" "$CACHE_DIR" "$NAME"
  done
fi


# ====== LLM (opcional; lista en models.yml: llm: [{name, url}, ...]) ======
LLM_LEN="$(yq -r '(.llm  // []) | length'  "$MODELS_FILE" 2>/dev/null || echo 0)"
if [[ -n "$LLM_LEN" && "$LLM_LEN" != "0" ]]; then
  echo "[LLM] Descargando modelos LLM..."
  for i in $(seq 0 $((LLM_LEN-1))); do
    NAME="$(yq -r ".llm[$i].name // \"\"" "$MODELS_FILE")"
    URL="$(yq -r  ".llm[$i].url  // \"\"" "$MODELS_FILE")"
    [[ -n "$URL" && "$URL" != "null" ]] || continue
    download_file_or_zip "$URL" "$CACHE_DIR" "$NAME"
  done
fi

# ====== Vosk (opcional; vosk: [{name, url}, ...]) ======
VOSK_LEN="$(yq -r '(.wake_word // []) | length' "$MODELS_FILE" 2>/dev/null || echo 0)"
if [[ -n "$VOSK_LEN" && "$VOSK_LEN" != "0" ]]; then
  echo "[VOSK] Descargando modelos Vosk..."
  for i in $(seq 0 $((VOSK_LEN-1))); do
    NAME="$(yq -r ".wake_word[$i].name // \"\"" "$MODELS_FILE")"
    URL="$(yq -r  ".wake_word[$i].url  // \"\"" "$MODELS_FILE")"
    [[ -n "$URL" && "$URL" != "null" ]] || continue
    download_file_or_zip "$URL" "$CACHE_DIR" "$NAME"
  done
fi

# ====== TTS (opcional; tts: [{name, url}, ...]) ======
TTS_LEN="$(yq -r '(.tts  // []) | length'  "$MODELS_FILE" 2>/dev/null || echo 0)"
if [[ -n "$TTS_LEN" && "$TTS_LEN" != "0" ]]; then
  echo "[TTS] Descargando modelos TTS..."
  for i in $(seq 0 $((TTS_LEN-1))); do
    NAME="$(yq -r ".tts[$i].name // \"\"" "$MODELS_FILE")"
    URL="$(yq -r  ".tts[$i].url  // \"\"" "$MODELS_FILE")"
    [[ -n "$URL" && "$URL" != "null" ]] || continue
    download_file_or_zip "$URL" "$CACHE_DIR" "$NAME"
  done
fi

echo "OK. Modelos listos en: $CACHE_DIR ✅ "
