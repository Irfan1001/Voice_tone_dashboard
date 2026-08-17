#!/usr/bin/env bash
# One-command local start: set up the venv, launch the API, open the dashboard.
#
#   ./run.sh              start the service and open http://127.0.0.1:8000
#   ./run.sh --no-open    start it without opening a browser
#   PORT=9000 ./run.sh    use a different port
#
# Binds to 127.0.0.1 only. The API therefore runs without a key unless you set
# VOICE_API_KEYS; do set one before binding to a public interface.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
VENV="${VENV:-.venv}"
URL="http://127.0.0.1:${PORT}"
OPEN_BROWSER=1

# The comment header below the shebang IS the help text, so they cannot drift.
usage() { sed -n '2,${/^#/!q; s/^# \{0,1\}//p;}' "$0"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-open) OPEN_BROWSER=0 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m/!\\\033[0m %s\n' "$1"; }
die() { printf '\033[1;31mERROR\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- python + venv
if [[ ! -x "$VENV/bin/python" ]]; then
  PY=""
  for cand in python3.12 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      v=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
      case "$v" in 3.1[2-9]) PY="$cand"; break ;; esac
    fi
  done
  [[ -n "$PY" ]] || die "need Python 3.12 or newer (found none). Install it, then re-run."
  say "creating $VENV with $PY ($($PY -V))"
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
fi

PYTHON="$VENV/bin/python"

if ! "$PYTHON" -c 'import uvicorn, torch, pyannote.audio' >/dev/null 2>&1; then
  say "installing dependencies (a few minutes: torch and friends are large)"
  "$PYTHON" -m pip install --quiet -r requirements.txt
fi

# ------------------------------------------------------------------- ffmpeg (m4a)
command -v ffmpeg >/dev/null 2>&1 || \
  warn "ffmpeg not found: wav/ogg/opus/mp3/flac still work, m4a and aac will not."

# ----------------------------------------------------------------- HF token gate
[[ -f .env ]] && set -a && . ./.env && set +a
if [[ -z "${HF_TOKEN:-}${HUGGINGFACE_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  cat <<'EOF'
/!\ HF_TOKEN is not set. Two pyannote models are GATED, and without them speaker
    overlap and customer isolation cannot run - requests will fail rather than
    guess. To fix:

      1. accept the terms on all three pages (same HuggingFace account):
           https://huggingface.co/pyannote/segmentation-3.0
           https://huggingface.co/pyannote/speaker-diarization-3.1
           https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
      2. create a READ token at https://huggingface.co/settings/tokens
      3. echo 'HF_TOKEN=hf_xxx' > .env

EOF
  read -r -p "Start anyway? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

if [[ -z "${VOICE_API_KEYS:-}" ]]; then
  warn "VOICE_API_KEYS unset: the API is open. Fine on 127.0.0.1, not in production."
fi

# ------------------------------------------------------------------------ launch
say "starting the API on $URL  (first request loads ~2 GB of weights: ~20 s)"
"$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$PORT" &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then
    say "up. dashboard: $URL   API docs: $URL/docs"
    if [[ "$OPEN_BROWSER" == 1 ]]; then
      if command -v open >/dev/null 2>&1; then open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
      else say "open $URL in a browser"; fi
    fi
    say "upload data/call_001.ogg to try it. Ctrl-C to stop."
    wait $API_PID
    exit $?
  fi
  kill -0 $API_PID 2>/dev/null || die "the API exited during startup - see the log above."
  sleep 1
done

die "the API did not answer /health within 60 s."
