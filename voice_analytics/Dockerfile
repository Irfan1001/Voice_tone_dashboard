# Voice Analytics API. CPU only - every model here runs on CPU by design.
FROM python:3.12-slim AS base

# ffmpeg is the fallback decoder for m4a/aac; soundfile handles the rest.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY api/ ./api/
COPY run_pipeline.py ./

# Weights (~2 GB) are baked in at BUILD time so a new container serves immediately
# and a HuggingFace outage cannot stop a deployment. pyannote models are gated, so
# the token must be present:
#
#   docker build --secret id=hf,env=HF_TOKEN -t voice-analytics .
RUN --mount=type=secret,id=hf,required=false \
    HF_TOKEN="$(cat /run/secrets/hf 2>/dev/null || true)" python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
token = os.environ.get("HF_TOKEN") or None
if not token:
    print("WARNING: no HF_TOKEN at build time. The gated pyannote models are NOT "
          "baked in and the first request will fail with incomplete_prediction. "
          "Rebuild with --secret id=hf,env=HF_TOKEN.")
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
for mid in ("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            "MIT/ast-finetuned-audioset-10-10-0.4593"):
    AutoFeatureExtractor.from_pretrained(mid); AutoModel.from_pretrained(mid)
    print("baked", mid)
AutoProcessor.from_pretrained("openai/whisper-tiny.en")
print("baked whisper-tiny.en")
if token:
    from pyannote.audio import Model, Pipeline
    Model.from_pretrained("pyannote/segmentation-3.0", token=token)
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    print("baked pyannote models")
PY

# OMP/MKL pinned to 1 for the same reason torch is: parallel float reductions reorder
# and change low-order bits, breaking byte-identical determinism. Scale with more
# CONTAINERS, never more threads.
ENV HF_HOME=/app/.cache/huggingface \
    VOICE_UPLOAD_DIR=/data \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN mkdir -p /data && useradd -u 10001 -m appuser \
    && chown -R appuser /data /app/.cache
USER appuser

EXPOSE 8000

# start-period covers the ~18 s weight load; a shorter grace period kills healthy
# containers during startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -qE '"status": ?"(ok|loading)"'

# One worker per container: a second would load another 2.9 GB of identical weights
# and contend for the same single torch thread.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
