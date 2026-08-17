FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY app/ ./app/
COPY api/ ./api/
COPY run_pipeline.py ./

# Runtime configuration
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_DATASETS_CACHE=/app/.cache/huggingface \
    VOICE_UPLOAD_DIR=/data \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Directories
RUN mkdir -p /app/.cache/huggingface /data

# Non-root user
RUN useradd -u 10001 -m appuser \
    && chown -R appuser:appuser /app /data

USER appuser

# Railway supplies PORT dynamically.
EXPOSE 8000

# Healthcheck
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=180s \
    --retries=5 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health \
        | grep -qE '"status": ?"(ok|loading)"'

# One Uvicorn worker.
# Your Worker already creates one AudioPipeline instance.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]