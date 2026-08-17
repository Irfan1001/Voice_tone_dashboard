"""Voice Analytics HTTP service: submit audio, poll, collect the nine-field JSON.

    GET  /health              liveness + whether the models finished loading
    POST /v1/analyze          one audio file  -> 202 with a job id
    POST /v1/batch            a ZIP of audio  -> 202 with a job id
    GET  /v1/jobs             recent jobs
    GET  /v1/jobs/{id}        status, progress, per-file results
    GET  /v1/jobs/{id}/csv    results as CSV
    GET  /                    dashboard

Submissions are asynchronous because the pipeline runs at 0.8x realtime: a blocking
endpoint would sit past most proxy timeouts. Clients get a job id and poll.

Auth is an `X-API-Key` header compared with `secrets.compare_digest`, from the
comma-separated `VOICE_API_KEYS`. Unset means OPEN mode - it starts anyway and says
so in the logs and in `/health`, because failing closed would block local
development. Never run it unset on a public interface.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.jobs import Item, JobState, JobStore, Worker  # noqa: E402
from api.uploads import (  # noqa: E402
    UploadRejected,
    extract_zip,
    is_audio,
    safe_name,
    save_stream,
)
from app.env import load_env  # noqa: E402
from app.schema import FIELD_ORDER  # noqa: E402

log = logging.getLogger("voice_api")
load_env()

UPLOAD_ROOT = Path(os.environ.get("VOICE_UPLOAD_DIR", tempfile.gettempdir())) / "voice_uploads"
KEEP_UPLOADS = os.environ.get("VOICE_KEEP_UPLOADS", "").lower() in ("1", "true", "yes")
API_KEYS = [k.strip() for k in os.environ.get("VOICE_API_KEYS", "").split(",") if k.strip()]
STARTED_AT = time.time()

store = JobStore()
worker = Worker(store, keep_uploads=KEEP_UPLOADS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    if not API_KEYS:
        log.warning("VOICE_API_KEYS is not set - the API is OPEN to anyone who can "
                    "reach this port. Set it before exposing the service.")
    worker.start()
    yield
    worker.stop()


app = FastAPI(
    title="Voice Analytics API",
    version="1.0",
    summary="Nine-field call analysis: emotion, background noise, quality, overlap, silence.",
    lifespan=lifespan,
)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    """Constant-time key check. No-op when no keys are configured (open mode)."""
    if not API_KEYS:
        return
    if not x_api_key or not any(
            secrets.compare_digest(x_api_key, k) for k in API_KEYS):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    """Liveness AND readiness, kept distinct: a worker still loading weights is not
    broken, and one that failed to load is not merely slow.
    """
    return {
        "status": "error" if worker.load_error else ("ok" if worker.ready else "loading"),
        "models_loaded": worker.ready,
        "load_error": worker.load_error,
        "queue_depth": store.queue_depth(),
        "jobs_processed": worker.processed,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "auth": "enabled" if API_KEYS else "OPEN - VOICE_API_KEYS is unset",
    }


def _work_dir() -> Path:
    """A fresh directory per request, created BEFORE the job is queued.

    Named from a random token, not the job id: the id only exists after submit(), by
    which point the worker may already be reading the files.
    """
    d = UPLOAD_ROOT / f"job_{secrets.token_hex(8)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post("/v1/analyze", status_code=202, dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)) -> dict:
    """Queue one audio file. Returns 202 and a job id; poll /v1/jobs/{id}."""
    name = safe_name(file.filename or "upload")
    if not is_audio(name):
        raise HTTPException(
            status_code=415,
            detail=f"{name!r} is not a supported audio file. Send wav, ogg, opus, "
                   "mp3, flac, m4a, aac or webm.")
    work = _work_dir()
    final = work / name
    try:
        save_stream(file.file, final)
    except UploadRejected as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    # Submitted only once the file is in place and complete.
    job = store.submit("single", [Item(name=name, path=final)], work_dir=work)
    return {**job.summary(), "poll": f"/v1/jobs/{job.id}"}


@app.post("/v1/batch", status_code=202, dependencies=[Depends(require_key)])
async def batch(file: UploadFile = File(...)) -> dict:
    """Queue every audio file inside a ZIP. One bad file fails only itself."""
    name = safe_name(file.filename or "batch.zip")
    tmp = UPLOAD_ROOT / f"upload_{secrets.token_hex(8)}.zip"
    try:
        save_stream(file.file, tmp)
    except UploadRejected as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    # Extract straight into the final directory - nothing moves after submit.
    work = _work_dir()
    try:
        extracted = extract_zip(tmp, work)
    except UploadRejected as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    items = [Item(name=n, path=p) for n, p in extracted.files]
    job = store.submit("batch", items, work_dir=work)
    return {
        **job.summary(),
        "source": name,
        "skipped": extracted.skipped,
        "poll": f"/v1/jobs/{job.id}",
    }


@app.get("/v1/jobs", dependencies=[Depends(require_key)])
def list_jobs(limit: int = 25) -> dict:
    return {"jobs": [j.summary() for j in store.recent(min(limit, 100))]}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id (jobs are held in "
                                                   "memory and lost on restart)")
    return job.to_dict()


def csv_cell(value: object) -> str:
    """Render one cell. Booleans are lowercased so the CSV matches the JSON output
    rather than Python's `repr`, which a spreadsheet reads as text."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


@app.get("/v1/jobs/{job_id}/csv", dependencies=[Depends(require_key)])
def get_job_csv(job_id: str) -> PlainTextResponse:
    """Flat CSV of the nine fields, one row per file - for spreadsheets and BI."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    header = ["name", "duration_s", "wall_s", *FIELD_ORDER, "error", "warnings"]
    rows = [",".join(header)]
    for item in job.items:
        res = item.result or {}
        cells = [item.name, str(item.duration_s or ""), str(item.wall_s or "")]
        cells += [csv_cell(res.get(f, "")) for f in FIELD_ORDER]
        cells += [item.error or "", " | ".join(item.warnings)]
        rows.append(",".join('"' + c.replace('"', '""') + '"' for c in cells))
    return PlainTextResponse("\n".join(rows) + "\n", media_type="text/csv")


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def delete_job(job_id: str) -> dict:
    """Drop a finished job's uploaded audio. Running jobs are refused, not truncated."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job.state in (JobState.QUEUED, JobState.RUNNING):
        raise HTTPException(status_code=409,
                            detail=f"job is {job.state}; wait for it to finish")
    if job.work_dir:
        shutil.rmtree(job.work_dir, ignore_errors=True)
    return {"job_id": job.id, "audio_deleted": True}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    page = Path(__file__).with_name("dashboard.html")
    if not page.is_file():
        return HTMLResponse("<h1>Voice Analytics API</h1>"
                            "<p>Dashboard file missing. See <a href='/docs'>/docs</a>.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))
