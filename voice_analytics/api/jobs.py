"""Job queue and a single sequential worker.

One worker, not a pool, for two measured reasons: a loaded pipeline holds ~2.9 GB
RSS, so four in-process copies would need ~12 GB; and torch is pinned to one thread
to keep the pipeline byte-for-byte deterministic, so extra threads would queue
behind it anyway. Scale with more containers, each with its own pipeline.

Jobs live in memory, so a restart loses queued, running and finished work. That is a
deliberate trade for a demo service; nothing here assumes the store is local, so
moving it to Postgres or Redis is a contained change.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Item:
    """One audio file inside a job. A batch job has many; a single job has one."""

    name: str
    path: Path
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float | None = None
    wall_s: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result,
            "error": self.error,
            "duration_s": self.duration_s,
            "wall_s": self.wall_s,
            "warnings": self.warnings,
        }


@dataclass
class Job:
    id: str
    kind: str                      # "single" | "batch"
    items: list[Item]
    # Where this job's audio lives. Set BEFORE the job is queued: the worker may start
    # the instant submit() returns, so moving files afterwards is a race.
    work_dir: Path | None = None
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.result is not None or i.error is not None)

    def summary(self) -> dict[str, Any]:
        total_audio = sum(i.duration_s or 0.0 for i in self.items)
        total_wall = sum(i.wall_s or 0.0 for i in self.items)
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": str(self.state),
            "total": len(self.items),
            "completed": self.completed,
            "failed": sum(1 for i in self.items if i.error),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "audio_seconds": round(total_audio, 1),
            "wall_seconds": round(total_wall, 1),
            "realtime_factor": (round(total_audio / total_wall, 2)
                                if total_wall > 0 else None),
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "items": [i.to_dict() for i in self.items]}


class JobStore:
    """Thread-safe job registry plus the queue feeding the worker."""

    def __init__(self, max_jobs: int = 500):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self.max_jobs = max_jobs

    def submit(self, kind: str, items: list[Item],
               work_dir: Path | None = None) -> Job:
        """Queue a job. Every item's path must already be final and readable."""
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, items=items, work_dir=work_dir)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Bound memory: forget the oldest FINISHED jobs first, never a live one.
            while len(self._order) > self.max_jobs:
                for i, jid in enumerate(self._order):
                    if self._jobs[jid].state in (JobState.DONE, JobState.FAILED):
                        self._order.pop(i)
                        self._jobs.pop(jid, None)
                        break
                else:
                    break
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 25) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order[-limit:])]

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def next_job(self, timeout: float = 1.0) -> Job | None:
        try:
            job_id = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return self.get(job_id)


class Worker(threading.Thread):
    """Drains the queue with ONE pipeline instance, loaded once and reused.

    Built lazily on the first job rather than at import, so the process can answer
    /health while the weights are still loading. `ready` and `load_error` distinguish
    "starting" from "broken".
    """

    daemon = True

    def __init__(self, store: JobStore, keep_uploads: bool = False):
        super().__init__(name="pipeline-worker")
        self.store = store
        self.keep_uploads = keep_uploads
        self.ready = False
        self.load_error: str | None = None
        self.processed = 0
        self._pipeline = None
        self._stop = threading.Event()

    def _get_pipeline(self):
        if self._pipeline is None:
            from app.config import DEFAULT
            from app.pipeline import AudioPipeline
            self._pipeline = AudioPipeline(DEFAULT)
            self.ready = True
        return self._pipeline

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._get_pipeline()
        except Exception as exc:               # a broken install must be visible
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        while not self._stop.is_set():
            job = self.store.next_job(timeout=1.0)
            if job is None:
                continue
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        job.state = JobState.RUNNING
        job.started_at = time.time()
        try:
            pipeline = self._get_pipeline()
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = f"pipeline unavailable: {type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            return

        for item in job.items:
            if self._stop.is_set():
                break
            t0 = time.perf_counter()
            try:
                result, diag = pipeline.run(str(item.path))
                item.result = result.to_dict()
                item.duration_s = round(diag.duration_s, 2)
                item.warnings = list(diag.warnings)
            except Exception as exc:
                # One bad file must not fail the other 499 in a batch.
                item.error = f"{type(exc).__name__}: {exc}"
            item.wall_s = round(time.perf_counter() - t0, 2)
            self.processed += 1
            if not self.keep_uploads:
                item.path.unlink(missing_ok=True)

        job.state = (JobState.FAILED
                     if all(i.error for i in job.items) else JobState.DONE)
        job.finished_at = time.time()
