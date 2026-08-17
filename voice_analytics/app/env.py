"""Minimal .env loader.

Called explicitly from entry points, never at import time, so nothing silently
mutates os.environ. Existing environment variables always win, so
`HF_TOKEN=... python run_pipeline.py ...` overrides the file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def load_env(path: str | Path = ".env", *, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from `path`. Returns the names that were set.

    Never logs values - only names - so a token cannot end up in a log file.
    """
    p = Path(path)
    if not p.is_file():
        return []
    loaded: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        loaded.append(key)
    if loaded:
        log.debug("loaded from %s: %s", p, ", ".join(loaded))
    return loaded
