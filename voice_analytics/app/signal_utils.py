"""Run-length helpers shared by preprocessing and feature extraction."""

from __future__ import annotations

import numpy as np


def runs(mask: np.ndarray) -> list[int]:
    """Lengths of each consecutive True run."""
    out: list[int] = []
    run = 0
    for v in mask:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def run_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, length) for each consecutive True run."""
    spans: list[tuple[int, int]] = []
    start = -1
    for i, v in enumerate(mask):
        if v and start < 0:
            start = i
        elif not v and start >= 0:
            spans.append((start, i - start))
            start = -1
    if start >= 0:
        spans.append((start, len(mask) - start))
    return spans


def clipping_run_ratio(
    x: np.ndarray, level: float, min_run: int, peak_fraction: float
) -> float:
    """Fraction of samples sitting in flat-topped runs at the signal's ceiling.

    A loud sample is not clipping - lossy codecs legitimately overshoot +-1.0 - so
    consecutive samples pinned at a ceiling are required. And the ceiling is
    relative to the signal's own peak, because clipping followed by gain reduction
    leaves flat tops well below full scale; the absolute rail is only a floor.

    Must be called BEFORE normalisation; rescaling destroys the evidence.
    """
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak <= 0:
        return 0.0
    ceiling = min(level, peak * peak_fraction)
    hot = np.abs(x) >= ceiling
    if not hot.any():
        return 0.0
    return float(sum(r for r in runs(hot) if r >= min_run)) / len(x)
