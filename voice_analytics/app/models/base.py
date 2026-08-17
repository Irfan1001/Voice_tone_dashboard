"""Predictor plugin contract and registry.

Adding or swapping a predictor is one class plus one @register line.

The division is strict: predictors MEASURE and return raw scores; the rule engine
DECIDES and alone turns scores into schema enum values. So a predictor never emits
`"emotional_tone": "upset"` - it emits arousal and valence, and every threshold
lives in one place.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from ..config import Config
from ..schema import AudioData, AudioFeatures, Prediction

SLOTS = ("emotion", "noise", "quality", "silence", "overlap")


class Predictor(Protocol):
    name: str
    slot: str

    def predict(self, audio: AudioData, features: AudioFeatures) -> Prediction: ...


def run_predictor(p: Predictor, audio: AudioData, features: AudioFeatures) -> Prediction:
    """Time a predictor and guarantee it never raises into the pipeline.

    A failed predictor yields `available=False` with the reason attached. The rule
    engine then raises IncompletePredictionError for the clip rather than filling
    the gap - see app/errors.py.
    """
    t0 = time.perf_counter()
    try:
        out = p.predict(audio, features)
    except Exception as exc:
        out = Prediction(
            prediction={},
            confidence=0.0,
            metadata={
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "predictor": p.name,
            },
        )
    out.latency_ms = (time.perf_counter() - t0) * 1000.0
    out.metadata.setdefault("available", True)
    out.metadata.setdefault("predictor", p.name)
    return out


REGISTRY: dict[str, dict[str, Callable[[Config], Predictor]]] = {s: {} for s in SLOTS}


def register(slot: str, name: str) -> Callable[[type], type]:
    if slot not in REGISTRY:
        raise KeyError(f"unknown slot {slot!r}; expected one of {SLOTS}")

    def deco(cls: type) -> type:
        REGISTRY[slot][name] = lambda cfg: cls(cfg)  # type: ignore[call-arg]
        return cls

    return deco


def get_predictor(slot: str, name: str, cfg: Config) -> Predictor:
    try:
        return REGISTRY[slot][name](cfg)
    except KeyError:
        have = ", ".join(sorted(REGISTRY.get(slot, {}))) or "(none registered)"
        raise KeyError(f"no {slot} predictor {name!r}; available: {have}") from None
