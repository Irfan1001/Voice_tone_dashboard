"""Speaker-overlap predictor, reading the shared overlapped-speech measurement.

Reports measured seconds and ratio; the rule engine applies the threshold. When
the detector is unavailable (gated model, no token, load failure) this reports
`available=False` rather than guessing a boolean.
"""

from __future__ import annotations

from ..config import Config
from ..schema import AudioData, AudioFeatures, Prediction
from .base import register


@register("overlap", "segmentation")
class SegmentationOverlapPredictor:
    name = "overlap:segmentation"
    slot = "overlap"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict(self, audio: AudioData, features: AudioFeatures) -> Prediction:
        seconds = features.extra.get("overlap_seconds")
        if not features.extra.get("overlap_available") or seconds is None:
            return Prediction(
                prediction={},
                confidence=0.0,
                metadata={
                    "available": False,
                    "reason": features.extra.get("overlap_reason")
                            or "overlapped-speech detection unavailable",
                    "hint": "set HF_TOKEN and accept the terms for "
                            "pyannote/segmentation-3.0 (the model is gated)",
                },
            )

        speech_time = float(features.extra.get("overlap_speech_seconds") or 0.0)
        ratio = seconds / speech_time if speech_time > 0 else 0.0
        return Prediction(
            prediction={
                "overlap_seconds": round(float(seconds), 2),
                "overlap_ratio": round(ratio, 4),
            },
            confidence=0.75,
            metadata={
                "speech_time_s": round(speech_time, 2),
                "detector": "pyannote/segmentation-3.0 powerset (2+ speakers/frame)",
                "dual_mono_source": audio.metadata.get("dual_mono"),
            },
        )
