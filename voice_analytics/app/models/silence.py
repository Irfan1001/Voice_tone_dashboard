"""Long-silence predictor, driven by the shared VAD segments.

Reports the measured longest silence; the rule engine applies the threshold.

"Dead air" means absence of SPEECH, so a stretch filled with background noise
is not silence. The measurement therefore requires both a non-speech VAD verdict
and energy well below the speech level.
"""

from __future__ import annotations

from ..config import Config
from ..schema import AudioData, AudioFeatures, Prediction
from .base import register


@register("silence", "vad")
class VadSilencePredictor:
    name = "silence:vad"
    slot = "silence"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict(self, audio: AudioData, features: AudioFeatures) -> Prediction:
        d = features.dsp
        threshold = self.cfg.silence.long_silence_s

        # Longest gap between consecutive speech segments, as a cross-check on
        # the frame-level energy measurement.
        gaps: list[float] = []
        segs = sorted(features.vad_segments, key=lambda s: s.start)
        prev_end = 0.0
        for seg in segs:
            if seg.start > prev_end:
                gaps.append(seg.start - prev_end)
            prev_end = max(prev_end, seg.end)
        if audio.duration > prev_end:
            gaps.append(audio.duration - prev_end)
        longest_gap = max(gaps) if gaps else 0.0

        # Distance from the threshold drives confidence: a 10 s silence is a far
        # safer call than a 4.1 s one.
        margin = abs(d.longest_silence_s - threshold)
        confidence = 0.60 + 0.30 * min(1.0, margin / 3.0)

        return Prediction(
            prediction={
                "longest_silence_s": round(d.longest_silence_s, 2),
                "longest_vad_gap_s": round(longest_gap, 2),
            },
            confidence=round(confidence, 3),
            metadata={
                "threshold_s": threshold,
                "speech_ratio": round(features.speech_ratio, 3),
                "pause_count": d.pause_count,
                "mean_pause_s": round(d.mean_pause_s, 2),
                "vad_backend": features.extra.get("vad_backend"),
                "n_speech_segments": len(segs),
            },
        )
