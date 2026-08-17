"""Feature extraction stage: builds one AudioFeatures for the whole pipeline.

Order is fixed and meaningful:
  VAD first (so "speech" has a single definition everywhere)
  -> acoustic features computed against that mask
  -> overlapped-speech detection (optional, gated)

Anything unavailable becomes None with a recorded reason; nothing here raises.
"""

from __future__ import annotations

from ..config import Config
from ..schema import AudioData, AudioFeatures
from .acoustic import AcousticExtractor
from .overlap_detect import OverlapDetector
from .roles import CustomerIsolator
from .vad import VoiceActivityDetector


class FeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vad = VoiceActivityDetector(cfg.audio, cfg.silence)
        self.acoustic = AcousticExtractor(cfg.audio, cfg.quality, cfg.silence)
        self.overlap = OverlapDetector(
            cfg.models.segmentation, cfg.models.hf_token, cfg.device, cfg.seed
        )
        self.isolator = CustomerIsolator(
            cfg.models.diarization, cfg.models.role_asr, cfg.models.hf_token,
            cfg.device, cfg.seed, cfg.rules.role_opening_s, cfg.rules.role_min_margin
        )

    def run(self, audio: AudioData) -> AudioFeatures:
        x, sr = audio.waveform, audio.sample_rate
        n = int(sr * self.cfg.audio.frame_ms / 1000)
        n_frames = max(1, (len(x) - n) // n + 1)
        frame_s = n / sr

        segments = self.vad.segments(x, sr)
        mask = self.vad.frame_mask(segments, n_frames, frame_s)

        spectral, dsp, speech_ratio = self.acoustic.compute(
            x, sr, mask, float(audio.metadata.get("clipping_run_ratio", 0.0))
        )

        overlap = self.overlap.measure(x, sr)
        iso = self.isolator.run(x, sr)

        extra = {
            "vad_backend": self.vad.backend,
            "overlap_available": overlap is not None,
            "overlap_reason": self.overlap.reason,
            "overlap_seconds": None if overlap is None else round(overlap[0], 3),
            "overlap_speech_seconds": None if overlap is None else round(overlap[1], 3),
            "source": dict(audio.metadata),
            # Empty means the role was not resolved and emotion covers ALL speech.
            "customer_spans": iso.customer_spans,
            "customer_isolation": iso.to_evidence(),
        }
        return AudioFeatures(
            spectral=spectral,
            speech_ratio=speech_ratio,
            vad_segments=segments,
            dsp=dsp,
            extra=extra,
        )
