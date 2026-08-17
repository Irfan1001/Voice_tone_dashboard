"""Voice activity detection via Silero VAD.

VAD runs once here and every downstream consumer uses the same speech mask, so
"speech" means one thing across the pipeline. There is deliberately no fallback
detector: a cruder VAD would shift every downstream number at once while the
output still looked normal, so a load failure raises instead.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import AudioConfig, SilenceConfig
from ..errors import VadUnavailableError
from ..schema import Segment

log = logging.getLogger(__name__)

_MODEL = None  # loaded once per process


def _load_silero():
    global _MODEL
    if _MODEL is None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_grad_enabled(False)
        _MODEL = load_silero_vad()
        _MODEL.eval()
    return _MODEL


class VoiceActivityDetector:
    def __init__(self, audio: AudioConfig, silence: SilenceConfig):
        self.audio = audio
        self.silence = silence
        self.backend = "unknown"

    def segments(self, x: np.ndarray, sr: int) -> list[Segment]:
        try:
            import torch
            from silero_vad import get_speech_timestamps

            model = _load_silero()
            with torch.inference_mode():
                stamps = get_speech_timestamps(
                    torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                    model,
                    sampling_rate=sr,
                    threshold=self.silence.vad_threshold,
                    min_speech_duration_ms=self.silence.min_speech_ms,
                    min_silence_duration_ms=self.silence.min_silence_ms,
                    return_seconds=True,
                )
        except Exception as exc:
            log.error("Silero VAD failed: %s: %s", type(exc).__name__, exc,
                      exc_info=True)
            raise VadUnavailableError(
                f"Silero VAD could not run ({type(exc).__name__}: {exc}). There is "
                "no fallback detector by design. Check that `silero-vad` and torch "
                "are installed and importable."
            ) from exc

        self.backend = "silero"
        segments = [Segment(float(s["start"]), float(s["end"])) for s in stamps]
        if not segments:
            log.warning("Silero VAD found no speech at all in this clip")
        return segments

    def frame_mask(self, segments: list[Segment], n_frames: int, frame_s: float
                   ) -> np.ndarray:
        """Convert time segments into a frame-level boolean mask."""
        mask = np.zeros(n_frames, dtype=bool)
        for seg in segments:
            lo = max(0, int(seg.start / frame_s))
            hi = min(n_frames, int(np.ceil(seg.end / frame_s)))
            if hi > lo:
                mask[lo:hi] = True
        return mask
