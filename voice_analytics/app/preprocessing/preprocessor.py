"""Preprocessing: any audio file -> canonical AudioData.

    decode -> capture source metadata -> MEASURE CLIPPING -> downmix
    -> resample -> normalise

The order matters: normalising rescales the waveform and destroys the clipping
evidence, and resampling erases the native sample rate, which is itself a genuine
audio-quality signal.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from ..config import AudioConfig, QualityConfig
from ..errors import AudioLoadError
from ..schema import AudioData
from ..signal_utils import clipping_run_ratio

# soundfile (bundled libsndfile) handles these with no system dependency.
NATIVE_EXT = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif", ".au"}
FFMPEG_EXT = {".m4a", ".aac", ".mp4", ".wma", ".webm", ".amr", ".3gp", ".mkv", ".mov"}
SUPPORTED_EXT = NATIVE_EXT | FFMPEG_EXT


def _decode_ffmpeg(path: Path, sr: int) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioLoadError(
            "decode_error", f"ffmpeg failed on {path.name}: {proc.stderr.decode()[:200]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


class Preprocessor:
    """Turns a path into AudioData. One responsibility, no analysis."""

    def __init__(self, audio: AudioConfig, quality: QualityConfig):
        self.audio = audio
        self.quality = quality

    def run(self, path: str | Path) -> AudioData:
        path = Path(path)
        if not path.is_file():
            raise AudioLoadError("not_found", f"not a file: {path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise AudioLoadError(
                "unsupported_format", f"unsupported extension: {ext or '(none)'}"
            )

        raw, native_sr, meta = self._decode(path, ext)
        if raw.size == 0:
            raise AudioLoadError("empty_audio", f"{path.name} decoded to zero samples")

        # --- measured BEFORE any gain change
        peak = float(np.abs(raw).max())
        mono_pre = librosa.to_mono(raw.T) if raw.ndim == 2 else raw
        clip_ratio = self._clipping(mono_pre)

        # --- channel independence: dual-mono destroys the strongest overlap cue
        dual_mono = False
        corr: float | None = None
        if raw.ndim == 2 and raw.shape[1] > 1:
            a, b = raw[:, 0], raw[:, 1]
            if a.std() > 0 and b.std() > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
                dual_mono = corr > 0.999
            else:
                dual_mono = bool(np.array_equal(a, b))

        # --- downmix, resample, normalise
        mono = mono_pre
        if native_sr != self.audio.sample_rate:
            mono = librosa.resample(
                mono, orig_sr=native_sr, target_sr=self.audio.sample_rate
            )
        mono = np.ascontiguousarray(mono, dtype=np.float32)

        new_peak = float(np.abs(mono).max())
        if new_peak > 0:
            mono = (mono * (self.audio.normalize_peak / new_peak)).astype(np.float32)

        duration = len(mono) / self.audio.sample_rate
        if duration < self.audio.min_duration_s:
            raise AudioLoadError(
                "empty_audio", f"{path.name} is only {duration:.2f}s of audio"
            )

        meta.update(
            {
                "filename": path.name,
                "native_sample_rate": native_sr,
                "source_peak": round(peak, 4),
                "clipping_run_ratio": clip_ratio,
                "dual_mono": dual_mono,
                "channel_correlation": corr,
                "size_bytes": path.stat().st_size,
                "normalized_to_peak": self.audio.normalize_peak,
            }
        )
        return AudioData(
            waveform=mono,
            sample_rate=self.audio.sample_rate,
            duration=duration,
            metadata=meta,
        )

    def _clipping(self, mono_pre: np.ndarray) -> float:
        q = self.quality
        return clipping_run_ratio(mono_pre, q.clip_level, q.clip_min_run,
                                  q.clip_peak_fraction)

    def _decode(self, path: Path, ext: str) -> tuple[np.ndarray, int, dict[str, Any]]:
        meta: dict[str, Any] = {"decoder": "soundfile"}
        try:
            info = sf.info(str(path))
            meta.update(
                {"container": info.format, "codec": info.subtype or "unknown",
                 "channels": info.channels}
            )
            raw, native_sr = sf.read(str(path), always_2d=True, dtype="float32")
            return raw, native_sr, meta
        except Exception as sf_err:
            if not shutil.which("ffmpeg"):
                if ext in FFMPEG_EXT:
                    raise AudioLoadError(
                        "unsupported_format",
                        f"{ext} requires ffmpeg and ffmpeg is not installed",
                    ) from sf_err
                raise AudioLoadError(
                    "decode_error",
                    f"could not decode {path.name} ({sf_err}); file may be corrupt "
                    "or truncated",
                ) from sf_err
            mono, native_sr = _decode_ffmpeg(path, self.audio.sample_rate)
            meta.update({"decoder": "ffmpeg", "container": ext.lstrip("."),
                         "codec": "unknown", "channels": 1})
            return mono[:, None], native_sr, meta
