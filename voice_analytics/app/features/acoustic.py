"""Acoustic feature computation: pitch, energy, spectral character, SNR, defects.

All deterministic and CPU-only.

The structurally important choice: noise statistics come from NON-SPEECH regions
(and from per-bin low-percentile tracking), while `bandwidth_99_hz` is measured on
SPEECH frames only. Keeping them apart is what lets the noise and quality
predictors reach independent conclusions - measuring bandwidth over the whole
signal lets low-frequency noise drag it down and leak noise into the quality
verdict.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
from scipy import signal as sps
from scipy import stats

from ..config import AudioConfig, QualityConfig, SilenceConfig
from ..schema import DspMetrics
from ..signal_utils import longest_run, run_spans, runs

log = logging.getLogger(__name__)

# 3 kHz separates the speech band from where impulsive noise lives. [principled]
HF_HIGHPASS_HZ = 3000.0
HF_ENVELOPE_FRAME_S = 0.010
PITCH_MIN_HZ = 60.0
PITCH_MAX_HZ = 400.0
MIN_PAUSE_S = 0.3


def _echo_prominence(x: np.ndarray, sr: int, cfg: QualityConfig) -> float:
    """Median cepstral peak prominence in the echo-delay quefrency band.

    A delayed copy appears as a cepstral peak at the delay. The cepstrum is what
    separates echo delay (20-400 ms) from voice pitch (2.5-12.5 ms); an
    envelope-autocorrelation detector instead measures syllabic rhythm and fires
    on clean speech.
    """
    win = int(1.0 * sr)
    hop = win // 2
    lo, hi = int(cfg.echo_quefrency_min_s * sr), int(cfg.echo_quefrency_max_s * sr)
    if len(x) < win or hi <= lo + 10:
        return 0.0
    w = np.hanning(win)
    vals: list[float] = []
    for i in range(0, len(x) - win, hop):
        seg = x[i : i + win]
        if np.sqrt((seg * seg).mean()) < 1e-4:
            continue
        spec = np.abs(np.fft.rfft(seg * w)) + 1e-12
        ceps = np.abs(np.fft.irfft(np.log(spec)))
        band = ceps[lo:hi]
        if len(band) < 10:
            continue
        med = float(np.median(band))
        mad = float(np.median(np.abs(band - med))) + 1e-12
        vals.append((float(band.max()) - med) / mad)
    return float(np.median(vals)) if vals else 0.0


def _dropout_count(x: np.ndarray, sr: int, speech: np.ndarray, frame_n: int,
                   min_ms: float) -> int:
    """Runs of exact digital silence inside speech - the packet-loss signature."""
    min_len = int(sr * min_ms / 1000.0)
    if min_len < 1:
        return 0
    count = 0
    for start, length in run_spans(x == 0.0):
        if length < min_len:
            continue
        f = start // frame_n
        if f < len(speech) and speech[max(0, f - 1) : f + 2].any():
            count += 1
    return count


class AcousticExtractor:
    """Computes all deterministic acoustic features for one clip."""

    def __init__(self, audio: AudioConfig, quality: QualityConfig,
                 silence: SilenceConfig):
        self.audio = audio
        self.quality = quality
        self.silence = silence

    def compute(
        self, x: np.ndarray, sr: int, speech_mask: np.ndarray, clipping_ratio: float
    ) -> tuple[dict[str, float], DspMetrics, float]:
        """Returns (spectral_scalars, dsp_metrics, speech_ratio)."""
        notes: dict[str, str] = {}
        n = int(sr * self.audio.frame_ms / 1000)
        frame_s = n / sr
        n_frames = max(1, (len(x) - n) // n + 1)

        frames = np.stack([x[i * n : i * n + n] for i in range(n_frames)])
        rms_db = 20.0 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)

        speech = self._align(speech_mask, n_frames)

        # Deterministic YIN, not probabilistic pYIN. No try/except: zeros here
        # would be indistinguishable from "no voiced speech detected".
        f0 = librosa.yin(x, fmin=PITCH_MIN_HZ, fmax=PITCH_MAX_HZ, sr=sr,
                         frame_length=1024)
        voiced = np.isfinite(f0) & (f0 > PITCH_MIN_HZ) & (f0 < PITCH_MAX_HZ)

        # ---- levels and the two noise estimators
        speech_db = rms_db[speech] if speech.any() else rms_db
        nonspeech_db = rms_db[~speech] if (~speech).any() else rms_db
        speech_level = float(np.median(speech_db))
        noise_floor = float(np.median(nonspeech_db))
        stationary_snr = speech_level - noise_floor

        S = np.abs(librosa.stft(x, n_fft=512, hop_length=n)) ** 2
        noise_bins = np.percentile(S, 10, axis=1)
        speech_bins = np.percentile(S, 90, axis=1)
        tracked_snr = float(
            10.0 * np.log10((speech_bins.sum() + 1e-12) / (noise_bins.sum() + 1e-12))
        )

        # ---- silence / pauses
        silent = (~speech) & (rms_db < speech_level + self.silence.silence_rel_db)
        longest_silence_s = longest_run(silent) * frame_s
        pause_runs = [r for r in runs(~speech) if r * frame_s >= MIN_PAUSE_S]
        mean_pause = float(np.mean(pause_runs) * frame_s) if pause_runs else 0.0

        # ---- spectral character of NON-SPEECH regions
        ns_idx = np.where(~speech)[0]
        if len(ns_idx) >= 3:
            ns_seg = np.concatenate([x[i * n : (i + 1) * n] for i in ns_idx])
        else:
            ns_seg = x
            notes["nonspeech"] = (
                "fewer than 3 non-speech frames: noise statistics fall back to "
                "the whole clip and are UNRELIABLE for this file"
            )
            log.warning("no usable non-speech frames; noise stats are unreliable")
        psd = (np.abs(librosa.stft(ns_seg, n_fft=512)) ** 2).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        hf_ratio = float(psd[freqs > self.quality.telephony_bw_hz].sum()
                         / (float(psd.sum()) + 1e-20))

        # ---- bandwidth on SPEECH frames only (keeps noise out of quality)
        sp_idx = np.where(speech)[0]
        if len(sp_idx) >= 3:
            sp_seg = np.concatenate([x[i * n : (i + 1) * n] for i in sp_idx])
        else:
            sp_seg = x
            notes["bandwidth"] = (
                "fewer than 3 speech frames: bandwidth measured on the whole "
                "clip and is UNRELIABLE for this file"
            )
            log.warning("no usable speech frames; bandwidth is unreliable")
        psd_sp = (np.abs(librosa.stft(sp_seg, n_fft=2048)) ** 2).mean(axis=1)
        fq = librosa.fft_frequencies(sr=sr, n_fft=2048)
        cum = np.cumsum(psd_sp) / (psd_sp.sum() + 1e-20)
        bandwidth_99 = float(fq[min(int(np.searchsorted(cum, 0.99)), len(fq) - 1)])

        # ---- impulsiveness above the speech band
        sos = sps.butter(6, HF_HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
        hf = sps.sosfilt(sos, x)
        m = int(sr * HF_ENVELOPE_FRAME_S)
        hf_n = max(1, (len(hf) - m) // m + 1)
        hf_env = np.sqrt(
            (np.stack([hf[i * m : i * m + m] for i in range(hf_n)]) ** 2).mean(axis=1)
        )
        hf_kurt = float(stats.kurtosis(hf_env))
        hf_rms = float(np.sqrt((hf**2).mean()))
        hf_crest = float(np.abs(hf).max() / hf_rms) if hf_rms > 0 else 0.0

        # ---- pacing
        onsets = librosa.onset.onset_detect(y=x, sr=sr, units="time")
        duration = len(x) / sr
        speaking_rate = float(len(onsets) / duration) if duration > 0 else 0.0

        spectral = {
            "centroid": float(librosa.feature.spectral_centroid(y=x, sr=sr).mean()),
            "rolloff": float(librosa.feature.spectral_rolloff(y=x, sr=sr).mean()),
            "bandwidth": float(librosa.feature.spectral_bandwidth(y=x, sr=sr).mean()),
            "flatness": float(librosa.feature.spectral_flatness(y=x).mean()),
            "zcr": float(librosa.feature.zero_crossing_rate(y=x).mean()),
            "voiced_ratio": float(np.mean(voiced)) if len(voiced) else 0.0,
        }

        metrics = DspMetrics(
            speech_level_db=speech_level,
            noise_floor_db=noise_floor,
            stationary_snr_db=stationary_snr,
            tracked_snr_db=tracked_snr,
            longest_silence_s=longest_silence_s,
            pause_count=len(pause_runs),
            mean_pause_s=mean_pause,
            speaking_rate=speaking_rate,
            hf_energy_ratio=hf_ratio,
            hf_env_kurtosis=hf_kurt,
            hf_crest_factor=hf_crest,
            clipping_run_ratio=clipping_ratio,
            echo_prominence=_echo_prominence(x, sr, self.quality),
            dropout_count=_dropout_count(x, sr, speech, n,
                                         self.quality.dropout_min_ms),
            bandwidth_99_hz=bandwidth_99,
            notes=notes,
        )
        return spectral, metrics, float(speech.mean())

    @staticmethod
    def _align(mask: np.ndarray, n_frames: int) -> np.ndarray:
        if len(mask) < n_frames:
            return np.pad(mask, (0, n_frames - len(mask)))
        return mask[:n_frames]
