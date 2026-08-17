"""Model identifiers and every threshold in the system.

Provenance tags say where a value came from:
    [principled] from signal-processing / telephony standards
    [validated]  tested against ground truth (synthetic set or CREMA-D)
    [measured]   informed by the 3 supplied calls (n=3 - weak)
    [arbitrary]  a starting guess

Nothing here may depend on randomness, wall-clock time or thread count: the
pipeline is required to be byte-for-byte deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers. Swapping a predictor is one line."""

    # dimensional (arousal/dominance/valence, MSP-Podcast conversational speech) beat
    # four alternatives on human agreement and stability; see README.
    emotion_dimensional: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    # LICENCE GATE: the default emotion model is CC-BY-NC-SA-4.0, non-commercial.
    # Resolve a commercial licence with audEERING before revenue traffic.
    MODEL_LICENCES: tuple[tuple[str, str], ...] = (
        ("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
         "cc-by-nc-sa-4.0 (NON-COMMERCIAL - research only)"),
        ("MIT/ast-finetuned-audioset-10-10-0.4593", "bsd-3-clause"),
        ("pyannote/segmentation-3.0", "mit (gated: terms must be accepted)"),
        ("pyannote/speaker-diarization-3.1", "mit (gated: terms must be accepted)"),
        ("openai/whisper-tiny.en", "apache-2.0"),
        ("snakers4/silero-vad", "mit"),
    )

    def licence_of(self, model_id: str | None) -> str:
        for mid, lic in self.MODEL_LICENCES:
            if mid == model_id:
                return lic
        return "unknown - check the model card before shipping"

    # AST over YAMNet: same AudioSet label space without shipping TensorFlow too.
    audio_events: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # Overlap: the SEGMENTATION model, not the diarization pipeline. pyannote 4.x's
    # pipeline returns a strictly exclusive timeline and can never report two speakers
    # at once. Gated on HuggingFace.
    segmentation: str = "pyannote/segmentation-3.0"
    # Customer isolation needs the clustering pipeline. Also gated.
    diarization: str = "pyannote/speaker-diarization-3.1"
    # Role resolution transcribes only the opening seconds, so tiny is enough.
    role_asr: str = "openai/whisper-tiny.en"

    hf_token_env: str = "HF_TOKEN"

    @property
    def hf_token(self) -> str | None:
        for var in (self.hf_token_env, "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            tok = os.environ.get(var)
            if tok:
                return tok
        return None


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000  # native rate for every model here [principled]
    frame_ms: int = 30  # webrtc/silero-compatible [principled]
    min_duration_s: float = 1.0  # [arbitrary]
    # Normalisation happens AFTER clipping is measured: lossy codecs overshoot +-1.0
    # (Opus decode reached 1.107 on audio labelled `clear`). [principled]
    normalize_peak: float = 0.95


@dataclass(frozen=True)
class SilenceConfig:
    # Dead-air convention is 3-5 s. DO NOT raise this to satisfy call_003: that clip is
    # labelled false but holds ~8 s of dead air, confirmed by listening. The label is
    # wrong, not the threshold. [validated by synthetic sweep: 3.5 s false, 4.5 s true]
    long_silence_s: float = 4.0
    silence_rel_db: float = -25.0  # below speech level to count as silence [principled]
    vad_threshold: float = 0.5  # [arbitrary]
    min_speech_ms: int = 250
    min_silence_ms: int = 100


@dataclass(frozen=True)
class NoiseConfig:
    # "Barely perceptible artifacts should not count as background noise." [measured]
    noise_floor_abs_db: float = -75.0

    # Textbook speech-quality bands: >30 dB excellent, 20-30 good, 10-20 fair.
    # [principled] bands, [measured] edges.
    snr_present_db: float = 30.0
    snr_low_db: float = 28.0
    snr_medium_db: float = 18.0

    # Gap-based SNR runs BACKWARDS on speech-like background (11 dB measured at 30 dB
    # true SNR) because television counts as speech and leaves no gaps. Above this
    # speech ratio, switch to min-statistics. Recall 81% -> 97%. [validated]
    gap_estimate_max_speech_ratio: float = 0.85
    snr_tracked_present_db: float = 48.0
    snr_tracked_low_db: float = 44.0
    snr_tracked_medium_db: float = 41.0

    # Two detectors are needed to DETECT impulsive noise; only kurtosis may NAME it.
    # Crest rates television 66 and static 77, so naming by crest relabelled a
    # television as "crackle". [validated]
    impulse_kurtosis_present: float = 60.0
    impulse_crest_present: float = 26.0

    # Minimum AudioSet score to name a noise type: the true background on a labelled
    # call scores 0.136, the best event on the clean call only 0.073.
    # [measured - thin margin]
    event_min_score: float = 0.10
    event_strong_score: float = 0.35  # audible noise on its own [arbitrary]

    # Splits impulsive noise into "sharp static" vs "crackle". [validated]
    static_hf_ratio: float = 0.005


@dataclass(frozen=True)
class QualityConfig:
    """DSP quality thresholds. [validated] on 60 synthetic clips: 10/10 injected
    defects detected, 33/33 noise-only clips stayed `clear`."""

    # Clipping is a FLAT TOP, not a loud sample, and the ceiling is relative to the
    # signal's own peak - clipping after gain reduction sits below full scale, and an
    # absolute rail missed every clipped clip.
    clip_level: float = 0.995
    clip_peak_fraction: float = 0.98
    clip_min_run: int = 3
    clip_ratio_slight: float = 0.0005
    clip_ratio_severe: float = 0.005

    low_speech_db: float = -52.0  # below call_002's -41.9 dB, labelled clear [measured]

    # Cepstral echo, not envelope autocorrelation: the latter fired on all three clean
    # calls because a 50-400 ms lag window measures syllabic rhythm, not echo.
    # Threshold 70: genuine echo 82-130, crackle 43-55, clean 11-14. [validated]
    echo_quefrency_min_s: float = 0.020
    echo_quefrency_max_s: float = 0.400
    echo_prominence: float = 70.0

    dropout_min_ms: float = 20.0  # exact digital silence inside speech [principled]
    dropout_count_slight: int = 1
    dropout_count_severe: int = 5

    # Telephony baseline, measured on SPEECH frames only: all supplied calls hold 99%
    # of energy below 2.1-3.2 kHz and are labelled `clear`. [validated]
    telephony_bw_hz: float = 3400.0
    muffled_bw_hz: float = 1500.0

    slight_points: int = 1
    severe_points: int = 3


@dataclass(frozen=True)
class RuleConfig:
    """Rule-engine thresholds: the only place scores become enum values."""

    # Intensity is arousal. Grid-fitting did not improve these (0.517 -> 0.512).
    # [validated]
    arousal_medium: float = 0.45
    arousal_high: float = 0.62

    # Arousal picks a band, dominance splits it, valence only breaks mid-band ties.
    # Valence is demoted deliberately: separation ratio 0.71 against 1.62 arousal and
    # 1.74 dominance, and it self-correlates at -0.01 across two halves of one
    # utterance. [validated]
    tone_arousal_high: float = 0.60
    tone_arousal_low: float = 0.36
    upset_dominance: float = 0.600
    grief_dominance: float = 0.400
    # Neutral is the band between these. 0.42/0.48 satisfies all 9 cases written from
    # the field definitions at the best macro F1 (0.395); 0.44 scores higher but calls
    # flat speech `satisfied`. [validated on the definitions + 240 CREMA-D clips]
    valence_positive: float = 0.48
    valence_negative: float = 0.42

    # Customer isolation: only turns starting inside role_opening_s are transcribed.
    # Below role_min_margin the role is reported UNDETERMINED rather than guessed.
    # Measured margins were +2, +2, +5. [validated: 3/3 on the supplied calls]
    role_opening_s: float = 20.0
    role_min_margin: int = 1

    # Seconds only. Measured 0.31/false, 0.73/true, 1.95/true - seconds order the
    # labels where the ratio does not. 0.5 sits mid-gap. [measured - n=3, WEAK]
    overlap_min_seconds: float = 0.5

    # Confidence: predictor confidence weighted by fields owned, then penalised.
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.98
    rule_penalty: float = 0.95
    quality_factor: tuple[float, float, float] = (1.0, 0.95, 0.85)
    short_clip_s: float = 5.0
    short_clip_penalty: float = 0.90


@dataclass(frozen=True)
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)

    emotion_strategy: str = "dimensional"
    noise_strategy: str = "audioset"
    quality_strategy: str = "dsp"
    silence_strategy: str = "vad"
    overlap_strategy: str = "segmentation"

    seed: int = 0
    device: str = "cpu"


DEFAULT = Config()
