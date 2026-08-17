"""The output contract and the pipeline's data objects.

Dataclasses rather than dictionaries, so every stage boundary is typed and
mistakes surface at construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# --------------------------------------------------------------- output enums


class EmotionalTone(StrEnum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(StrEnum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


# -------------------------------------------------------------- final result


@dataclass
class AnalysisResult:
    """The required JSON output. Field order here IS the output order."""

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float

    def __post_init__(self) -> None:
        self.emotional_tone = EmotionalTone(self.emotional_tone)
        self.emotional_intensity = EmotionalIntensity(self.emotional_intensity)
        self.background_noise_severity = NoiseSeverity(self.background_noise_severity)
        self.audio_quality = AudioQuality(self.audio_quality)
        self.background_noise_present = bool(self.background_noise_present)
        self.speaker_overlap_present = bool(self.speaker_overlap_present)
        self.long_silence_present = bool(self.long_silence_present)
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if not isinstance(self.background_noise_type, str):
            raise TypeError("background_noise_type must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotional_tone": str(self.emotional_tone),
            "emotional_intensity": str(self.emotional_intensity),
            "background_noise_present": self.background_noise_present,
            "background_noise_type": self.background_noise_type,
            "background_noise_severity": str(self.background_noise_severity),
            "audio_quality": str(self.audio_quality),
            "speaker_overlap_present": self.speaker_overlap_present,
            "long_silence_present": self.long_silence_present,
            "confidence": self.confidence,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


FIELD_ORDER: tuple[str, ...] = (
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
    "confidence",
)


# --------------------------------------------------------------- data objects


@dataclass
class AudioData:
    """Decoded, normalized audio ready for analysis."""

    waveform: np.ndarray  # mono float32 at sample_rate
    sample_rate: int
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    start: float
    end: float
    label: str = "speech"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DspMetrics:
    """Derived scalar measurements shared by the noise and quality predictors."""

    # levels and noise estimates
    speech_level_db: float
    noise_floor_db: float
    stationary_snr_db: float  # gap-based (needs usable pauses)
    tracked_snr_db: float  # min-statistics (valid even without pauses)

    # silence / pacing
    longest_silence_s: float
    pause_count: int
    mean_pause_s: float
    speaking_rate: float

    # spectral character of non-speech regions
    hf_energy_ratio: float  # above the telephony band

    # impulsiveness (crackle / static)
    hf_env_kurtosis: float
    hf_crest_factor: float

    # technical defects
    clipping_run_ratio: float
    echo_prominence: float
    dropout_count: int
    bandwidth_99_hz: float  # measured on SPEECH frames only

    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class AudioFeatures:
    """Shared features computed once per clip and read by every predictor."""

    spectral: dict[str, float]
    speech_ratio: float
    vad_segments: list[Segment]
    dsp: DspMetrics
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def snr(self) -> float:
        """The canonical SNR: the gap-based estimate.

        Derived, not stored, so only one place holds the number. See
        `dsp.tracked_snr_db` for the variant that survives a clip with no pauses.
        """
        return self.dsp.stationary_snr_db


@dataclass
class Prediction:
    """What every predictor returns.

    `prediction` holds only the schema fields this predictor owns, so the rule
    engine can merge without guessing.
    """

    prediction: dict[str, Any]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
