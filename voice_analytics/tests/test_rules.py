"""The rule engine: scores in, schema values out.

Model-free — the engine is the one module that turns measurements into enum values,
so its whole decision surface is testable without loading anything.

The tone cases are written straight from the field definitions in the brief, not from
the model's score distribution. That is the point: if a future retune improves a
benchmark but breaks these, the retune is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DEFAULT
from app.rules.engine import (
    Diagnostics,
    RuleEngine,
    map_dimensional,
    map_intensity,
    normalize_noise,
)
from app.schema import (
    AudioData,
    AudioQuality,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
    Prediction,
)

R = DEFAULT.rules


def diag() -> Diagnostics:
    return Diagnostics(filename="t.wav", duration_s=30.0)


def silence(seconds: float) -> AudioData:
    n = int(16000 * seconds)
    return AudioData(np.zeros(n, dtype="float32"), 16000, seconds)


# ------------------------------------------------------------------ tone mapping


@pytest.mark.parametrize(
    "arousal,dominance,valence,expected",
    [
        # high arousal splits on dominance: in control vs overwhelmed
        (0.80, 0.80, 0.40, EmotionalTone.UPSET),
        (0.80, 0.20, 0.40, EmotionalTone.DISTRESSED),
        # low arousal: quiet grief vs flat
        (0.20, 0.20, 0.45, EmotionalTone.DISTRESSED),
        (0.20, 0.60, 0.45, EmotionalTone.NEUTRAL),
        # mid band is where valence finally decides
        (0.45, 0.50, 0.60, EmotionalTone.SATISFIED),
        (0.45, 0.50, 0.30, EmotionalTone.FRUSTRATED),
        (0.45, 0.50, 0.45, EmotionalTone.NEUTRAL),
    ],
)
def test_tone_follows_the_field_definitions(arousal, dominance, valence, expected):
    tone, _ = map_dimensional(arousal, dominance, valence, R)
    assert tone is expected


def test_a_panicked_caller_is_distressed_not_frustrated():
    """High arousal + low dominance must be `distressed` ("escalated, overwhelmed").

    A nearest-centroid mapping fitted for benchmark F1 placed `distressed` at LOW
    arousal and returned `frustrated` here, which inverts the definition.
    """
    tone, intensity = map_dimensional(0.85, 0.25, 0.30, R)
    assert tone is EmotionalTone.DISTRESSED
    assert intensity is EmotionalIntensity.HIGH


def test_valence_cannot_override_the_arousal_band():
    """Valence only breaks mid-band ties. A furious customer who happens to score
    positive valence must not come out `satisfied` - valence is the axis with the
    least separation (0.71 vs 1.62/1.74), so it must not outvote arousal."""
    tone, _ = map_dimensional(0.90, 0.85, 0.99, R)
    assert tone is EmotionalTone.UPSET


def test_tone_is_deterministic_on_the_band_edges():
    """Exactly on a threshold, the answer must be stable and defined."""
    for _ in range(3):
        assert map_dimensional(R.tone_arousal_high, R.upset_dominance, 0.5, R)[0] \
            is EmotionalTone.UPSET


# --------------------------------------------------------------------- intensity


def test_intensity_is_arousal_banded():
    assert map_intensity(0.90, R) is EmotionalIntensity.HIGH
    assert map_intensity(0.50, R) is EmotionalIntensity.MEDIUM
    assert map_intensity(0.10, R) is EmotionalIntensity.LOW


def test_intensity_thresholds_are_inclusive_at_the_lower_edge():
    assert map_intensity(R.arousal_high, R) is EmotionalIntensity.HIGH
    assert map_intensity(R.arousal_medium, R) is EmotionalIntensity.MEDIUM


# ------------------------------------------------------------ noise consistency


def test_no_noise_clears_type_and_severity():
    """R1: a clip with no noise cannot carry a noise name or a severity."""
    fired: list[str] = []
    present, ntype, sev = normalize_noise(False, "television", NoiseSeverity.HIGH, fired)
    assert (present, ntype, sev) == (False, "", NoiseSeverity.NONE)
    assert len(fired) == 2


def test_noise_present_cannot_have_severity_none():
    """R2: `present` with severity `none` is self-contradictory."""
    fired: list[str] = []
    _, _, sev = normalize_noise(True, "television", NoiseSeverity.NONE, fired)
    assert sev is NoiseSeverity.LOW
    assert any("R2" in f for f in fired)


def test_noise_present_always_gets_a_description():
    """R3: the field must never be present-but-unnamed."""
    fired: list[str] = []
    _, ntype, _ = normalize_noise(True, "", NoiseSeverity.MEDIUM, fired)
    assert ntype == "unspecified background noise"
    assert any("R3" in f for f in fired)


def test_a_consistent_noise_state_fires_no_rules():
    fired: list[str] = []
    out = normalize_noise(True, "television", NoiseSeverity.MEDIUM, fired)
    assert out == (True, "television", NoiseSeverity.MEDIUM)
    assert fired == []


# ------------------------------------------------- diagnostics must not lie


def test_noise_type_source_is_reported_as_the_predictor_measured_it():
    """Diagnostics exist to say WHERE an answer came from.

    `sharp static` is named by the DSP kurtosis signature, not by the AudioSet
    classifier. Deriving the source from "is event_type non-empty" reported
    `audioset_event` for a DSP-named type - the diagnostics contradicted the code.
    """
    engine = RuleEngine(DEFAULT)
    d = diag()
    p = Prediction(
        prediction={
            "stationary": False, "impulsive": True, "strong_event": False,
            "snr_db_used": 43.9, "severity_bands": (28.0, 18.0),
            "event_type": "sharp static",
            "type_source": "dsp_impulsive_kurtosis",
        },
        confidence=0.7,
    )
    present, ntype, _ = engine._noise(p, d)
    assert (present, ntype) == (True, "sharp static")
    assert d.evidence["noise"]["type_source"] == "dsp_impulsive_kurtosis"


def test_audioset_named_noise_keeps_its_label_in_the_source():
    engine = RuleEngine(DEFAULT)
    d = diag()
    p = Prediction(
        prediction={
            "stationary": True, "impulsive": False, "strong_event": False,
            "snr_db_used": 25.3, "severity_bands": (28.0, 18.0),
            "event_type": "television",
            "type_source": "audioset_event:Television",
        },
        confidence=0.7,
    )
    engine._noise(p, d)
    assert d.evidence["noise"]["type_source"] == "audioset_event:Television"


def test_a_noise_prediction_missing_its_bands_fails_loudly():
    """A predictor that skipped a decision-bearing key is a contract violation.

    Defaulting to hardcoded bands would duplicate config (so a retune would silently
    disagree with it) and would emit a plausible severity nobody measured.
    """
    engine = RuleEngine(DEFAULT)
    p = Prediction(
        prediction={"stationary": True, "impulsive": False, "strong_event": False},
        confidence=0.7,
    )
    with pytest.raises(KeyError):
        engine._noise(p, diag())


def test_a_silence_prediction_missing_its_measurement_fails_loudly():
    engine = RuleEngine(DEFAULT)
    with pytest.raises(KeyError):
        engine._silence(Prediction(prediction={"unrelated": 1}, confidence=0.8), diag())


# ------------------------------------------------------- thresholded booleans


def test_overlap_keys_on_seconds_not_ratio():
    """The ratio ranks a negative above a positive on the labelled calls, so it must
    not gate. Here the ratio is tiny while the seconds clearly clear the bar."""
    engine = RuleEngine(DEFAULT)
    d = diag()
    p = Prediction(prediction={"overlap_seconds": 1.95, "overlap_ratio": 0.0170},
                   confidence=0.75)
    assert engine._overlap(p, d) is True


def test_brief_crosstalk_is_not_overlap():
    engine = RuleEngine(DEFAULT)
    p = Prediction(prediction={"overlap_seconds": 0.31, "overlap_ratio": 0.0251},
                   confidence=0.75)
    assert engine._overlap(p, diag()) is False


def test_long_silence_uses_the_dead_air_threshold():
    engine = RuleEngine(DEFAULT)
    assert engine._silence(
        Prediction(prediction={"longest_silence_s": 7.35}, confidence=0.8), diag()
    ) is True
    assert engine._silence(
        Prediction(prediction={"longest_silence_s": 2.79}, confidence=0.8), diag()
    ) is False


# -------------------------------------------------------------- confidence


def test_confidence_stays_inside_its_bounds_and_penalises_short_clips():
    engine = RuleEngine(DEFAULT)
    preds = {
        slot: Prediction(prediction={"x": 1}, confidence=0.9)
        for slot in ("emotion", "noise", "quality", "silence", "overlap")
    }
    long_conf = engine._confidence(silence(30.0), preds, AudioQuality.CLEAR, diag())
    short_conf = engine._confidence(silence(2.0), preds, AudioQuality.CLEAR, diag())
    assert short_conf < long_conf
    assert R.confidence_floor <= short_conf <= R.confidence_ceiling


def test_worse_audio_quality_lowers_confidence():
    engine = RuleEngine(DEFAULT)
    audio = silence(30.0)
    preds = {"emotion": Prediction(prediction={"x": 1}, confidence=0.9)}
    clear = engine._confidence(audio, preds, AudioQuality.CLEAR, diag())
    bad = engine._confidence(audio, preds, AudioQuality.SEVERELY_IMPAIRED, diag())
    assert bad < clear
