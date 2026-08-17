"""Deterministic rule engine: raw predictor scores -> the final JSON.

The only module that produces schema enum values. Predictors measure; this
decides. Same scores in, same JSON out, always.

Deliberately absent: any rule linking `audio_quality` to a noise field in either
direction. A clip can carry loud static while its speech stays intelligible, so
any correlation must come from independent measurement, not from this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, RuleConfig
from ..errors import IncompletePredictionError
from ..schema import (
    AnalysisResult,
    AudioData,
    AudioQuality,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
    Prediction,
)

# No placeholder values anywhere in this module: a predictor that cannot run raises
# IncompletePredictionError and the clip emits nothing, because a fabricated
# "neutral" is indistinguishable from a measured one in the JSON.


@dataclass
class Diagnostics:
    """Everything useful that is NOT part of the required output schema."""

    filename: str
    duration_s: float
    strategies: dict[str, str] = field(default_factory=dict)
    rules_fired: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    predictor_confidence: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "duration_s": round(self.duration_s, 2),
            "strategies": self.strategies,
            "rules_fired": self.rules_fired,
            "warnings": self.warnings,
            "predictor_confidence": self.predictor_confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
            "total_latency_ms": round(self.total_latency_ms, 1),
            "cost_usd": self.cost_usd,
            "evidence": self.evidence,
        }


log = logging.getLogger(__name__)

_QUALITY_INDEX = {
    AudioQuality.CLEAR: 0,
    AudioQuality.SLIGHTLY_IMPAIRED: 1,
    AudioQuality.SEVERELY_IMPAIRED: 2,
}
_SEVERITY_RANK = (
    NoiseSeverity.NONE, NoiseSeverity.LOW, NoiseSeverity.MEDIUM, NoiseSeverity.HIGH,
)


def _worse(a: NoiseSeverity, b: NoiseSeverity) -> NoiseSeverity:
    return max(a, b, key=_SEVERITY_RANK.index)


def normalize_noise(
    present: bool, noise_type: str, severity: NoiseSeverity, fired: list[str]
) -> tuple[bool, str, NoiseSeverity]:
    """Force the three noise fields into a mutually consistent state.

      R1  no noise      -> type must be empty and severity `none`
      R2  noise present -> severity `none` contradicts presence
      R3  noise present -> a description is required

    A separate pass rather than logic woven into the derivation, so the rules stay
    reachable and testable instead of true by construction.
    """
    if not present:
        if noise_type:
            fired.append("R1: cleared background_noise_type (no noise present)")
            noise_type = ""
        if severity != NoiseSeverity.NONE:
            fired.append("R1: forced severity=none (no noise present)")
            severity = NoiseSeverity.NONE
        return present, noise_type, severity

    if severity == NoiseSeverity.NONE:
        fired.append("R2: promoted severity none -> low (noise present)")
        severity = NoiseSeverity.LOW
    if not noise_type:
        fired.append("R3: filled empty background_noise_type")
        noise_type = "unspecified background noise"
    return present, noise_type, severity


def map_intensity(arousal: float, r: RuleConfig) -> EmotionalIntensity:
    """Intensity is arousal, directly.

    Grid-fitting these thresholds on 680 CREMA-D clips moved macro F1 from 0.517 to
    0.512, so the hand-picked values are already at the optimum. Left alone.
    """
    if arousal >= r.arousal_high:
        return EmotionalIntensity.HIGH
    if arousal >= r.arousal_medium:
        return EmotionalIntensity.MEDIUM
    return EmotionalIntensity.LOW


def _tone_arousal_primary(
    arousal: float, dominance: float, valence: float, r: RuleConfig
) -> EmotionalTone:
    """Arousal picks the band; dominance splits it; valence only breaks ties.

    Arousal goes first because it separates the classes best (separation ratio
    2.45). In the high band dominance distinguishes an angry customer in control
    (`upset`) from a panicking one (`distressed`) - both are equally aroused. The
    low band is the second `distressed` region: grief is quiet. Valence decides
    only inside the middle band, the one place it separates anything at all.
    """
    if arousal >= r.tone_arousal_high:
        return (EmotionalTone.UPSET if dominance >= r.upset_dominance
                else EmotionalTone.DISTRESSED)
    if arousal < r.tone_arousal_low:
        return (EmotionalTone.DISTRESSED if dominance < r.grief_dominance
                else EmotionalTone.NEUTRAL)
    if valence >= r.valence_positive:
        return EmotionalTone.SATISFIED
    if valence <= r.valence_negative:
        return EmotionalTone.FRUSTRATED
    return EmotionalTone.NEUTRAL


def map_dimensional(
    arousal: float, dominance: float, valence: float, r: RuleConfig
) -> tuple[EmotionalTone, EmotionalIntensity]:
    """(arousal, dominance, valence) -> (tone, intensity).

    A free function so evaluation harnesses exercise the exact code that ships.

    Tone keys on arousal and dominance rather than valence: across 680 CREMA-D
    clips valence's entire spread over the five tones (0.065) is smaller than the
    variation inside one tone (0.090), a separation ratio of 0.72. Arousal (2.45)
    and dominance (2.53) both separate the classes.
    """
    return (_tone_arousal_primary(arousal, dominance, valence, r),
            map_intensity(arousal, r))


class RuleEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------- emotion
    def _emotion(self, p: Prediction, diag: Diagnostics
                 ) -> tuple[EmotionalTone, EmotionalIntensity]:
        r = self.cfg.rules
        arousal = float(p.prediction["arousal"])
        valence = float(p.prediction["valence"])
        dominance = float(p.prediction["dominance"])
        tone, intensity = map_dimensional(arousal, dominance, valence, r)

        diag.evidence.setdefault("emotion", {}).update(
            {"arousal": round(arousal, 3), "valence": round(valence, 3),
             "dominance": round(dominance, 3),
             "mapped_via": "dimensional: arousal band -> dominance -> valence",
             "scored_on": p.metadata.get("scored_on", "unknown")}
        )
        return tone, intensity

    # --------------------------------------------------------------- noise
    def _noise(self, p: Prediction, diag: Diagnostics
               ) -> tuple[bool, str, NoiseSeverity]:
        pr = p.prediction
        stationary = bool(pr.get("stationary"))
        impulsive = bool(pr.get("impulsive"))
        strong_event = bool(pr.get("strong_event"))
        present = stationary or impulsive or strong_event

        # Indexed, not `.get`-with-a-default: these are contract keys the predictor
        # always sends. A default here would duplicate config (and silently disagree
        # with it after a retune) while emitting a severity nobody measured.
        severity = NoiseSeverity.NONE
        if stationary:
            low_edge, med_edge = pr["severity_bands"]
            snr = float(pr["snr_db_used"])
            if snr >= low_edge:
                severity = _worse(severity, NoiseSeverity.LOW)
            elif snr >= med_edge:
                severity = _worse(severity, NoiseSeverity.MEDIUM)
            else:
                severity = _worse(severity, NoiseSeverity.HIGH)
        if impulsive:
            severity = _worse(severity, NoiseSeverity.MEDIUM)

        # Only the AudioSet classifier may NAME the noise; hand-written spectral
        # rules capped near 70% on 60 controlled clips. When it abstains,
        # normalize_noise writes "unspecified background noise".
        noise_type = pr.get("event_type") or ""

        present, noise_type, severity = normalize_noise(
            present, noise_type, severity, diag.rules_fired
        )

        diag.evidence.setdefault("noise", {}).update(
            {"stationary": stationary, "impulsive": impulsive,
             "strong_event": strong_event,
             "snr_db_used": pr.get("snr_db_used"),
             # Passed through, not re-derived: only the predictor knows whether the
             # name came from the AudioSet classifier or the DSP kurtosis signature.
             "type_source": pr.get("type_source") or "none"}
        )
        return present, noise_type, severity

    # ------------------------------------------------------------- quality
    def _quality(self, p: Prediction, diag: Diagnostics) -> AudioQuality:
        q = self.cfg.quality
        points = int(p.prediction["impairment_points"])
        if points >= q.severe_points:
            verdict = AudioQuality.SEVERELY_IMPAIRED
        elif points >= q.slight_points:
            verdict = AudioQuality.SLIGHTLY_IMPAIRED
        else:
            verdict = AudioQuality.CLEAR
        diag.evidence.setdefault("quality", {}).update(
            {"impairment_points": points,
             "impairments": p.prediction.get("impairments", [])}
        )
        return verdict

    # ------------------------------------------------------------- silence
    def _silence(self, p: Prediction, diag: Diagnostics) -> bool:
        longest = float(p.prediction["longest_silence_s"])
        threshold = self.cfg.silence.long_silence_s
        diag.evidence.setdefault("silence", {}).update(
            {"longest_silence_s": longest, "threshold_s": threshold,
             "longest_vad_gap_s": p.prediction.get("longest_vad_gap_s")}
        )
        return longest >= threshold

    # ------------------------------------------------------------- overlap
    def _overlap(self, p: Prediction, diag: Diagnostics) -> bool:
        seconds = float(p.prediction["overlap_seconds"])
        ratio = float(p.prediction.get("overlap_ratio", 0.0))   # reported, not used
        # Seconds alone: a fraction of a second of turn-taking crosstalk is normal.
        # `overlap_ratio` is reported but does not gate - it ranks a negative above
        # a positive on the labelled calls, so it can only turn verdicts wrong.
        r = self.cfg.rules
        present = seconds >= r.overlap_min_seconds
        diag.evidence.setdefault("overlap", {}).update(
            {"overlap_seconds": seconds, "overlap_ratio": ratio,
             "threshold_s": r.overlap_min_seconds,
             "detector": p.metadata.get("detector")}
        )
        return present

    # ---------------------------------------------------------------- run
    def run(self, audio: AudioData, predictions: dict[str, Prediction],
            strategies: dict[str, str]) -> tuple[AnalysisResult, Diagnostics]:
        diag = Diagnostics(
            filename=str(audio.metadata.get("filename", "")),
            duration_s=audio.duration,
            strategies=dict(strategies),
        )
        for slot, p in predictions.items():
            diag.predictor_confidence[slot] = round(p.confidence, 3)
            diag.latency_ms[slot] = p.latency_ms
            diag.cost_usd += p.cost_usd
            if not p.metadata.get("available", True):
                diag.warnings.append(
                    f"{slot} predictor unavailable: {p.metadata.get('reason') or p.metadata.get('error')}"
                )

        # ---- strict gate: refuse to fabricate before deriving anything
        unavailable = {
            slot: str(p.metadata.get("reason") or p.metadata.get("error")
                      or "predictor reported unavailable")
            for slot, p in predictions.items()
            if not p.metadata.get("available", True) or not p.prediction
        }
        if unavailable:
            for slot, why in unavailable.items():
                log.error("predictor %s unavailable: %s", slot, why)
            raise IncompletePredictionError(unavailable)

        # A tone measured over both parties does not answer "the customer's tone",
        # so say so in the output rather than letting it pass as if it did.
        iso = predictions["emotion"].metadata.get("customer_isolation") or {}
        if iso and not iso.get("customer_isolated"):
            diag.warnings.append(
                "emotional_tone/intensity cover ALL speech, not the customer alone: "
                + str(iso.get("reason") or "role could not be resolved")
            )

        tone, intensity = self._emotion(predictions["emotion"], diag)
        present, ntype, severity = self._noise(predictions["noise"], diag)
        quality = self._quality(predictions["quality"], diag)
        long_silence = self._silence(predictions["silence"], diag)
        overlap = self._overlap(predictions["overlap"], diag)

        confidence = self._confidence(audio, predictions, quality, diag)

        result = AnalysisResult(
            emotional_tone=tone,
            emotional_intensity=intensity,
            background_noise_present=present,
            background_noise_type=ntype,
            background_noise_severity=severity,
            audio_quality=quality,
            speaker_overlap_present=overlap,
            long_silence_present=long_silence,
            confidence=confidence,
        )
        diag.total_latency_ms = sum(diag.latency_ms.values())
        return result, diag

    def _confidence(self, audio: AudioData, predictions: dict[str, Prediction],
                    quality: AudioQuality, diag: Diagnostics) -> float:
        """Weighted predictor confidence, then deterministic penalties.

        Weights are the number of schema fields each predictor informs, so the
        emotion and noise predictors count for more than the single-field ones.
        """
        r = self.cfg.rules
        weights = {"emotion": 2, "noise": 3, "quality": 1, "silence": 1, "overlap": 1}
        total_w = 0
        acc = 0.0
        for slot, w in weights.items():
            p = predictions.get(slot)
            if p is None:
                continue
            acc += p.confidence * w
            total_w += w
        base = acc / total_w if total_w else r.confidence_floor

        rule_pen = r.rule_penalty ** len(diag.rules_fired)
        q_pen = r.quality_factor[_QUALITY_INDEX[quality]]
        short_pen = (
            r.short_clip_penalty if audio.duration < r.short_clip_s else 1.0
        )
        conf = base * rule_pen * q_pen * short_pen
        conf = max(r.confidence_floor, min(r.confidence_ceiling, conf))
        conf = round(conf, 2)

        diag.confidence_breakdown = {
            "weighted_base": round(base, 3),
            "rule_penalty": round(rule_pen, 3),
            "audio_quality_factor": q_pen,
            "short_clip_penalty": short_pen,
            "final": conf,
            "note": "deterministic formula, not calibrated against labels",
        }
        return conf
