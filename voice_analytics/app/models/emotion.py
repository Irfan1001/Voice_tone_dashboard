"""Emotion predictor: arousal / dominance / valence from a wav2vec2 model.

Dimensional rather than categorical because it maps onto the required taxonomy
directly - intensity essentially IS arousal - and because MSP-Podcast is natural
conversational speech rather than acted studio recordings. Four alternatives were
compared on human agreement and window-to-window stability; see the README.

Emits raw scores only; the rule engine turns them into tone and intensity.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..schema import AudioData, AudioFeatures, Prediction
from .base import register

WINDOW_S = 8.0
HOP_S = 8.0
MIN_WINDOW_S = 1.0
MIN_SEGMENT_S = 0.4   # shorter fragments carry too little prosody to score


def _speech_windows(x: np.ndarray, sr: int, features: AudioFeatures
                    ) -> list[np.ndarray]:
    """Fixed windows covering the speech to be scored.

    Windowing over speech rather than the whole clip keeps long silences from
    diluting the estimate. Fixed size, fixed hop, no sampling.

    Prefers the customer's turns when role resolution succeeded. Falls back to all
    speech otherwise, flagged as `customer_isolated: false` in diagnostics.
    """
    win, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    customer = features.extra.get("customer_spans") or []
    if customer:
        spans = [(int(s * sr), int(e * sr)) for s, e in customer
                 if e - s >= MIN_SEGMENT_S]
    else:
        spans = [
            (int(s.start * sr), int(s.end * sr))
            for s in features.vad_segments
            if s.duration >= MIN_SEGMENT_S
        ]
    if not spans:
        spans = [(0, len(x))]
    out: list[np.ndarray] = []
    for lo, hi in spans:
        lo, hi = max(0, lo), min(len(x), hi)
        if hi - lo < int(MIN_WINDOW_S * sr):
            continue
        for s in range(lo, hi, hop):
            chunk = x[s : min(s + win, hi)]
            if len(chunk) >= int(MIN_WINDOW_S * sr):
                out.append(chunk)
    if not out:
        out = [x[: int(WINDOW_S * sr)]]
    return out


class _DimensionalBackend:
    """Loads the MSP-Podcast arousal/dominance/valence regression model."""

    def __init__(self, model_id: str, device: str, seed: int):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self._model = None
        self._extractor = None
        self.reason = ""
        self.loaded_report: dict[str, int] = {}

    def load(self) -> bool:
        """Build the model directly and load its weights by hand.

        The model card subclasses Wav2Vec2PreTrainedModel, which couples this to
        transformers' internals and breaks on 5.x. Composing a plain nn.Module
        around Wav2Vec2Model does the same arithmetic, version-independently.
        """
        if self._model is not None:
            return True
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model

            torch.manual_seed(self.seed)
            torch.set_grad_enabled(False)

            class RegressionHead(nn.Module):
                def __init__(self, hidden: int, n_out: int, dropout: float):
                    super().__init__()
                    self.dense = nn.Linear(hidden, hidden)
                    self.dropout = nn.Dropout(dropout)
                    self.out_proj = nn.Linear(hidden, n_out)

                def forward(self, features):
                    x = self.dropout(features)
                    x = torch.tanh(self.dense(x))
                    x = self.dropout(x)
                    return self.out_proj(x)

            class EmotionModel(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.wav2vec2 = Wav2Vec2Model(config)
                    self.classifier = RegressionHead(
                        config.hidden_size,
                        config.num_labels,
                        getattr(config, "final_dropout", 0.1),
                    )

                def forward(self, input_values):
                    hidden = torch.mean(self.wav2vec2(input_values)[0], dim=1)
                    return hidden, self.classifier(hidden)

            config = AutoConfig.from_pretrained(self.model_id)
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            model = EmotionModel(config)

            state = self._checkpoint(self.model_id)
            missing, unexpected = model.load_state_dict(state, strict=False)
            critical = [k for k in missing if k.startswith("classifier.")]
            if critical:
                raise RuntimeError(
                    f"regression head weights absent from checkpoint: {critical}"
                )
            self.loaded_report = {
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
            self._model = model.to(self.device)
            self._model.eval()
            return True
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            return False

    @staticmethod
    def _checkpoint(model_id: str) -> dict:
        """Fetch the checkpoint, preferring safetensors over pickle."""
        import torch
        from huggingface_hub import hf_hub_download

        try:
            from safetensors.torch import load_file

            path = hf_hub_download(model_id, "model.safetensors")
            return load_file(path)
        except Exception:
            path = hf_hub_download(model_id, "pytorch_model.bin")
            return torch.load(path, map_location="cpu", weights_only=True)

    def scores(self, windows: list[np.ndarray], sr: int) -> tuple[np.ndarray, int]:
        """Mean arousal/dominance/valence across windows, plus the window count."""
        import torch

        vals: list[np.ndarray] = []
        with torch.inference_mode():
            for chunk in windows:
                inputs = self._extractor(chunk, sampling_rate=sr, return_tensors="pt")
                x = inputs["input_values"].to(self.device)
                _, logits = self._model(x)
                vals.append(logits.squeeze(0).cpu().numpy())
        arr = np.stack(vals)
        return arr.mean(axis=0), len(vals)


@register("emotion", "dimensional")
class DimensionalEmotionPredictor:
    name = "emotion:dimensional"
    slot = "emotion"

    # The model returns three values in this order.
    DIMS = ("arousal", "dominance", "valence")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = _DimensionalBackend(
            cfg.models.emotion_dimensional, cfg.device, cfg.seed
        )

    def predict(self, audio: AudioData, features: AudioFeatures) -> Prediction:
        if not self.backend.load():
            return Prediction(
                prediction={},
                confidence=0.0,
                metadata={"available": False, "reason": self.backend.reason,
                          "model": self.cfg.models.emotion_dimensional},
            )
        windows = _speech_windows(audio.waveform, audio.sample_rate, features)
        mean, n = self.backend.scores(windows, audio.sample_rate)
        scores = {d: float(v) for d, v in zip(self.DIMS, mean)}

        # Distance from the neutral midpoint: a clip reading 0.50/0.50 is
        # genuinely ambiguous, not a confident neutral.
        decisiveness = max(
            abs(scores["valence"] - 0.5), abs(scores["arousal"] - 0.5)
        ) * 2.0
        confidence = 0.45 + 0.40 * min(1.0, decisiveness)

        return Prediction(
            prediction=scores,
            confidence=round(confidence, 3),
            metadata={
                "model": self.cfg.models.emotion_dimensional,
                "model_licence": self.cfg.models.licence_of(
                    self.cfg.models.emotion_dimensional
                ),
                "windows_scored": n,
                "window_s": WINDOW_S,
                "scored_on": ("customer turns only"
                              if features.extra.get("customer_spans")
                              else "ALL speech (customer NOT isolated)"),
                "customer_isolation": features.extra.get("customer_isolation"),
                "scale": "0..1 for each dimension",
                "prosody": {
                    "speech_level_db": round(features.dsp.speech_level_db, 1),
                    "speaking_rate": round(features.dsp.speaking_rate, 2),
                    "voiced_ratio": round(features.spectral.get("voiced_ratio", 0.0), 3),
                },
            },
        )
