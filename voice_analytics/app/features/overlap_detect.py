"""Overlapped-speech detection via `pyannote/segmentation-3.0`.

The segmentation model, not the diarization pipeline: pyannote 4.x's pipeline
returns a strictly exclusive timeline and can never report two speakers at once.
Segmentation-3.0 is a powerset model - each 17 ms frame is classified as
"nobody" / "speaker 1" / ... / "speakers 1+2" - so simultaneous speech is a native
prediction, made before the clustering step that collapses it. Decoding the
powerset to per-speaker activity and counting frames with two or more speakers
active measures overlap directly, and skips the embedding/clustering cost.

Known limit: the model does not flag two clean recordings summed to mono at equal
level, so only real call audio can validate this field.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEGMENTATION = "pyannote/segmentation-3.0"


class OverlapDetector:
    """Measures seconds of simultaneous speech. Reports unavailable, never guesses."""

    def __init__(self, model_id: str, token: str | None, device: str = "cpu",
                 seed: int = 0):
        self.model_id = model_id or DEFAULT_SEGMENTATION
        self.token = token
        self.device = device
        self.seed = seed
        self._model = None
        self._powerset = None
        self.available = False
        self.reason = ""

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.token:
            self.reason = ("no HF token found (set HF_TOKEN) and the pyannote "
                           "segmentation model is gated")
            return False
        try:
            import torch
            from pyannote.audio import Model
            from pyannote.audio.utils.powerset import Powerset

            torch.manual_seed(self.seed)
            model = Model.from_pretrained(self.model_id, token=self.token)
            if model is None:
                self.reason = (f"pyannote returned no model for {self.model_id}; "
                               "the terms are probably not accepted for this account")
                return False
            model.eval()
            spec = model.specifications
            if not getattr(spec, "powerset", False):
                self.reason = (f"{self.model_id} is not a powerset model, so it "
                               "cannot report simultaneous speakers")
                return False
            self._model = model.to(torch.device(self.device))
            self._powerset = Powerset(len(spec.classes), spec.powerset_max_classes)
            self.available = True
            return True
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            return False

    def measure(self, x: np.ndarray, sr: int) -> tuple[float, float] | None:
        """(overlap_seconds, speech_seconds), or None when unavailable.

        Non-overlapping chunks of the model's native window, so each frame maps to
        exactly one moment in time and no aggregation heuristic is involved.
        """
        if not self._load():
            return None
        import torch

        try:
            window = int(float(self._model.specifications.duration) * sr)
            overlap_s = speech_s = 0.0
            for start in range(0, len(x), window):
                chunk = x[start:start + window]
                if len(chunk) < sr:      # ignore a final sliver under one second
                    continue
                padded = np.zeros(window, dtype=np.float32)
                padded[:len(chunk)] = np.asarray(chunk, dtype=np.float32)
                tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    logits = self._model(tensor.to(torch.device(self.device)))
                active = (self._powerset
                          .to_multilabel(logits, soft=False)
                          .squeeze(0).cpu().numpy()
                          .sum(axis=1))
                frame_s = float(self._model.specifications.duration) / len(active)
                # Only count frames inside the real audio, not the zero padding.
                valid = min(len(active), int(round(len(chunk) / sr / frame_s)))
                active = active[:valid]
                overlap_s += float((active >= 2).sum()) * frame_s
                speech_s += float((active >= 1).sum()) * frame_s
            return overlap_s, speech_s
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            self.available = False
            return None
