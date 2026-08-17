"""AudioPipeline: the single orchestrator.

    result, diagnostics = AudioPipeline().run("call.ogg")

    Audio -> Preprocessor -> Feature Extraction
          -> {Emotion, Noise, Quality, Overlap, Silence}
          -> Rule Engine -> Final JSON

Models are built once per instance and reused, so batch processing does not
reload weights per file.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import DEFAULT, Config
from ..features.extractor import FeatureExtractor
from ..models import base as predictor_base
from ..rules.engine import Diagnostics, RuleEngine
from ..preprocessing.preprocessor import Preprocessor
from ..schema import AnalysisResult, Prediction


class AudioPipeline:
    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        self._seed()
        self.preprocessor = Preprocessor(cfg.audio, cfg.quality)
        self.features = FeatureExtractor(cfg)
        self.rules = RuleEngine(cfg)

        self.strategies = {
            "emotion": cfg.emotion_strategy,
            "noise": cfg.noise_strategy,
            "quality": cfg.quality_strategy,
            "silence": cfg.silence_strategy,
            "overlap": cfg.overlap_strategy,
        }
        self.predictors = {
            slot: predictor_base.get_predictor(slot, name, cfg)
            for slot, name in self.strategies.items()
        }

    def _seed(self) -> None:
        """Fixed seeds, inference mode, single-threaded reductions.

        Thread count is pinned because parallel float reductions reorder and change
        low-order bits, which would break byte-identical reruns.

        Deliberately not wrapped in try/except: torch is a hard dependency, and a
        failure here silently costs determinism while the output still looks normal.
        """
        import random

        import numpy as np
        import torch

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------ run
    def run(self, path: str | Path) -> tuple[AnalysisResult, Diagnostics]:
        t0 = time.perf_counter()
        audio = self.preprocessor.run(path)
        t_pre = time.perf_counter()
        features = self.features.run(audio)
        t_feat = time.perf_counter()

        predictions: dict[str, Prediction] = {
            slot: predictor_base.run_predictor(p, audio, features)
            for slot, p in self.predictors.items()
        }

        result, diag = self.rules.run(audio, predictions, self.strategies)

        diag.latency_ms["preprocess"] = (t_pre - t0) * 1000.0
        diag.latency_ms["features"] = (t_feat - t_pre) * 1000.0
        diag.total_latency_ms = (time.perf_counter() - t0) * 1000.0
        diag.evidence["features"] = {
            "vad_backend": features.extra.get("vad_backend"),
            "n_speech_segments": len(features.vad_segments),
            "speech_ratio": round(features.speech_ratio, 3),
            "snr_db": round(features.snr, 1),
            "customer_isolation": features.extra.get("customer_isolation"),
            "overlap_available": features.extra.get("overlap_available"),
            "overlap_reason": features.extra.get("overlap_reason"),
            "source": features.extra.get("source"),
        }
        diag.evidence["predictor_metadata"] = {
            slot: p.metadata for slot, p in predictions.items()
        }
        # Zero: every model runs locally. Cost per audio minute is derived from
        # wall-clock time instead; see the README.
        diag.cost_usd = sum(p.cost_usd for p in predictions.values())
        return result, diag
