#!/usr/bin/env python
"""Which emotion model gives the most stable, most human-agreeing signal?

    python -m evaluation.compare_emotion_models --limit 120
    python -m evaluation.compare_emotion_models --limit 120 --models audeering,voxprofile

Judged on two things, not on a benchmark score:

1. Human agreement, against CREMA-D's voice-only listening study (`VoiceVote`)
   rather than the actor's intended emotion, which no listener recovers either.
2. Stability: the same utterance split in half and scored twice. Invisible in any
   accuracy number, and no threshold placement fixes it.

Each dimensional model gets its OWN nearest-centroid mapping, fitted on TRAIN actors
and reported on TEST actors in its own scaled space. Judging every model through the
shipping thresholds would measure resemblance to audeering, not quality. Splits are
grouped by actor, so nothing can score by recognising a voice.

Evaluation only - nothing here is imported by `app/` or `api/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT  # noqa: E402
from app.env import load_env  # noqa: E402
from app.schema import EmotionalTone  # noqa: E402
from evaluation.human_agreement import load_clips, read  # noqa: E402
from evaluation.metrics import evaluate, format_report  # noqa: E402

SR = 16000
TONES = [str(t) for t in EmotionalTone]

# Any categorical model's class label -> our five tones. `surprised` maps to neutral:
# the brief defines `satisfied` as "pleased, relieved, appreciative", which surprise
# is not.
CLASS_TO_TONE = {
    "neutral": "neutral", "neu": "neutral", "calm": "neutral",
    "surprised": "neutral", "surprise": "neutral", "ps": "neutral", "other": "neutral",
    "unknown": "neutral",
    "happy": "satisfied", "happiness": "satisfied", "hap": "satisfied",
    "joy": "satisfied", "excited": "satisfied",
    "angry": "upset", "anger": "upset", "ang": "upset",
    "disgust": "frustrated", "disgusted": "frustrated", "dis": "frustrated",
    "contempt": "frustrated", "frustrated": "frustrated",
    "fear": "distressed", "fearful": "distressed", "fea": "distressed",
    "sad": "distressed", "sadness": "distressed",
}


@dataclass
class Scored:
    """One model's output for one clip: a vector (dimensional) or a label."""

    vec: np.ndarray | None
    label: str


# --------------------------------------------------------------- adapters


class Audeering:
    """The shipping model: wav2vec2 on MSP-Podcast, arousal/dominance/valence."""

    kind = "dimensional"
    licence = "cc-by-nc-sa-4.0 NON-COMMERCIAL"
    domain = "MSP-Podcast (real conversational)"

    def __init__(self):
        from app.models.emotion import _DimensionalBackend
        self.b = _DimensionalBackend(DEFAULT.models.emotion_dimensional, "cpu", 0)
        if not self.b.load():
            raise RuntimeError(self.b.reason)

    def score(self, x: np.ndarray) -> Scored:
        m, _ = self.b.scores([x], SR)
        return Scored(np.asarray(m, dtype=float), "")


class VoxProfile:
    """tiantiaf/wavlm-large-msp-podcast-emotion-dim - the INTERSPEECH 2025 winner.

    Its wrapper class lives in a GitHub repo licensed NOASSERTION, so it is fetched
    into a scratch directory for evaluation only and never vendored. The one
    speechbrain import is shimmed to an all-ones padding mask, which is what a single
    unpadded chunk needs anyway.
    """

    kind = "dimensional"
    licence = "openrail (weights); wrapper code NOASSERTION"
    domain = "MSP-Podcast (real conversational)"

    def __init__(self, cache: Path):
        import types
        import urllib.request

        import torch

        cache.mkdir(parents=True, exist_ok=True)
        src = cache / "wavlm_emotion_dim.py"
        if not src.is_file():
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/"
                "main/src/model/emotion/wavlm_emotion_dim.py", src)
        text = src.read_text()
        shim = ("def make_padding_masks(x, wav_len=None):\n"
                "    import torch as _t\n"
                "    return _t.ones(x.shape[0], x.shape[-1], dtype=_t.bool,"
                " device=x.device)\n")
        text = text.replace(
            "from speechbrain.lobes.models.huggingface_transformers.huggingface "
            "import make_padding_masks", shim)
        mod = types.ModuleType("vox_wavlm_dim")
        mod.__dict__["__file__"] = str(src)
        exec(compile(text, str(src), "exec"), mod.__dict__)
        self.model = mod.WavLMWrapper.from_pretrained(
            "tiantiaf/wavlm-large-msp-podcast-emotion-dim")
        self.model.eval()
        self._torch = torch

    def score(self, x: np.ndarray) -> Scored:
        t = self._torch
        # the model card documents a 15 s maximum and unreliability below 3 s
        seg = x[: 15 * SR]
        with t.inference_mode():
            a, v, d = self.model(t.from_numpy(seg).float().unsqueeze(0))
        vec = np.array([float(a.squeeze()), float(d.squeeze()), float(v.squeeze())])
        return Scored(vec, "")   # ordered arousal, dominance, valence to match ours


class Categorical:
    """Any plain transformers audio-classification model."""

    kind = "categorical"

    def __init__(self, model_id: str, licence: str, domain: str,
                 trust_remote_code: bool = False):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.licence, self.domain = licence, domain
        self.fx = AutoFeatureExtractor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def score(self, x: np.ndarray) -> Scored:
        import torch
        inp = self.fx(x, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inp).logits
        raw = str(self.id2label[int(logits.argmax())])
        key = raw.lower().strip().replace("_", "").replace("-", "")
        return Scored(None, CLASS_TO_TONE.get(key, "UNMAPPED:" + raw))


class Emotion2Vec:
    """emotion2vec_plus_large via FunASR. 9 classes, multi-corpus training."""

    kind = "categorical"
    licence = "other (check the model card)"
    domain = "multi-corpus"

    def __init__(self, model_id: str = "iic/emotion2vec_plus_base"):
        # _base substitutes for the 1.94 GB _large checkpoint: same family and
        # architecture. Note the substitution against published _large numbers.
        from funasr import AutoModel
        self.model = AutoModel(model=model_id, disable_update=True)

    def score(self, x: np.ndarray) -> Scored:
        res = self.model.generate(x, granularity="utterance", extract_embedding=False)
        item = res[0]
        labels, scores = item["labels"], item["scores"]
        raw = str(labels[int(np.argmax(scores))])
        # funasr labels look like "生气/angry" - take the English half
        key = raw.split("/")[-1].lower().strip()
        return Scored(None, CLASS_TO_TONE.get(key, "UNMAPPED:" + raw))


# ------------------------------------------------- per-model tone mapping


def stability_stats(pairs: list[tuple[str, str]]) -> dict:
    """Self-consistency on two halves of one utterance, corrected for chance.

    Raw agreement is gameable: a model answering `upset` for 88% of clips agrees with
    itself ~78% of the time by arithmetic alone, so reporting it alone would rank a
    degenerate model first. kappa = (observed - expected) / (1 - expected), with
    expected from the model's own prediction distribution: 0 means no better than its
    prior, negative means worse.
    """
    if not pairs:
        return {"observed": 0.0, "expected": 0.0, "kappa": 0.0, "max_share": 0.0}
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)
    counts: dict[str, int] = {}
    for a, b in pairs:                     # both halves inform the prior
        for lab in (a, b):
            counts[lab] = counts.get(lab, 0) + 1
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    expected = sum(pr * pr for pr in probs)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return {"observed": observed, "expected": expected, "kappa": kappa,
            "max_share": max(probs)}


def fit_centroids(vecs: np.ndarray, tones: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                               list[str]]:
    """Class centroids and pooled within-class sd, in this model's own units."""
    labels = sorted(set(tones))
    cents, residuals = [], []
    for lab in labels:
        sel = vecs[[i for i, t in enumerate(tones) if t == lab]]
        mu = sel.mean(axis=0)
        cents.append(mu)
        residuals.append(sel - mu)
    res = np.vstack(residuals)
    scales = np.sqrt((res ** 2).mean(axis=0))
    scales[scales == 0] = 1.0
    return np.stack(cents), scales, labels


def nearest(vec: np.ndarray, cents: np.ndarray, scales: np.ndarray,
            labels: list[str]) -> str:
    d = (((vec - cents) / scales) ** 2).sum(axis=1)
    return labels[int(d.argmin())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=120, help="clips per human-voted tone")
    ap.add_argument("--min-agreement", type=float, default=0.6)
    ap.add_argument("--models", default="audeering,voxprofile,whisper,superb,emotion2vec")
    ap.add_argument("--cache", default="/tmp/voxprofile_eval")
    load_env()
    args = ap.parse_args(argv)

    clips = load_clips(args.limit, args.min_agreement)
    if not clips:
        raise SystemExit("no clips - is data/cremad present?")
    # actor-grouped split: actor id is the filename prefix
    actors = sorted({c[0].name.split("_")[0] for c in clips})
    test_actors = {a for i, a in enumerate(actors) if i % 3 == 0}
    print(f"{len(clips)} clips, {len(actors)} actors "
          f"({len(actors) - len(test_actors)} train / {len(test_actors)} test), "
          f"min listener agreement {args.min_agreement:.0%}")

    audio = [(read(p), h, p.name.split("_")[0] in test_actors) for p, h, _i, _a in clips]
    human = [h for _p, h, _i, _a in clips]

    builders = {
        "audeering": lambda: Audeering(),
        "voxprofile": lambda: VoxProfile(Path(args.cache)),
        "whisper": lambda: Categorical(
            "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
            "apache-2.0", "acted corpora"),
        "superb": lambda: Categorical("superb/wav2vec2-base-superb-er",
                                      "apache-2.0", "IEMOCAP (acted dyads)"),
        "emotion2vec": lambda: Emotion2Vec(),
    }

    summary = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in builders:
            print(f"\n--- {name}: unknown, skipped")
            continue
        print(f"\n{'=' * 74}\n--- {name}")
        try:
            model = builders[name]()
        except Exception as exc:
            print(f"    UNAVAILABLE: {type(exc).__name__}: {str(exc)[:200]}")
            summary.append((name, None, None, None, None,
                            f"unavailable: {type(exc).__name__}"))
            continue
        print(f"    licence: {model.licence}\n    domain : {model.domain}")

        t0 = time.perf_counter()
        full, halves = [], []
        for x, _h, _is_test in audio:
            full.append(model.score(x))
            mid = len(x) // 2
            if mid > int(0.5 * SR):
                halves.append((model.score(x[:mid]), model.score(x[mid:])))
            else:
                halves.append(None)
        ms = (time.perf_counter() - t0) / len(audio) * 1000

        if model.kind == "dimensional":
            vecs = np.stack([s.vec for s in full])
            tr = [i for i, (_x, _h, is_t) in enumerate(audio) if not is_t]
            te = [i for i, (_x, _h, is_t) in enumerate(audio) if is_t]
            cents, scales, labels = fit_centroids(vecs[tr], [human[i] for i in tr])
            pred = [nearest(vecs[i], cents, scales, labels) for i in range(len(vecs))]
            # score-level stability: correlation of each dimension across halves
            pairs = [(a.vec, b.vec) for hp in halves if hp for a, b in [hp]]
            A = np.stack([p[0] for p in pairs]); B = np.stack([p[1] for p in pairs])
            dim_r = [float(np.corrcoef(A[:, k], B[:, k])[0, 1]) for k in range(A.shape[1])]
            half_lab = [(nearest(a, cents, scales, labels),
                         nearest(b, cents, scales, labels)) for a, b in pairs]
        else:
            pred = [s.label for s in full]
            te = [i for i, (_x, _h, is_t) in enumerate(audio) if is_t]
            dim_r = None
            half_lab = [(a.label, b.label) for hp in halves if hp for a, b in [hp]]
            bad = {p for p in pred if p.startswith("UNMAPPED")}
            if bad:
                print(f"    !! classes with no enum mapping: {bad}")

        rep = evaluate([human[i] for i in te], [pred[i] for i in te], TONES)
        st = stability_stats(half_lab)
        stable = st["kappa"]
        print(format_report(rep, f"{name}: agreement with HUMAN voice-only vote (TEST actors)"))
        print(f"\n    stability (same utterance, two halves): "
              f"{st['observed']:.1%} raw, {st['expected']:.1%} expected by its own "
              f"prior -> kappa {st['kappa']:+.2f}")
        print(f"    largest predicted-class share: {st['max_share']:.1%}"
              + ("   <-- DEGENERATE: raw stability is meaningless here"
                 if st["max_share"] > 0.5 else ""))
        if dim_r:
            print(f"    half-to-half score correlation: arousal {dim_r[0]:.2f}  "
                  f"dominance {dim_r[1]:.2f}  valence {dim_r[2]:.2f}")
        print(f"    {ms:.0f} ms per clip")
        summary.append((name, rep.macro_f1, rep.accuracy, stable, ms, model.licence))

    print("\n" + "=" * 74)
    print("SUMMARY - human agreement on TEST actors, and self-consistency")
    print("=" * 74)
    print(f"  {'model':<13}{'macroF1':>9}{'acc':>8}{'stab.kappa':>12}{'ms/clip':>9}  licence")
    for name, f1, acc, st, ms, lic in summary:
        if f1 is None:
            print(f"  {name:<13}{'-':>9}{'-':>8}{'-':>11}{'-':>9}  {lic}")
            continue
        print(f"  {name:<13}{f1:>9.3f}{acc:>8.1%}{st:>+12.2f}{ms:>9.0f}  {lic}")
    print("\n  stab.kappa is chance-corrected self-consistency on two halves of ONE")
    print("  utterance. 0 = no better than the model's own class prior; negative =")
    print("  worse. Raw agreement is NOT usable here: a model that answers one class")
    print("  88% of the time scored 62.3% raw against a 78.2% chance level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
