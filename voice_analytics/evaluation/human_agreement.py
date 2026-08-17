#!/usr/bin/env python
"""Score an emotion model against HUMAN PERCEPTION, and measure its stability.

    python -m evaluation.human_agreement --limit 250
    python -m evaluation.human_agreement --limit 250 --min-agreement 0.8

The target is `VoiceVote` - the majority verdict of listeners who heard the audio
only - not the emotion the actor was DIRECTED to perform. Humans recover that intent
only 41.6% of the time from voice alone (SAD 16.4%), so scoring against it penalises
a model for agreeing with the average listener.

Two numbers, both required:

* agreement with the human voice-only vote, on clips where humans agreed
* stability: the same utterance split in half and scored twice. A verdict that flips
  between halves of one sentence cannot hold across a call, whatever the benchmark
  says. Needs no labels at all.

Ratings are not redistributed here; they download on first use from the CREMA-D
project (github.com/CheyneyComputerScience/CREMA-D, ODbL alongside the audio).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT  # noqa: E402
from app.env import load_env  # noqa: E402
from app.rules.engine import map_dimensional  # noqa: E402
from app.schema import EmotionalTone  # noqa: E402
from evaluation.metrics import evaluate, format_report  # noqa: E402

SR = 16000
AUDIO = Path("data/cremad/data/AudioWAV")
RATINGS = Path("data/cremad/summaryTable.csv")
VOTES = Path("data/cremad/tabulatedVotes.csv")
RATINGS_URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/"
               "master/processedResults/summaryTable.csv")
VOTES_URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/"
             "master/processedResults/tabulatedVotes.csv")

# Human vote letter / CREMA-D code -> the brief's five tones.
TONE_OF = {
    "N": "neutral", "NEU": "neutral",
    "H": "satisfied", "HAP": "satisfied",
    "A": "upset", "ANG": "upset",
    "D": "frustrated", "DIS": "frustrated",
    "F": "distressed", "FEA": "distressed",
    "S": "distressed", "SAD": "distressed",
}
TONES = [str(t) for t in EmotionalTone]


def fetch(path: Path, url: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {path.name} from the CREMA-D project ...")
    urllib.request.urlretrieve(url, path)


def voice_agreement() -> dict[str, float]:
    """Per-clip agreement among voice-only listeners.

    `tabulatedVotes.csv` holds three rows per clip, one per presentation modality,
    identified by id prefix: 1xxxxx voice, 2xxxxx face, 3xxxxx multimodal.
    """
    rows = list(csv.DictReader(VOTES.open(encoding="utf-8")))
    prefixes = Counter(str(r[""])[0] for r in rows)
    if prefixes.get("1", 0) * 3 != len(rows):
        print(f"  WARNING: unexpected id prefixes {dict(prefixes)}; "
              "treating the lowest-id row per clip as the voice-only rating")
    out: dict[str, float] = {}
    for r in rows:
        if str(r[""]).startswith("1"):
            try:
                out[r["fileName"]] = float(r["agreement"])
            except (TypeError, ValueError):
                continue
    return out


def load_clips(limit_per_tone: int, min_agreement: float
               ) -> list[tuple[Path, str, str, float]]:
    """(path, human_tone, intended_tone, agreement), stratified by HUMAN tone."""
    fetch(RATINGS, RATINGS_URL)
    fetch(VOTES, VOTES_URL)
    agree = voice_agreement()
    buckets: dict[str, list] = {}
    for row in csv.DictReader(RATINGS.open(encoding="utf-8")):
        name = row["FileName"]
        vote = row["VoiceVote"].strip()
        if vote not in TONE_OF:          # ties like "N:S" - humans disagreed
            continue
        parts = name.split("_")
        if len(parts) != 4 or parts[2] not in TONE_OF:
            continue
        a = agree.get(name, 0.0)
        if a < min_agreement:
            continue
        path = AUDIO / f"{name}.wav"
        if not path.is_file():
            continue
        buckets.setdefault(TONE_OF[vote], []).append(
            (path, TONE_OF[vote], TONE_OF[parts[2]], a))
    out = []
    for tone in sorted(buckets):
        out.extend(sorted(buckets[tone], key=lambda t: t[0].name)[:limit_per_tone])
    return out


def read(path: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=250,
                    help="clips per HUMAN-voted tone")
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="drop clips where voice-only listeners agreed less than this")
    load_env()
    args = ap.parse_args(argv)

    clips = load_clips(args.limit, args.min_agreement)
    if not clips:
        raise SystemExit("no clips matched - is data/cremad present?")
    print(f"\n{len(clips)} clips, min listener agreement {args.min_agreement:.0%}")
    print("  human-voted tones: " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(c[1] for c in clips).items())))

    from app.models.emotion import _DimensionalBackend
    backend = _DimensionalBackend(DEFAULT.models.emotion_dimensional, "cpu", 0)
    if not backend.load():
        raise SystemExit(backend.reason)

    human, intent, pred, first_half, second_half = [], [], [], [], []
    t0 = time.perf_counter()
    for i, (path, h_tone, i_tone, _a) in enumerate(clips, 1):
        x = read(path)
        mean, _ = backend.scores([x], SR)
        tone, _ = map_dimensional(float(mean[0]), float(mean[1]), float(mean[2]),
                                  DEFAULT.rules)
        human.append(h_tone)
        intent.append(i_tone)
        pred.append(str(tone))
        # stability: the SAME utterance, split down the middle, scored twice
        half = len(x) // 2
        if half > int(0.5 * SR):
            for store, seg in ((first_half, x[:half]), (second_half, x[half:])):
                m, _ = backend.scores([seg], SR)
                t, _ = map_dimensional(float(m[0]), float(m[1]), float(m[2]),
                                       DEFAULT.rules)
                store.append(str(t))
        if i % 100 == 0:
            print(f"  scored {i}/{len(clips)}")
    elapsed = time.perf_counter() - t0

    print("\n" + format_report(evaluate(human, pred, TONES),
                               "AGREEMENT WITH HUMAN VOICE-ONLY VOTE"))
    print("\n" + format_report(evaluate(intent, pred, TONES),
                               "AGREEMENT WITH THE ACTOR'S INTENT (the old target)"))
    print("\n" + format_report(evaluate(human, intent, TONES),
                               "CEILING: humans vs the actor's intent"))

    if first_half:
        flips = sum(1 for a, b in zip(first_half, second_half) if a != b)
        print(f"\nSTABILITY (same utterance, two halves): "
              f"{len(first_half) - flips}/{len(first_half)} consistent "
              f"= {1 - flips / len(first_half):.1%}")
        print("  A verdict that changes between halves of one sentence will not hold "
              "across a call.")
        pairs = Counter(tuple(sorted((a, b)))
                        for a, b in zip(first_half, second_half) if a != b)
        for (a, b), n in pairs.most_common(5):
            print(f"    {a} <-> {b}: {n}")

    print(f"\n{elapsed / len(clips) * 1000:.0f} ms per clip "
          f"({DEFAULT.models.emotion_dimensional})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
