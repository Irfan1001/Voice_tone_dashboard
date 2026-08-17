#!/usr/bin/env python
"""Run the pipeline over a set of clips and report every metric, reproducibly.

    python -m evaluation.evaluate data/ --out out/run1
    python -m evaluation.evaluate data/ --labels data/labels.csv --out out/run1

Every run writes `run_meta.json` with the model ids, every threshold that could
change an output, the seed and library versions. Two runs with matching
`run_meta.json` must produce identical `predictions.csv`; if not, that is a
determinism bug, not noise.

`--labels` is optional, and the report always says what it scored against. The three
supplied labels have known defects, so a headline accuracy against them would
mislead. `background_noise_type` is free text and is reported as exact-match plus the
raw pairs, because "television" against a label of "TV" is not a miss in any useful
sense and no string metric will tell you that.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT, Config  # noqa: E402
from app.env import load_env  # noqa: E402
from app.pipeline import AudioPipeline  # noqa: E402
from app.schema import (  # noqa: E402
    AudioQuality,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
)
from evaluation.metrics import evaluate as score_field  # noqa: E402
from evaluation.metrics import format_report  # noqa: E402

AUDIO_EXT = {".wav", ".ogg", ".opus", ".mp3", ".flac", ".m4a"}

LABEL_SETS: dict[str, list[str]] = {
    "emotional_tone": [str(t) for t in EmotionalTone],
    "emotional_intensity": [str(i) for i in EmotionalIntensity],
    "background_noise_severity": [str(s) for s in NoiseSeverity],
    "audio_quality": [str(q) for q in AudioQuality],
    "background_noise_present": ["false", "true"],
    "speaker_overlap_present": ["false", "true"],
    "long_silence_present": ["false", "true"],
}
FREE_TEXT = ["background_noise_type"]
NUMERIC = ["confidence"]


def run_meta(cfg: Config, files: list[Path]) -> dict:
    r = cfg.rules
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "unavailable"
    return {
        "models": {
            "emotion": cfg.models.emotion_dimensional,
            "audio_events": cfg.models.audio_events,
            "segmentation": cfg.models.segmentation,
            "diarization": cfg.models.diarization,
            "role_asr": cfg.models.role_asr,
        },
        "strategies": {
            "emotion": cfg.emotion_strategy, "noise": cfg.noise_strategy,
            "quality": cfg.quality_strategy, "silence": cfg.silence_strategy,
            "overlap": cfg.overlap_strategy,
        },
        "decision_thresholds": {
            "arousal_medium": r.arousal_medium, "arousal_high": r.arousal_high,
            "tone_arousal_low": r.tone_arousal_low,
            "tone_arousal_high": r.tone_arousal_high,
            "upset_dominance": r.upset_dominance, "grief_dominance": r.grief_dominance,
            "valence_negative": r.valence_negative,
            "valence_positive": r.valence_positive,
            "overlap_min_seconds": r.overlap_min_seconds,
            "long_silence_s": cfg.silence.long_silence_s,
            "role_opening_s": r.role_opening_s, "role_min_margin": r.role_min_margin,
        },
        "seed": cfg.seed,
        "device": cfg.device,
        "versions": {"python": platform.python_version(), "torch": torch_v,
                     "platform": platform.platform()},
        "inputs": [{"name": f.name, "bytes": f.stat().st_size} for f in files],
    }


def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["name"]] = json.loads(row["result_json"])
    return rows


def collect(root: Path, limit: int | None) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXT)
    return files[:limit] if limit else files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="file or directory of audio")
    ap.add_argument("--labels", help="CSV: name,result_json (optional)")
    ap.add_argument("--out", default="out/eval", help="output directory")
    ap.add_argument("--limit", type=int, help="only the first N files")
    load_env()
    args = ap.parse_args(argv)

    cfg = DEFAULT
    files = collect(Path(args.audio), args.limit)
    if not files:
        raise SystemExit(f"no audio found under {args.audio}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "run_meta.json").write_text(
        json.dumps(run_meta(cfg, files), indent=2) + "\n")

    pipe = AudioPipeline(cfg)
    rows, failures = [], []
    audio_s = wall_s = 0.0
    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        try:
            result, diag = pipe.run(str(path))
        except Exception as exc:
            failures.append({"name": path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}/{len(files)}] {path.name}: FAILED {type(exc).__name__}")
            continue
        elapsed = time.perf_counter() - t0
        wall_s += elapsed
        audio_s += diag.duration_s
        got = result.to_dict()
        em = diag.evidence.get("emotion", {})
        iso = (diag.evidence.get("features", {}) or {}).get("customer_isolation") or {}
        rows.append({
            "name": path.name,
            "duration_s": round(diag.duration_s, 2),
            "wall_s": round(elapsed, 2),
            **{k: str(v) for k, v in got.items()},
            "arousal": em.get("arousal"), "dominance": em.get("dominance"),
            "valence": em.get("valence"), "scored_on": em.get("scored_on"),
            "customer_isolated": iso.get("customer_isolated"),
            "role_margin": iso.get("role_margin"),
            "warnings": " | ".join(diag.warnings),
        })
        print(f"  [{i}/{len(files)}] {path.name}: {got['emotional_tone']}/"
              f"{got['emotional_intensity']}  {elapsed:.1f}s")

    if rows:
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    latency = {
        "clips": len(rows), "failed": len(failures),
        "audio_seconds": round(audio_s, 1), "wall_seconds": round(wall_s, 1),
        "seconds_per_audio_minute": round(wall_s / audio_s * 60, 1) if audio_s else None,
        "realtime_factor": round(audio_s / wall_s, 2) if wall_s else None,
        "compute_cost_usd_per_audio_minute": (
            round(wall_s / audio_s * 60 / 3600 * 0.04, 8) if audio_s else None),
        "assumed_machine_rate_usd_per_hour": 0.04,
        "ceiling_usd_per_audio_minute": 0.003,
    }

    report_lines: list[str] = []
    metrics: dict = {"latency": latency}

    if args.labels:
        labels = load_labels(Path(args.labels))
        missing = [r["name"] for r in rows if r["name"] not in labels]
        if missing:
            report_lines.append(f"NOTE: no label for {missing}")
        paired = [(r, labels[r["name"]]) for r in rows if r["name"] in labels]
        report_lines.append(
            f"SCORED AGAINST {args.labels} - read this module's docstring before "
            "quoting any number below.\n")
        for field, label_set in LABEL_SETS.items():
            true = [str(lab[field]).lower() for _, lab in paired]
            pred = [str(r[field]).lower() for r, _ in paired]
            rep = score_field(true, pred, label_set)
            metrics[field] = rep.to_dict()
            report_lines.append(format_report(rep, f"{field}") + "\n")
        for field in FREE_TEXT:
            pairs = [(str(lab[field]), str(r[field])) for r, lab in paired]
            exact = sum(1 for a, b in pairs if a.strip().lower() == b.strip().lower())
            metrics[field] = {"exact_match": exact, "n": len(pairs),
                              "pairs": [{"label": a, "predicted": b} for a, b in pairs]}
            report_lines.append(
                f"{field}: {exact}/{len(pairs)} exact. Free text - judge these by eye:")
            for a, b in pairs:
                report_lines.append(f"   label={a!r:<20} predicted={b!r}")
            report_lines.append("")
        for field in NUMERIC:
            vals = [(float(lab[field]), float(r[field])) for r, lab in paired]
            mae = sum(abs(a - b) for a, b in vals) / len(vals) if vals else None
            distinct = len({a for a, _ in vals})
            metrics[field] = {"mae": round(mae, 4) if mae is not None else None,
                              "distinct_label_values": distinct}
            report_lines.append(f"{field}: MAE {mae:.3f} over {len(vals)} clips")
            if distinct <= 1:
                report_lines.append(
                    "   WARNING: every label has the same value, so this measures "
                    "nothing about ranking or calibration - only distance to a "
                    "constant. Do not tune against it.")
            report_lines.append("")

    report = "\n".join([
        f"clips={latency['clips']} failed={latency['failed']}",
        f"audio={latency['audio_seconds']}s wall={latency['wall_seconds']}s "
        f"-> {latency['seconds_per_audio_minute']}s per audio minute, "
        f"{latency['realtime_factor']}x realtime",
        f"compute cost ${latency['compute_cost_usd_per_audio_minute']} per audio "
        f"minute (ceiling ${latency['ceiling_usd_per_audio_minute']})",
        "", *report_lines,
    ])
    (out / "report.txt").write_text(report + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("\n" + report)
    print(f"wrote {out}/predictions.csv, metrics.json, report.txt, run_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
