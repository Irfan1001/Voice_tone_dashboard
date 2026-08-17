#!/usr/bin/env python
"""Analyse one audio file and print the required JSON.

    python run_pipeline.py data/call_001.ogg
    python run_pipeline.py data/call_001.ogg --diagnostics
    python run_pipeline.py data/call_001.ogg --out result.json

Only the nine-field result goes to stdout, so the command pipes into jq or a
file. Diagnostics and warnings go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import DEFAULT
from app.env import load_env
from app.errors import PipelineError
from app.pipeline import AudioPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("audio", help="path to an audio file")
    p.add_argument("--out", help="write the JSON result here instead of stdout")
    p.add_argument("--diagnostics", action="store_true",
                   help="print evidence, latency and confidence breakdown to stderr")
    p.add_argument("--compact", action="store_true", help="single-line JSON")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="log every stage to stderr (DEBUG level)")
    return p


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        result, diag = AudioPipeline(DEFAULT).run(args.audio)
    except PipelineError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}, indent=2),
              file=sys.stderr)
        return 2

    payload = result.to_json(indent=None if args.compact else 2)
    if args.out:
        Path(args.out).write_text(payload + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload)

    if args.diagnostics:
        print(json.dumps(diag.to_dict(), indent=2, default=str), file=sys.stderr)

    for w in diag.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
