#!/usr/bin/env python3
"""Evaluate adjacent simultaneous-training checkpoints in both directions."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.evaluation.play_priors import run_play_priors
from jaxborg.recipe import load


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play each current Blue/Red policy against its prior opponent")
    parser.add_argument("--model", required=True, help="Final model used to locate the training run directory")
    parser.add_argument("--recipe", required=True, help="Resolved recipe sidecar")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    args = parser.parse_args(argv)
    run_play_priors(args.model, load(args.recipe), output=args.output)


if __name__ == "__main__":
    main()
