#!/usr/bin/env python3
"""Evaluate every durable checkpoint against the scripted Reds with CIA."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _main() -> None:
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from jaxborg.evaluation.checkpoint_scripted_reds import run_checkpoint_scripted_reds
    from jaxborg.recipe import load

    parser = argparse.ArgumentParser(description="Scripted-Red and CIA curves across a run's checkpoints")
    parser.add_argument("--model", required=True, help="Final model used to locate the training run directory")
    parser.add_argument("--recipe", required=True, help="Resolved recipe sidecar")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    args = parser.parse_args()
    run_checkpoint_scripted_reds(args.model, load(args.recipe), output=args.output)


if __name__ == "__main__":
    _main()
