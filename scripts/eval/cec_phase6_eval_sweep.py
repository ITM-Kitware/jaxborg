"""Phase 6 Test 2 eval sweep — single-process driver.

Loops through (checkpoint, held-out red) cells sequentially in ONE Python
process so JIT compilation amortizes: 4 reds × 1 compile each, not 24
compiles across separate processes. ~5× faster on a single machine.

Output: one phase6_*.jsonl row per cell, identical schema to
``cec_phase6_eval_jax.py``.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/eval/cec_phase6_eval_sweep.py \\
        --episodes 300 --seed 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Import the run_eval helper from the per-cell script — shares the JIT
# rollout body so behavior is identical.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))
from cec_phase6_eval_jax import run_eval  # noqa: E402

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()

ARMS = ("C00", "C11")
SEEDS = (42, 142, 242)
REDS = ("fsm", "cia_c", "cia_i", "cia_a")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Test 2 sequential eval sweep")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2000, help="Rollout PRNG seed")
    parser.add_argument("--reds", nargs="+", default=list(REDS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    out_dir = EXP_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for arm in args.arms:
        for ts in args.train_seeds:
            tag = f"cec_phase6_{arm}_seed{ts}"
            model = EXP_DIR / "ippo_jax" / tag / f"model_{tag}.safetensors"
            if not model.is_file():
                print(f"SKIP missing checkpoint: {model}", flush=True)
                continue
            for red in args.reds:
                cells.append((tag, model, red))

    print(f"=== sweep: {len(cells)} cells, episodes={args.episodes}, seed={args.seed} ===", flush=True)
    t0_all = time.perf_counter()

    # Group by red so JIT cache hits within each red across all 6 checkpoints.
    cells.sort(key=lambda c: (c[2], c[0]))

    for i, (tag, model, red) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {tag} vs {red}", flush=True)
        t0 = time.perf_counter()
        row = run_eval(model_path=model, eval_red=red, episodes=args.episodes, seed=args.seed)
        eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.seed}_{red}"
        row["eval_id"] = eval_id
        out_path = out_dir / f"phase6_{row['recipe_name']}_{model.stem}_{eval_id}.jsonl"
        out_path.write_text(json.dumps(row, indent=2) + "\n")
        print(f"  mean={row['mean_reward']:.1f} ± {row['std_reward']:.1f} n={row['n_episodes']} wall={time.perf_counter() - t0:.1f}s", flush=True)
        print(f"  wrote {out_path.name}", flush=True)

    print(f"\nTotal wall: {time.perf_counter() - t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
