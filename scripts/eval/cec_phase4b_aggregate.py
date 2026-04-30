"""Phase 4b aggregator: tighter ZSC reward + CIA breakdown.

Consumes:
  <root>/cec_phase4b/<arm>/seed<N>/discovery.json       summary (mean reward)
  <root>/cec_phase4b/<arm>/seed<N>/discovery_traj/*.jsonl  per-ep trajectories

Reports (all against `DiscoveryFSRed`):
  1. Per-arm mean reward + R_mean / C_mean / I_mean / A_mean (mean +- stderr).
  2. Diversity effect: each diverse arm minus fixed-nomsg baseline,
     for reward AND each CIA component.

Usage::
    uv run python scripts/eval/cec_phase4b_aggregate.py jaxborg-exp
"""

# ruff: noqa: E402

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jaxborg.eval.cia.cc4_cia_metric import score_trajectory_file

ARMS = (
    "gen-fixed-nomsg",
    "gen-mission-nomsg",
    "gen-mission-10x-hidden",
    "gen-mission-10x-visible",
)
SEEDS = (1, 2, 3)
METRICS = ("reward", "R_mean", "C_mean", "I_mean", "A_mean")


def _stderr(xs):
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def _aggregate_seed(traj_dir: Path) -> dict:
    """Score every .jsonl in `traj_dir`; return per-seed mean of each metric."""
    files = sorted(traj_dir.glob("*.jsonl"))
    if not files:
        return {}
    rows = [score_trajectory_file(p) for p in files]
    return {
        "reward": sum(r.total_reward for r in rows) / len(rows),
        "R_mean": sum(r.R_mean for r in rows) / len(rows),
        "C_mean": sum(r.C_mean for r in rows) / len(rows),
        "I_mean": sum(r.I_mean for r in rows) / len(rows),
        "A_mean": sum(r.A_mean for r in rows) / len(rows),
        "n_episodes": len(rows),
    }


def _collect(root: Path):
    """by_arm[arm][metric] = [per-seed mean]."""
    out = {arm: {m: [] for m in METRICS} for arm in ARMS}
    n_eps = {arm: [] for arm in ARMS}
    base = root / "cec_phase4b"
    if not base.is_dir():
        print(f"missing: {base}", file=sys.stderr)
        return out, n_eps
    for arm in ARMS:
        for seed in SEEDS:
            traj_dir = base / arm / f"seed{seed}" / "discovery_traj"
            if not traj_dir.is_dir():
                print(f"  missing {traj_dir}", file=sys.stderr)
                continue
            seed_stats = _aggregate_seed(traj_dir)
            if not seed_stats:
                continue
            for m in METRICS:
                out[arm][m].append(seed_stats[m])
            n_eps[arm].append(seed_stats["n_episodes"])
    return out, n_eps


def _print_table(by_arm, n_eps):
    print("\nPer-arm summary against DiscoveryFSRed (mean +- stderr over 3 seeds)\n")
    header = f"{'arm':<28} {'n_ep':>6}" + "".join(f"{m:>16}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        row = f"{arm:<28} {sum(n_eps[arm]):>6}"
        for m in METRICS:
            xs = by_arm[arm][m]
            if not xs:
                row += f"{'--':>16}"
            else:
                mean = sum(xs) / len(xs)
                se = _stderr(xs)
                row += f"{mean:+8.3f}+/-{se:>5.3f}".rjust(16)
        print(row)
    print()


def _diversity_effect(by_arm):
    print("Diversity effect: each diverse arm - gen-fixed-nomsg baseline")
    print("(positive = diverse training does better than fixed)\n")
    base = "gen-fixed-nomsg"
    base_data = by_arm[base]
    diverse = [a for a in ARMS if a != base]
    header = f"{'arm':<28}" + "".join(f"{m:>18}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for arm in diverse:
        row = f"{arm:<28}"
        for m in METRICS:
            base_xs = base_data.get(m, [])
            arm_xs = by_arm[arm].get(m, [])
            n = min(len(base_xs), len(arm_xs))
            if n == 0:
                row += f"{'--':>18}"
                continue
            diffs = [arm_xs[i] - base_xs[i] for i in range(n)]
            mean = sum(diffs) / len(diffs)
            se = _stderr(diffs)
            row += f"{mean:+9.3f}+/-{se:>6.3f}".rjust(18)
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="jaxborg-exp dir holding cec_phase4b/")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    by_arm, n_eps = _collect(root)
    _print_table(by_arm, n_eps)
    _diversity_effect(by_arm)


if __name__ == "__main__":
    main()
