"""Aggregate Phase 4 ZSC eval JSONs across the 2 arms x 3 seeds x 3 reds grid.

Consumes:
  <root>/cec_phase4_zsc/<arm>/seed<N>/<red>.json   (mean reward over EPISODES)

Reports:
  1. Per-arm x per-red mean reward (mean +- stderr over seeds).
  2. ZSC drop: held-out_red - training_red, per arm.
  3. Visibility main effect on held-out reds (visible - hidden) per red.

Usage::
    python scripts/eval/cec_phase4_aggregate.py jaxborg-exp
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ARMS = (
    "gen-fixed-nomsg",  # Phase 2: no diversity baseline (fixed topo + fixed mission)
    "gen-mission-nomsg",  # Phase 2: 5x mission bank, varied topology
    "gen-mission-10x-hidden",  # Phase 3: 10x mission bank, goal hidden
    "gen-mission-10x-visible",  # Phase 3: 10x mission bank, goal in obs
)
SEEDS = (1, 2, 3)
REDS = ("fsm", "discovery", "random")
TRAINING_RED = "fsm"


def _stderr(xs):
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def _collect(root: Path):
    """by_arm[arm][red] = [mean reward per seed]."""
    out = {arm: {red: [] for red in REDS} for arm in ARMS}
    zsc_root = root / "cec_phase4_zsc"
    if not zsc_root.is_dir():
        print(f"missing: {zsc_root}", file=sys.stderr)
        return out
    for arm in ARMS:
        for seed in SEEDS:
            for red in REDS:
                jp = zsc_root / arm / f"seed{seed}" / f"{red}.json"
                if not jp.exists():
                    print(f"  missing {jp}", file=sys.stderr)
                    continue
                d = json.loads(jp.read_text())
                out[arm][red].append(d["mean"])
    return out


def _print_table(by_arm):
    print("\nPer-arm x per-red mean reward (mean +- stderr over 3 seeds, 30 episodes each)\n")
    header = f"{'arm':<28} " + "".join(f"{red:>20}" for red in REDS)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        row = f"{arm:<28} "
        for red in REDS:
            xs = by_arm[arm][red]
            if not xs:
                row += f"{'--':>20}"
            else:
                m = sum(xs) / len(xs)
                se = _stderr(xs)
                row += f"{m:+10.2f}+/-{se:>6.2f}".rjust(20)
        print(row)
    print()


def _zsc_drop(by_arm):
    print("ZSC drop: held-out red reward - training red (fsm) reward, per arm")
    print("(positive = held-out is easier; negative = generalization gap)\n")
    header = f"{'arm':<28} " + "".join(f"{red:>20}" for red in REDS if red != TRAINING_RED)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        train_xs = by_arm[arm][TRAINING_RED]
        if not train_xs:
            continue
        row = f"{arm:<28} "
        for red in REDS:
            if red == TRAINING_RED:
                continue
            held_xs = by_arm[arm][red]
            n = min(len(train_xs), len(held_xs))
            if n == 0:
                row += f"{'--':>20}"
                continue
            diffs = [held_xs[i] - train_xs[i] for i in range(n)]
            m = sum(diffs) / len(diffs)
            se = _stderr(diffs)
            row += f"{m:+10.2f}+/-{se:>6.2f}".rjust(20)
        print(row)
    print()


def _diversity_effect(by_arm):
    print("\nDiversity effect: each diverse arm minus the no-diversity baseline (fixed-nomsg), per red")
    print("(positive = diversity helps; negative = diversity hurts)\n")
    base = "gen-fixed-nomsg"
    base_by_red = by_arm.get(base, {})
    diverse_arms = [a for a in ARMS if a != base]
    print(f"{'arm':<28} " + "".join(f"{red:>20}" for red in REDS))
    print("-" * (28 + 20 * len(REDS)))
    for arm in diverse_arms:
        row = f"{arm:<28} "
        for red in REDS:
            base_xs = base_by_red.get(red, [])
            arm_xs = by_arm[arm].get(red, [])
            n = min(len(base_xs), len(arm_xs))
            if n == 0:
                row += f"{'--':>20}"
                continue
            diffs = [arm_xs[i] - base_xs[i] for i in range(n)]
            m = sum(diffs) / len(diffs)
            se = _stderr(diffs)
            row += f"{m:+10.2f}+/-{se:>6.2f}".rjust(20)
        print(row)
    print()


def _visibility_effect(by_arm):
    print("Visibility main effect on held-out reds (visible - hidden) per red")
    print("(positive = visibility helps generalize; negative = visibility hurts)\n")
    print(f"{'red':<14} {'visible_mean':>14} {'hidden_mean':>14} {'diff':>12} {'stderr':>10}")
    print("-" * 70)
    for red in REDS:
        v = by_arm["gen-mission-10x-visible"][red]
        h = by_arm["gen-mission-10x-hidden"][red]
        if not v or not h:
            continue
        v_m = sum(v) / len(v)
        h_m = sum(h) / len(h)
        n = min(len(v), len(h))
        diffs = [v[i] - h[i] for i in range(n)]
        se = _stderr(diffs)
        marker = ""
        if red != TRAINING_RED:
            marker = "  <- ZSC test"
        print(f"{red:<14} {v_m:>14.2f} {h_m:>14.2f} {(v_m - h_m):>+12.2f} {se:>10.2f}{marker}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="jaxborg-exp dir holding cec_phase4_zsc/")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    by_arm = _collect(root)
    _print_table(by_arm)
    _zsc_drop(by_arm)
    _diversity_effect(by_arm)
    _visibility_effect(by_arm)


if __name__ == "__main__":
    main()
