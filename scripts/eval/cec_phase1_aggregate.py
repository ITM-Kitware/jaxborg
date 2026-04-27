"""Aggregate CEC Phase 1 axis-B eval JSONs across arms × seeds.

Reads ``eval_phase1.json`` files produced by :mod:`cec_phase1_eval`, computes
per-arm mean ± stderr over seeds for each testbed, and prints the primary
research-question DV: ``(held-out − train) CIA gap`` per arm.

Usage::

    python scripts/eval/cec_phase1_aggregate.py <train_root>
"""

# ruff: noqa: E402

import argparse
import json
import statistics
import sys
from pathlib import Path

ARMS = ("gen-fixed", "gen-base", "gen-router", "gen-router-rewards")
TESTBEDS = ("train", "heldout", "heldout_fsm")
METRICS = ("reward", "C_mean", "I_mean", "A_mean", "R_mean")


def _stderr(xs):
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_root")
    args = ap.parse_args()

    root = Path(args.train_root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Collect: by_arm[arm][bed][metric] = [seed_values]
    by_arm: dict[str, dict[str, dict[str, list[float]]]] = {}
    for arm in ARMS:
        by_arm[arm] = {bed: {m: [] for m in METRICS} for bed in TESTBEDS}
        for seed_dir in sorted(root.glob(f"{arm}/seed*")):
            ej = seed_dir / "eval_phase1.json"
            if not ej.exists():
                continue
            data = json.loads(ej.read_text())
            for bed in TESTBEDS:
                if bed not in data["testbeds"]:
                    continue
                bed_data = data["testbeds"][bed]
                for m in METRICS:
                    if m in bed_data and "mean" in bed_data[m]:
                        by_arm[arm][bed][m].append(bed_data[m]["mean"])

    # Print per-arm × per-bed table (mean ± stderr over seeds)
    print(f"\nPer-arm × per-testbed summary (mean ± stderr over seeds, root={root})\n")
    header = f"{'arm':<12} {'testbed':<14}"
    for m in METRICS:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        for bed in TESTBEDS:
            row = f"{arm:<12} {bed:<14}"
            for m in METRICS:
                xs = by_arm[arm][bed][m]
                if not xs:
                    row += f" {'--':>16}"
                else:
                    mu = sum(xs) / len(xs)
                    se = _stderr(xs)
                    row += f" {mu:+.3f} ± {se:.3f}".rjust(17)
            print(row)
        print()

    # Primary DV: (heldout − train) gap per arm, in R_mean
    print("\nPrimary DV: (heldout − train) gap, per arm, in R_mean (composite CIA)")
    print(f"{'arm':<12} {'train':>16} {'heldout':>16} {'heldout − train':>18}")
    print("-" * 64)
    for arm in ARMS:
        train_xs = by_arm[arm]["train"]["R_mean"]
        held_xs = by_arm[arm]["heldout"]["R_mean"]
        if not train_xs or not held_xs:
            print(f"{arm:<12} {'--':>16} {'--':>16} {'--':>18}")
            continue
        # Per-seed gap (paired)
        n = min(len(train_xs), len(held_xs))
        gaps = [held_xs[i] - train_xs[i] for i in range(n)]
        mu_t = sum(train_xs) / len(train_xs)
        mu_h = sum(held_xs) / len(held_xs)
        mu_g = sum(gaps) / len(gaps)
        se_g = _stderr(gaps)
        print(f"{arm:<12} {mu_t:+.3f}  {mu_h:+.3f}  {mu_g:+.3f} ± {se_g:.3f}")

    # CEC prediction: gen-router gap should be SMALLER (less negative or more
    # positive) than gen-fixed and gen-base.
    if by_arm["gen-fixed"]["train"]["R_mean"] and by_arm["gen-router"]["train"]["R_mean"]:
        gf = sum(by_arm["gen-fixed"]["heldout"]["R_mean"]) / len(by_arm["gen-fixed"]["heldout"]["R_mean"]) - sum(
            by_arm["gen-fixed"]["train"]["R_mean"]
        ) / len(by_arm["gen-fixed"]["train"]["R_mean"])
        gb = sum(by_arm["gen-base"]["heldout"]["R_mean"]) / len(by_arm["gen-base"]["heldout"]["R_mean"]) - sum(
            by_arm["gen-base"]["train"]["R_mean"]
        ) / len(by_arm["gen-base"]["train"]["R_mean"])
        gr = sum(by_arm["gen-router"]["heldout"]["R_mean"]) / len(by_arm["gen-router"]["heldout"]["R_mean"]) - sum(
            by_arm["gen-router"]["train"]["R_mean"]
        ) / len(by_arm["gen-router"]["train"]["R_mean"])
        print("\nCEC prediction: gen-router gap should be ≥ gen-fixed and gen-base gaps")
        print(f"  gen-fixed  gap: {gf:+.3f}")
        print(f"  gen-base   gap: {gb:+.3f}")
        print(f"  gen-router gap: {gr:+.3f}")
        if gr > gb and gr > gf:
            print("  → gen-router has the BEST (least negative / most positive) gap. Consistent with CEC.")
        else:
            print("  → gen-router does NOT have the best gap. CEC effect not observed at this scale.")


if __name__ == "__main__":
    main()
