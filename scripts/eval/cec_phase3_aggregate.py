"""Aggregate Phase 3 eval JSONs and probe logs across the 2x3 matrix.

Consumes:
  - <root>/cec_phase3_eval/<arm>/seed<N>/eval_phase3.json
  - <root>/cec_phase3_probe/<arm>-seed<N>.log    (total L1 spread line)

Reports:
  1. Per-arm × per-testbed reward / R_mean / C_mean / A_mean (mean +- stderr).
  2. Visibility main effect (visible - hidden) per testbed for reward / R_mean.
  3. Symmetric-testbed pattern (heldout_unseen-train, heldout_thinner-train)
     per arm in R_mean.
  4. Per-mission R_mean variance per arm in the per_mission testbed.
  5. PRIMARY DV: per-mission action-distribution L1 spread (from probe logs)
     per arm, vs the falsification thresholds (>=0.15 confirm, <=0.06 null).

Usage::

    python scripts/eval/cec_phase3_aggregate.py jaxborg-exp
"""

# ruff: noqa: E402

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ARMS = ("gen-mission-10x-hidden", "gen-mission-10x-visible")
TESTBEDS = ("train", "heldout_fsm", "per_mission", "heldout_unseen", "heldout_thinner")
METRICS = ("reward", "R_mean", "C_mean", "A_mean")

# Falsification thresholds from cec-phase3-plan.md.
SPREAD_CONFIRM = 0.15
SPREAD_NULL = 0.06

SPREAD_RE = re.compile(r"total L1 spread:\s*([0-9.]+)")


def _stderr(xs):
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def _stats(xs):
    if not xs:
        return None
    n = len(xs)
    m = sum(xs) / n
    return dict(mean=m, stderr=_stderr(xs), n=n)


def _collect_eval(root: Path):
    """by_arm[arm][bed][metric] = [seed values]."""
    eval_root = root / "cec_phase3_eval"
    out: dict = {arm: {bed: {} for bed in TESTBEDS} for arm in ARMS}
    if not eval_root.is_dir():
        return out
    for arm in ARMS:
        for seed_dir in sorted(eval_root.glob(f"{arm}/seed*")):
            ej = seed_dir / "eval_phase3.json"
            if not ej.exists():
                continue
            data = json.loads(ej.read_text())
            for bed in TESTBEDS:
                if bed not in data["testbeds"]:
                    continue
                bd = data["testbeds"][bed]
                slot = out[arm][bed]
                for m in METRICS:
                    if m in bd and "mean" in bd[m]:
                        slot.setdefault(m, []).append(bd[m]["mean"])
                if bed == "per_mission" and "per_mission_profile" in bd:
                    R_means = [v["R_mean"]["mean"] for v in bd["per_mission_profile"].values() if "R_mean" in v]
                    if len(R_means) >= 2:
                        slot.setdefault("R_mean_per_mission_var", []).append(statistics.pvariance(R_means))
    return out


def _collect_probe(root: Path):
    """by_arm[arm] = [total L1 spread per seed]."""
    probe_root = root / "cec_phase3_probe"
    out = {arm: [] for arm in ARMS}
    if not probe_root.is_dir():
        return out
    for arm in ARMS:
        for log in sorted(probe_root.glob(f"{arm}-seed*.log")):
            text = log.read_text()
            m = SPREAD_RE.search(text)
            if m:
                out[arm].append(float(m.group(1)))
    return out


def _print_eval_table(by_arm: dict):
    print("\nPer-arm x per-testbed summary (mean +- stderr over seeds)\n")
    cols = list(METRICS)
    header = f"{'arm':<26} {'testbed':<16}" + "".join(f" {c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        for bed in TESTBEDS:
            row = f"{arm:<26} {bed:<16}"
            for c in cols:
                xs = by_arm[arm][bed].get(c, [])
                if not xs:
                    row += f" {'--':>16}"
                else:
                    s = _stats(xs)
                    row += f" {s['mean']:+.3f}+/-{s['stderr']:.3f}".rjust(17)
            print(row)
        print()


def _visibility_effect(by_arm: dict):
    print("\nVisibility main effect (visible - hidden) per testbed\n")
    for bed in TESTBEDS:
        for metric in ("reward", "R_mean"):
            v = by_arm["gen-mission-10x-visible"][bed].get(metric, [])
            h = by_arm["gen-mission-10x-hidden"][bed].get(metric, [])
            if not v or not h:
                continue
            v_mean = sum(v) / len(v)
            h_mean = sum(h) / len(h)
            print(f"  {bed:<16} {metric:<8} visible={v_mean:+.3f}  hidden={h_mean:+.3f}  diff={v_mean - h_mean:+.3f}")
        print()


def _symmetric_pattern(by_arm: dict):
    print("\nSymmetric-testbed pattern (R_mean drop relative to train)\n")
    print(f"{'arm':<26} {'unseen-train':>14} {'thinner-train':>16}")
    print("-" * 58)
    for arm in ARMS:
        train = by_arm[arm]["train"].get("R_mean", [])
        unseen = by_arm[arm]["heldout_unseen"].get("R_mean", [])
        thinner = by_arm[arm]["heldout_thinner"].get("R_mean", [])
        u = [unseen[i] - train[i] for i in range(min(len(unseen), len(train)))]
        t = [thinner[i] - train[i] for i in range(min(len(thinner), len(train)))]
        u_str = f"{sum(u) / len(u):+.3f}+/-{_stderr(u):.3f}" if u else "--"
        t_str = f"{sum(t) / len(t):+.3f}+/-{_stderr(t):.3f}" if t else "--"
        print(f"{arm:<26} {u_str:>14} {t_str:>16}")


def _per_mission_variance(by_arm: dict):
    print("\nPer-mission R_mean variance (per_mission testbed)\n")
    print(f"{'arm':<26} {'R_mean variance':>18}")
    print("-" * 46)
    for arm in ARMS:
        xs = by_arm[arm]["per_mission"].get("R_mean_per_mission_var", [])
        s = f"{sum(xs) / len(xs):.4f}+/-{_stderr(xs):.4f}" if xs else "--"
        print(f"{arm:<26} {s:>18}")
    print(
        "\nCEC prediction: visible arm shrinks per-mission R_mean variance\n"
        "(specializes uniformly across goals); hidden arm flat or grows."
    )


def _spread_summary(by_arm_spread: dict):
    print("\n*** PRIMARY DV: per-mission action-distribution L1 spread ***\n")
    print(f"{'arm':<26} {'n':>3} {'mean':>8} {'sd':>8} {'stderr':>8} {'verdict':>20}")
    print("-" * 80)
    for arm in ARMS:
        xs = by_arm_spread[arm]
        if not xs:
            print(f"{arm:<26} {'-':>3} {'--':>8} {'--':>8} {'--':>8} {'no probe data':>20}")
            continue
        m = sum(xs) / len(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        se = sd / (len(xs) ** 0.5) if len(xs) > 1 else 0.0
        if m >= SPREAD_CONFIRM:
            verdict = "CONFIRM (>=0.15)"
        elif m <= SPREAD_NULL:
            verdict = "NULL (<=0.06)"
        else:
            verdict = "indeterminate"
        print(f"{arm:<26} {len(xs):>3} {m:>8.4f} {sd:>8.4f} {se:>8.4f} {verdict:>20}")

    print(
        "\nFalsification verdicts (per cec-phase3-plan.md):\n"
        "  - CEC compositional confirmed: visible >= 0.15  AND  hidden <= 0.06\n"
        "  - Action-space limitation:    both <= 0.06\n"
        "  - Bigger multipliers alone:   both > 0.06 with visible > hidden\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="jaxborg-exp dir holding cec_phase3_eval/ and cec_phase3_probe/")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    by_arm = _collect_eval(root)
    by_arm_spread = _collect_probe(root)

    _print_eval_table(by_arm)
    _visibility_effect(by_arm)
    _symmetric_pattern(by_arm)
    _per_mission_variance(by_arm)
    _spread_summary(by_arm_spread)


if __name__ == "__main__":
    main()
