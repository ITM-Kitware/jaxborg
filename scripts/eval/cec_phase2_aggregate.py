"""Aggregate CEC Phase 2 eval JSONs across the 4×3 matrix.

Consumes ``eval_phase2.json`` files produced by :mod:`cec_phase1_eval` after
Phase 2 changes, computes per-arm mean ± stderr over seeds for each testbed
and DV, and tests the falsification criteria from
``plans/jax/cc4/prompts/cec-phase2-plan.md``:

  1. Comms main effect: ``*-msg`` arms beat ``*-nomsg`` on reward / R_mean /
     per-mission CIA variance.
  2. Mission-family main effect: ``gen-mission-*`` beats ``gen-fixed-*``.
  3. Symmetric-testbed pattern: ``gen-mission-msg`` should have the smallest
     ``train → heldout_unseen`` drop and the largest ``train → heldout_thinner``
     drop.
  4. Message-content structure: per-byte std + entropy elevated in ``*-msg``
     arms over ``*-nomsg``.
  5. Norm scorecard: any arm beats ``gen-fixed-nomsg`` on ≥3 of 5 norm rules.

Usage::

    python scripts/eval/cec_phase2_aggregate.py jaxborg-exp/cec_phase2_eval
"""

# ruff: noqa: E402

import argparse
import json
import statistics
import sys
from pathlib import Path

ARMS = ("gen-fixed-nomsg", "gen-fixed-msg", "gen-mission-nomsg", "gen-mission-msg")
TESTBEDS = ("train", "heldout_fsm", "per_mission", "heldout_unseen", "heldout_thinner")
METRICS = ("reward", "R_mean", "C_mean", "A_mean")
NORM_KEYS = ("sleep_fraction", "analyse_fraction", "restore_fraction", "decoy_fraction", "sleep_when_quiet")
MSG_KEYS = ("msg_per_byte_std_mean", "msg_quantile_entropy_mean")


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


def _collect(root: Path):
    """by_arm[arm][bed]['<metric>'|'norms'|'messages'] = [seed values]."""
    out: dict = {}
    for arm in ARMS:
        out[arm] = {bed: {} for bed in TESTBEDS}
        for seed_dir in sorted(root.glob(f"{arm}/seed*")):
            ej = seed_dir / "eval_phase2.json"
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
                if "norms" in bd:
                    for nk in NORM_KEYS:
                        if nk in bd["norms"]:
                            slot.setdefault(f"norm_{nk}", []).append(bd["norms"][nk]["mean"])
                if "messages" in bd:
                    for mk in MSG_KEYS:
                        if mk in bd["messages"]:
                            slot.setdefault(f"msg_{mk}", []).append(bd["messages"][mk]["mean"])
                # Per-mission CIA variance: spread of R_mean across profiles in the per_mission testbed.
                if bed == "per_mission" and "per_mission_profile" in bd:
                    profs = bd["per_mission_profile"]
                    R_means = [v["R_mean"]["mean"] for v in profs.values() if "R_mean" in v]
                    if len(R_means) >= 2:
                        slot.setdefault("R_mean_per_mission_var", []).append(
                            statistics.pvariance(R_means)
                        )
    return out


def _print_table(by_arm: dict):
    print("\nPer-arm × per-testbed summary (mean ± stderr over seeds)")
    print()
    cols = ["reward", "R_mean", "C_mean", "A_mean"]
    header = f"{'arm':<22} {'testbed':<16}" + "".join(f" {c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        for bed in TESTBEDS:
            row = f"{arm:<22} {bed:<16}"
            for c in cols:
                xs = by_arm[arm][bed].get(c, [])
                if not xs:
                    row += f" {'--':>16}"
                else:
                    s = _stats(xs)
                    row += f" {s['mean']:+.3f}±{s['stderr']:.3f}".rjust(17)
            print(row)
        print()


def _gap(by_arm, arm, bed_a, bed_b, metric):
    """Per-seed paired gap (bed_a − bed_b) on metric."""
    a = by_arm[arm][bed_a].get(metric, [])
    b = by_arm[arm][bed_b].get(metric, [])
    n = min(len(a), len(b))
    return [a[i] - b[i] for i in range(n)]


def _main_effects(by_arm: dict):
    print("\nMain effects (averaged over the other axis, mean ± stderr over 6 seeds)")
    print()
    for bed in TESTBEDS:
        print(f"  testbed={bed}")
        for metric in ("reward", "R_mean"):
            msg_xs = by_arm["gen-fixed-msg"][bed].get(metric, []) + by_arm["gen-mission-msg"][bed].get(metric, [])
            nomsg_xs = by_arm["gen-fixed-nomsg"][bed].get(metric, []) + by_arm["gen-mission-nomsg"][bed].get(metric, [])
            mission_xs = by_arm["gen-mission-nomsg"][bed].get(metric, []) + by_arm["gen-mission-msg"][bed].get(metric, [])
            fixed_xs = by_arm["gen-fixed-nomsg"][bed].get(metric, []) + by_arm["gen-fixed-msg"][bed].get(metric, [])
            comms_diff = (sum(msg_xs) / max(len(msg_xs), 1)) - (sum(nomsg_xs) / max(len(nomsg_xs), 1))
            mission_diff = (sum(mission_xs) / max(len(mission_xs), 1)) - (sum(fixed_xs) / max(len(fixed_xs), 1))
            print(
                f"    {metric:<10}  comms(msg−nomsg)={comms_diff:+.3f}   "
                f"mission(mission−fixed)={mission_diff:+.3f}"
            )
        print()


def _symmetric_pattern(by_arm: dict):
    """heldout_unseen − train AND heldout_thinner − train per arm, in R_mean."""
    print("\nSymmetric-testbed pattern (R_mean drop relative to train)")
    print(f"{'arm':<22} {'unseen − train':>16} {'thinner − train':>18}")
    print("-" * 60)
    for arm in ARMS:
        u_gap = _gap(by_arm, arm, "heldout_unseen", "train", "R_mean")
        t_gap = _gap(by_arm, arm, "heldout_thinner", "train", "R_mean")
        u_str = (
            f"{sum(u_gap) / len(u_gap):+.3f}±{_stderr(u_gap):.3f}" if u_gap else "--"
        )
        t_str = (
            f"{sum(t_gap) / len(t_gap):+.3f}±{_stderr(t_gap):.3f}" if t_gap else "--"
        )
        print(f"{arm:<22} {u_str:>16} {t_str:>18}")
    print(
        "\nCEC prediction: gen-mission-msg has the smallest (least negative) "
        "unseen−train gap and the largest (most negative) thinner−train gap."
    )


def _comms_content(by_arm: dict):
    print("\nMessage-content metrics (mean over seeds, train testbed)")
    print(f"{'arm':<22} {'msg_std':>10} {'msg_entropy':>14}")
    print("-" * 50)
    for arm in ARMS:
        std = by_arm[arm]["train"].get("msg_msg_per_byte_std_mean", [])
        ent = by_arm[arm]["train"].get("msg_msg_quantile_entropy_mean", [])
        s_str = f"{sum(std) / len(std):.4f}" if std else "--"
        e_str = f"{sum(ent) / len(ent):.3f}" if ent else "--"
        print(f"{arm:<22} {s_str:>10} {e_str:>14}")


def _norm_scorecard(by_arm: dict):
    print("\nNorm scorecard (mean over seeds, train testbed)")
    cols = NORM_KEYS
    header = f"{'arm':<22}" + "".join(f" {c:>17}" for c in cols)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        row = f"{arm:<22}"
        for nk in cols:
            xs = by_arm[arm]["train"].get(f"norm_{nk}", [])
            row += f" {sum(xs) / len(xs):.3f}".rjust(18) if xs else f" {'--':>17}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root")
    args = ap.parse_args()
    root = Path(args.eval_root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    by_arm = _collect(root)
    _print_table(by_arm)
    _main_effects(by_arm)
    _symmetric_pattern(by_arm)
    _comms_content(by_arm)
    _norm_scorecard(by_arm)


if __name__ == "__main__":
    main()
