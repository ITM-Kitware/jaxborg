"""Aggregate CIA scores across multiple checkpoint trajectory directories.

For the CEC-pilot fixed-vs-stoch comparison: produces a per-condition table of
mean ± stderr for reward and CIA scalars across seeds, plus the headline
fixed-vs-stoch deltas with simple paired tests.
"""

# ruff: noqa: E402

import argparse
import json
import os
import re
import statistics as stats
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jaxborg.evaluation.cia import score_trajectory_file


def _stderr(xs):
    if len(xs) < 2:
        return 0.0
    return stats.stdev(xs) / (len(xs) ** 0.5)


def aggregate_dir(traj_dir: Path) -> dict:
    files = sorted(traj_dir.glob("*.jsonl"))
    if not files:
        return {"n": 0}
    rows = [score_trajectory_file(p) for p in files]
    rewards = [r.total_reward for r in rows]
    out = {
        "n_episodes": len(rows),
        "reward_mean": stats.mean(rewards),
        "reward_stderr": _stderr(rewards),
    }
    for k in ("C_mean", "I_mean", "A_mean", "R_mean"):
        vals = [getattr(r, k) for r in rows]
        out[k] = stats.mean(vals)
        out[f"{k}_stderr"] = _stderr(vals)
    # Action counts
    red_total = {}
    blue_total = {}
    for r in rows:
        for k, v in r.red_event_counts.items():
            red_total[k] = red_total.get(k, 0) + v
        for k, v in r.blue_event_counts.items():
            blue_total[k] = blue_total.get(k, 0) + v
    out["red_events"] = red_total
    out["blue_events"] = blue_total
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate CIA scores across CEC-pilot checkpoint dirs")
    parser.add_argument("root", help="Parent dir containing one subdir per checkpoint")
    parser.add_argument("--pattern", default=r"cec_cyborg_(?P<cond>fixed|stoch)_s(?P<seed>\d+)")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    pat = re.compile(args.pattern)
    by_cond: dict[str, list] = {"fixed": [], "stoch": []}
    per_seed = {}

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        cond = m.group("cond")
        seed = int(m.group("seed"))
        agg = aggregate_dir(d)
        if agg.get("n_episodes", 0) == 0:
            continue
        by_cond[cond].append({"seed": seed, **agg})
        per_seed[d.name] = agg

    print(f"\n{'checkpoint':<40s}  {'n':>3s}  {'reward':>11s}  {'C':>9s}  {'I':>9s}  {'A':>9s}  {'R':>9s}")
    for cond in ("fixed", "stoch"):
        for r in sorted(by_cond[cond], key=lambda x: x["seed"]):
            tag = f"cec_cyborg_{cond}_s{r['seed']}"
            print(
                f"{tag:<40s}  {r['n_episodes']:>3d}  "
                f"{r['reward_mean']:>+8.1f}±{r['reward_stderr']:.1f}  "
                f"{r['C_mean']:+.3f}±{r['C_mean_stderr']:.3f}  "
                f"{r['I_mean']:+.3f}±{r['I_mean_stderr']:.3f}  "
                f"{r['A_mean']:+.3f}±{r['A_mean_stderr']:.3f}  "
                f"{r['R_mean']:+.3f}±{r['R_mean_stderr']:.3f}"
            )

    print("\n=== per-condition aggregate (mean across seeds) ===")
    cond_summary = {}
    for cond in ("fixed", "stoch"):
        rows = by_cond[cond]
        if not rows:
            continue
        n = len(rows)
        rew = [r["reward_mean"] for r in rows]
        cias = {k: [r[k] for r in rows] for k in ("C_mean", "I_mean", "A_mean", "R_mean")}
        cond_summary[cond] = {
            "n_seeds": n,
            "reward_mean": stats.mean(rew),
            "reward_stderr": _stderr(rew),
            **{k: {"mean": stats.mean(v), "stderr": _stderr(v)} for k, v in cias.items()},
        }
        print(
            f"  {cond:<8s} n={n}  reward {stats.mean(rew):+9.1f}±{_stderr(rew):.1f}  "
            f"C {stats.mean(cias['C_mean']):+.3f}±{_stderr(cias['C_mean']):.3f}  "
            f"I {stats.mean(cias['I_mean']):+.3f}±{_stderr(cias['I_mean']):.3f}  "
            f"A {stats.mean(cias['A_mean']):+.3f}±{_stderr(cias['A_mean']):.3f}  "
            f"R {stats.mean(cias['R_mean']):+.3f}±{_stderr(cias['R_mean']):.3f}"
        )

    if "fixed" in cond_summary and "stoch" in cond_summary:
        print("\n=== fixed → stoch delta (CEC effect direction; positive = stoch>fixed) ===")
        for k in ("reward_mean", "C_mean", "I_mean", "A_mean", "R_mean"):
            if k == "reward_mean":
                fm, fs = cond_summary["fixed"]["reward_mean"], cond_summary["fixed"]["reward_stderr"]
                sm, ss = cond_summary["stoch"]["reward_mean"], cond_summary["stoch"]["reward_stderr"]
                delta = sm - fm
                pooled = (fs**2 + ss**2) ** 0.5
            else:
                fm, fs = cond_summary["fixed"][k]["mean"], cond_summary["fixed"][k]["stderr"]
                sm, ss = cond_summary["stoch"][k]["mean"], cond_summary["stoch"][k]["stderr"]
                delta = sm - fm
                pooled = (fs**2 + ss**2) ** 0.5
            z = delta / pooled if pooled > 0 else 0.0
            print(f"  {k:<14s}  fixed {fm:+8.3f}  stoch {sm:+8.3f}  Δ {delta:+8.3f}  z≈{z:+.2f}")

    print("\n=== red event totals (fixed vs stoch, sum across all episodes) ===")
    fr = {}
    sr = {}
    for r in by_cond["fixed"]:
        for k, v in r["red_events"].items():
            fr[k] = fr.get(k, 0) + v
    for r in by_cond["stoch"]:
        for k, v in r["red_events"].items():
            sr[k] = sr.get(k, 0) + v
    keys = sorted(set(fr) | set(sr))
    print(f"  {'event':<35s}  {'fixed':>8s}  {'stoch':>8s}")
    for k in keys:
        print(f"  {k:<35s}  {fr.get(k, 0):>8d}  {sr.get(k, 0):>8d}")

    print("\n=== blue event totals ===")
    fb = {}
    sb = {}
    for r in by_cond["fixed"]:
        for k, v in r["blue_events"].items():
            fb[k] = fb.get(k, 0) + v
    for r in by_cond["stoch"]:
        for k, v in r["blue_events"].items():
            sb[k] = sb.get(k, 0) + v
    keys = sorted(set(fb) | set(sb))
    print(f"  {'event':<35s}  {'fixed':>8s}  {'stoch':>8s}")
    for k in keys:
        print(f"  {k:<35s}  {fb.get(k, 0):>8d}  {sb.get(k, 0):>8d}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"per_seed": per_seed, "by_condition": cond_summary}, indent=2) + "\n")
        print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
