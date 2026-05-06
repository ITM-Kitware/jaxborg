"""Score CC4 trajectory JSONL files with the resilience CIA metric.

Reads files produced by `cc4_trajectory_eval.py`, computes per-episode
C/I/A/Resilience, and emits a summary CSV (or JSON) plus impact counts per
host.

Re-runnable: rolling out CybORG is expensive; metric tweaks should be cheap.
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jaxborg.evaluation.cia import get_cia_scorer
from jaxborg.recipe import load, project_eval


def _stats(xs):
    if not xs:
        return 0.0, 0.0
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Score CC4 trajectory files with CIA + resilience")
    parser.add_argument("traj_dir", help="Directory containing *.jsonl trajectories")
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--per-episode-json", default=None)
    parser.add_argument("--recipe", default=None, help="Path or name of recipe yaml")
    args = parser.parse_args()

    eval_cfg = project_eval(load(args.recipe)) if args.recipe is not None else {}
    scorer = get_cia_scorer(eval_cfg)

    traj_dir = Path(args.traj_dir)
    files = sorted(traj_dir.glob(args.glob))
    if not files:
        print(f"no files matching {args.glob} in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in files:
        s = scorer(p)
        rows.append(
            {
                "file": p.name,
                "steps": s.steps,
                "reward": s.total_reward,
                "C_mean": s.C_mean,
                "I_mean": s.I_mean,
                "A_mean": s.A_mean,
                "R_mean": s.R_mean,
                "C_min": s.C_min,
                "I_min": s.I_min,
                "A_min": s.A_min,
                "R_min": s.R_min,
                "impact_counts": s.impact_counts,
            }
        )

    print(f"\n{'file':<60s}  {'rew':>9s}  {'C':>7s}  {'I':>7s}  {'A':>7s}  {'R':>7s}")
    for r in rows:
        print(
            f"{r['file']:<60s}  {r['reward']:>+9.1f}  "
            f"{r['C_mean']:>+7.3f}  {r['I_mean']:>+7.3f}  "
            f"{r['A_mean']:>+7.3f}  {r['R_mean']:>+7.3f}"
        )

    print("\n=== aggregate ===")
    rew_m, rew_s = _stats([r["reward"] for r in rows])
    print(f"reward     {rew_m:+10.3f} ± {rew_s:.3f}  (n={len(rows)})")
    for k in ("C_mean", "I_mean", "A_mean", "R_mean"):
        m, s = _stats([r[k] for r in rows])
        print(f"{k:<10s} {m:+10.3f} ± {s:.3f}")

    impact_total: dict[str, int] = {}
    for r in rows:
        for k, v in r["impact_counts"].items():
            impact_total[k] = impact_total.get(k, 0) + v
    print("\nimpact counts (per role-tagged host):")
    for k, v in sorted(impact_total.items(), key=lambda kv: -kv[1]):
        print(f"  {k:35s}  {v}")

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n": len(rows),
                    "reward_mean": rew_m,
                    "reward_stdev": rew_s,
                    "cia_means": {k: _stats([r[k] for r in rows])[0] for k in ("C_mean", "I_mean", "A_mean", "R_mean")},
                    "cia_stdev": {k: _stats([r[k] for r in rows])[1] for k in ("C_mean", "I_mean", "A_mean", "R_mean")},
                    "impact_counts": impact_total,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote summary: {out}")
    if args.per_episode_json:
        out = Path(args.per_episode_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote per-episode: {out}")


if __name__ == "__main__":
    main()
