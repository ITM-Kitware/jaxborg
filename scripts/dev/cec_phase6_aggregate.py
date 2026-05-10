"""Phase 6 Test 2 aggregator — paired delta C11−C00 across 3 seeds × 5 held-out reds.

Reads result rows from ``$JAXBORG_EXP_DIR/eval/phase6_*.jsonl`` (written by
``cec_phase6_eval_jax.py``), pivots into a (arm, seed, red) → mean_reward
table, computes:

  - Per-red mean across seeds, ± stderr.
  - Paired delta C11 − C00 per seed (pair by seed), then mean ± stderr.
  - Pre-registered falsification verdict per the plan:
      * confirmed: paired delta ≥ +200 reward AND lower bound > 0
      * refuted:   paired delta ≤ +50 reward OR sign flip on ≥1 seed
      * inconclusive otherwise

Usage:
    uv run python scripts/dev/cec_phase6_aggregate.py
    uv run python scripts/dev/cec_phase6_aggregate.py --eval-dir /custom/eval/dir
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ARMS = ("C00", "C11")
REDS = ("fsm", "cia_c", "cia_i", "cia_a", "random")
SEEDS = (42, 142, 242)

CONFIRM_REWARD_DELTA = 200.0
REFUTE_REWARD_DELTA = 50.0


def _load_rows(eval_dir: Path) -> list[dict]:
    files = sorted(eval_dir.glob("phase6_*.jsonl"))
    rows = []
    for f in files:
        try:
            row = json.loads(f.read_text())
            rows.append(row)
        except Exception as e:
            print(f"WARN: failed to parse {f}: {e}")
    return rows


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def _arm_from_recipe(recipe_name: str) -> str:
    for arm in ARMS:
        if arm in recipe_name:
            return arm
    return "?"


def _seed_from_model(model_path: str) -> int:
    import re

    m = re.search(r"seed(\d+)", model_path)
    return int(m.group(1)) if m else -1


def aggregate(eval_dir: Path) -> dict:
    rows = _load_rows(eval_dir)
    table: dict[tuple[str, int, str], float] = {}
    for row in rows:
        arm = _arm_from_recipe(row.get("recipe_name", ""))
        seed = _seed_from_model(row.get("model", ""))
        red = row.get("eval_red", "?")
        if arm in ARMS and seed in SEEDS and red in REDS:
            key = (arm, seed, red)
            if key in table:
                print(f"WARN: duplicate row for {key}; keeping latest by file order")
            table[key] = float(row["mean_reward"])

    summary = {
        "per_arm_red": {},
        "paired_deltas": {},
        "verdicts": {},
        "missing_cells": [],
    }
    for arm in ARMS:
        for red in REDS:
            vals = [table.get((arm, s, red)) for s in SEEDS]
            present = [v for v in vals if v is not None]
            for s, v in zip(SEEDS, vals):
                if v is None:
                    summary["missing_cells"].append((arm, s, red))
            mean = sum(present) / len(present) if present else float("nan")
            stderr = _stderr(present) if len(present) >= 2 else 0.0
            summary["per_arm_red"][f"{arm}/{red}"] = {
                "n": len(present),
                "mean": mean,
                "stderr": stderr,
                "per_seed": vals,
            }

    for red in REDS:
        deltas = []
        for s in SEEDS:
            v00 = table.get(("C00", s, red))
            v11 = table.get(("C11", s, red))
            if v00 is None or v11 is None:
                continue
            deltas.append(v11 - v00)
        n = len(deltas)
        if n == 0:
            summary["paired_deltas"][red] = None
            summary["verdicts"][red] = "no-data"
            continue
        mean = sum(deltas) / n
        stderr = _stderr(deltas) if n >= 2 else 0.0
        lb = mean - stderr
        signs_match = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        summary["paired_deltas"][red] = {
            "n": n,
            "mean": mean,
            "stderr": stderr,
            "lower_bound": lb,
            "per_seed": deltas,
            "signs_match": signs_match,
        }
        if mean >= CONFIRM_REWARD_DELTA and lb > 0:
            verdict = "CONFIRMED"
        elif mean <= REFUTE_REWARD_DELTA or not signs_match:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        summary["verdicts"][red] = verdict

    return summary


def _print_table(summary: dict) -> None:
    print("\n=== Per-arm × held-out red mean reward (across seeds) ===")
    print(f"{'arm/red':<12} {'n':>3} {'mean':>10} {'± stderr':>10}  per-seed")
    for k, v in summary["per_arm_red"].items():
        ps = ", ".join(f"{x:.0f}" if x is not None else "—" for x in v["per_seed"])
        print(f"{k:<12} {v['n']:>3} {v['mean']:>10.1f} {v['stderr']:>10.1f}  [{ps}]")

    print("\n=== Paired delta (C11 − C00), per held-out red ===")
    print(f"{'red':<10} {'n':>3} {'Δmean':>10} {'± stderr':>10} {'lower_bound':>12}  verdict")
    for red in REDS:
        d = summary["paired_deltas"].get(red)
        verdict = summary["verdicts"].get(red, "—")
        if d is None:
            print(f"{red:<10} {0:>3} {'—':>10} {'—':>10} {'—':>12}  {verdict}")
        else:
            print(f"{red:<10} {d['n']:>3} {d['mean']:>10.1f} {d['stderr']:>10.1f} {d['lower_bound']:>12.1f}  {verdict}")

    if summary["missing_cells"]:
        print(f"\nMISSING ({len(summary['missing_cells'])} cells):")
        for arm, seed, red in summary["missing_cells"]:
            print(f"  {arm} seed={seed} red={red}")

    print("\nPre-registered thresholds:")
    print(f"  CONFIRMED: paired Δ ≥ +{CONFIRM_REWARD_DELTA:.0f} reward AND mean−stderr > 0")
    print(f"  REFUTED:   paired Δ ≤ +{REFUTE_REWARD_DELTA:.0f}  OR sign flip across seeds")
    print("  INCONCLUSIVE band in between → escalate (e.g. mission_bank_amplify=10 or more seeds)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 6 Test 2 eval results")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp") + "/eval",
        help="Directory containing phase6_*.jsonl eval rows",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        raise SystemExit(f"eval-dir not found: {eval_dir}")

    summary = aggregate(eval_dir)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        _print_table(summary)


if __name__ == "__main__":
    main()
