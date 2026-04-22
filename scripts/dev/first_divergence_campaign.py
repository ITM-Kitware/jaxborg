"""Run differential harness across many seeds and histogram first divergence.

Unlike `parity_loop.sh`, this does not stop on first mismatch — it records the
first divergence per seed (if any), then prints a histogram by step + field.

Usage:
    uv run python scripts/dev/first_divergence_campaign.py \\
        --seeds 100 --steps 500 --output /tmp/first_divergence.json

    uv run python scripts/dev/first_divergence_campaign.py \\
        --seed-start 10000 --seeds 50 --steps 500 \\
        --output /tmp/held_out_divergence.json
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _worker(task):
    # Lazy imports so child processes pay the cost, not the parent.
    from tests.differential.fuzzer import _fuzz_one_seed

    seed, steps, blue_agent, blue_action_source, mismatch_mode = task
    t0 = time.time()
    report = _fuzz_one_seed(
        seed=seed,
        max_steps=steps,
        mismatch_mode=mismatch_mode,
        blue_agent=blue_agent,
        blue_action_source=blue_action_source,
        strict_random_sync=True,
        check_obs=True,
        check_masks=True,
        strip_inactive_knowledge=False,
    )
    elapsed = time.time() - t0
    entry = {"seed": seed, "elapsed_s": round(elapsed, 2)}
    if report is None:
        entry["clean"] = True
    else:
        entry["clean"] = False
        entry["first_step"] = report.step
        entry["first_field"] = report.field_name
        entry["host_or_agent"] = report.host_or_agent
        entry["cyborg_value"] = str(report.cyborg_value)
        entry["jax_value"] = str(report.jax_value)
    return entry


def run_campaign(seeds, steps, blue_agent, blue_action_source, mismatch_mode, verbose, workers):
    tasks = [(s, steps, blue_agent, blue_action_source, mismatch_mode) for s in seeds]
    per_seed = []
    if workers == 1:
        for idx, task in enumerate(tasks):
            entry = _worker(task)
            per_seed.append(entry)
            if verbose:
                tag = "CLEAN" if entry["clean"] else f"DIV step={entry['first_step']} field={entry['first_field']}"
                print(f"[{idx + 1}/{len(tasks)}] seed={entry['seed']} {tag} ({entry['elapsed_s']}s)", flush=True)
        return per_seed

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        done = 0
        total = len(tasks)
        for entry in pool.imap_unordered(_worker, tasks, chunksize=1):
            per_seed.append(entry)
            done += 1
            if verbose:
                tag = "CLEAN" if entry["clean"] else f"DIV step={entry['first_step']} field={entry['first_field']}"
                print(f"[{done}/{total}] seed={entry['seed']} {tag} ({entry['elapsed_s']}s)", flush=True)
    per_seed.sort(key=lambda e: e["seed"])
    return per_seed


def summarize(per_seed):
    n = len(per_seed)
    clean = sum(1 for e in per_seed if e["clean"])
    diverged = n - clean
    step_hist = Counter(e["first_step"] for e in per_seed if not e["clean"])
    field_hist = Counter(e["first_field"] for e in per_seed if not e["clean"])
    return {
        "n_seeds": n,
        "n_clean": clean,
        "n_diverged": diverged,
        "divergence_rate": diverged / n if n else 0.0,
        "first_step_histogram": dict(sorted(step_hist.items())),
        "first_field_histogram": dict(sorted(field_hist.items(), key=lambda x: -x[1])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seeds", type=int, default=100, help="Number of seeds")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--blue-agent", default="random", choices=["random", "sleep", "monitor"])
    p.add_argument("--blue-action-source", default="cyborg_policy", choices=["sleep", "cyborg_policy"])
    p.add_argument("--mismatch-mode", default="error", choices=["error", "all"])
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes (1 = serial)")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(
        f"Campaign: seeds={args.seed_start}..{args.seed_start + args.seeds - 1} "
        f"steps={args.steps} blue_agent={args.blue_agent} "
        f"blue_action_source={args.blue_action_source} mismatch_mode={args.mismatch_mode}",
        flush=True,
    )

    wall_start = time.time()
    per_seed = run_campaign(
        seeds,
        args.steps,
        args.blue_agent,
        args.blue_action_source,
        args.mismatch_mode,
        verbose=True,
        workers=args.workers,
    )
    wall = time.time() - wall_start

    summary = summarize(per_seed)
    summary["wall_seconds"] = round(wall, 1)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        out = {"summary": summary, "per_seed": per_seed, "config": vars(args)}
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
