"""Probe (A): per-mission-profile action-distribution analysis.

Loads one or more Phase 2 checkpoints, runs rollouts on the
``per_mission`` testbed (which forces ``vary_mission_profile=True`` so
all 4 profiles are sampled), and reports the action-class histogram
bucketed by mission profile.

Compositional behavior signature: action distributions differ across
profiles within a single arm.  Generic regularization signature: action
distributions are flat across profiles.

Usage::

    python scripts/eval/cec_phase2_action_probe.py \
        --checkpoint <path> \
        --label <arm/seed> \
        [--episodes 30]
"""

# ruff: noqa: E402

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from transfer import load_checkpoint, make_scan_eval_fn  # noqa: E402

from jaxborg.actions.encoding import (  # noqa: E402
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_END,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_END,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_END,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_END,
    BLUE_REMOVE_START,
    BLUE_RESTORE_END,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.fsm_red_env import FsmRedCC4Env  # noqa: E402

ACTION_CLASSES = [
    ("Sleep", BLUE_SLEEP, BLUE_SLEEP + 1),
    ("Monitor", BLUE_MONITOR, BLUE_MONITOR + 1),
    ("Analyse", BLUE_ANALYSE_START, BLUE_ANALYSE_END),
    ("Remove", BLUE_REMOVE_START, BLUE_REMOVE_END),
    ("Restore", BLUE_RESTORE_START, BLUE_RESTORE_END),
    ("Decoy", BLUE_DECOY_START, BLUE_DECOY_END),
    ("Block", BLUE_BLOCK_TRAFFIC_START, BLUE_BLOCK_TRAFFIC_END),
    ("Allow", BLUE_ALLOW_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_END),
]


def classify(actions: np.ndarray) -> dict:
    flat = actions.reshape(-1)
    counts = {}
    total = flat.size
    for name, lo, hi in ACTION_CLASSES:
        counts[name] = float(np.mean((flat >= lo) & (flat < hi)))
    counts["__total"] = total
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True, help="Display label, e.g. gen-mission-msg/seed1")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=50000, help="per_mission testbed seed offset")
    args = ap.parse_args()

    print(f"[{args.label}] loading checkpoint {args.checkpoint}", flush=True)
    policy, params, policy_kind = load_checkpoint(args.checkpoint)

    # per_mission testbed: vary_mission_profile=True (so we span all profiles)
    env = FsmRedCC4Env(num_steps=500, topology_mode="generative", vary_mission_profile=True)
    keys = jnp.stack([jax.random.PRNGKey(args.seed_offset + ep) for ep in range(args.episodes)])
    obs, env_state = jax.vmap(env.reset)(keys)
    scan_fn = make_scan_eval_fn(env, policy, policy_kind, deterministic=False)

    print(f"[{args.label}] running {args.episodes} episodes ...", flush=True)
    _, step_data = jax.vmap(scan_fn, in_axes=(None, 0, 0, 0))(params, keys, env_state, obs)

    actions = np.asarray(step_data["actions"])  # (E, T, NUM_BLUE_AGENTS)
    mission_idx = np.asarray(env_state.const.mission_profile_index)  # (E,)

    print(f"[{args.label}] mission_profile distribution: {np.bincount(mission_idx, minlength=4).tolist()}")

    # Bucket episodes by profile index, then aggregate action histogram per profile.
    by_profile = defaultdict(list)
    for ep in range(args.episodes):
        by_profile[int(mission_idx[ep])].append(actions[ep])

    print()
    print(f"[{args.label}] action-class fractions per mission profile")
    profile_names = {0: "default", 1: "avail-heavy", 2: "prod-heavy", 3: "CI-heavy"}
    cols = [name for name, _, _ in ACTION_CLASSES]
    header = f"{'profile':<14} {'n':>3}" + "".join(f" {c:>9}" for c in cols)
    print(header)
    print("-" * len(header))
    rows = {}
    for profile in sorted(by_profile.keys()):
        eps = np.stack(by_profile[profile])
        h = classify(eps)
        n = len(by_profile[profile])
        row_str = f"{profile_names.get(profile, str(profile)):<14} {n:>3}"
        for c in cols:
            row_str += f" {h[c]:.4f}".rjust(10)
        print(row_str)
        rows[profile] = h

    # Compute per-action-class spread across profiles (max − min) — the
    # discrimination metric.  High spread = mission-aware behavior.
    print()
    print(f"[{args.label}] per-action-class spread (max − min across profiles)")
    spreads = []
    for c in cols:
        values = [rows[p][c] for p in rows]
        spread = max(values) - min(values)
        spreads.append((c, spread))
    spreads.sort(key=lambda x: -x[1])
    for c, s in spreads:
        bar = "█" * int(s * 200)
        print(f"  {c:<10} {s:.4f}  {bar}")
    total_spread = sum(s for _, s in spreads)
    print(f"  total L1 spread: {total_spread:.4f}")


if __name__ == "__main__":
    main()
