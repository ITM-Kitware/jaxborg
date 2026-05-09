"""Phase 6 Test 1 (Axis B) — heuristic-spread spike for the mission-profile bank.

Mirrors the canonical Phase 5 diversity spike: rolls out N short episodes under
two conditions and reports the σ-ratio of per-episode reward.

* ENV-FIXED: default mission profile (no ``mission_bank``); fixed topology.
* ENV-AXIS-B: ``mission_bank=[(1,1,1), (3,1,1), (1,3,1), (1,1,3)]``; same
  fixed topology.

Pre-registered pass threshold: σ-ratio (AXIS-B / FIXED) ≥ 1.5.

Both arms run a *sleep policy* — see the axis-A spike for rationale.

Env vars:
    CEC_SPIKE_EPISODES   episodes per arm (default 32, matching Phase 5)
    CEC_SPIKE_STEPS      steps per episode (default 500)
    CEC_SPIKE_VARIANT    variant name (default ``cc4_stock``)
    CEC_SPIKE_AMPLIFY    mission_bank_amplify (default 1.0; 10× is the
                         pre-registered escalation if 3× is weak)

Exit code: 0 on PASS (σ-ratio ≥ 1.5), 1 on FAIL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.scenarios.cc4.game_variants import VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
BANK_DIR = REPO_ROOT / "scripts" / "dev" / "topology_bank"

DEFAULT_MISSION_BANK = [
    [1.0, 1.0, 1.0],
    [3.0, 1.0, 1.0],
    [1.0, 3.0, 1.0],
    [1.0, 1.0, 3.0],
]
PASS_THRESHOLD = 1.5


def _fixed_topology() -> Path:
    """Pin a single bank snapshot so axis-B isolates the mission knob."""
    paths = sorted(BANK_DIR.glob("shape_*.snapshot.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No topology snapshots found in {BANK_DIR}. Run scripts/dev/build_topology_bank.py first."
        )
    return paths[0]


def _rollout_episode_rewards(
    env,
    *,
    num_episodes: int,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    """Run ``num_episodes`` independent episodes with a sleep policy; return totals."""

    blue_agents = tuple(f"blue_{i}" for i in range(NUM_BLUE_AGENTS))

    @jax.jit
    def _run_one(key):
        reset_key, scan_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key)

        def step_fn(carry, _):
            state, k = carry
            k, step_key = jax.random.split(k)
            actions = {a: jnp.int32(0) for a in blue_agents}
            _, new_state, rewards, _, _ = env.step(step_key, state, actions)
            mean_reward = jnp.stack([rewards[a] for a in blue_agents]).mean()
            return (new_state, k), mean_reward

        (_, _), per_step = jax.lax.scan(step_fn, (env_state, scan_key), None, length=num_steps)
        return per_step.sum()

    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    totals = jax.vmap(_run_one)(keys)
    return np.asarray(totals)


def main() -> int:
    n_eps = int(os.environ.get("CEC_SPIKE_EPISODES", "32"))
    n_steps = int(os.environ.get("CEC_SPIKE_STEPS", "500"))
    variant_name = os.environ.get("CEC_SPIKE_VARIANT", "cc4_stock")
    amplify = float(os.environ.get("CEC_SPIKE_AMPLIFY", "1.0"))
    variant = VARIANTS[variant_name]

    fixed_topology = _fixed_topology()
    print(
        f"[axis-b] variant={variant_name} episodes={n_eps} steps={n_steps} "
        f"amplify={amplify} bank_size={len(DEFAULT_MISSION_BANK)} "
        f"topology={fixed_topology.name}",
        flush=True,
    )

    fixed_env = make_jax_env(variant, topology_path=fixed_topology)
    bank_env = make_jax_env(
        variant,
        topology_path=fixed_topology,
        mission_bank=DEFAULT_MISSION_BANK,
        mission_bank_amplify=amplify,
    )

    print("[axis-b] rolling out ENV-FIXED ...", flush=True)
    fixed_rewards = _rollout_episode_rewards(fixed_env, num_episodes=n_eps, num_steps=n_steps, seed=20260509)

    print("[axis-b] rolling out ENV-AXIS-B ...", flush=True)
    axis_rewards = _rollout_episode_rewards(bank_env, num_episodes=n_eps, num_steps=n_steps, seed=20260509)

    sigma_fixed = float(np.std(fixed_rewards, ddof=1)) if n_eps > 1 else 0.0
    sigma_axis = float(np.std(axis_rewards, ddof=1)) if n_eps > 1 else 0.0
    mean_fixed = float(np.mean(fixed_rewards))
    mean_axis = float(np.mean(axis_rewards))
    ratio = sigma_axis / sigma_fixed if sigma_fixed > 0 else float("inf")

    print(f"[axis-b] ENV-FIXED   reward mean={mean_fixed:+.2f} σ={sigma_fixed:.3f}")
    print(f"[axis-b] ENV-AXIS-B  reward mean={mean_axis:+.2f} σ={sigma_axis:.3f}")
    print(f"[axis-b] σ-ratio (AXIS-B / FIXED) = {ratio:.3f}")

    if ratio >= PASS_THRESHOLD:
        print(f"[axis-b] PASS — σ-ratio {ratio:.3f} ≥ {PASS_THRESHOLD}")
        return 0
    print(f"[axis-b] FAIL — σ-ratio {ratio:.3f} < {PASS_THRESHOLD}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
