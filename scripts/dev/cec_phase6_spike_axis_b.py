"""Phase 6 Test 1 (Axis B) — heuristic-spread spike for the mission-profile bank.

Mirrors the canonical Phase 5 diversity spike: rolls out N short episodes under
two conditions and reports the σ-ratio of per-episode reward.

* ENV-FIXED: default mission profile (no ``mission_bank``); fixed topology.
* ENV-AXIS-B: ``mission_bank=[(1,1,1), (3,1,1), (1,3,1), (1,1,3)]``; same
  fixed topology.

Pre-registered pass threshold: σ-ratio (AXIS-B / FIXED) ≥ 1.5.

Default: load the matched-v2 ``default_seed42`` 3M-step checkpoint and
roll out with deterministic argmax. Set ``CEC_SPIKE_CHECKPOINT=sleep`` for
the always-action-0 baseline.

Env vars:
    CEC_SPIKE_EPISODES   episodes per arm (default 128, plan default)
    CEC_SPIKE_STEPS      steps per episode (default 500)
    CEC_SPIKE_VARIANT    variant name (default ``cc4_stock``)
    CEC_SPIKE_AMPLIFY    mission_bank_amplify (default 1.0; 10× is the
                         pre-registered escalation if 3× is weak)
    CEC_SPIKE_CHECKPOINT path to .safetensors, or ``sleep`` for sleep policy
                         (default: default_seed42 model in JAXBORG_EXP_DIR)

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
DEFAULT_CKPT = (
    "/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp/ippo_jax/"
    "default_seed42/model_default_seed42.safetensors"
)
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
    policy=None,
    params=None,
) -> np.ndarray:
    """Run ``num_episodes`` independent episodes; return per-episode totals.

    When ``policy``/``params`` are given, action selection is deterministic
    argmax of the masked actor head. Otherwise blue plays sleep (action 0).
    """

    blue_agents = tuple(f"blue_{i}" for i in range(NUM_BLUE_AGENTS))
    use_policy = policy is not None and params is not None

    @jax.jit
    def _run_one(key):
        reset_key, scan_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key)
        mask = env.get_avail_actions(env_state) if use_policy else None

        def step_fn(carry, _):
            state, obs, mask, k = carry
            k, step_key = jax.random.split(k)
            if use_policy:
                obs_stack = jnp.stack([obs[a] for a in blue_agents])
                mask_stack = jnp.stack([mask[a] for a in blue_agents])

                def _fwd(o, m):
                    pi, _ = policy.apply(params, o, m)
                    return pi.logits

                logits = jax.vmap(_fwd)(obs_stack, mask_stack)
                acts = jnp.argmax(logits, axis=-1)
                actions = {a: acts[i] for i, a in enumerate(blue_agents)}
            else:
                actions = {a: jnp.int32(0) for a in blue_agents}
            new_obs, new_state, rewards, _, _ = env.step(step_key, state, actions)
            new_mask = env.get_avail_actions(new_state) if use_policy else mask
            mean_reward = jnp.stack([rewards[a] for a in blue_agents]).mean()
            return (new_state, new_obs, new_mask, k), mean_reward

        init_carry = (env_state, obs, mask, scan_key)
        (_, _, _, _), per_step = jax.lax.scan(step_fn, init_carry, None, length=num_steps)
        return per_step.sum()

    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    totals = jax.vmap(_run_one)(keys)
    return np.asarray(totals)


def main() -> int:
    n_eps = int(os.environ.get("CEC_SPIKE_EPISODES", "128"))
    n_steps = int(os.environ.get("CEC_SPIKE_STEPS", "500"))
    variant_name = os.environ.get("CEC_SPIKE_VARIANT", "cc4_stock")
    amplify = float(os.environ.get("CEC_SPIKE_AMPLIFY", "1.0"))
    ckpt_arg = os.environ.get("CEC_SPIKE_CHECKPOINT", DEFAULT_CKPT)
    variant = VARIANTS[variant_name]

    use_sleep = ckpt_arg.lower() == "sleep"
    policy = params = None
    if not use_sleep:
        from jaxborg.evaluation.jax_runner import load_jax_checkpoint

        ckpt = Path(ckpt_arg)
        if not ckpt.is_file():
            print(f"[axis-b] checkpoint not found: {ckpt} — falling back to sleep", flush=True)
            use_sleep = True
        else:
            policy, params, _recipe = load_jax_checkpoint(ckpt)
            print(f"[axis-b] policy: {ckpt.name}", flush=True)
    if use_sleep:
        print("[axis-b] policy: sleep (action 0)", flush=True)

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
    fixed_rewards = _rollout_episode_rewards(
        fixed_env, num_episodes=n_eps, num_steps=n_steps, seed=20260510, policy=policy, params=params
    )

    print("[axis-b] rolling out ENV-AXIS-B ...", flush=True)
    axis_rewards = _rollout_episode_rewards(
        bank_env, num_episodes=n_eps, num_steps=n_steps, seed=20260510, policy=policy, params=params
    )

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
