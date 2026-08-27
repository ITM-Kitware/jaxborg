"""Phase 5 Test 1 (rerun) — diversity audit with a TRAINED policy.

Same protocol as ``cec_phase5_diversity_spike.py`` (sleep variant), but
the blue agents are driven by a 3M-step matched-v2 IPPO checkpoint
instead of always selecting Sleep. Sleep showed a 1.15 std ratio (FAIL);
a responsive policy ought to react to topology variation, so its reward
should be more sensitive to the fixed-vs-diverse env distinction than a
pure noise-floor policy is.

Default checkpoint: ``default_seed42`` (no resilience red, fixed env at
training time) — the parity-gate baseline. Override via
``CEC_SPIKE_CHECKPOINT``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.evaluation.jax_runner import load_jax_checkpoint
from jaxborg.parity.fsm_red_env import make_fsm_red_env

DEFAULT_CKPT = (
    "/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp/ippo_jax/"
    "default_seed42/model_default_seed42.safetensors"
)

N_EPISODES = int(os.environ.get("CEC_SPIKE_EPISODES", "128"))
N_STEPS = int(os.environ.get("CEC_SPIKE_STEPS", "500"))
CKPT = Path(os.environ.get("CEC_SPIKE_CHECKPOINT", DEFAULT_CKPT))


def make_rollout_fn(env, policy, params):
    agents = [f"blue_{i}" for i in range(5)]

    def policy_step(obs_dict, mask_dict):
        # Stack across 5 agents, vmap policy.apply.
        obs_stack = jnp.stack([obs_dict[a] for a in agents])
        mask_stack = jnp.stack([mask_dict[a] for a in agents])

        def fwd(o, m):
            pi, _ = policy.apply(params, o, m)
            return pi.logits

        logits = jax.vmap(fwd)(obs_stack, mask_stack)
        actions = jnp.argmax(logits, axis=-1)
        return {a: actions[i] for i, a in enumerate(agents)}

    def rollout(init_key: jax.Array, rollout_key: jax.Array) -> jax.Array:
        obs0, state0 = env.reset(init_key)
        mask0 = env.get_avail_actions(state0)

        def step_fn(carry, k):
            st, obs, mask, total = carry
            actions = policy_step(obs, mask)
            new_obs, new_st, r, _, _ = env.step(k, st, actions)
            new_mask = env.get_avail_actions(new_st)
            return (new_st, new_obs, new_mask, total + r["blue_0"]), None

        step_keys = jax.random.split(rollout_key, N_STEPS)
        (_, _, _, total), _ = jax.lax.scan(
            step_fn, (state0, obs0, mask0, jnp.float32(0.0)), step_keys
        )
        return total

    return rollout


def main() -> int:
    print(f"=== CEC Phase 5 Test 1 — diversity spike (TRAINED policy) ===")
    print(f"Checkpoint: {CKPT}")
    print(f"N_EPISODES={N_EPISODES}, N_STEPS={N_STEPS}")
    print(f"JAX backend: {jax.default_backend()} ({jax.devices()})")

    if not CKPT.is_file():
        print(f"ERROR: checkpoint not found: {CKPT}", file=sys.stderr)
        return 2

    policy, params, recipe = load_jax_checkpoint(CKPT)
    red_agent = recipe.get("train", {}).get("red_agent", "fsm")
    print(f"Recipe red_agent: {red_agent}")

    env = make_fsm_red_env(num_steps=N_STEPS, topology_mode="generative", red_agent=red_agent)
    rollout = make_rollout_fn(env, policy, params)
    rollout_batched = jax.jit(jax.vmap(rollout, in_axes=(0, 0)))

    rng = jax.random.PRNGKey(0)
    rng_split = jax.random.split(rng, 3)
    init_keys_diverse = jax.random.split(rng_split[0], N_EPISODES)
    rollout_keys_a = jax.random.split(rng_split[1], N_EPISODES)
    rollout_keys_b = jax.random.split(rng_split[2], N_EPISODES)

    fixed_init_key = jax.random.PRNGKey(42)
    init_keys_fixed = jnp.broadcast_to(fixed_init_key[None, :], (N_EPISODES, fixed_init_key.shape[0]))

    print("\nCompiling + running ENV-FIXED scan...")
    t0 = time.perf_counter()
    rewards_fixed = rollout_batched(init_keys_fixed, rollout_keys_a)
    rewards_fixed.block_until_ready()
    t_fixed = time.perf_counter() - t0
    print(f"  wall: {t_fixed:.1f}s")

    print("\nRunning ENV-DIVERSE scan (re-uses compiled cache)...")
    t0 = time.perf_counter()
    rewards_diverse = rollout_batched(init_keys_diverse, rollout_keys_b)
    rewards_diverse.block_until_ready()
    t_div = time.perf_counter() - t0
    print(f"  wall: {t_div:.1f}s")

    rf = np.asarray(rewards_fixed)
    rd = np.asarray(rewards_diverse)

    print("\n--- Per-episode reward summary (TRAINED policy, deterministic) ---")
    fmt = "{:14s}  mean={:8.1f}   std={:6.1f}   min={:8.1f}   max={:8.1f}   range={:7.1f}"
    print(fmt.format("ENV-FIXED", rf.mean(), rf.std(), rf.min(), rf.max(), rf.max() - rf.min()))
    print(fmt.format("ENV-DIVERSE", rd.mean(), rd.std(), rd.min(), rd.max(), rd.max() - rd.min()))

    ratio = rd.std() / max(rf.std(), 1e-9)
    print(f"\nstd ratio (diverse / fixed): {ratio:.2f}")

    THRESHOLD = 1.5
    if ratio >= THRESHOLD:
        print(f"VERDICT: PASS  (ratio {ratio:.2f} >= {THRESHOLD})  — env-diverse has real spread under a trained policy.")
        return 0
    print(f"VERDICT: FAIL  (ratio {ratio:.2f} < {THRESHOLD})  — even a trained policy doesn't see meaningful spread.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
