"""Phase 5 Test 1 — heuristic-spread diversity audit (no training).

Run a deterministic blue policy (sleep) across two env distributions:
  A. ENV-FIXED:    reset() with the same topology key every episode; only
                   red/green stochasticity varies between rollouts.
  B. ENV-DIVERSE:  reset() with a fresh topology key each episode; what
                   `topology_mode='generative'` produces by default on this
                   branch.

Per-episode total reward is recorded for each. Pre-registered verdict:
the env-diverse distribution is "meaningful" if its per-episode reward
std is at least 1.5x the env-fixed std (i.e., topology variation
contributes meaningful spread on top of red/green noise).

Heuristic-spread spike — diagnoses whether what the cec branch already
varies (host counts, services, PIDs) actually changes the problem the
agent faces, before we either trust it or port phase-reward / topology
banks from the diversity branch.
"""

from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.parity.fsm_red_env import make_fsm_red_env


N_EPISODES = int(os.environ.get("CEC_SPIKE_EPISODES", "32"))
N_STEPS = int(os.environ.get("CEC_SPIKE_STEPS", "500"))


def make_rollout_fn(env):
    sleep = {f"blue_{i}": jnp.int32(0) for i in range(5)}

    def rollout(init_key: jax.Array, rollout_key: jax.Array) -> jax.Array:
        _, state = env.reset(init_key)

        def step_fn(carry, k):
            st, total = carry
            _, st, r, _, _ = env.step(k, st, sleep)
            return (st, total + r["blue_0"]), None

        step_keys = jax.random.split(rollout_key, N_STEPS)
        (_, total), _ = jax.lax.scan(step_fn, (state, jnp.float32(0.0)), step_keys)
        return total

    return rollout


def main() -> int:
    print(f"=== CEC Phase 5 Test 1 — diversity spike (N={N_EPISODES} eps, sleep policy) ===")
    print(f"JAX backend: {jax.default_backend()} ({jax.devices()})")

    env = make_fsm_red_env(num_steps=N_STEPS, topology_mode="generative")
    rollout = make_rollout_fn(env)
    rollout_batched = jax.jit(jax.vmap(rollout, in_axes=(0, 0)))

    rng = jax.random.PRNGKey(0)
    rng_split = jax.random.split(rng, 3)
    init_keys_diverse = jax.random.split(rng_split[0], N_EPISODES)
    rollout_keys_a = jax.random.split(rng_split[1], N_EPISODES)
    rollout_keys_b = jax.random.split(rng_split[2], N_EPISODES)

    # SAME topology key every episode (env-fixed): broadcast a single key.
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

    print("\n--- Per-episode reward summary ---")
    fmt = "{:14s}  mean={:8.1f}   std={:6.1f}   min={:8.1f}   max={:8.1f}   range={:7.1f}"
    print(fmt.format("ENV-FIXED", rf.mean(), rf.std(), rf.min(), rf.max(), rf.max() - rf.min()))
    print(fmt.format("ENV-DIVERSE", rd.mean(), rd.std(), rd.min(), rd.max(), rd.max() - rd.min()))

    ratio = rd.std() / max(rf.std(), 1e-9)
    print(f"\nstd ratio (diverse / fixed): {ratio:.2f}")

    # Pre-registered threshold: 1.5x. Below = topology variation is not
    # producing meaningful spread on top of red/green noise.
    THRESHOLD = 1.5
    if ratio >= THRESHOLD:
        print(f"VERDICT: PASS  (ratio {ratio:.2f} >= {THRESHOLD})  — env-diverse has real spread.")
        return 0
    else:
        print(f"VERDICT: FAIL  (ratio {ratio:.2f} < {THRESHOLD})  — env-diverse spread is not meaningfully larger than red/green noise.")
        print("  → port phase-reward / topology banks from diversity branch before any Phase 5 training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
