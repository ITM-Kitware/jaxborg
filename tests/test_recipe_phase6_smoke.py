"""Phase 6 C11 integration smoke test.

Builds the env that arm C11 (both topology bank + mission bank) trains on,
resets a vmapped batch, steps through one full episode with sampled actions,
and asserts:

  1. No NaN/Inf in rewards or observations.
  2. Both banks were exercised — at least 3 distinct topology snapshots and
     at least 3 distinct mission-multiplier triples were observed across the
     batch (catches a regression where one bank silently degrades to a
     singleton, e.g. PRNG splitting bug).

Uses a smaller batch than the production training run (64 envs × 50 steps
vs 1024 × 500) to keep the test fast under CPU JAX. Bank diversity is still
observable: 16-entry topology bank with 64 draws → expected ~4 hits/entry
under uniform sampling; same logic for the 4-entry mission bank.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.recipe import load, project_jax

pytestmark = pytest.mark.slow

NUM_ENVS = 64
NUM_STEPS = 50


def _build_c11_env():
    recipe = load("cec_phase6_C11")
    cfg = project_jax(recipe)

    topology_bank = cfg.get("TOPOLOGY_BANK") or None
    return cfg, make_jax_env(
        cfg["TRAIN_VARIANT"],
        training_mode=True,
        topology_path=list(topology_bank) if topology_bank else None,
        mission_bank=cfg.get("MISSION_BANK"),
        mission_bank_amplify=cfg.get("MISSION_BANK_AMPLIFY", 1.0),
    )


def test_c11_smoke_no_nan_and_bank_diversity():
    cfg, env = _build_c11_env()

    assert cfg.get("TOPOLOGY_BANK"), "C11 must have a non-empty TOPOLOGY_BANK"
    assert cfg.get("MISSION_BANK"), "C11 must have a non-empty MISSION_BANK"
    assert len(cfg["TOPOLOGY_BANK"]) == 16
    assert len(cfg["MISSION_BANK"]) == 4

    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    obs, state = jax.vmap(env.reset)(keys)

    # Bank diversity check, post-reset.
    # Topology: const.host_subnet is one of 16 distinct snapshots.
    host_subnets = np.asarray(state.const.host_subnet)  # (NUM_ENVS, GLOBAL_MAX_HOSTS)
    topo_signatures = {tuple(row.tolist()) for row in host_subnets}
    assert len(topo_signatures) >= 3, (
        f"topology bank under-sampled: only {len(topo_signatures)} distinct snapshots "
        f"in {NUM_ENVS} resets (expected ≥3 from a 16-entry bank)"
    )

    # Mission: const.phase_rewards is the unscaled bank entry × multiplier triple.
    # Extract phase 0, subnet 0 — its (LWF, ASF, RIA) row encodes the active triple
    # up to a per-snapshot constant. Stacking across the batch yields the
    # diversity signal regardless of which topology snapshot was drawn.
    pr = np.asarray(state.const.phase_rewards)  # (NUM_ENVS, MISSION_PHASES, NUM_SUBNETS, 3)

    # Normalize per-env by the per-snapshot baseline so the mission multiplier
    # ratio is what we count. Use the first non-zero ratio across phases/subnets
    # as the signature.
    def _ratio_signature(env_idx: int) -> tuple[float, float, float]:
        slab = pr[env_idx]  # (PHASES, SUBNETS, 3)
        # find a (phase, subnet) where the row is not all zero
        nonzero = np.nonzero(np.linalg.norm(slab, axis=-1))
        if nonzero[0].size == 0:
            return (0.0, 0.0, 0.0)
        p, s = int(nonzero[0][0]), int(nonzero[1][0])
        triple = slab[p, s]
        # quantize to 4 decimals to defang float dust
        peak = float(np.max(np.abs(triple))) or 1.0
        return tuple(round(float(x) / peak, 4) for x in triple)

    mission_signatures = {_ratio_signature(i) for i in range(NUM_ENVS)}
    assert len(mission_signatures) >= 3, (
        f"mission bank under-sampled: only {len(mission_signatures)} distinct triples "
        f"in {NUM_ENVS} resets (expected ≥3 from a 4-entry bank): {mission_signatures}"
    )

    # JIT'd scan rollout — random actions per env per step, accumulating
    # max(|reward|) and max(|obs|) so we can assert finiteness once at the end.
    agents = list(env.agents)
    action_dim = env.action_space(agents[0]).n

    @jax.jit
    def _rollout(state, rng):
        def _step(carry, _):
            state, rng, max_r, max_o = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            act_keys = jax.random.split(rng_act, NUM_ENVS)
            actions = {
                a: jax.vmap(
                    lambda k, off=i: jax.random.randint(jax.random.fold_in(k, off), (), 0, action_dim, dtype=jnp.int32)
                )(act_keys)
                for i, a in enumerate(agents)
            }
            step_keys = jax.random.split(rng_step, NUM_ENVS)
            obs, state, rewards, dones, info = jax.vmap(env.step)(step_keys, state, actions)
            r_stack = jnp.stack([rewards[a] for a in agents])
            o_stack = jnp.concatenate([obs[a].reshape(-1) for a in agents])
            max_r = jnp.maximum(max_r, jnp.max(jnp.abs(r_stack)))
            max_o = jnp.maximum(max_o, jnp.max(jnp.abs(o_stack)))
            return (state, rng, max_r, max_o), None

        init_carry = (state, rng, jnp.float32(0.0), jnp.float32(0.0))
        (final_state, _, max_r, max_o), _ = jax.lax.scan(_step, init_carry, None, NUM_STEPS)
        return final_state, max_r, max_o

    _, max_r, max_o = _rollout(state, jax.random.PRNGKey(1))
    max_r = float(max_r)
    max_o = float(max_o)
    assert np.isfinite(max_r), f"NaN/Inf in rewards across rollout (max |r| = {max_r})"
    assert np.isfinite(max_o), f"NaN/Inf in observations across rollout (max |o| = {max_o})"
