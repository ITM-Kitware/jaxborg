"""Empirical check: does each registered red selector bias as advertised?

For each selector, roll out N episodes of a sleep-blue env and count which
hosts red actually attacked. Group counts by role (NONE / AUTH / DB / WEB) and
print share-of-attacks per role.

Expectations from the PR description:
  fsm        — ~92% attacks on NONE (untagged) hosts (no bias)
  resilience — AUTH+DB+WEB share lifts ~5× over fsm (target_weight=5)
  cia_c      — AUTH+DB share lifts ~10× over fsm; tagged hosts at FSM_R get
               Impact/Degrade vs vanilla's Discover (action shift visible if
               we count action types — out of scope here)
  cia_i      — AUTH+WEB share lifts ~10×
  cia_a      — AUTH+DB+WEB share lifts ~10×

This script doesn't enforce thresholds — it prints, you eyeball.
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from jaxborg.parity.fsm_red_env import make_fsm_red_env
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_NONE, ROLE_WEB

NUM_EPISODES = 3
EPISODE_STEPS = 30  # short — enough for bias to surface in expectation
SELECTORS = ("fsm", "resilience", "cia_c", "cia_i", "cia_a")
_ROLE_LABEL = {ROLE_NONE: "NONE", ROLE_AUTH: "AUTH", ROLE_DB: "DB", ROLE_WEB: "WEB"}


def rollout_one(env, key) -> Counter:
    """Run one episode; return counter of (role, attacked) for each red attack step."""
    obs, state = env.reset(key)
    counts: Counter = Counter()
    blue_actions = {a: jnp.int32(0) for a in env.agents}
    roles = state.extras["host_resilience_role"]  # (GLOBAL_MAX_HOSTS,) int32

    step_fn = jax.jit(env.step)
    for _ in range(EPISODE_STEPS):
        key, sk = jax.random.split(key)
        obs, state, _rew, dones, _info = step_fn(sk, state, blue_actions)
        # Each red agent's pending_target_host is the host they currently target.
        # Active agents only: red_agent_active[r] && red_pending_ticks[r] >= 0.
        targets = state.state.red_pending_target_host  # (NUM_RED,)
        active = state.state.red_agent_active  # (NUM_RED,)
        for r in range(targets.shape[0]):
            if not bool(active[r]):
                continue
            host_idx = int(targets[r])
            role = int(roles[host_idx])
            counts[role] += 1
        if bool(dones["__all__"]):
            break
    return counts


def main():
    print(f"\n{'selector':<12s}  {'NONE':>10s}  {'AUTH':>10s}  {'DB':>10s}  {'WEB':>10s}  {'tagged%':>8s}")
    print("-" * 70)
    print("(All selectors run with role_assignment='resilience' so the baseline is\n"
          " apples-to-apples — fsm shows uniform-attack share on tagged hosts.)\n")
    for name in SELECTORS:
        # Force role assignment for all selectors so "NONE %" reflects the same
        # underlying tag set; biased rows then show their lift vs the fsm baseline.
        env = make_fsm_red_env(num_steps=EPISODE_STEPS, red_agent=name, role_assignment="resilience")
        agg: Counter = Counter()
        for ep in range(NUM_EPISODES):
            agg += rollout_one(env, jax.random.PRNGKey(1000 + ep))
        total = sum(agg.values()) or 1
        tagged = agg[ROLE_AUTH] + agg[ROLE_DB] + agg[ROLE_WEB]
        share = lambda r: f"{agg[r] / total * 100:>9.1f}%"  # noqa: E731
        print(
            f"{name:<12s}  {share(ROLE_NONE)}  {share(ROLE_AUTH)}  "
            f"{share(ROLE_DB)}  {share(ROLE_WEB)}  {tagged / total * 100:>7.1f}%"
        )


if __name__ == "__main__":
    main()
