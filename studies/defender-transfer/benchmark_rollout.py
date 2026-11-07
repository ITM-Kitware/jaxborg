"""Shared stock-FSM rollout for archived LSTM and imported H-MARL systems.

Both systems see the same enhanced-v2 observable evidence and exact snapshot.
Their observation encodings, memory and policy action restrictions remain their
own. This compares defender systems, not architecture or published paper scores.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint
from jaxborg.evaluation.jax_scripted_red import _blue_policy_action_masks
from jaxborg.evaluation.matchup_runner import load_matchup_policy
from jaxborg.evaluation.stateful_blue import StatefulBluePolicy
from jaxborg.parity.fsm_red_env import FsmRedCC4Env
from jaxborg.policies import initial_carry, is_recurrent, policy_step
from jaxborg.pretrained.hmarl import load_policy as load_hmarl_policy
from jaxborg.scenarios.cc4.topology import load_topology

COMPONENTS = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
KINDS = ("lstm", "hmarl_expert", "hmarl_meta")


def load_policy(model, *, kind):
    """Load a frozen defender, preserving its checked observation contract."""
    if kind not in KINDS:
        raise ValueError(f"Unknown benchmark policy kind: {kind}")
    if kind.startswith("hmarl_"):
        return load_hmarl_policy(model, variant=kind.removeprefix("hmarl_"))
    policy = load_matchup_policy(model, team="blue", backend="jax")
    if not is_recurrent(policy.module) or policy.source["observation_dim"] != 450:
        raise ValueError("This benchmark requires the archived enhanced-v2 450-input LSTM")
    return policy


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")
    return int(value)


def _step_keys(policy_key, dynamics_key, tick):
    return jax.random.fold_in(policy_key, tick), jax.random.fold_in(dynamics_key, tick)


def _same_tree(left, right):
    """Include every state/const/extras leaf and static tree metadata."""
    if jax.tree.structure(left) != jax.tree.structure(right):
        return jnp.bool_(False)
    comparisons = [jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)]
    return jnp.all(jnp.stack(comparisons)) if comparisons else jnp.bool_(True)


@lru_cache(maxsize=4)
def _environment(num_steps):
    # No role factory, reward amendment, server-count override, or traffic mask.
    return FsmRedCC4Env(
        num_steps=num_steps,
        red_reward="zero_sum",
        blue_block_policy="cc4",
        cage4_enhanced_obs=True,
        blue_observation_version=2,
    )


def _initial_state(env, const):
    observations, inner = env._env._reset_from_const(const)
    return env._wrap_reset(observations, inner, jax.random.PRNGKey(0))


@lru_cache(maxsize=12)
def _runner(module, num_steps):
    """One executable per architecture; checkpoints and snapshots are dynamic."""
    env = _environment(num_steps)
    stateful = isinstance(module, StatefulBluePolicy)
    agents = tuple(env.agents)

    @jax.jit
    def run(weights, const, policy_key, dynamics_key):
        obs, state = _initial_state(env, const)
        before = state
        if stateful:
            state, memory = module.initialize(state)
        else:
            memory = initial_carry(module, len(agents))
        unchanged = _same_tree(before, state)
        services_match = jnp.array_equal(state.state.host_services, const.initial_services)
        pids_match = jnp.array_equal(state.state.host_max_pid, const.host_initial_max_pid)
        roles_absent = jnp.all(state.extras["host_resilience_role"] == 0)
        observation_width = obs[agents[0]].shape[0]

        def step(carry, tick):
            current_obs, current, memory = carry
            action_key, dynamics_tick_key = _step_keys(policy_key, dynamics_key, tick)
            if stateful:
                actions, memory = module.select_actions(weights, current, action_key, memory, deterministic=False)
            else:
                observations = jnp.stack([current_obs[agent] for agent in agents])
                masks = _blue_policy_action_masks(env, current, agents)
                distribution, _, memory = policy_step(module, weights, observations, masks, carry=memory)
                actions = distribution.sample(seed=action_key)
            next_obs, following, rewards, dones, info = env.step_env(
                dynamics_tick_key,
                current,
                {agent: actions[index] for index, agent in enumerate(agents)},
            )
            row = {
                "reward": rewards[agents[0]],
                "components": jnp.stack([info[name] for name in COMPONENTS]),
                "actions": actions,
                "done": dones["__all__"],
                "phase": following.state.mission_phase,
            }
            return (next_obs, following, memory), row

        (_, final, _), rows = jax.lax.scan(step, (obs, state, memory), jnp.arange(num_steps, dtype=jnp.uint32))
        checks = {
            "initializer_preserved_world": unchanged,
            "initial_services_match": services_match,
            "initial_pids_match": pids_match,
            "roles_absent": roles_absent,
            "observation_width": observation_width,
            "final_time": final.state.time,
            "const_unchanged": _same_tree(const, final.const),
        }
        return rows, checks

    return run


def _validate_rows(rows, checks, num_steps):
    for name in (
        "initializer_preserved_world",
        "initial_services_match",
        "initial_pids_match",
        "roles_absent",
        "const_unchanged",
    ):
        if not bool(checks[name]):
            raise RuntimeError(f"Benchmark game contract failed: {name}")
    if int(checks["observation_width"]) != 450:
        raise RuntimeError("Benchmark did not produce enhanced-v2 450-input observations")
    done = np.asarray(rows["done"], dtype=bool)
    if done.shape != (num_steps,) or np.any(done[:-1]) or not done[-1] or int(checks["final_time"]) != num_steps:
        raise RuntimeError("Episode terminated outside the fixed transition horizon")
    rewards = np.asarray(rows["reward"], dtype=np.float64)
    components = np.asarray(rows["components"], dtype=np.float64)
    if (
        rewards.shape != (num_steps,)
        or components.shape != (num_steps, len(COMPONENTS))
        or not np.isfinite(rewards).all()
        or not np.isfinite(components).all()
        or not np.allclose(rewards, components.sum(axis=1), rtol=0, atol=1e-6)
    ):
        raise RuntimeError("Benchmark reward differs from the unchanged simulator components")
    if np.asarray(rows["actions"]).shape != (num_steps, 5):
        raise RuntimeError("Expected five Blue actions per transition")


def run_episode(policy, *, snapshot_path, policy_seed, dynamics_seed, num_steps=500):
    """Return a JSON-ready, fixed-500-transition episode; never write or train.

    The snapshot supplies exact initial arrays. Actor keys and dynamics keys use
    separate roots, each folded with the transition index. A shared dynamics seed
    does not imply identical trajectories once policies choose different actions.
    """
    if num_steps != 500 or isinstance(num_steps, bool):
        raise ValueError("The frozen benchmark horizon is exactly 500 transitions")
    if policy.team != "blue" or policy.backend != "jax":
        raise ValueError("Expected a JAX Blue policy")
    policy_seed, dynamics_seed = _seed(policy_seed, "policy_seed"), _seed(dynamics_seed, "dynamics_seed")
    path = Path(snapshot_path).expanduser().resolve()
    const = load_topology(path)
    if int(const.max_steps) != num_steps:
        raise ValueError("Snapshot horizon differs from the fixed benchmark horizon")
    fingerprint = canonical_topology_fingerprint(const)
    # These flags select observations/bookkeeping, never topology/reward arrays.
    const = const.replace(cage4_enhanced_obs=True, blue_observation_version=2)
    rows, checks = jax.device_get(
        _runner(policy.module, num_steps)(
            policy.weights, const, jax.random.PRNGKey(policy_seed), jax.random.PRNGKey(dynamics_seed)
        )
    )
    _validate_rows(rows, checks, num_steps)
    rewards = np.asarray(rows["reward"], dtype=np.float64)
    components = np.asarray(rows["components"], dtype=np.float64)
    return {
        "reward": float(rewards.sum()),
        "steps": num_steps,
        "policy_seed": policy_seed,
        "dynamics_seed": dynamics_seed,
        "topology_fingerprint": fingerprint,
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy": policy.source,
        "deterministic": False,
        "red": "stock_finite_state",
        "blue_block_policy": "cc4",
        "resilience_roles": False,
        "blue_observation_version": 2,
        "horizon_contract": "fixed_transitions",
        "initializer_preserved_world": True,
        "initial_services_match": True,
        "initial_pids_match": True,
        "const_unchanged": True,
        "component_returns": {name: float(components[:, index].sum()) for index, name in enumerate(COMPONENTS)},
        "per_step_returns": rewards.tolist(),
        "per_step_components": [
            {name: float(row[index]) for index, name in enumerate(COMPONENTS)} for row in components
        ],
        "per_step_actions": np.asarray(rows["actions"], dtype=np.int32).tolist(),
        "per_step_done": np.asarray(rows["done"], dtype=bool).tolist(),
        "per_step_phases": np.asarray(rows["phase"], dtype=np.int32).tolist(),
    }
