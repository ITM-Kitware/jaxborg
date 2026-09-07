from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from jaxborg.evaluation.cia.fixed_topology import EvaluationCase
from jaxborg.evaluation.jax_scripted_red import run_jax_scripted_red_episode
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy, run_matchup_episode
from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE


class _Distribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self, seed):
        del seed
        return jnp.argmax(self.logits, axis=-1)


class _PolicyModule:
    def apply(self, weights, obs, mask):
        del weights, obs
        logits = jnp.where(mask, jnp.float32(0.0), jnp.float32(-1e9))
        return _Distribution(logits), jnp.zeros(mask.shape[0], dtype=jnp.float32)


@struct.dataclass
class _SimState:
    time: jnp.ndarray
    ot_service_stopped: jnp.ndarray
    host_service_reliability: jnp.ndarray
    host_decoy_reliability: jnp.ndarray


@struct.dataclass
class _JointState:
    state: _SimState


@struct.dataclass
class _FsmState:
    state: _SimState
    const: jnp.ndarray
    extras: dict


def _initial_sim_state() -> _SimState:
    return _SimState(
        time=jnp.int32(0),
        ot_service_stopped=jnp.zeros(3, dtype=jnp.bool_),
        host_service_reliability=jnp.full((3, 1), 100, dtype=jnp.int32),
        host_decoy_reliability=jnp.full((3, 1), 100, dtype=jnp.int32),
    )


def _observations(agents):
    return {agent: jnp.zeros(2, dtype=jnp.float32) for agent in agents}


def _masks(agents):
    return {agent: jnp.ones(2, dtype=jnp.bool_) for agent in agents}


class _JointScanEnv:
    blue_agents = ("blue_0",)
    red_agents = ("red_0",)

    def __init__(self):
        self.trace_count = 0

    def reset_at_topology(self, key, topology_index):
        del key, topology_index
        agents = self.blue_agents + self.red_agents
        return _observations(agents), _JointState(_initial_sim_state())

    def reset(self, key):
        return self.reset_at_topology(key, jnp.int32(0))

    def get_avail_actions(self, state):
        del state
        return _masks(self.blue_agents + self.red_agents)

    def step_env(self, key, state, actions):
        del key, actions
        self.trace_count += 1
        next_time = state.state.time + jnp.int32(1)
        sim = state.state.replace(
            time=next_time,
            ot_service_stopped=state.state.ot_service_stopped.at[0].set(True),
        )
        next_state = state.replace(state=sim)
        agents = self.blue_agents + self.red_agents
        rewards = {"blue_0": jnp.float32(1.5), "red_0": jnp.float32(-1.5)}
        done = next_time >= 2
        dones = {agent: done for agent in agents} | {"__all__": done}
        return _observations(agents), next_state, rewards, dones, {}


class _FsmScanEnv:
    agents = ("blue_0",)

    def __init__(self):
        self.trace_count = 0

    def reset_at_topology(self, key, topology_index):
        del key, topology_index
        state = _FsmState(
            state=_initial_sim_state(),
            const=jnp.int32(0),
            extras={"host_resilience_role": jnp.zeros(3, dtype=jnp.int32)},
        )
        return _observations(self.agents), state

    def get_avail_actions(self, state):
        del state
        return _masks(self.agents)

    def step_env(self, key, state, actions):
        del key, actions
        self.trace_count += 1
        next_time = state.state.time + jnp.int32(1)
        # Stand in for a role-biased scripted Red selector: impact whichever
        # host was injected as AUTH.
        auth_host = jnp.argmax(state.extras["host_resilience_role"] == 1)
        sim = state.state.replace(
            time=next_time,
            ot_service_stopped=state.state.ot_service_stopped.at[auth_host].set(True),
        )
        next_state = state.replace(state=sim)
        rewards = {"blue_0": jnp.float32(2.0)}
        done = next_time >= 2
        dones = {"blue_0": done, "__all__": done}
        return _observations(self.agents), next_state, rewards, dones, {}


class _JointKeyEnv(_JointScanEnv):
    def step_env(self, key, state, actions):
        del actions
        next_state = state.replace(state=state.state.replace(time=state.state.time + 1))
        reward = jax.random.uniform(key, ())
        dones = {"blue_0": True, "red_0": True, "__all__": True}
        agents = self.blue_agents + self.red_agents
        return (
            _observations(agents),
            next_state,
            {"blue_0": reward, "red_0": -reward},
            dones,
            {},
        )


class _FsmKeyEnv(_FsmScanEnv):
    def step_env(self, key, state, actions):
        del actions
        next_state = state.replace(state=state.state.replace(time=state.state.time + 1))
        reward = jax.random.uniform(key, ())
        dones = {"blue_0": True, "__all__": True}
        return _observations(self.agents), next_state, {"blue_0": reward}, dones, {}


def _jax_policy(team: str, module: _PolicyModule) -> LoadedMatchupPolicy:
    return LoadedMatchupPolicy(team, "jax", module, {}, {})


def test_learned_matchup_jax_path_scans_episode_and_masks_post_terminal_steps():
    env = _JointScanEnv()
    module = _PolicyModule()
    policies = {
        "blue": _jax_policy("blue", module),
        "red": _jax_policy("red", module),
    }
    variant = replace(CIA_RESILIENCE, num_steps=5)
    roles = jnp.asarray([1, 0, 0], dtype=jnp.int32)

    result = run_matchup_episode(
        policies,
        variant=variant,
        seed=7,
        deterministic=True,
        env=env,
        topology_index=0,
        host_resilience_role=roles,
    )
    traced_after_first_call = env.trace_count
    repeated = run_matchup_episode(
        policies,
        variant=variant,
        seed=8,
        deterministic=True,
        env=env,
        topology_index=0,
        host_resilience_role=roles,
    )

    assert result == (3.0, [-10.0, -10.0, -10.0])
    assert repeated == result
    assert traced_after_first_call > 0
    assert env.trace_count == traced_after_first_call


def test_scripted_red_jax_path_scans_episode_with_injected_fixed_roles():
    env = _FsmScanEnv()
    module = _PolicyModule()
    policy = _jax_policy("blue", module)
    variant = replace(CIA_RESILIENCE, num_steps=5)
    case = EvaluationCase(
        topology_index=0,
        topology_path=Path("fixed.snapshot.npz"),
        topology_fingerprint="fingerprint",
        base_seed=11,
        replicate_index=0,
        episode_seed=11,
        host_roles=(0, 1, 0),
        role_map_id="role-map",
    )

    result = run_jax_scripted_red_episode(
        policy,
        env=env,
        variant=variant,
        case=case,
        deterministic=True,
    )
    traced_after_first_call = env.trace_count
    repeated = run_jax_scripted_red_episode(
        policy,
        env=env,
        variant=variant,
        case=case,
        deterministic=True,
    )

    assert result.reward == 4.0
    assert result.cia == (-10.0, -10.0, -10.0)
    assert repeated == result
    assert traced_after_first_call > 0
    assert env.trace_count == traced_after_first_call


def test_scan_paths_preserve_environment_step_transition_key_conventions():
    module = _PolicyModule()
    variant = replace(CIA_RESILIENCE, num_steps=1)
    seed = 23

    matchup_key = jax.random.PRNGKey(seed)
    matchup_key, _ = jax.random.split(matchup_key)
    matchup_key, _ = jax.random.split(matchup_key)  # Blue policy key.
    matchup_key, _ = jax.random.split(matchup_key)  # Red policy key.
    _, submitted_step_key = jax.random.split(matchup_key)
    expected_matchup_key, _ = jax.random.split(submitted_step_key)

    matchup = run_matchup_episode(
        {
            "blue": _jax_policy("blue", module),
            "red": _jax_policy("red", module),
        },
        variant=variant,
        seed=seed,
        deterministic=True,
        env=_JointKeyEnv(),
        topology_index=0,
    )
    assert matchup == pytest.approx(float(jax.random.uniform(expected_matchup_key, ())))

    scripted_key = jax.random.PRNGKey(seed)
    scripted_key, _ = jax.random.split(scripted_key)
    scripted_key, _ = jax.random.split(scripted_key)  # Blue policy key.
    _, submitted_step_key = jax.random.split(scripted_key)
    expected_scripted_key = jax.random.split(submitted_step_key, 3)[0]
    case = EvaluationCase(
        topology_index=0,
        topology_path=Path("fixed.snapshot.npz"),
        topology_fingerprint="fingerprint",
        base_seed=seed,
        replicate_index=0,
        episode_seed=seed,
        host_roles=(1, 0, 0),
        role_map_id="role-map",
    )
    scripted = run_jax_scripted_red_episode(
        _jax_policy("blue", module),
        env=_FsmKeyEnv(),
        variant=variant,
        case=case,
        deterministic=True,
    )
    assert scripted.reward == pytest.approx(float(jax.random.uniform(expected_scripted_key, ())))
