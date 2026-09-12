from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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


@pytest.mark.parametrize("count", [1, 4, 7])
def test_matchup_padded_batches_preserve_episode_results_and_compile_one_shape(count):
    from jaxborg.evaluation.matchup_runner import _run_jax_matchup_episodes_batched

    env = _JointScanEnv()
    module = _PolicyModule()
    policies = {team: _jax_policy(team, module) for team in ("blue", "red")}
    variant = replace(CIA_RESILIENCE, num_steps=5)
    seeds = list(range(31, 31 + count))
    # Distinct role maps ensure padding cannot silently shift CIA rows.
    roles = [jnp.asarray([1, 0, 0] if i % 2 else [0, 1, 0]) for i in range(count)]
    returns, cia = _run_jax_matchup_episodes_batched(
        policies,
        variant=variant,
        env=env,
        episode_seeds=seeds,
        topology_indices=[0] * count,
        role_arrays=roles,
        deterministic=False,
        batch_size=4,
    )
    assert env.trace_count == 1  # Includes the short final batch.
    expected = [
        run_matchup_episode(
            policies,
            variant=variant,
            seed=seed,
            deterministic=False,
            env=env,
            topology_index=0,
            host_resilience_role=role,
        )
        for seed, role in zip(seeds, roles, strict=True)
    ]
    np.testing.assert_array_equal(returns, [row[0] for row in expected])
    np.testing.assert_array_equal(cia, [row[1] for row in expected])


def test_padded_matchup_preserves_distinct_random_keys():
    from jaxborg.evaluation.matchup_runner import _run_jax_matchup_episodes_batched

    env = _JointKeyEnv()
    module = _PolicyModule()
    policies = {team: _jax_policy(team, module) for team in ("blue", "red")}
    variant = replace(CIA_RESILIENCE, num_steps=1)
    seeds = [2, 7, 101, 33, 29]
    returns, _ = _run_jax_matchup_episodes_batched(
        policies,
        variant=variant,
        env=env,
        episode_seeds=seeds,
        topology_indices=None,
        role_arrays=None,
        deterministic=False,
        batch_size=4,
    )
    expected = [
        run_matchup_episode(policies, variant=variant, seed=seed, deterministic=False, env=env) for seed in seeds
    ]
    np.testing.assert_array_equal(returns, expected)
    assert len(set(returns)) == len(seeds)


@pytest.mark.parametrize("count", [1, 4, 7])
def test_scripted_padded_batches_preserve_cia_and_compile_one_shape(count):
    from jaxborg.evaluation.jax_scripted_red import _run_jax_scripted_red_episodes_batched

    env = _FsmScanEnv()
    policy = _jax_policy("blue", _PolicyModule())
    variant = replace(CIA_RESILIENCE, num_steps=5)
    cases = [
        EvaluationCase(
            topology_index=i % 2,
            topology_path=Path(f"topology-{i % 2}.npz"),
            topology_fingerprint=str(i % 2),
            base_seed=100 + i,
            replicate_index=0,
            episode_seed=100 + i,
            host_roles=(1, 0, 0) if i % 2 else (0, 1, 0),
            role_map_id=str(i % 2),
        )
        for i in range(count)
    ]
    actual = _run_jax_scripted_red_episodes_batched(
        policy,
        env=env,
        variant=variant,
        cases=cases,
        deterministic=False,
        batch_size=4,
    )
    assert env.trace_count == 1
    expected = [
        run_jax_scripted_red_episode(policy, env=env, variant=variant, case=case, deterministic=False) for case in cases
    ]
    assert actual == expected


class _WeightedPolicyModule:
    def apply(self, weights, obs, mask):
        logits = jnp.broadcast_to(weights, mask.shape)
        return _Distribution(logits), jnp.zeros(obs.shape[0])


class _ActionRewardEnv(_JointScanEnv):
    def step_env(self, key, state, actions):
        obs, state, _, dones, info = super().step_env(key, state, actions)
        reward = (actions["blue_0"] - actions["red_0"]).astype(jnp.float32)
        return obs, state, {"blue_0": reward, "red_0": -reward}, dones, info


def test_matchup_context_reuses_compilation_but_uses_each_checkpoints_weights(monkeypatch, tmp_path):
    from jaxborg.evaluation import matchup_runner as runner

    monkeypatch.setenv("JAXBORG_EVAL_BATCH_SIZE", "4")
    env = _ActionRewardEnv()
    module = _WeightedPolicyModule()
    loaded, constructed = [], []

    def load(path, *, team, backend):
        loaded.append((str(path), team))
        weights = jnp.asarray([0.0, 1.0] if "new" in str(path) else [1.0, 0.0])
        return LoadedMatchupPolicy(team, backend, module, weights, {"path": str(path)})

    def make_env(*args, **kwargs):
        constructed.append(kwargs)
        return env

    monkeypatch.setattr(runner, "load_matchup_policy", load)
    monkeypatch.setattr(runner, "make_joint_jax_env", make_env)
    context = runner.MatchupEvaluationContext()
    kwargs = dict(
        backend="jax",
        variant=replace(CIA_RESILIENCE, num_steps=5),
        seeds=[11, 12, 13, 14, 15],
        context=context,
        progress=False,
    )
    old = runner.evaluate_matchup(tmp_path / "old-blue", tmp_path / "old-red", **kwargs)
    traces = env.trace_count
    new = runner.evaluate_matchup(tmp_path / "new-blue", tmp_path / "old-red", **kwargs)
    assert old.blue_returns == [0.0] * 5
    assert new.blue_returns == [2.0] * 5
    assert env.trace_count == traces == 1
    assert len(constructed) == 1
    assert len(loaded) == 3
    assert new.policies["blue"]["path"].endswith("new-blue")
