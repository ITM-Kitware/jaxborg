import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, RED_WITHDRAW_END
from jaxborg.constants import BLUE_OBS_SIZE, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.env import ScenarioEnv, ScenarioEnvState

pytestmark = pytest.mark.slow


@pytest.fixture
def env():
    return ScenarioEnv(num_steps=50)


@pytest.fixture
def reset_data(env):
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    return obs, state


class TestCC4EnvSmoke:
    def test_reset_returns_obs_and_state(self, env, reset_data):
        obs, state = reset_data
        assert isinstance(state, ScenarioEnvState)
        assert len(obs) == NUM_BLUE_AGENTS + NUM_RED_AGENTS
        for b in range(NUM_BLUE_AGENTS):
            assert obs[f"blue_{b}"].shape == (BLUE_OBS_SIZE,)
        for r in range(NUM_RED_AGENTS):
            assert obs[f"red_{r}"].shape == (BLUE_OBS_SIZE,)

    def test_step_returns_correct_structure(self, env, reset_data):
        obs, state = reset_data
        key = jax.random.PRNGKey(1)
        actions = {}
        for b in range(NUM_BLUE_AGENTS):
            actions[f"blue_{b}"] = jnp.int32(0)
        for r in range(NUM_RED_AGENTS):
            actions[f"red_{r}"] = jnp.int32(0)

        obs2, state2, rewards, dones, info = env.step(key, state, actions)

        assert isinstance(state2, ScenarioEnvState)
        for agent in env.agents:
            assert agent in obs2
            assert agent in rewards
            assert agent in dones
        assert "__all__" in dones

    def test_step_jit_compatible(self, env, reset_data):
        _, state = reset_data
        actions = {}
        for b in range(NUM_BLUE_AGENTS):
            actions[f"blue_{b}"] = jnp.int32(0)
        for r in range(NUM_RED_AGENTS):
            actions[f"red_{r}"] = jnp.int32(0)

        step_fn = jax.jit(env.step)
        key = jax.random.PRNGKey(2)
        obs, state2, rewards, dones, info = step_fn(key, state, actions)
        assert state2.state.time == 1

    def test_episode_terminates(self, env):
        key = jax.random.PRNGKey(7)
        obs, state = env.reset(key)

        actions = {}
        for b in range(NUM_BLUE_AGENTS):
            actions[f"blue_{b}"] = jnp.int32(0)
        for r in range(NUM_RED_AGENTS):
            actions[f"red_{r}"] = jnp.int32(0)

        for t in range(env.num_steps):
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        assert bool(dones["__all__"])

    def test_reset_produces_different_topologies(self, env):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        obs1, state1 = env.reset(k1)
        obs2, state2 = env.reset(k2)
        differ = (
            not jnp.array_equal(state1.const.host_active, state2.const.host_active)
            or not jnp.array_equal(state1.const.initial_services, state2.const.initial_services)
            or not jnp.array_equal(state1.const.red_start_hosts, state2.const.red_start_hosts)
        )
        assert differ

    def test_action_spaces_correct(self, env):
        for agent in env.blue_agents:
            assert env.action_spaces[agent].n == BLUE_ALLOW_TRAFFIC_END
        for agent in env.red_agents:
            assert env.action_spaces[agent].n == RED_WITHDRAW_END

    def test_observation_spaces_correct(self, env):
        for agent in env.agents:
            assert env.observation_spaces[agent].shape == (BLUE_OBS_SIZE,)
