import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.constants import BLUE_OBS_SIZE, NUM_BLUE_AGENTS
from jaxborg.env import ScenarioEnvState
from jaxborg.parity.fsm_red_env import FsmRedCC4Env

pytestmark = pytest.mark.slow


@pytest.fixture
def env():
    return FsmRedCC4Env(num_steps=50)


@pytest.fixture
def reset_data(env):
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    return obs, state


class TestFsmRedEnvSmoke:
    def test_agents_are_blue_only(self, env):
        assert len(env.agents) == NUM_BLUE_AGENTS
        for a in env.agents:
            assert a.startswith("blue_")

    def test_action_spaces(self, env):
        for a in env.agents:
            assert env.action_spaces[a].n == BLUE_ALLOW_TRAFFIC_END

    def test_observation_spaces(self, env):
        for a in env.agents:
            assert env.observation_spaces[a].shape == (BLUE_OBS_SIZE,)

    def test_reset_returns_blue_only(self, env, reset_data):
        obs, state = reset_data
        assert isinstance(state, ScenarioEnvState)
        assert len(obs) == NUM_BLUE_AGENTS
        for b in range(NUM_BLUE_AGENTS):
            assert f"blue_{b}" in obs
            assert obs[f"blue_{b}"].shape == (BLUE_OBS_SIZE,)

    def test_step_returns_blue_only(self, env, reset_data):
        obs, state = reset_data
        key = jax.random.PRNGKey(1)
        actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}

        obs2, state2, rewards, dones, info = env.step(key, state, actions)

        assert len(obs2) == NUM_BLUE_AGENTS
        assert len(rewards) == NUM_BLUE_AGENTS
        assert "__all__" in dones
        for a in env.agents:
            assert a in obs2
            assert a in rewards
            assert a in dones

    def test_step_jit_compatible(self, env, reset_data):
        _, state = reset_data
        actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}

        step_fn = jax.jit(env.step)
        key = jax.random.PRNGKey(2)
        obs, state2, rewards, dones, info = step_fn(key, state, actions)
        assert state2.state.time == 1

    def test_episode_terminates(self, env):
        key = jax.random.PRNGKey(7)
        obs, state = env.reset(key)
        actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}

        for t in range(env._env.num_steps):
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        assert bool(dones["__all__"])

    def test_fsm_red_is_active(self):
        env = FsmRedCC4Env(num_steps=100)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}

        initial_sessions = int(jnp.sum(state.state.red_sessions))

        for t in range(50):
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        final_sessions = int(jnp.sum(state.state.red_sessions))
        final_discovered = int(jnp.sum(state.state.red_discovered_hosts))

        assert final_sessions > initial_sessions, "FSM red should create sessions"
        assert final_discovered > 1, "FSM red should discover hosts"

    def test_name(self, env):
        assert env.name == "FsmRedCC4"
