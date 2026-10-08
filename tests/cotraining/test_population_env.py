import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.env import ScenarioEnvState
from jaxborg.opponent_population import OpponentPopulationSettings
from jaxborg.population_env import PopulationCC4Env
from jaxborg.scenarios.cc4.game_variant import GameVariant


@struct.dataclass
class _Sim:
    time: jax.Array
    knowledge: jax.Array


class _Joint:
    blue_agents = ("blue_0",)
    red_agents = ("red_0",)

    def __init__(self):
        self._env = self

    def reset(self, key):
        state = ScenarioEnvState(_Sim(jnp.int32(0), jnp.int32(7)), jax.random.randint(key, (), 0, 16))
        return self.get_obs(state), state

    def reset_at_topology(self, key, index):
        _, state = self.reset(key)
        state = state.replace(const=jnp.int32(index))
        return self.get_obs(state), state

    def reset_batch(self, keys, topology_key):
        assert len(keys) <= 16
        indices = jax.random.permutation(topology_key, 16)[: len(keys)]
        return jax.vmap(self.reset_at_topology)(keys, indices)

    def _reset_state(self, state, key):
        return self.reset(key)[1]

    def get_obs(self, state):
        return {
            name: jnp.stack((state.state.knowledge, state.const)).astype(jnp.float32)
            for name in self.blue_agents + self.red_agents
        }

    def step_env(self, key, state, actions):
        del key, actions
        state = state.replace(state=state.state.replace(time=state.state.time + 1))
        done = state.state.time >= 2
        return (
            self.get_obs(state),
            state,
            {"blue_0": jnp.float32(0), "red_0": jnp.float32(0)},
            {"blue_0": done, "red_0": done, "__all__": done},
            {"source": jnp.int32(0)},
        )


class _Scripted(_Joint):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def _strip_inactive_red_reset_knowledge(self, state):
        return state.replace(state=state.state.replace(knowledge=jnp.int32(-1)))

    def step_env(self, key, state, actions):
        assert set(actions) == {"blue_0"}
        assert "host_resilience_role" in state.extras
        _, state, _, dones, _ = super().step_env(key, state, actions)
        # Represent selector-side and post-step FSM bookkeeping, both of
        # which the adapter must preserve verbatim rather than reimplement.
        state = state.replace(state=state.state.replace(knowledge=state.state.knowledge + self.source))
        return (
            {"blue_0": self.get_obs(state)["blue_0"]},
            state,
            {"blue_0": jnp.float32(self.source)},
            {"blue_0": dones["blue_0"], "__all__": dones["__all__"]},
            {"source": jnp.int32(self.source)},
        )


@pytest.fixture
def cheap_native_envs(monkeypatch):
    import jaxborg.population_env as module

    monkeypatch.setattr(module, "make_joint_jax_env", lambda *a, **kw: _Joint())
    ids = {"fsm": 1, "cia_c": 2, "cia_i": 3, "cia_a": 4}
    monkeypatch.setattr(module, "make_jax_env", lambda variant, **kw: _Scripted(ids[variant.red_agent]))
    monkeypatch.setattr(module, "assign_resilience_roles_from_const", lambda const, key: jnp.array([1, 2, 3]))
    return module


def _env(percentages=(75, 25, 0, 0, 0), preserve=False):
    return PopulationCC4Env(GameVariant("population_test"), OpponentPopulationSettings(percentages, preserve))


def test_population_adapter_dispatches_all_native_opponents_and_keeps_fsm_state(cheap_native_envs):
    env = _env((20, 20, 20, 20, 20))
    _, state = env.reset(jax.random.PRNGKey(1))
    for opponent_id in range(5):
        selected = state.replace(opponent_id=jnp.int32(opponent_id))
        obs, new, rewards, dones, info = env.step_env(
            jax.random.PRNGKey(2), selected, {"blue_0": jnp.int32(0), "red_0": jnp.int32(0)}
        )
        assert info["source"] == opponent_id
        assert new.state.knowledge == selected.state.knowledge + opponent_id
        assert new.opponent_id == opponent_id
        assert set(obs) == set(rewards) == {"blue_0", "red_0"}
        assert set(dones) == {"blue_0", "red_0", "__all__"}
        assert obs["blue_0"].shape == (2,)  # No opponent identity appended.
        np.testing.assert_array_equal(new.extras["host_resilience_role"], [1, 2, 3])


def test_population_reset_applies_only_scripted_reset_knowledge(cheap_native_envs):
    env = _env((50, 50, 0, 0, 0))
    _, states = env.reset_batch(jax.random.split(jax.random.PRNGKey(4), 16), jax.random.PRNGKey(5))
    assert set(np.asarray(states.opponent_id)) == {0, 1}
    np.testing.assert_array_equal(states.state.knowledge, jnp.where(states.opponent_id == 0, 7, -1))
    assert len(set(np.asarray(states.const))) == 16


@pytest.mark.parametrize("preserve", [False, True])
def test_population_identity_changes_only_on_reset_and_auxiliary_stays_learned(cheap_native_envs, preserve):
    env = _env((50, 50, 0, 0, 0), preserve)
    count = 32 if preserve else 16
    keys = jax.random.split(jax.random.PRNGKey(6), count)
    _, original = env.reset_batch(keys, jax.random.PRNGKey(7))
    actions = {"blue_0": jnp.zeros(count, jnp.int32), "red_0": jnp.zeros(count, jnp.int32)}
    _, next_state, _, dones, _ = env.step_batch(keys, original, actions, jax.random.PRNGKey(8))
    assert not np.any(dones["__all__"])
    np.testing.assert_array_equal(next_state.opponent_id, original.opponent_id)
    _, reset, _, dones, _ = env.step_batch(keys, next_state, actions, jax.random.PRNGKey(9))
    assert np.all(dones["__all__"])
    assert np.any(np.asarray(reset.opponent_id[:16]) != np.asarray(original.opponent_id[:16]))
    assert len(set(np.asarray(reset.const[:16]))) == 16
    if preserve:
        np.testing.assert_array_equal(original.opponent_id[16:], 0)
        np.testing.assert_array_equal(reset.opponent_id[16:], 0)
        assert len(set(np.asarray(reset.const[16:]))) == 16


def test_population_partial_reset_leaves_live_games_untouched(cheap_native_envs):
    env = _env()
    keys = jax.random.split(jax.random.PRNGKey(11), 4)
    _, states = env.reset_batch(keys, jax.random.PRNGKey(12))
    states = states.replace(state=states.state.replace(time=jnp.array([0, 1, 0, 1])))
    _, after, _, dones, _ = env.step_batch(
        keys, states, {"blue_0": jnp.zeros(4, jnp.int32), "red_0": jnp.zeros(4, jnp.int32)}, jax.random.PRNGKey(13)
    )
    np.testing.assert_array_equal(dones["__all__"], [False, True, False, True])
    np.testing.assert_array_equal(after.state.time, [1, 0, 1, 0])
    np.testing.assert_array_equal(after.opponent_id[::2], states.opponent_id[::2])
    np.testing.assert_array_equal(after.const[::2], states.const[::2])


def test_population_cia_rejects_incompatible_topologies():
    with pytest.raises(ValueError, match="op-zone server candidates"):
        PopulationCC4Env(GameVariant("insufficient_roles"), OpponentPopulationSettings((0, 0, 100, 0, 0)))


def test_production_population_reset_and_step_have_compatible_jax_branches():
    env = PopulationCC4Env(
        GameVariant("trace_population", op_zone_servers=3, num_steps=2),
        OpponentPopulationSettings((20, 20, 20, 20, 20)),
        training_mode=True,
    )
    key = jax.random.PRNGKey(12)
    obs, state = jax.eval_shape(env.reset, key)
    actions = {name: jnp.int32(0) for name in env.agents}
    new_obs, new_state, rewards, dones, _ = jax.eval_shape(env.step_env, key, state, actions)
    assert set(new_obs) == set(obs) == set(rewards) == set(env.agents)
    assert new_state.opponent_id.shape == ()
    assert dones["__all__"].shape == ()


def test_production_population_batched_scan_traces_preserved_batches_and_reset():
    env = PopulationCC4Env(
        GameVariant("trace_preserved_population", num_steps=2),
        OpponentPopulationSettings((75, 25, 0, 0, 0), preserve_red_batch_size=True),
        training_mode=True,
    )
    keys = jax.random.split(jax.random.PRNGKey(15), 4)
    _, state = jax.eval_shape(env.reset_batch, keys, jax.random.PRNGKey(16))
    actions = {name: jnp.zeros(4, jnp.int32) for name in env.agents}
    _, state, _, dones, _ = jax.eval_shape(env.step_batch, keys, state, actions, jax.random.PRNGKey(17))
    assert state.opponent_id.shape == (4,)
    assert dones["__all__"].shape == (4,)
