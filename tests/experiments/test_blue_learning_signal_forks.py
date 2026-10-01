"""Check terminal scoring and distinguish biased bootstrap credit from outcomes."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from jaxborg.policies.categorical import Categorical
from scripts.experiments import blue_learning_signal_forks as probe


@struct.dataclass
class Dynamic:
    time: jax.Array
    blocked_zones: jax.Array
    bad: jax.Array


@struct.dataclass
class State:
    state: Dynamic
    const: object = None


class Policy:
    def apply(self, params, obs, masks=None):
        del params, masks
        # A deliberately biased value prediction after the bad intervention.
        value = jnp.where(obs[..., 0] == 2, 100.0 * obs[..., 1], 0.0)
        return Categorical(jnp.zeros((*obs.shape[:-1], 2))), value


class Env:
    blue_agents = ("blue_0", "blue_1")
    red_agents = ("red_0",)

    def get_obs(self, state):
        return {
            n: jnp.array([state.state.time, state.state.bad], dtype=float)
            for n in (*self.blue_agents, *self.red_agents)
        }

    def get_avail_actions(self, state):
        del state
        return {n: jnp.ones(2, dtype=bool) for n in (*self.blue_agents, *self.red_agents)}

    def step_env(self, key, state, actions):
        del key
        first = state.state.time == 1
        # Only Allow yields a point, while the other simultaneous actions must
        # remain fixed. Subsequent actions have no effect on this toy outcome.
        r = jnp.where(first, 1 - actions["blue_0"] + 10 * actions["blue_1"] + 100 * actions["red_0"], 0)
        bad = jnp.where(first, actions["blue_0"], state.state.bad)
        state = state.replace(state=state.state.replace(time=state.state.time + 1, bad=bad))
        parts = {
            "reward_ria": r.astype(float),
            "reward_asf": jnp.float32(0),
            "reward_lwf": jnp.float32(0),
            "action_cost": jnp.float32(0),
        }
        return self.get_obs(state), state, {}, {"__all__": state.state.time >= 3}, parts


def test_legal_fork_measures_wrong_gae_ranking_and_telescopes_at_lambda_one(monkeypatch):
    monkeypatch.setattr(probe, "decode_blue_action", lambda *args: (0, 0, 0, 0, 1))
    env = Env()
    networks = {"blue": Policy(), "red": Policy()}
    state = State(Dynamic(jnp.int32(1), jnp.zeros((2, 2), dtype=bool), jnp.int32(0)))
    actions = {"blue_0": jnp.int32(1), "blue_1": jnp.int32(1), "red_0": jnp.int32(1)}

    def run(chosen):
        return probe.learning_fork(
            env,
            networks,
            state,
            actions,
            jax.random.PRNGKey(1),
            jnp.int32(0),
            jnp.int32(chosen),
            jax.random.PRNGKey(2),
            {"blue": None, "red": None},
            jnp.full(500, 400.0),
        )

    bad, good = run(1), run(0)
    np.testing.assert_array_equal(bad["undiscounted"], [110, 0, 0, 0])
    np.testing.assert_array_equal(good["undiscounted"], [111, 0, 0, 0])
    assert float(good["normalized_mc_advantage"]) > float(bad["normalized_mc_advantage"])
    assert float(good["gae_advantage"]) < float(bad["gae_advantage"])
    for r in (bad, good):
        np.testing.assert_allclose(r["gae_lambda1_advantage"], r["normalized_mc_advantage"], atol=2e-5)
        assert int(r["route_open_ticks"]) == 2


def test_terminal_intervention_includes_reward_without_critic_bootstrap(monkeypatch):
    monkeypatch.setattr(probe, "decode_blue_action", lambda *args: (0, 0, 0, 0, 1))
    env = Env()
    state = State(Dynamic(jnp.int32(2), jnp.zeros((2, 2), dtype=bool), jnp.int32(1)))
    actions = {"blue_0": jnp.int32(1), "blue_1": jnp.int32(1), "red_0": jnp.int32(1)}
    r = probe.learning_fork(
        env,
        {"blue": Policy(), "red": Policy()},
        state,
        actions,
        jax.random.PRNGKey(1),
        jnp.int32(0),
        jnp.int32(0),
        jax.random.PRNGKey(2),
        {"blue": None, "red": None},
        jnp.full(500, 400.0),
    )
    np.testing.assert_array_equal(r["undiscounted"], [0, 0, 0, 0])
    assert float(r["initial_value"]) == 100
    assert float(r["gae_advantage"]) == float(r["normalized_mc_advantage"]) == -100
    assert int(r["route_open_ticks"]) == 1
