"""Recording must leave policy sampling, recurrent ticks and outcomes intact."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.evaluation import behavior_trace, jax_scripted_red, matchup_runner


@struct.dataclass
class Sim:
    time: object
    value: object
    blue_pending_ticks: object
    red_agent_active: object


@struct.dataclass
class State:
    state: object
    extras: dict


class Env:
    blue_agents = ("blue_0",)
    agents = blue_agents
    red_agents = ("red_0",)
    blue_block_policy = "cc4"

    def __init__(self):
        self.events = []

    def reset_at_topology(self, key, index):
        value = jax.random.uniform(key) + index
        state = State(Sim(jnp.int32(0), value, jnp.zeros(1, jnp.int32), jnp.ones(1, bool)), {})
        return self.obs(state), state

    def obs(self, state):
        return {name: jnp.array([state.state.value]) for name in self.blue_agents + self.red_agents}

    def get_avail_actions(self, state):
        idle = state.state.blue_pending_ticks[0] == 0
        return {"blue_0": jnp.array([True, idle, idle]), "red_0": jnp.ones(3, bool)}

    def step_env(self, key, state, actions):
        sim = state.state
        blue = actions["blue_0"]
        red = actions.get("red_0", jnp.int32(0))
        ticks = jnp.where(sim.blue_pending_ticks > 0, 0, jnp.where(blue > 0, 1, 0))
        value = sim.value + (blue + 3 * red) * (9.0**sim.time) + jax.random.uniform(key)
        next_state = state.replace(state=sim.replace(time=sim.time + 1, value=value, blue_pending_ticks=ticks))
        jax.debug.callback(
            lambda b, r, v, t: self.events.append((int(b), int(r), float(v), int(t[0]))),
            blue,
            red,
            value,
            ticks,
            ordered=True,
        )
        return self.obs(next_state), next_state, {"blue_0": value}, {"__all__": sim.time >= 3}, {}


def _policy(module, weights, obs, mask, *, carry=None, reset=None):
    del module, weights
    if reset is not None:
        carry = jnp.where(reset, 0, carry)
    logits = jnp.stack((jnp.zeros_like(carry), carry * 0.1, jnp.ones_like(carry)), axis=-1)
    logits = logits + jnp.sin(obs[:, :1]) * jnp.array([0.0, 1.0, -1.0])
    logits = jnp.where(mask, logits, -1e9)
    distribution = SimpleNamespace(logits=logits, sample=lambda seed: jax.random.categorical(seed, logits))
    return distribution, None, carry + 1


@pytest.mark.parametrize("kind", ["fsm", "learned"])
def test_tracing_preserves_sampling_busy_ticks_and_outcome(monkeypatch, kind):
    env = Env()
    module = object()
    target = jax_scripted_red if kind == "fsm" else matchup_runner
    monkeypatch.setattr(target, "policy_step", _policy)
    monkeypatch.setattr(target, "initial_carry", lambda module, count: jnp.zeros(count))
    monkeypatch.setattr(
        behavior_trace,
        "behavior_fields",
        lambda before, after, actions, masks, info, **kwargs: {
            "blue_busy": before.state.blue_pending_ticks > 0,
            "blue_action_masks": masks,
            "state_value_after": after.state.value,
            "state_time_after": after.state.time,
        },
    )
    monkeypatch.setattr(jax_scripted_red, "score_resilience_state", lambda state, roles: jnp.repeat(state.value, 3))
    kwargs = dict(env=env, num_steps=6, deterministic=False)
    if kind == "fsm":
        scan = jax_scripted_red._run_jax_scripted_red_episode_scan
        args = ({}, jax.random.PRNGKey(17), jnp.int32(0), jnp.zeros(1, jnp.int32))
        kwargs["policy_module"] = module
    else:
        monkeypatch.setattr(matchup_runner, "is_recurrent", lambda module: True)
        scan = matchup_runner._run_jax_matchup_episode_scan
        args = ({}, {}, jax.random.PRNGKey(17), jnp.int32(0), jnp.zeros(1, jnp.int32))
        kwargs.update(blue_module=module, red_module=module, use_topology_index=True, score_cia=False)
    plain = scan(*args, **kwargs)
    jax.block_until_ready(plain)
    plain_events = list(env.events)
    env.events.clear()
    traced = scan(*args, **kwargs, record_trajectory=True)
    jax.block_until_ready(traced)
    assert env.events == plain_events
    assert len(plain_events) == 4
    np.testing.assert_array_equal(plain[0], traced[0])
    np.testing.assert_array_equal(plain[1], traced[1])
    trace = traced[2] if kind == "fsm" else traced[3]
    np.testing.assert_array_equal(trace["valid"], [True, True, True, True, False, False])
    np.testing.assert_array_equal(trace["state_time_after"][:4], [1, 2, 3, 4])
    busy = np.asarray(trace["blue_busy"][:4, 0])
    assert np.any(busy)
    np.testing.assert_array_equal(np.asarray(trace["blue_actions"][:4, 0])[busy], 0)
    np.testing.assert_array_equal(trace["state_value_after"][:4], trace["blue_reward"][:4])
    # A second recording of the same case must preserve the entire chosen action sequence.
    again = scan(*args, **kwargs, record_trajectory=True)
    other_trace = again[2] if kind == "fsm" else again[3]
    for key in trace:
        np.testing.assert_array_equal(trace[key], other_trace[key])


def test_effective_completion_is_distinct_from_submission_and_success():
    from jaxborg.actions.encoding import BLUE_RESTORE_START
    from jaxborg.state import create_initial_const, create_initial_state

    const = create_initial_const()
    sim = create_initial_state().replace(
        blue_pending_ticks=jnp.array([1, 2, 0, 0, 0], dtype=jnp.int32),
        blue_pending_action=jnp.full(5, BLUE_RESTORE_START, dtype=jnp.int32),
    )
    after = sim.replace(blue_recovered_this_step=sim.blue_recovered_this_step.at[0].set(True))
    submitted = jnp.array([0, 0, BLUE_RESTORE_START, 1, 0], dtype=jnp.int32)
    fields = behavior_trace.behavior_fields(
        SimpleNamespace(state=sim, const=const),
        SimpleNamespace(state=after, const=const),
        submitted,
        jnp.ones((5, 242), bool),
        {"action_cost": -1},
    )
    np.testing.assert_array_equal(fields["blue_busy"], [True, True, False, False, False])
    np.testing.assert_array_equal(fields["blue_submitted_type"], [0, 0, 4, 1, 0])
    np.testing.assert_array_equal(fields["blue_effective_type"], [4, 4, 4, 1, 0])
    np.testing.assert_array_equal(fields["blue_effective_completes"], [True, False, False, True, True])
    assert bool(fields["blue_recovery_attempt_completed"][0])
    assert not np.asarray(fields["post_host_compromised"]).any()
    assert not np.asarray(fields["post_red_sessions"]).any()
    assert float(fields["action_cost"]) == -1
