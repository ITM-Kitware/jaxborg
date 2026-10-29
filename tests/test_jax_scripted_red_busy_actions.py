"""Scripted evaluation must not charge for resubmitting an in-flight action."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.evaluation import jax_scripted_red
from jaxborg.evaluation.cia.fixed_topology import EvaluationCase
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE


@struct.dataclass
class _Sim:
    blue_pending_ticks: object


@struct.dataclass
class _State:
    state: _Sim
    extras: dict
    const: object = None


class _DurationEnv:
    agents = ("blue_0", "blue_1")

    def reset_at_topology(self, key, topology_index):
        del key, topology_index
        return self.get_obs(), _State(_Sim(jnp.array([0, 2])), {"host_resilience_role": jnp.zeros(3)})

    def get_obs(self):
        return {agent: jnp.zeros(2) for agent in self.agents}

    def get_avail_actions(self, state):
        del state
        # Toy actions: Sleep, unavailable action, charged three-tick recovery.
        return {agent: jnp.array([True, False, True]) for agent in self.agents}

    def step_env(self, key, state, actions):
        del key
        selected = jnp.stack([actions[agent] for agent in self.agents])
        ticks = state.state.blue_pending_ticks
        next_ticks = jnp.where(ticks > 0, ticks - 1, jnp.where(selected == 2, 2, 0))
        # Charge every caller-submitted recovery, including ignored busy ones,
        # as the real CC4 reward calculator does for Restore.
        reward = -jnp.sum(selected == 2).astype(jnp.float32)
        return (
            self.get_obs(),
            state.replace(state=_Sim(next_ticks)),
            dict.fromkeys(self.agents, reward),
            {"__all__": jnp.bool_(False)},
            {},
        )


def test_busy_mask_preserves_idle_legality_and_reopens_after_completion():
    env = _DurationEnv()
    _, state = env.reset_at_topology(None, None)
    masks = jax_scripted_red._blue_policy_action_masks(env, state, env.agents)
    np.testing.assert_array_equal(masks, [[True, False, True], [True, False, False]])
    ready = state.replace(state=_Sim(jnp.zeros(2, dtype=jnp.int32)))
    np.testing.assert_array_equal(
        jax_scripted_red._blue_policy_action_masks(env, ready, env.agents),
        [[True, False, True], [True, False, True]],
    )


@pytest.mark.parametrize("backend", ["jax", "cyborg"])
def test_episode_charges_only_new_recovery_decisions(monkeypatch, tmp_path, backend):
    def policy_step(module, weights, obs, mask, *, carry):
        del module, weights, obs
        logits = jnp.where(mask, jnp.array([0.0, 1.0, 2.0]), -1e10)
        return SimpleNamespace(logits=logits), jnp.zeros(2), carry

    monkeypatch.setattr(jax_scripted_red, "policy_step", policy_step)
    monkeypatch.setattr(jax_scripted_red, "initial_carry", lambda *args: None)
    monkeypatch.setattr(jax_scripted_red, "score_resilience_state", lambda *args: jnp.zeros(3))
    monkeypatch.setattr(jax_scripted_red, "cyborg_blue_flat_to_jax_lookup", lambda *args: np.arange(3))
    monkeypatch.setattr(
        jax_scripted_red,
        "_torch_blue_actions",
        lambda policy, obs, mask, lookups, **kwargs: np.argmax(np.where(mask, [0, 1, 2], -1e10), axis=-1),
    )
    case = EvaluationCase(0, tmp_path / "topology.npz", "fp", 1000, 0, 1000, (0, 0, 0), "roles")
    result = jax_scripted_red.run_jax_scripted_red_episode(
        LoadedMatchupPolicy("blue", backend, object(), {}, {}),
        env=_DurationEnv(),
        variant=replace(CIA_RESILIENCE, num_steps=6),
        case=case,
        deterministic=True,
    )
    # Each agent starts two recoveries. Before the fix all twelve submitted
    # actions were charged, including the eight ignored submissions.
    assert result.reward == -4.0
    assert result.cia == (0.0, 0.0, 0.0)
