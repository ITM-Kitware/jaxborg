"""Evaluation must submit the saved Blue message head, including ablations."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import struct

from jaxborg.evaluation import cyborg_runner, matchup_runner
from jaxborg.evaluation.jax_runner import _policy_dist
from jaxborg.evaluation.message_diagnostics import summarize_message_statistics
from jaxborg.policies import init_policy_params, policy_from_arch
from jaxborg.policies.message_override import MessageOverridePolicy
from jaxborg.policies.torch_messages import TorchMessageOverride
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.eval.eval_comm_transfer import transfer_scores


@struct.dataclass
class Sim:
    time: object
    red_agent_active: object


@struct.dataclass
class Const:
    blue_obs_subnets: object


@struct.dataclass
class State:
    state: Sim
    const: Const


class TinyMessageEnv:
    blue_agents = tuple(f"blue_{i}" for i in range(5))
    red_agents = tuple(f"red_{i}" for i in range(6))

    def reset(self, key):
        state = State(Sim(jnp.int32(0), jnp.ones(6, dtype=bool)), Const(jnp.array([[0, -1, -1]] * 4 + [[1, 2, 3]])))
        return self.obs(), state

    def reset_at_topology(self, key, index):
        return self.reset(key)

    def obs(self):
        return {a: jnp.zeros(210) for a in self.blue_agents + self.red_agents}

    def get_avail_actions(self, state):
        return {a: jnp.ones(242, dtype=bool) for a in self.blue_agents + self.red_agents}

    def step_env(self, key, state, actions):
        assert "red_messages" not in actions
        reward = actions.get("blue_messages", jnp.zeros((5, 8))).sum()
        state = state.replace(state=state.state.replace(time=state.state.time + 1))
        return self.obs(), state, {a: reward for a in self.blue_agents}, {"__all__": state.state.time >= 2}, {}

    step = step_env


@pytest.mark.parametrize("mute,foreign,expected", [(False, False, 80), (True, False, 0), (False, True, 0)])
def test_compiled_jax_evaluation_sends_messages_and_collects_diagnostics(mute, foreign, expected):
    arch = dict(name="shared", hidden_dim=4, hidden_layers=1)
    blue = policy_from_arch({**arch, "message_dim": 8}, action_dim=242)
    red = policy_from_arch(arch, action_dim=242)
    bw, rw = [init_policy_params(m, jax.random.PRNGKey(i), 210) for i, m in enumerate((blue, red))]
    bw["params"]["actor_message"]["bias"] = jnp.full(8, 30.0)
    sw = copy.deepcopy(bw)
    sw["params"]["actor_message"]["bias"] = jnp.full(8, -30.0)
    if mute or foreign:
        blue = MessageOverridePolicy(blue, blue if foreign else None, mute)
        bw = dict(actor=bw, sender=sw)
    reward, _, stats = matchup_runner._run_jax_matchup_episode_scan(
        bw,
        rw,
        jax.random.PRNGKey(2),
        jnp.int32(0),
        jnp.zeros(1, dtype=jnp.int32),
        blue_module=blue,
        red_module=red,
        env=TinyMessageEnv(),
        num_steps=2,
        deterministic=True,
        use_topology_index=False,
        score_cia=False,
        collect_messages=True,
    )
    assert reward == expected
    summary = summarize_message_statistics(stats)
    assert summary["samples"] == 10
    np.testing.assert_allclose(summary["bit_mean"], expected / 80)
    np.testing.assert_allclose(summary["evidence_mutual_information_bits"], 0)


@pytest.mark.parametrize("mute,foreign,expected", [(False, False, 80), (True, False, 0), (False, True, 0)])
def test_torch_evaluation_sends_messages_and_supports_controls(mute, foreign, expected):
    arch = dict(name="shared", hidden_dim=4, hidden_layers=1)
    blue = policy_from_arch({**arch, "message_dim": 8}, backend="cyborg", obs_dim=210, action_dim=242)
    red = policy_from_arch(arch, backend="cyborg", obs_dim=210, action_dim=242)
    with torch.no_grad():
        blue.message_head.bias.fill_(30)
    sender = copy.deepcopy(blue)
    with torch.no_grad():
        sender.message_head.bias.fill_(-30)
    if mute or foreign:
        blue = TorchMessageOverride(blue, sender if foreign else None, mute)
    policies = {
        t: matchup_runner.LoadedMatchupPolicy(t, "cyborg", m, {}, {}) for t, m in [("blue", blue), ("red", red)]
    }
    diagnostics = []
    reward = matchup_runner.run_matchup_episode(
        policies,
        variant=replace(CC4_STOCK, num_steps=2),
        seed=3,
        deterministic=True,
        env=TinyMessageEnv(),
        message_diagnostics=diagnostics,
    )
    assert reward == expected
    assert diagnostics[0]["samples"] == 10


def test_native_cyborg_runner_passes_bits_to_step(monkeypatch):
    agent = policy_from_arch(
        dict(name="shared", hidden_dim=4, hidden_layers=1, message_dim=8), backend="cyborg", obs_dim=210, action_dim=242
    )
    with torch.no_grad():
        agent.message_head.bias.fill_(30)
    obs = {a: np.zeros(210, dtype=np.float32) for a in cyborg_runner.AGENT_IDS}
    info = {a: {"action_mask": np.ones(242)} for a in obs}
    monkeypatch.setattr(cyborg_runner, "reset_cyborg_env", lambda *a, **kw: SimpleNamespace(obs=obs, info=info))

    class Env:
        def step(self, actions, messages):
            assert set(messages) == set(obs)
            for bits in messages.values():
                np.testing.assert_array_equal(bits, np.ones(8, dtype=bool))
            return obs, dict.fromkeys(obs, 1.0), {"__all__": True}, {}, info

    assert cyborg_runner.rollout_episode(Env(), CC4_STOCK, 3, agent, deterministic=True) == 1


def test_jax_native_inference_preserves_message_head():
    module = policy_from_arch(dict(name="shared", hidden_dim=4, hidden_layers=1, message_dim=8), action_dim=3)
    weights = init_policy_params(module, jax.random.PRNGKey(0), 210)
    pi, _ = _policy_dist(module, weights, jnp.ones(210), jnp.ones(3))
    assert pi.message_logits.shape == (8,)


def test_message_information_probe_and_transfer_report():
    counts = np.zeros((8, 2, 2))
    counts[:, 0, 0] = counts[:, 1, 1] = 5
    stats = np.concatenate([[0.0], counts.flatten(), [10.0]])
    np.testing.assert_allclose(summarize_message_statistics(stats)["evidence_mutual_information_bits"], 1)
    assert transfer_scores(10.0, 6.0, 8.0) == dict(comm_gain=4.0, transfer=2.0, transfer_fraction=0.5)
    assert transfer_scores(6.0, 6.0, 8.0)["transfer_fraction"] is None
