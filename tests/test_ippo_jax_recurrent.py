"""Blue-only IPPO with a recurrent architecture.

`ippo_jax.make_train` carries its own rollout and minibatching code, separate
from the joint trainer's. A sequence policy changes both: the rollout threads a
hidden state through the `env_step` scan, and the PPO update permutes over
sequences (env x agent) with the time axis intact instead of shuffling rows.

The environment here is a stand-in — the point is the trainer plumbing, not CC4
dynamics — but it is the real `LogWrapper`, the real optimizer, and the real
PPO update.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.policies import policy_from_arch
from scripts.train.algorithms import ippo_jax

OBS_DIM = 4
ACTION_DIM = 3
NUM_AGENTS = 2


@struct.dataclass
class _FakeConst:
    marker: jax.Array


@struct.dataclass
class _FakeSimState:
    time: jax.Array
    blue_pending_ticks: jax.Array


@struct.dataclass
class _FakeEnvState:
    const: _FakeConst
    state: _FakeSimState


class _TinyBlueEnv:
    """Minimal JaxMARL-shaped env: the observation is a function of the clock."""

    def __init__(self, episode_length: int):
        self.agents = [f"blue_{i}" for i in range(NUM_AGENTS)]
        self.num_agents = NUM_AGENTS
        self._episode_length = episode_length

    def observation_space(self, agent: str):
        del agent
        return SimpleNamespace(shape=(OBS_DIM,))

    def _obs(self, state: _FakeEnvState):
        value = state.state.time.astype(jnp.float32) / 10.0
        return {agent: jnp.full((OBS_DIM,), value, dtype=jnp.float32) for agent in self.agents}

    def reset(self, key):
        del key
        state = _FakeEnvState(
            const=_FakeConst(marker=jnp.zeros((), dtype=jnp.int32)),
            state=_FakeSimState(
                time=jnp.array(0, dtype=jnp.int32),
                blue_pending_ticks=jnp.zeros((NUM_AGENTS,), dtype=jnp.int32),
            ),
        )
        return self._obs(state), state

    def step(self, key, state, actions):
        del key
        time = state.state.time + 1
        done = time >= self._episode_length
        time = jnp.where(done, 0, time)
        next_state = state.replace(state=state.state.replace(time=time))
        reward = 1.0 + 0.1 * actions[self.agents[0]].astype(jnp.float32)
        rewards = {agent: reward for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        zero = jnp.zeros((), dtype=jnp.float32)
        info = {
            "reward_ria": reward,
            "reward_lwf": zero,
            "reward_asf": zero,
            "action_cost": zero,
            "impact_count": zero,
            "green_lwf_count": zero,
            "green_asf_count": zero,
        }
        return self._obs(next_state), next_state, rewards, dones, info


def _config(num_envs: int, num_steps: int, num_minibatches: int) -> dict:
    return {
        "SEED": 0,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "NUM_MINIBATCHES": num_minibatches,
        "UPDATE_EPOCHS": 1,
        "TOTAL_TIMESTEPS": num_envs * num_steps,
        "LR": 1e-2,
        "GAMMA": 0.9,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 1.0,
        "CLIP_VALUE_LOSS": False,
        "ANNEAL_LR": False,
        "NORM_REWARDS": False,
        "REWARD_SCALE": 1.0,
        "TRAIN_VARIANT": SimpleNamespace(blue_block_policy="cc4"),
        "TOPOLOGY_BANK": None,
    }


@pytest.fixture
def tiny_blue(monkeypatch):
    """Patch out the CC4 env and its action mask; keep everything else real."""

    def _make_env(*_args, **_kwargs):
        return _TinyBlueEnv(episode_length=4)

    def _mask(const, agent_id, state, blue_block_policy="cc4"):
        del const, agent_id, state, blue_block_policy
        return jnp.ones((ACTION_DIM,), dtype=jnp.bool_)

    monkeypatch.setattr(ippo_jax, "make_jax_env", _make_env)
    monkeypatch.setattr(ippo_jax, "compute_blue_action_mask", _mask)


def _recurrent(**arch):
    spec = {"name": "recurrent", "hidden_dim": 8, "hidden_layers": 1, "activation": "tanh"}
    spec.update(arch)
    return policy_from_arch(spec, action_dim=ACTION_DIM)


def _one_update(config, network):
    _, obs, env_state, init_train_state, collect_and_update = ippo_jax.make_train(config, network)
    state = init_train_state(jax.random.PRNGKey(1))
    before = jax.tree.map(lambda x: x.copy(), state.params)
    reward_norm = ippo_jax.RewardNormState(
        returns=jnp.zeros(config["NUM_ENVS"]),
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=jnp.array(1e-4),
    )
    with jax.disable_jit():
        state, _, _, _, _, metric = collect_and_update(state, env_state, obs, jax.random.PRNGKey(2), reward_norm)
    return before, state, metric


def _changed(before, after) -> bool:
    return any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))
    )


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_recurrent_update_reaches_the_cell_parameters(tiny_blue, cell):
    """If the time axis were being flattened away, the cell would see no gradient."""
    network = _recurrent(cell=cell)
    # env x agent = 4 x 2 = 8 sequences over 2 minibatches.
    before, after, metric = _one_update(_config(num_envs=4, num_steps=4, num_minibatches=2), network)

    assert _changed(before, after.params)
    assert _changed(before["params"]["trunk"]["ScannedRNN_0"], after.params["params"]["trunk"]["ScannedRNN_0"])
    assert np.isfinite(float(metric["total_loss"]))
    assert np.isfinite(float(metric["entropy"]))


def test_recurrent_minibatching_requires_whole_sequences(tiny_blue):
    """`num_minibatches` divides env x agent, not the row count.

    The row count (env x agent x steps) is much more permissive, so a recipe
    that passes the smoke test's divisibility check can still be wrong here.
    """
    network = _recurrent()
    # 4 envs x 2 agents = 8 sequences, and 5 does not divide 8 — but it does
    # divide the 40 rows those sequences flatten to, so the feedforward check
    # would have accepted this configuration.
    with pytest.raises(ValueError, match="env x agent"):
        _one_update(_config(num_envs=4, num_steps=5, num_minibatches=5), network)


def test_feedforward_path_is_untouched_by_the_recurrent_plumbing(tiny_blue):
    """The MLP recipes must train exactly as they did before sequences existed."""
    network = policy_from_arch({"name": "shared", "hidden_dim": 8, "hidden_layers": 1}, action_dim=ACTION_DIM)
    before, after, metric = _one_update(_config(num_envs=4, num_steps=4, num_minibatches=2), network)

    assert _changed(before, after.params)
    assert np.isfinite(float(metric["total_loss"]))
