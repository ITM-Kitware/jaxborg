"""Exercise the Blue/FSM trainer with real JIT and repeated recurrent updates.

The synthetic environment keeps compilation small while retaining the real
LogWrapper state, topology-bank batch path, reward normalizer, LSTM and PPO.
In particular, initial log buffers can alias; eager-only tests miss invalid
buffer donation at the compiled trainer boundary.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from jaxborg.policies import policy_from_arch
from scripts.train.algorithms import ippo_jax


@struct.dataclass
class _Const:
    marker: jax.Array


@struct.dataclass
class _State:
    time: jax.Array
    blue_pending_ticks: jax.Array


@struct.dataclass
class _EnvState:
    const: _Const
    state: _State


class _EpisodeEnv:
    agents = ("blue_0", "blue_1")
    num_agents = 2

    def observation_space(self, agent):
        del agent
        return SimpleNamespace(shape=(4,))

    def _obs(self, state):
        clock = state.state.time.astype(jnp.float32) / 4.0
        return {name: jnp.full((4,), clock + index / 10.0) for index, name in enumerate(self.agents)}

    def reset(self, key):
        del key
        state = _EnvState(
            const=_Const(marker=jnp.zeros((), dtype=jnp.int32)),
            state=_State(time=jnp.zeros((), dtype=jnp.int32), blue_pending_ticks=jnp.zeros(2, dtype=jnp.int32)),
        )
        return self._obs(state), state

    def reset_batch(self, keys, topology_key):
        del topology_key
        return jax.vmap(self.reset)(keys)

    def step(self, key, state, actions):
        del key
        next_tick = state.state.time + 1
        done = next_tick == 4
        following = state.replace(state=state.state.replace(time=jnp.where(done, 0, next_tick)))
        reward = 1.0 + 0.1 * actions["blue_0"].astype(jnp.float32)
        rewards = {name: reward for name in self.agents}
        dones = {name: done for name in (*self.agents, "__all__")}
        zero = jnp.zeros((), dtype=jnp.float32)
        info = dict.fromkeys(
            ("reward_lwf", "reward_asf", "action_cost", "impact_count", "green_lwf_count", "green_asf_count"),
            zero,
        )
        info["reward_ria"] = reward
        return self._obs(following), following, rewards, dones, info

    def step_batch(self, keys, states, actions, topology_key):
        del topology_key
        return jax.vmap(self.step)(keys, states, actions)


def test_compiled_topology_bank_lstm_runs_two_complete_updates(monkeypatch):
    """Compile, cross an episode boundary, and update both recurrent trunks."""
    monkeypatch.setattr(ippo_jax, "make_jax_env", lambda *_args, **_kwargs: _EpisodeEnv())
    monkeypatch.setattr(
        ippo_jax,
        "compute_blue_action_mask",
        lambda *_args, **_kwargs: jnp.ones(3, dtype=jnp.bool_),
    )
    config = {
        "SEED": 42,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 2,
        "TOTAL_TIMESTEPS": 32,
        "LR": 1e-2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 0.5,
        "CLIP_VALUE_LOSS": False,
        "ANNEAL_LR": False,
        "NORM_REWARDS": True,
        "REWARD_SCALE": 1.0,
        "TRAIN_VARIANT": SimpleNamespace(blue_block_policy="cc4", num_steps=4),
        "TOPOLOGY_BANK": tuple(f"test-topology-{index}.npz" for index in range(4)),
    }
    network = policy_from_arch(
        {"name": "recurrent", "hidden_dim": 8, "hidden_layers": 1, "cell": "lstm", "trunk": "separate"},
        action_dim=3,
    )
    # Explicitly override an inherited JAX_DISABLE_JIT so this regression cannot
    # silently turn into an eager test. A real CC4 smoke covers the game itself.
    with jax.disable_jit(False):
        _, obs, env_state, initialize, update = ippo_jax.make_train(config, network)
        train_state = initialize(jax.random.PRNGKey(43))
        before = jax.tree.map(lambda leaf: np.array(leaf, copy=True), train_state.params)
        norm = ippo_jax.RewardNormState(
            returns=jnp.zeros(4), mean=jnp.zeros(()), var=jnp.ones(()), count=jnp.array(1e-4)
        )
        rng = jax.random.PRNGKey(44)
        for completed in (1, 2):
            train_state, env_state, obs, rng, norm, metrics = update(train_state, env_state, obs, rng, norm)
            metrics = jax.device_get(metrics)
            assert all(np.isfinite(float(value)) for value in metrics.values())
            np.testing.assert_array_equal(np.asarray(env_state.env_state.state.time), np.zeros(4))
            np.testing.assert_array_equal(np.asarray(env_state.episode_lengths), np.zeros((4, 2)))
            np.testing.assert_array_equal(np.asarray(env_state.returned_episode_lengths), np.full((4, 2), 4))
            np.testing.assert_array_equal(np.asarray(norm.returns), np.zeros(4))
            assert int(train_state.step) == completed * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
            np.testing.assert_allclose(float(norm.count), completed * 16 + 1e-4, rtol=0, atol=1e-5)

    for trunk in ("actor_trunk", "critic_trunk"):
        old = jax.tree.leaves(before["params"][trunk]["ScannedRNN_0"])
        new = jax.tree.leaves(train_state.params["params"][trunk]["ScannedRNN_0"])
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in new)
        assert any(not np.array_equal(a, np.asarray(b)) for a, b in zip(old, new, strict=True)), trunk
