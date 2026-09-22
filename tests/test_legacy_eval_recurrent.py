import importlib
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation import jax_runner
from jaxborg.policies import init_policy_params, policy_from_arch
from jaxborg.policies.base import RecurrentPolicy
from jaxborg.policies.categorical import Categorical
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.eval import cec_phase6_eval_jax


class CountingPolicy(RecurrentPolicy):
    def initialize_carry(self, batch_size):
        return jnp.zeros(batch_size, dtype=jnp.int32)

    def apply(self, params, carry, obs, mask, resets):
        carry = carry + 1
        logits = jax.nn.one_hot(carry % 3, 3) * 10
        return carry, Categorical(logits=logits[None]), jnp.zeros(obs.shape[:-1])


def test_phase6_threads_recurrent_memory_and_resets_between_episodes(monkeypatch, tmp_path):
    class Env:
        def reset(self, key):
            return {f"blue_{i}": jnp.zeros(1) for i in range(NUM_BLUE_AGENTS)}, jnp.int32(0)

        def get_avail_actions(self, state):
            return {f"blue_{i}": jnp.ones(3, dtype=bool) for i in range(NUM_BLUE_AGENTS)}

        def step(self, key, state, actions):
            obs, _ = self.reset(key)
            rewards = {name: action.astype(jnp.float32) for name, action in actions.items()}
            return obs, state + 1, rewards, {"__all__": state == 2}, {}

    recipe = {"meta": {"name": "recurrent"}, "train": {}}
    monkeypatch.setattr(jax_runner, "load_jax_checkpoint", lambda _: (CountingPolicy(), {}, recipe))
    monkeypatch.setattr(
        cec_phase6_eval_jax, "_build_eval_env", lambda *args, **kwargs: (replace(CC4_STOCK, num_steps=3), Env())
    )
    monkeypatch.setattr(
        "jaxborg.recipe.project_eval", lambda *args, **kwargs: {"TOPOLOGY_BANK": [], "TOPOLOGY_SAMPLING": "random"}
    )

    row = cec_phase6_eval_jax.run_eval(model_path=tmp_path / "model.safetensors", eval_red="fsm", episodes=2, seed=7)

    # The action sequence must be 1, 2, 0 for both fresh episodes.
    assert row["per_episode"] == [3.0, 3.0]


@pytest.mark.parametrize("arch", ["shared", "recurrent"])
def test_trajectory_policy_supports_lstm_memory_and_deterministic_actions(monkeypatch, arch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/eval"))
    trajectories = importlib.import_module("scripts.eval.generate_cynex_trajectories")
    config = {"name": arch, "hidden_dim": 4, "hidden_layers": 1}
    if arch == "recurrent":
        config["cell"] = "lstm"
    policy = policy_from_arch(config, action_dim=3)
    params = init_policy_params(policy, jax.random.PRNGKey(42), obs_dim=2)
    monkeypatch.setattr(jax_runner, "load_jax_checkpoint", lambda _: (policy, params, {"arch": config}))
    step, _ = trajectories._load_jax_model("model.safetensors")
    obs = jnp.ones((32, 2))
    masks = jnp.ones((32, 3), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(7), 32)

    actions, logits, carry = step(obs, masks, keys, deterministic=True)
    np.testing.assert_array_equal(actions, jnp.argmax(logits, axis=-1))
    _, next_logits, _ = step(obs, masks, keys, carry, deterministic=True)
    _, fresh_logits, _ = step(obs, masks, keys, deterministic=True)
    np.testing.assert_array_equal(logits, fresh_logits)
    if arch == "recurrent":
        assert not np.allclose(next_logits, fresh_logits)
