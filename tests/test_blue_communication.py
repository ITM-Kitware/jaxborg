"""Blue messaging must be chosen, transported, and restored from checkpoints."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxborg.checkpoint import PolicyBundleEntry, load_jax_bundle, load_torch_bundle, save_jax_bundle, save_torch_bundle
from jaxborg.policies import init_policy_params, initial_carry, policy_from_arch, policy_sequence, policy_step
from jaxborg.policies.bernoulli import Bernoulli
from jaxborg.policies.message_override import MessageOverridePolicy
from jaxborg.recipe import load, project_cleanrl, team_recipe


def test_recipe_switch_defaults_off_and_enables_only_blue():
    recipe = load("cotraining")
    recipe.pop("use_messages", None)
    assert team_recipe(recipe, "blue")["arch"].get("message_dim", 0) == 0
    recipe["use_messages"] = True
    assert team_recipe(recipe, "blue")["arch"]["message_dim"] == 8
    assert team_recipe(recipe, "red")["arch"].get("message_dim", 0) == 0
    assert project_cleanrl(recipe, team="blue")["use_messages"]
    assert not project_cleanrl(recipe, team="red")["use_messages"]
    recipe["train"].setdefault("team_overrides", {}).setdefault("red", {}).setdefault("arch", {})["message_dim"] = 8
    with pytest.raises(ValueError, match="Blue only"):
        team_recipe(recipe, "red")
    recipe["use_messages"] = "false"
    with pytest.raises(ValueError, match="YAML boolean"):
        team_recipe(recipe, "blue")


def test_bernoulli_log_prob_entropy_and_gradients():
    dist = Bernoulli(jnp.array([[0.0, 2.0, -3.0], [1000.0, -1000.0, 0.0]]))
    bits = jnp.array([[0, 1, 1], [1, 0, 0]], dtype=jnp.float32)
    reference = torch.distributions.Bernoulli(logits=torch.tensor(np.asarray(dist.logits)))
    np.testing.assert_allclose(
        dist.log_prob(bits), reference.log_prob(torch.tensor(np.asarray(bits))).sum(-1), atol=1e-6
    )
    np.testing.assert_allclose(dist.entropy(), reference.entropy().sum(-1), atol=1e-6)
    gradient = jax.grad(lambda x: Bernoulli(x).log_prob(bits).sum())(dist.logits)
    assert np.isfinite(gradient).all()
    assert set(np.asarray(dist.sample(jax.random.PRNGKey(4))).flatten()) <= {0, 1}


@pytest.mark.parametrize("name", ["shared", "separate", "mappo", "recurrent", "recurrent_mappo"])
def test_jax_message_head_replays_and_legacy_parameters_stay_compatible(name):
    arch = dict(name=name, hidden_dim=8, hidden_layers=1)
    legacy = policy_from_arch(arch, action_dim=3)
    explicit_off = policy_from_arch({**arch, "message_dim": 0}, action_dim=3)
    old_params = init_policy_params(legacy, jax.random.PRNGKey(0), 4)
    off_params = init_policy_params(explicit_off, jax.random.PRNGKey(0), 4)
    for a, b in zip(jax.tree.leaves(old_params), jax.tree.leaves(off_params)):
        np.testing.assert_array_equal(a, b)
    module = policy_from_arch({**arch, "message_dim": 8}, action_dim=3)
    params = init_policy_params(module, jax.random.PRNGKey(1), 4)
    obs = jax.random.normal(jax.random.PRNGKey(2), (3, 2, 4))
    masks = jnp.broadcast_to(jnp.array([True, False, False]), (3, 2, 3))
    carry = initial_carry(module, 2)
    seq, _, _ = policy_sequence(module, params, obs, masks, carry=carry)
    messages = []
    for t in range(3):
        pi, _, carry = policy_step(module, params, obs[t], masks[t], carry=carry)
        messages.append(pi.message_logits)
        np.testing.assert_array_equal(pi.sample(jax.random.PRNGKey(t)), 0)
        assert pi.messages.entropy().min() > 0  # Busy agents still have choices.
    np.testing.assert_allclose(jnp.stack(messages), seq.message_logits, atol=1e-6)
    assert seq.message_logits.shape == (3, 2, 8)


@pytest.mark.parametrize("backend", ["jax", "cyborg"])
def test_message_checkpoint_round_trip_and_legacy_load(tmp_path, backend):
    for dimension in (0, 8):
        arch = dict(name="shared", hidden_dim=8, hidden_layers=1, message_dim=dimension)
        module = policy_from_arch(arch, action_dim=3, backend=backend, obs_dim=4)
        if backend == "jax":
            weights = init_policy_params(module, jax.random.PRNGKey(0), 4)
            save, read, suffix = save_jax_bundle, load_jax_bundle, ".safetensors"
        else:
            weights = module.state_dict()
            save, read, suffix = save_torch_bundle, load_torch_bundle, ".pt"
        path = tmp_path / f"blue{dimension}{suffix}"
        metadata = arch if dimension else {k: v for k, v in arch.items() if k != "message_dim"}
        save(path, {"blue": PolicyBundleEntry(weights=weights, team="blue", obs_dim=4, action_dim=3, arch=metadata)})
        entry = read(path).policies["blue"]
        restored = policy_from_arch(entry.arch, action_dim=3, backend=backend, obs_dim=4)
        assert restored.message_dim == dimension
        if backend == "cyborg":
            restored.load_state_dict(entry.weights)
        else:
            pi, _, _ = policy_step(restored, entry.weights, jnp.ones((1, 4)))
            assert (pi.messages is not None) == bool(dimension)


def test_mute_and_foreign_sender_preserve_actor_and_use_separate_carry():
    module = policy_from_arch(dict(name="shared", hidden_dim=8, hidden_layers=1, message_dim=8), action_dim=3)
    weights = init_policy_params(module, jax.random.PRNGKey(0), 4)
    foreign = copy.deepcopy(weights)
    foreign["params"]["actor_message"]["bias"] = jnp.full(8, 20.0)
    obs = jnp.ones((2, 4))
    original, _, _ = policy_step(module, weights, obs)
    for mute in (True, False):
        wrapper = MessageOverridePolicy(module, None if mute else module, mute)
        pi, _, _ = policy_step(wrapper, {"actor": weights, "sender": foreign}, obs, carry=initial_carry(wrapper, 2))
        np.testing.assert_array_equal(pi.logits, original.logits)
        np.testing.assert_array_equal(pi.sample_messages(jax.random.PRNGKey(1), deterministic=True), 0 if mute else 1)


def test_native_and_jax_deliver_through_real_step_and_clear_messages():
    from jaxborg.cyborg_joint import BLUE_AGENT_IDS, POLICY_AGENT_IDS, CyborgJointAdapter
    from jaxborg.env import ScenarioEnvState
    from jaxborg.joint_env import JointPolicyCC4Env
    from jaxborg.observations import SUBNET_BLOCK_SIZE
    from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
    from jaxborg.state import create_initial_state

    native = CyborgJointAdapter(CC4_STOCK, seed=7)
    native.reset(ep_seed=11)
    try:
        const = build_const_from_cyborg(native.raw_env).replace(max_steps=jnp.int32(3))
        state = ScenarioEnvState(create_initial_state().replace(host_services=const.initial_services), const)
        env = JointPolicyCC4Env()
        native_actions = {a: native._blue_sleep_index(a) if a.startswith("blue") else 0 for a in POLICY_AGENT_IDS}
        jax_actions = {a: jnp.int32(0) for a in env.agents}
        bits = (np.arange(40).reshape(5, 8) % 3 == 0).astype(np.float32)
        # No direct state.messages writes: both backends must transport the submission.
        for payload in (bits, None):
            na, ja = dict(native_actions), dict(jax_actions)
            if payload is not None:
                na["blue_messages"], ja["blue_messages"] = payload, jnp.asarray(payload)
            native_obs, *_ = native.step(na)
            obs, state, *_ = env.step_env(jax.random.PRNGKey(2), state, ja)
            for receiver, name in enumerate(BLUE_AGENT_IDS):
                start = 1 + int((np.asarray(const.blue_obs_subnets[receiver]) >= 0).sum()) * SUBNET_BLOCK_SIZE
                expected = np.delete(bits, receiver, axis=0).flatten() if payload is not None else np.zeros(32)
                np.testing.assert_array_equal(obs[f"blue_{receiver}"][start : start + 32], expected)
                np.testing.assert_array_equal(native_obs[name][start : start + 32], expected)
            np.testing.assert_array_equal(state.state.messages[np.arange(5), np.arange(5)], 0)
        # Terminal auto-reset cannot leak the last episode's messages.
        _, reset_state, _, dones, _ = env.step(
            jax.random.PRNGKey(3), state, {**jax_actions, "blue_messages": jnp.asarray(bits)}
        )
        assert dones["__all__"]
        np.testing.assert_array_equal(reset_state.state.messages, 0)
        with pytest.raises(ValueError, match="Blue only"):
            native.step({**native_actions, "red_messages": bits})
        with pytest.raises(ValueError, match="Blue only"):
            env.step_env(jax.random.PRNGKey(2), state, {**jax_actions, "red_messages": bits})
    finally:
        native.close()
