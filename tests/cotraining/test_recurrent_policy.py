"""Contract tests for `arch.name: recurrent`.

Three things have to hold for a sequence policy to be trainable by PPO here:

1. It actually remembers — otherwise the whole point (cause 2 in
   docs/cotraining_collapse.md, evidence that ages out in ~2 steps) is lost.
2. Stepping it one row at a time reproduces replaying the whole window at
   once. PPO's importance ratio compares a stored rollout log-prob against a
   re-computed one; if those two paths disagree, every ratio is wrong.
3. A reset flag really blanks the state, so one episode — or one Red agent's
   intrusion — cannot leak into the next.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.policies import (
    POLICY_REGISTRY,
    init_policy_params,
    initial_carry,
    is_recurrent,
    make_torch_policy,
    policy_from_arch,
    policy_sequence,
    policy_step,
)

OBS_DIM = 7
ACTION_DIM = 5


def _arch(**overrides):
    arch = {
        "name": "recurrent",
        "hidden_dim": 12,
        "hidden_layers": 1,
        "activation": "tanh",
        "cell": "gru",
        "trunk": "shared",
    }
    arch.update(overrides)
    return arch


def _built(**overrides):
    module = policy_from_arch(_arch(**overrides), action_dim=ACTION_DIM)
    params = init_policy_params(module, jax.random.PRNGKey(0), OBS_DIM)
    return module, params


def test_registered_and_flagged_recurrent():
    assert "recurrent" in POLICY_REGISTRY
    module, _ = _built()
    assert is_recurrent(module)
    assert not is_recurrent(policy_from_arch({"name": "shared"}, action_dim=ACTION_DIM))


@pytest.mark.parametrize("cell", ["gru", "lstm"])
@pytest.mark.parametrize("trunk", ["shared", "separate"])
def test_stepwise_matches_whole_window_replay(cell, trunk):
    """The rollout path and the update path must compute the same policy.

    The rollout steps one row at a time and stores a log-prob; the PPO update
    replays the stored window in one call. A mismatch would silently corrupt
    every importance ratio, and nothing else in the trainer would notice.
    """
    module, params = _built(cell=cell, trunk=trunk)
    steps, rows = 6, 3
    obs = jax.random.normal(jax.random.PRNGKey(1), (steps, rows, OBS_DIM))
    resets = jnp.zeros((steps, rows), dtype=jnp.bool_).at[3, 1].set(True)

    seq_pi, seq_value, _ = policy_sequence(module, params, obs, None, carry=initial_carry(module, rows), reset=resets)

    carry = initial_carry(module, rows)
    logits, values = [], []
    for t in range(steps):
        pi, value, carry = policy_step(module, params, obs[t], None, carry=carry, reset=resets[t])
        logits.append(pi.logits)
        values.append(value)

    np.testing.assert_allclose(np.asarray(jnp.stack(logits)), np.asarray(seq_pi.logits), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.stack(values)), np.asarray(seq_value), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_identical_observation_reads_differently_after_a_different_history(cell):
    """The defining property: this policy is not a function of the frame alone."""
    module, params = _built(cell=cell)
    probe = jnp.ones((1, OBS_DIM), dtype=jnp.float32)
    history = jax.random.normal(jax.random.PRNGKey(2), (4, 1, OBS_DIM))

    def logits_after(prefix):
        carry = initial_carry(module, 1)
        for row in prefix:
            _, _, carry = policy_step(module, params, row, None, carry=carry)
        pi, _, _ = policy_step(module, params, probe, None, carry=carry)
        return np.asarray(pi.logits)

    assert not np.allclose(logits_after(history), logits_after(history[:2]), atol=1e-6)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_reset_blanks_the_hidden_state(cell):
    """A set reset flag must be indistinguishable from a fresh episode.

    Red agents go dormant and are revived by session reassignment mid-episode;
    without this the revived agent inherits its predecessor's belief about a
    part of the network it no longer has a session on.
    """
    module, params = _built(cell=cell)
    probe = jnp.ones((1, OBS_DIM), dtype=jnp.float32)

    carry = initial_carry(module, 1)
    for row in jax.random.normal(jax.random.PRNGKey(3), (4, 1, OBS_DIM)):
        _, _, carry = policy_step(module, params, row, None, carry=carry)
    after_reset, _, _ = policy_step(module, params, probe, None, carry=carry, reset=jnp.array([True]))
    fresh, _, _ = policy_step(module, params, probe, None, carry=initial_carry(module, 1))

    np.testing.assert_allclose(np.asarray(after_reset.logits), np.asarray(fresh.logits), rtol=1e-6, atol=1e-7)


def test_reset_is_per_row_not_per_batch():
    module, params = _built()
    obs = jnp.tile(jnp.ones((1, OBS_DIM), dtype=jnp.float32), (2, 1))

    carry = initial_carry(module, 2)
    for row in jax.random.normal(jax.random.PRNGKey(4), (3, 2, OBS_DIM)):
        _, _, carry = policy_step(module, params, row, None, carry=carry)
    pi, _, _ = policy_step(module, params, obs, None, carry=carry, reset=jnp.array([True, False]))

    assert not np.allclose(np.asarray(pi.logits[0]), np.asarray(pi.logits[1]), atol=1e-6)


def test_lstm_carries_a_cell_and_hidden_state_gru_carries_one():
    gru, _ = _built(cell="gru")
    lstm, _ = _built(cell="lstm")
    assert len(jax.tree.leaves(initial_carry(gru, 3))) == 1
    assert len(jax.tree.leaves(initial_carry(lstm, 3))) == 2


def test_separate_trunk_gives_actor_and_critic_their_own_state():
    module, _ = _built(trunk="separate")
    carry = initial_carry(module, 3)
    assert set(carry) == {"actor", "critic"}


def test_available_action_mask_is_applied():
    module, params = _built()
    mask = jnp.zeros((1, ACTION_DIM), dtype=jnp.bool_).at[0, 2].set(True)
    pi, _, _ = policy_step(module, params, jnp.ones((1, OBS_DIM)), mask, carry=initial_carry(module, 1))
    assert int(jnp.argmax(pi.logits[0])) == 2


def test_a_call_site_that_forgot_the_carry_fails_loudly():
    """Silently running memoryless would be the expensive failure mode."""
    module, params = _built()
    with pytest.raises(ValueError, match="time-major"):
        module.apply(params, jnp.ones((3, OBS_DIM)), jnp.ones((3, ACTION_DIM)))
    with pytest.raises(ValueError, match="needs a carry"):
        policy_step(module, params, jnp.ones((3, OBS_DIM)))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cell": "rnn"}, "arch.cell"),
        ({"trunk": "tied"}, "arch.trunk"),
        ({"hidden_layers": 0}, "hidden_layers"),
        ({"depth": 3}, "unknown recurrent arch options"),
    ],
)
def test_bad_arch_options_are_rejected_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        policy_from_arch(_arch(**overrides), action_dim=ACTION_DIM)


def test_torch_backend_refuses_rather_than_training_a_different_model():
    with pytest.raises(NotImplementedError, match="no CybORG/torch backend"):
        make_torch_policy("recurrent", obs_dim=OBS_DIM, action_dim=ACTION_DIM)


def test_feedforward_archs_still_ignore_the_carry_plumbing():
    """The helpers are the single call path, so they must be a no-op for MLPs."""
    module = policy_from_arch({"name": "shared", "hidden_dim": 8, "hidden_layers": 1}, action_dim=ACTION_DIM)
    params = init_policy_params(module, jax.random.PRNGKey(5), OBS_DIM)
    assert initial_carry(module, 4) is None

    obs = jnp.ones((4, OBS_DIM), dtype=jnp.float32)
    pi, value, carry = policy_step(module, params, obs, None, carry=None, reset=jnp.ones(4, dtype=jnp.bool_))
    direct_pi, direct_value = module.apply(params, obs, None)

    assert carry is None
    np.testing.assert_array_equal(np.asarray(pi.logits), np.asarray(direct_pi.logits))
    np.testing.assert_array_equal(np.asarray(value), np.asarray(direct_value))


def test_recurrent_only_arch_options_are_refused_by_feedforward_archs():
    """`cell` / `trunk` under `arch.name: shared` must fail, not be ignored.

    `recipes/config_all.yaml` documents both fields as commented-out entries in
    the `arch` block, so uncommenting one without changing `name` is an easy
    slip — and silently dropping it would train a different architecture than
    the recipe asks for.
    """
    with pytest.raises(ValueError, match="does not accept the extra arch options"):
        policy_from_arch({"name": "shared", "hidden_dim": 8, "cell": "gru"}, action_dim=ACTION_DIM)
