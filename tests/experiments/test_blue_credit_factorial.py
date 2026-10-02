"""Verify actor credit and critic-target separation against execution-revision PPO."""

import subprocess
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from jaxborg.blue_learning_probe import first_minibatch
from jaxborg.policies import make_jax_policy
from scripts.experiments.blue_learning_mechanism import reference_update
from scripts.train.algorithms import ippo_jax_joint as trainer


@pytest.fixture
def sample():
    network = make_jax_policy("shared", action_dim=3, hidden_dim=8, hidden_layers=1)
    params = trainer.init_policy_params(network, jax.random.PRNGKey(11), 4)
    state = TrainState.create(apply_fn=network.apply, params=params, tx=optax.adam(0.003, eps=1e-5))
    obs = jnp.arange(4 * 2 * 1 * 4, dtype=jnp.float32).reshape((4, 2, 1, 4)) / 10
    avail = jnp.ones((4, 2, 1, 3), dtype=bool)
    pi, value, _ = trainer.policy_step(network, params, obs, avail)
    actions = jnp.array([[[0], [1]], [[2], [0]], [[1], [2]], [[2], [1]]])
    traj = trainer.TeamTransition(
        done=jnp.zeros((4, 2, 1)).at[2, 0].set(1),
        action=actions,
        value=value,
        reward=jnp.array([1.0, -2.0, 3.0, 0.5, -4.0, 5.0, 2.0, -1.0]).reshape((4, 2, 1)),
        log_prob=pi.log_prob(actions),
        obs=obs,
        avail_actions=avail,
        actor_mask=jnp.ones((4, 2, 1)).at[1].set(0),
        critic_mask=jnp.ones((4, 2, 1)),
    )
    config = {
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 2,
        "MAX_GRAD_NORM": 0.5,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
    }
    return network, state, traj, jnp.array([[1.5], [-0.75]]), jax.random.PRNGKey(12), config


def reference():
    source = subprocess.check_output(
        ["git", "show", "0d577faab9ac3e0a06376e99e9592aadf128c130:scripts/train/algorithms/ippo_jax_joint.py"]
    )
    module = ModuleType("original_joint_ppo_for_test")
    exec(compile(source, "original-execution-joint-ppo.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("actor,critic", [(0.95, 0.95), (0.95, 1.0), (1.0, 0.95), (1.0, 1.0)])
def test_complete_update_matches_independently_composed_original_updater(sample, actor, critic):
    network, state, traj, last, key, base = sample
    config = {**base, "ACTOR_GAE_LAMBDA": actor, "CRITIC_TARGET_LAMBDA": critic}
    ref = reference()
    original_fn = ref.compute_gae
    expected = reference_update(ref, network, config, state, traj, last, key)
    actual = jax.jit(trainer._make_team_updater(network, config))(state, traj, last, key)
    assert ref.compute_gae is original_fn
    # Include parameters, Adam moments, step, shuffled RNG and loss diagnostics.
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)


def test_changing_critic_lambda_preserves_actor_credit_and_changing_actor_preserves_targets(sample):
    _, _, traj, last, _, config = sample
    estimates = {
        (a, c): trainer.compute_actor_credit_and_value_targets(
            traj, last, {**config, "ACTOR_GAE_LAMBDA": a, "CRITIC_TARGET_LAMBDA": c}
        )
        for a in (0.95, 1.0)
        for c in (0.95, 1.0)
    }
    for lam in (0.95, 1.0):
        np.testing.assert_array_equal(estimates[lam, 0.95][0], estimates[lam, 1.0][0])
        np.testing.assert_array_equal(estimates[0.95, lam][1], estimates[1.0, lam][1])
        legacy = trainer.compute_gae(traj, last, gamma=0.99, gae_lambda=lam)
        for a, b in zip(estimates[lam, lam], legacy):
            np.testing.assert_array_equal(a, b)
    assert not np.array_equal(estimates[0.95, 0.95][0], estimates[1.0, 0.95][0])
    assert not np.array_equal(estimates[0.95, 0.95][1], estimates[0.95, 1.0][1])


def test_busy_rows_still_contribute_to_critic_targets_and_rollout_boundary_bootstraps(sample):
    _, _, traj, last, _, config = sample
    _, targets = trainer.compute_actor_credit_and_value_targets(traj, last, {**config, "CRITIC_TARGET_LAMBDA": 1.0})
    expected = np.zeros(traj.value.shape, dtype=np.float32)
    running = np.asarray(last)
    for tick in reversed(range(len(traj.reward))):
        running = np.asarray(traj.reward[tick]) + 0.99 * running * (1 - np.asarray(traj.done[tick]))
        expected[tick] = running
    np.testing.assert_allclose(targets, expected, atol=2e-6)
    assert not np.allclose(np.asarray(targets[1]), np.asarray(traj.value[1]))


def test_captured_minibatch_uses_critic_targets_and_only_normalizes_actor(sample):
    _, _, traj, last, key, base = sample
    config = {**base, "ACTOR_GAE_LAMBDA": 0.95, "CRITIC_TARGET_LAMBDA": 1.0}
    _, mini_adv, mini_target, raw, normalized, targets, indices = first_minibatch(trainer, traj, last, key, config)
    actor_adv, _ = trainer.compute_gae(traj, last, gamma=0.99, gae_lambda=0.95)
    _, critic_targets = trainer.compute_gae(traj, last, gamma=0.99, gae_lambda=1.0)
    np.testing.assert_array_equal(raw, actor_adv)
    np.testing.assert_array_equal(targets, critic_targets)
    np.testing.assert_array_equal(normalized, trainer._masked_normalize(actor_adv, traj.actor_mask))
    np.testing.assert_array_equal(mini_adv, normalized.reshape(-1)[indices])
    np.testing.assert_array_equal(mini_target, critic_targets.reshape(-1)[indices])


def test_factorial_analysis_recovers_known_conditional_effects_and_checks_state(tmp_path):
    import json

    import yaml

    from scripts.experiments.analyze_blue_credit_factorial import ARMS, LAMBDAS, analyze

    root = tmp_path / "study"
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    base = {
        "source_steps": 9600000,
        "training_seed": 22001,
        "warm_updates": 20,
        "normalization_calibration_updates": 4,
        "seeds": {"warm_rollout": 10000001},
    }
    eval_cfg = {**base, "confirmation_episodes": 3, "seeds": {"confirmation_start": 11200000, "bootstrap": 11400001}}
    (evaluation / "config.yaml").write_text(yaml.safe_dump(eval_cfg))
    scores = {"a095-c095": 0, "a100-c095": 10, "a095-c100": -3, "a100-c100": 9, "original": 2, "initial": -1}
    episodes = {
        label + suffix: [
            {
                "seed": 11200000 + i,
                "blue_return": score - i,
                "reward_ria": score - i,
                "reward_lwf": 0,
                "reward_asf": 0,
                "action_cost": 0,
                "illegal_actions": 0,
                "block": 0,
            }
            for i in range(3)
        ]
        for label, score in scores.items()
        for suffix in ("", "-no-block")
    }
    (evaluation / "per-episode.json").write_text(json.dumps(episodes))
    for label, directory in ARMS.items():
        p = root / directory
        (p / "captures").mkdir(parents=True)
        (p / "manifest.json").write_text(
            json.dumps(
                {"status": "FINISHED", "run_id": label, "source": {"git_commit": "pinned"}, "actual_steps": 960000}
            )
        )
        (p / "config.yaml").write_text(yaml.safe_dump(base))
        actor, critic = LAMBDAS[label]
        cfg = {
            "GAE_LAMBDA": 0.95,
            "ACTOR_GAE_LAMBDA": actor,
            "CRITIC_TARGET_LAMBDA": critic,
            "NUM_ENVS": 96,
            "NUM_STEPS": 500,
            "GAMMA": 0.99,
        }
        (p / "effective-config.json").write_text(json.dumps({t: cfg for t in ("blue", "red")}))
        np.savez(p / "captures/calibrated-state.npz", leaf0=np.array([0, 1, 2]))
        (p / "captures/calibrated-state.leaves.json").write_text(json.dumps([{"shape": [3]}]))
        (p / "training-metrics.json").write_text(
            json.dumps(
                [{"update": i, "warm_steps": i * 48000, "blue": {"raw_rollout_return": i}} for i in range(1, 21)]
            )
        )
        (p / "gradient-components.json").write_text(json.dumps([{"reference_max_parameter_error": 0}]))
        (p / "learning-signals.json").write_text("[]")
    result = analyze(root, evaluation, tmp_path / "analysis")
    assert result["independent_training_seeds_in_factorial"] == 1
    assert result["factorial_effects"]["actor-main-effect"]["mean"] == 11
    assert result["factorial_effects"]["critic-main-effect"]["mean"] == -2
    assert result["factorial_effects"]["interaction"]["mean"] == 2
    assert result["comparisons"]["a100-c100-minus-original"]["mean"] == 7
    np.savez(root / "a095-c100/captures/calibrated-state.npz", leaf0=np.array([0, 1, 99]))
    with pytest.raises(ValueError, match="starting state"):
        analyze(root, evaluation, tmp_path / "bad")
