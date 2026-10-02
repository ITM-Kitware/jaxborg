"""Controls for a new training seed and a same-cohort original defender comparison."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from jaxborg.actions import action_defs as action
from jaxborg.blue_learning_probe import traffic_groups
from scripts.experiments.analyze_blue_credit_replication import audit_pair
from scripts.experiments.blue_learning_mechanism import archived_baseline, confirmation_comparisons, load_reference


@pytest.mark.parametrize("source,seed", [(9600000, 22001), (49968000, 33001)])
def test_archived_backend_check_uses_actual_source_and_training_seed(tmp_path, source, seed):
    path = tmp_path / f"eval/blue_diagnosis/ippo-{source}/evaluations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "confirmation": {
                    "seed-11001-step-1920000": {"per_episode_blue_returns": [-1]},
                    f"seed-{seed}-step-1920000": {"per_episode_blue_returns": [-2]},
                }
            }
        )
    )
    assert archived_baseline({"source_steps": source, "training_seed": seed}, tmp_path)["per_episode_blue_returns"] == [
        -2
    ]


def confirmation():
    rows = {}
    for label, offset in (("original", 0), ("warm-0", -10), ("lambda095-final", -30), ("warm-final", 30)):
        for suffix in ("", "-no-block"):
            rows[label + suffix] = [{"seed": i, "blue_return": -100 - i + offset} for i in (1, 2, 3)]
    return rows


def test_both_final_policies_and_starting_policy_compare_with_same_original_episodes():
    estimates = confirmation_comparisons(confirmation(), seed=10400001)
    for suffix in ("", "-no-block"):
        for label, expected in (
            ("initial-minus-original", -10),
            ("warm-final-minus-original", 30),
            ("lambda095-minus-original", -30),
            ("lambda1-minus-lambda095", 60),
            ("warm-final-minus-initial", 40),
        ):
            assert estimates[label + suffix] == {"mean": expected, "ci95": [expected, expected], "n": 3}


def test_confirmation_rejects_differently_ordered_original_seed_cohort():
    rows = confirmation()
    rows["original"] = list(reversed(rows["original"]))
    with pytest.raises(ValueError, match="unpaired"):
        confirmation_comparisons(rows)


def test_reference_updater_rejects_a_different_execution_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.experiments.blue_learning_mechanism.subprocess.check_output", lambda *_: b"different-revision\n"
    )
    with pytest.raises(ValueError, match="original fresh-Blue execution"):
        load_reference(tmp_path / "scripts/train/algorithms/ippo_jax_joint.py")


def test_open_mission_block_and_reopening_allow_require_the_reverse_direction_open():
    shape = (1, 1, 4, action.BLUE_TRAFFIC_SLOTS)
    observed = {
        "permitted": jnp.ones(shape, dtype=bool),
        "blocked": jnp.zeros(shape, dtype=bool).at[:, :, 2:].set(True),
        "reverse_blocked": jnp.zeros(shape, dtype=bool).at[:, :, (1, 3)].set(True),
    }
    traj = SimpleNamespace(
        actor_mask=jnp.ones((1, 1, 4)),
        action=jnp.array([[[action.BLUE_BLOCK_TRAFFIC_START] * 2 + [action.BLUE_ALLOW_TRAFFIC_START] * 2]]),
    )
    groups = traffic_groups(traj, observed)
    np.testing.assert_array_equal(groups["new_open_mission_block"], [[[True, False, False, False]]])
    np.testing.assert_array_equal(groups["mission_route_reopening_allow"], [[[False, False, True, False]]])
    # The legacy classification includes a newly set own bit even if the
    # reverse bit already blocked Green; never silently relabel those data.
    np.testing.assert_array_equal(groups["harmful_new_block"], [[[True, True, False, False]]])
    observed.pop("reverse_blocked")
    assert "new_open_mission_block" not in traffic_groups(traj, observed)


def matched_pair(tmp_path):
    control, variant = tmp_path / "control", tmp_path / "variant"
    protocol = {
        "source_steps": 9600000,
        "training_seed": 22001,
        "warm_updates": 20,
        "validation_episodes": 32,
        "confirmation_episodes": 3,
        "include_original_confirmation": True,
        "seeds": {"confirmation_start": 1},
    }
    for directory, lam in ((control, 0.95), (variant, 1.0)):
        (directory / "captures").mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"status": "FINISHED"}))
        (directory / "config.yaml").write_text(yaml.safe_dump(protocol))
        (directory / "effective-config.json").write_text(
            json.dumps(
                {t: {"GAE_LAMBDA": lam, "NUM_ENVS": 96, "NUM_STEPS": 500, "GAMMA": 0.99} for t in ("blue", "red")}
            )
        )
        np.savez(directory / "captures/calibrated-state.npz", leaf0=np.array([0, 1, 2]))
        (directory / "captures/calibrated-state.leaves.json").write_text(json.dumps([{"shape": [3]}]))
        (directory / "training-metrics.json").write_text(
            json.dumps([{"update": i, "warm_steps": i * 48000} for i in range(1, 21)])
        )
        episodes = confirmation()
        for rows in episodes.values():
            for row in rows:
                row.update(
                    reward_ria=row["blue_return"], reward_lwf=0, reward_asf=0, action_cost=0, illegal_actions=0, block=0
                )
        if directory == control:
            for suffix in ("", "-no-block"):
                episodes["warm-final" + suffix] = episodes.pop("lambda095-final" + suffix)
        (directory / "confirmation-episodes.json").write_text(json.dumps(episodes))
    return control, variant


def test_pair_audit_checks_numeric_start_and_original_on_registered_cohort(tmp_path):
    control, variant = matched_pair(tmp_path)
    result = audit_pair(control, variant)
    assert result["independent_training_seeds_in_this_pair"] == 1
    assert result["both_arms_confirmed_on_same_cohort"]
    assert result["comparisons"]["lambda1-minus-original"]["mean"] == 30
    assert result["supports_lambda_within_seed"]
    np.savez(variant / "captures/calibrated-state.npz", leaf0=np.array([0, 1, 99]))
    with pytest.raises(ValueError, match="starting state"):
        audit_pair(control, variant)


def test_pair_audit_rejects_a_different_reserved_cohort_and_missing_original(tmp_path):
    control, variant = matched_pair(tmp_path)
    p = variant / "confirmation-episodes.json"
    episodes = json.loads(p.read_text())
    episodes["original"][0]["seed"] = 999
    p.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match="registered cohort"):
        audit_pair(control, variant)
    for directory in (control, variant):
        p = directory / "confirmation-episodes.json"
        episodes = json.loads(p.read_text())
        for suffix in ("", "-no-block"):
            episodes.pop("original" + suffix)
        p.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match="defender evaluation is missing"):
        audit_pair(control, variant)
