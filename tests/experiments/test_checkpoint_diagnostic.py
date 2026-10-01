"""Candidate and seed boundaries for saved-checkpoint diagnosis."""

import pytest
import yaml

from jaxborg.checkpoint_diagnostic import candidates, confirmation_candidates, load_config, select_checkpoint


def config(tmp_path, **changes):
    value = {
        "schema_version": 1,
        "resources": {"partition": "community", "gpus_per_job": 1},
        "checkpoint_steps": [1920000, 4800000, "final"],
        "episodes": {"validation": {"seed_start": 100, "count": 10}, "confirmation": {"seed_start": 200, "count": 20}},
    }
    value.update(changes)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def test_separate_seeds(tmp_path):
    assert load_config(config(tmp_path))["episodes"]["confirmation"]["count"] == 20
    with pytest.raises(ValueError, match="overlap"):
        load_config(
            config(
                tmp_path,
                episodes={
                    "validation": {"seed_start": 100, "count": 10},
                    "confirmation": {"seed_start": 105, "count": 20},
                },
            )
        )


@pytest.mark.parametrize("partition,gpus", [("priority", 1), ("community", 2)])
def test_resource_bounds(tmp_path, partition, gpus):
    with pytest.raises(ValueError, match="community"):
        load_config(config(tmp_path, resources={"partition": partition, "gpus_per_job": gpus}))


def test_candidates_confirmation_and_ties():
    protocol = {
        "trainable_team": "blue",
        "source": {"checkpoint": "runs:/source/model", "original_training_steps": 9600000},
        "training_seeds": [11, 22],
        "oracle_budget_per_attempt": {"completed_steps": 9984000},
    }
    state = {
        "training": {
            f"seed-{seed}": {
                "run_id": f"run-{seed}",
                "actual_steps": 9984000,
                "final_checkpoint": f"runs:/run-{seed}/final",
            }
            for seed in (11, 22)
        }
    }
    pool = candidates(protocol, state, [1920000, 4800000, "final"])
    assert len(pool) == 7
    assert pool[1]["checkpoint"] == "runs:/run-11/checkpoints/checkpoint_1920000.safetensors"
    assert pool[3]["checkpoint"] == "runs:/run-11/final"
    order = [c["name"] for c in pool]
    assert select_checkpoint(dict.fromkeys(order, -100), order) == "original"
    scores = dict.fromkeys(order, -100)
    scores["seed-22-step-4800000"] = -50
    selected = select_checkpoint(scores, order)
    confirmation = confirmation_candidates(pool, selected)
    assert confirmation == [
        "original",
        "seed-11-step-1920000",
        "seed-11-step-final",
        "seed-22-step-1920000",
        "seed-22-step-final",
        selected,
    ]
    with pytest.raises(ValueError, match="every"):
        select_checkpoint({"original": -100}, order)


def test_reject_incomplete_attempt():
    protocol = {
        "trainable_team": "blue",
        "source": {"checkpoint": "source", "original_training_steps": 9600000},
        "training_seeds": [11],
        "oracle_budget_per_attempt": {"completed_steps": 9984000},
    }
    with pytest.raises(ValueError, match="missing"):
        candidates(protocol, {"training": {}}, [1920000, 4800000, "final"])
