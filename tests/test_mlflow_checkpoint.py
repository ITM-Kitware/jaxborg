from pathlib import Path

import pytest

from jaxborg.mlflow_setup import (
    CheckpointEvalSettings,
    MlflowCheckpointEvaluator,
    checkpoint_eval_due,
)


def _recipe(*, teams: str = "both", **checkpoint_eval):
    settings = {
        "every_steps": 100,
        "episodes_per_seed": 7,
        "seed": 123,
        "deterministic": True,
    }
    settings.update(checkpoint_eval)
    return {
        "meta": {"name": "checkpoint-eval-test"},
        "train": {"teams": teams},
        "mlflow": {"checkpoint_eval": settings},
    }


class FakeMlflow:
    def __init__(self):
        self.artifacts = []
        self.metrics = []

    def log_artifact(self, path, *, artifact_path):
        self.artifacts.append((path, artifact_path))

    def log_metrics(self, metrics, *, step):
        self.metrics.append((metrics, step))


def test_checkpoint_eval_settings_default_to_disabled():
    settings = CheckpointEvalSettings.from_recipe({"meta": {"name": "defaults"}, "train": {"teams": "blue"}})

    assert settings.every_steps == 0
    assert settings.episodes_per_seed == 10
    assert settings.seed is None
    assert settings.deterministic is False


def test_checkpoint_eval_settings_read_nested_mlflow_config():
    settings = CheckpointEvalSettings.from_recipe(_recipe())

    assert settings.every_steps == 100
    assert settings.episodes_per_seed == 7
    assert settings.seed == 123
    assert settings.deterministic is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"every_steps": True}, "every_steps"),
        ({"every_steps": -1}, "every_steps"),
        ({"episodes_per_seed": True}, "episodes_per_seed"),
        ({"episodes_per_seed": 0}, "episodes_per_seed"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"deterministic": 1}, "deterministic"),
    ],
)
def test_checkpoint_eval_settings_validate_values(override, message):
    with pytest.raises(ValueError, match=message):
        CheckpointEvalSettings.from_recipe(_recipe(**override))


@pytest.mark.parametrize(
    "recipe",
    [
        {"train": {"teams": "blue"}, "mlflow": []},
        {"train": {"teams": "blue"}, "mlflow": {"checkpoint_eval": []}},
    ],
)
def test_checkpoint_eval_settings_reject_non_mapping_sections(recipe):
    with pytest.raises(ValueError, match="must be a mapping"):
        CheckpointEvalSettings.from_recipe(recipe)


def test_checkpoint_eval_settings_accept_legacy_episodes_key():
    recipe = _recipe()
    recipe["mlflow"]["checkpoint_eval"].pop("episodes_per_seed")
    recipe["mlflow"]["checkpoint_eval"]["episodes"] = 4

    settings = CheckpointEvalSettings.from_recipe(recipe)

    assert settings.episodes_per_seed == 4
    assert settings.episodes == 4


def test_checkpoint_eval_settings_reject_both_episode_count_names():
    recipe = _recipe()
    recipe["mlflow"]["checkpoint_eval"]["episodes"] = 4

    with pytest.raises(ValueError, match="not both"):
        CheckpointEvalSettings.from_recipe(recipe)


def test_checkpoint_eval_due_when_step_boundary_is_crossed():
    assert not checkpoint_eval_due(0, 99, 100)
    assert checkpoint_eval_due(99, 100, 100)
    assert not checkpoint_eval_due(100, 199, 100)
    assert checkpoint_eval_due(199, 250, 100)
    assert checkpoint_eval_due(0, 350, 100)


def test_checkpoint_eval_due_on_final_update_without_crossing_boundary():
    assert checkpoint_eval_due(100, 150, 100, final=True)
    assert not checkpoint_eval_due(100, 150, 0, final=True)
    assert not checkpoint_eval_due(0, 100, 0)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 100),
        (100, 99, 100),
        (0, 100, -1),
        (True, 100, 100),
        (0, 100, True),
    ],
)
def test_checkpoint_eval_due_validates_inputs(args):
    with pytest.raises(ValueError):
        checkpoint_eval_due(*args)


def test_disabled_evaluator_is_a_noop():
    fake_mlflow = FakeMlflow()
    evaluator = MlflowCheckpointEvaluator(
        {"meta": {"name": "disabled"}, "train": {"teams": "blue"}},
        mlflow_module=fake_mlflow,
    )

    assert not evaluator.enabled
    assert not evaluator.due(0, 100, final=True)
    assert (
        evaluator.on_checkpoint(
            "unused.safetensors",
            "unused.yaml",
            env_steps=100,
            evaluate_fn=lambda _: pytest.fail("disabled evaluator ran evaluation"),
        )
        == {}
    )
    assert fake_mlflow.artifacts == []
    assert fake_mlflow.metrics == []


def test_evaluator_due_uses_configured_step_interval():
    evaluator = MlflowCheckpointEvaluator(_recipe(every_steps=250), mlflow_module=FakeMlflow())

    assert evaluator.enabled
    assert not evaluator.due(0, 249)
    assert evaluator.due(249, 250)
    assert evaluator.due(250, 300, final=True)


def test_checkpoint_is_artifacted_evaluated_and_logged_for_both_teams(tmp_path):
    fake_mlflow = FakeMlflow()
    evaluator = MlflowCheckpointEvaluator(_recipe(), mlflow_module=fake_mlflow)
    checkpoint = tmp_path / "checkpoint_120.safetensors"
    sidecar = tmp_path / "recipe_checkpoint_120.yaml"
    checkpoint.write_bytes(b"model")
    sidecar.write_text("meta:\n  name: test\n")
    calls = []

    def evaluate(episodes):
        calls.append(episodes)
        return {"blue": -20.0, "red": 20}

    means = evaluator.on_checkpoint(
        checkpoint,
        sidecar,
        env_steps=120,
        evaluate_fn=evaluate,
    )

    assert calls == [7]
    assert means == {"blue": -20.0, "red": 20.0}
    assert fake_mlflow.artifacts == [
        (str(checkpoint), "checkpoints/step-120"),
        (str(sidecar), "checkpoints/step-120"),
    ]
    assert fake_mlflow.metrics == [
        (
            {
                "eval.checkpoint.blue.mean_reward": -20.0,
                "eval.checkpoint.red.mean_reward": 20.0,
            },
            120,
        )
    ]


@pytest.mark.parametrize(
    ("teams", "expected"),
    [
        ("blue", {"blue": -4.0}),
        ("red", {"red": 4.0}),
        ("both", {"blue": -4.0, "red": 4.0}),
    ],
)
def test_checkpoint_metrics_include_only_trained_teams(tmp_path, teams, expected):
    fake_mlflow = FakeMlflow()
    evaluator = MlflowCheckpointEvaluator(_recipe(teams=teams), mlflow_module=fake_mlflow)
    checkpoint = tmp_path / f"{teams}.safetensors"
    sidecar = tmp_path / f"{teams}.yaml"
    checkpoint.touch()
    sidecar.touch()

    means = evaluator.on_checkpoint(
        checkpoint,
        sidecar,
        env_steps=200,
        evaluate_fn=lambda _: {"blue": -4.0, "red": 4.0},
    )

    assert means == expected
    expected_metrics = {f"eval.checkpoint.{team}.mean_reward": reward for team, reward in expected.items()}
    assert fake_mlflow.metrics == [(expected_metrics, 200)]


def test_checkpoint_paths_accept_pathlike_values(tmp_path):
    fake_mlflow = FakeMlflow()
    evaluator = MlflowCheckpointEvaluator(_recipe(teams="blue"), mlflow_module=fake_mlflow)
    checkpoint: Path = tmp_path / "checkpoint.safetensors"
    sidecar: Path = tmp_path / "recipe.yaml"

    evaluator.on_checkpoint(
        checkpoint,
        sidecar,
        env_steps=1,
        evaluate_fn=lambda _: {"blue": 1.0},
    )

    assert fake_mlflow.artifacts == [
        (str(checkpoint), "checkpoints/step-1"),
        (str(sidecar), "checkpoints/step-1"),
    ]
