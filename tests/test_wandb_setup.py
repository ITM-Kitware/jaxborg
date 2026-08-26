from pathlib import Path

import pytest

from jaxborg.wandb_setup import WandbCallback, WandbSettings, milestones_for_update


def _recipe(**overrides):
    recipe = {
        "meta": {"name": "metadata-name"},
        "train": {"teams": "both"},
        "wandb": True,
        "__source_path__": "/configs/blue-vs-red.yaml",
    }
    recipe.update(overrides)
    return recipe


class FakeArtifact:
    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeRun:
    id = "run-123"

    def __init__(self):
        self.defined_metrics = []
        self.logged = []
        self.artifacts = []
        self.finish_calls = 0

    def define_metric(self, name, **kwargs):
        self.defined_metrics.append((name, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def log_artifact(self, artifact, *, aliases):
        self.artifacts.append((artifact, aliases))

    def finish(self):
        self.finish_calls += 1


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.init_kwargs = None
        self.created_artifacts = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name, *, type, metadata):
        artifact = FakeArtifact(name, type=type, metadata=metadata)
        self.created_artifacts.append(artifact)
        return artifact


def test_settings_default_to_source_filename_then_meta_name():
    settings = WandbSettings.from_recipe(
        {
            "meta": {"name": "metadata-name"},
            "train": {},
            "__source_path__": "/tmp/configs/experiment.v2.yaml",
        }
    )
    assert settings == WandbSettings(
        enabled=False,
        project="jaxborg",
        run_name="experiment.v2",
        eval_episodes=10,
    )

    without_source = WandbSettings.from_recipe({"meta": {"name": "from-meta"}, "train": {}})
    assert without_source.run_name == "from-meta"


def test_settings_honor_explicit_overrides():
    settings = WandbSettings.from_recipe(
        _recipe(
            wandb_project="cyber-project",
            wandb_run_name="custom-run",
            wandb_eval_episodes=7,
        )
    )
    assert settings.enabled
    assert settings.project == "cyber-project"
    assert settings.run_name == "custom-run"
    assert settings.eval_episodes == 7


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"wandb": 1}, "wandb must be a boolean"),
        ({"wandb_project": ""}, "wandb_project"),
        ({"wandb_run_name": " "}, "wandb_run_name"),
        ({"wandb_eval_episodes": True}, "must be an integer"),
        ({"wandb_eval_episodes": 0}, "must be positive"),
    ],
)
def test_settings_validate_recipe_values(override, message):
    with pytest.raises(ValueError, match=message):
        WandbSettings.from_recipe(_recipe(**override))


def test_milestones_use_integer_threshold_crossings():
    due_by_update = [milestones_for_update(update, 20) for update in range(1, 21)]
    assert due_by_update == [[percent] for percent in range(5, 101, 5)]


def test_short_run_coalesces_all_due_aliases_on_each_update():
    due_by_update = [milestones_for_update(update, 3) for update in range(1, 4)]
    assert due_by_update == [
        [5, 10, 15, 20, 25, 30],
        [35, 40, 45, 50, 55, 60, 65],
        [70, 75, 80, 85, 90, 95, 100],
    ]
    assert [percent for due in due_by_update for percent in due] == list(range(5, 101, 5))


def test_milestones_always_include_terminal_100_percent():
    assert milestones_for_update(3, 3, interval_percent=30) == [90, 100]
    assert milestones_for_update(0, 3) == []


@pytest.mark.parametrize(
    "args",
    [
        (1, 0, 5),
        (-1, 10, 5),
        (11, 10, 5),
        (1, 10, 0),
        (1, 10, 101),
        (True, 10, 5),
    ],
)
def test_milestones_validate_inputs(args):
    with pytest.raises(ValueError):
        milestones_for_update(*args)


def test_disabled_callback_never_imports_or_calls_wandb(monkeypatch):
    def reject_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("jaxborg.wandb_setup.importlib.import_module", reject_import)
    callback = WandbCallback(
        {"meta": {"name": "disabled"}, "train": {"teams": "blue"}},
        backend="jax",
        seed=42,
    )
    callback.log_training({})
    result = callback.on_checkpoint(
        "unused.safetensors",
        "unused.yaml",
        100,
        [5],
        lambda _: pytest.fail("disabled callback evaluated a checkpoint"),
    )
    callback.finish()

    assert not callback.enabled
    assert result == {}


def test_disabled_callback_schedules_no_milestone_checkpoints():
    callback = WandbCallback(
        {"meta": {"name": "disabled"}, "train": {"teams": "blue"}},
        backend="jax",
        seed=42,
    )

    assert callback.milestones(1, 20) == []


def test_enabled_callback_initializes_run_and_custom_axes():
    fake_wandb = FakeWandb()
    callback = WandbCallback(_recipe(), backend="jax", seed=17, wandb_module=fake_wandb)

    assert callback.enabled
    assert fake_wandb.init_kwargs["project"] == "jaxborg"
    assert fake_wandb.init_kwargs["name"] == "blue-vs-red"
    assert fake_wandb.init_kwargs["config"]["backend"] == "jax"
    assert fake_wandb.init_kwargs["config"]["seed"] == 17
    assert "__source_path__" not in fake_wandb.init_kwargs["config"]
    assert fake_wandb.run.defined_metrics == [
        ("global_step", {}),
        ("train/*", {"step_metric": "global_step"}),
        ("eval/*", {"step_metric": "global_step"}),
    ]
    assert callback.milestones(1, 20) == [5]


def test_training_log_includes_numeric_metrics_and_joint_reward_aliases():
    fake_wandb = FakeWandb()
    callback = WandbCallback(_recipe(), backend="jax", seed=1, wandb_module=fake_wandb)
    callback.log_training(
        {
            "env_steps": 24000,
            "update_idx": 1,
            "loss_policy": 0.25,
            "team.blue.return": -12.5,
            "team.red.return": 12.5,
            "team.blue.trainable": True,
            "label": "ignored",
        }
    )

    payload = fake_wandb.run.logged[-1]
    assert payload["global_step"] == 24000
    assert payload["train/loss_policy"] == pytest.approx(0.25)
    assert payload["train/blue_reward"] == pytest.approx(-12.5)
    assert payload["train/red_reward"] == pytest.approx(12.5)
    assert "train/team.blue.trainable" not in payload
    assert "train/label" not in payload


def test_legacy_training_reward_falls_back_to_top_level_for_trainable_team():
    fake_wandb = FakeWandb()
    recipe = _recipe(train={"teams": "blue"})
    callback = WandbCallback(recipe, backend="jax", seed=1, wandb_module=fake_wandb)
    callback.log_training(
        {
            "env_steps": 100,
            "train_episode_reward_mean": -4.0,
            "team.red.return": 4.0,
        }
    )

    payload = fake_wandb.run.logged[-1]
    assert payload["train/blue_reward"] == pytest.approx(-4.0)
    assert "train/red_reward" not in payload


def test_checkpoint_evaluates_logs_and_versions_artifact(tmp_path):
    fake_wandb = FakeWandb()
    callback = WandbCallback(
        _recipe(wandb_eval_episodes=7),
        backend="jax",
        seed=1,
        wandb_module=fake_wandb,
    )
    checkpoint = tmp_path / "checkpoint_120.safetensors"
    sidecar = tmp_path / "recipe_checkpoint_120.yaml"
    checkpoint.write_bytes(b"model")
    sidecar.write_text("meta:\n  name: test\n")
    calls = []

    def evaluate(episodes):
        calls.append(episodes)
        return {"blue": -20.0, "red": 20.0}

    means = callback.on_checkpoint(checkpoint, sidecar, 120, [10, 5, 10], evaluate)

    assert calls == [7]
    assert means == {"blue": -20.0, "red": 20.0}
    assert fake_wandb.run.logged[-1] == {
        "global_step": 120,
        "eval/blue_reward": -20.0,
        "eval/red_reward": 20.0,
    }
    artifact, aliases = fake_wandb.run.artifacts[-1]
    assert artifact.name == "checkpoint-run-123"
    assert artifact.type == "model"
    assert artifact.files == [str(checkpoint), str(sidecar)]
    assert artifact.metadata == {
        "global_step": 120,
        "training_percent": 10,
        "training_milestones": [5, 10],
        "eval/blue_reward": -20.0,
        "eval/red_reward": 20.0,
    }
    assert aliases == ["percent-05", "percent-10", "step-120", "latest"]


def test_empty_checkpoint_milestones_skip_evaluation_and_artifact():
    fake_wandb = FakeWandb()
    callback = WandbCallback(_recipe(), backend="jax", seed=1, wandb_module=fake_wandb)

    means = callback.on_checkpoint("unused", "unused", 120, [], lambda _: pytest.fail("unexpected evaluation"))

    assert means == {}
    assert fake_wandb.run.artifacts == []


def test_finish_is_idempotent():
    fake_wandb = FakeWandb()
    callback = WandbCallback(_recipe(), backend="jax", seed=1, wandb_module=fake_wandb)

    callback.finish()
    callback.finish()

    assert fake_wandb.run.finish_calls == 1


def test_checkpoint_paths_accept_pathlike_values(tmp_path):
    fake_wandb = FakeWandb()
    callback = WandbCallback(_recipe(), backend="jax", seed=1, wandb_module=fake_wandb)
    checkpoint: Path = tmp_path / "checkpoint.safetensors"
    sidecar: Path = tmp_path / "recipe.yaml"
    callback.on_checkpoint(checkpoint, sidecar, 1, [100], lambda _: {"blue": 1.0})

    assert fake_wandb.created_artifacts[0].files == [str(checkpoint), str(sidecar)]
