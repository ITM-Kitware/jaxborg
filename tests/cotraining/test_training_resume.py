import copy
import json

import jax.numpy as jnp
import pytest

from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle, write_sidecar
from jaxborg.training_resume import preserve_resume_metrics, select_joint_resume


@pytest.fixture
def recipe():
    return {
        "meta": {"name": "resume_test"},
        "algorithm": "ippo",
        "arch": {"name": "shared"},
        "train": {"teams": "both", "total_timesteps": 100},
        "jax": {"num_envs": 2},
    }


def export(path, recipe, steps, seed=42):
    policies = {
        team: PolicyBundleEntry(
            weights={"weight": jnp.array([steps], dtype=jnp.float32)},
            team=team,
            obs_dim=3,
            action_dim=2,
            arch=recipe["arch"],
        )
        for team in ("blue", "red")
    }
    save_jax_bundle(path, policies, provenance={"recipe": "resume_test", "seed": seed, "total_steps": steps})
    stem = path.stem.removeprefix("model_")
    write_sidecar(path.with_name(f"recipe_{stem}.yaml"), recipe, seed=seed, total_steps=steps, backend="jax")


def test_latest_intact_checkpoint_skips_truncated_write_and_missing_sidecar(tmp_path, recipe):
    export(tmp_path / "checkpoint_20.safetensors", recipe, 20)
    export(tmp_path / "checkpoint_80.safetensors", recipe, 80)
    (tmp_path / "recipe_checkpoint_80.yaml").unlink()
    (tmp_path / "checkpoint_100.safetensors").write_bytes(b"truncated")
    path, bundle, update = select_joint_resume(tmp_path, "test", recipe, 42, 10, 10)
    assert path.name == "checkpoint_20.safetensors"
    assert update == 2
    assert set(bundle.policies) == {"blue", "red"}
    assert float(bundle.policies["red"].weights["weight"][0]) == 20


def test_valid_final_model_takes_precedence_and_eval_changes_are_allowed(tmp_path, recipe):
    export(tmp_path / "checkpoint_80.safetensors", recipe, 80)
    export(tmp_path / "model_test.safetensors", recipe, 100)
    recipe["eval"] = {"after_training": []}
    path, _, update = select_joint_resume(tmp_path, "test", recipe, 42, 10, 10)
    assert path.name == "model_test.safetensors"
    assert update == 10


@pytest.mark.parametrize("change", ["seed", "recipe", "steps", "sidecar"])
def test_incompatible_resume_fails_without_falling_back(tmp_path, recipe, change):
    export(tmp_path / "checkpoint_20.safetensors", recipe, 20)
    saved = copy.deepcopy(recipe)
    if change == "recipe":
        saved["jax"]["num_envs"] = 4
    export(
        tmp_path / "checkpoint_80.safetensors",
        saved,
        81 if change == "steps" else 80,
        seed=100 if change == "seed" else 42,
    )
    if change == "sidecar":
        write_sidecar(tmp_path / "recipe_checkpoint_80.yaml", recipe, seed=42, total_steps=70, backend="jax")
    with pytest.raises(ValueError):
        select_joint_resume(tmp_path, "test", recipe, 42, 10, 10)


def test_no_progress_starts_fresh_but_damaged_progress_does_not(tmp_path, recipe):
    assert select_joint_resume(tmp_path, "test", recipe, 42, 10, 10) is None
    # MAPPO crashed during its first compilation, after opening an empty log.
    (tmp_path / "metrics.jsonl").touch()
    assert select_joint_resume(tmp_path, "test", recipe, 42, 10, 10) is None
    (tmp_path / "checkpoint_20.safetensors").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="No intact checkpoint"):
        select_joint_resume(tmp_path, "test", recipe, 42, 10, 10)
    (tmp_path / "checkpoint_20.safetensors").unlink()
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="metrics exist"):
        select_joint_resume(tmp_path, "test", recipe, 42, 10, 10)


def test_resume_metrics_archives_original_and_removes_uncheckpointed_progress(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    original = "\n".join(json.dumps({"env_steps": step}) for step in [10, 20, 30]) + '\n{"env_steps":'
    metrics.write_text(original)
    preserve_resume_metrics(metrics, 20)
    assert [json.loads(line)["env_steps"] for line in metrics.read_text().splitlines()] == [10, 20]
    assert next(tmp_path.glob("metrics_before_resume_*.jsonl")).read_text() == original
