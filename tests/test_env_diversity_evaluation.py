from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from jaxborg.evaluation import env_diversity as comparison
from jaxborg.recipe import load


@pytest.fixture
def recipes():
    a, b = load("cotraining_mappo"), load("cotraining_mappo_env_diversity")
    for recipe in (a, b):
        recipe["eval"]["topology_generation"]["count"] = 1
    return a, b


def write_model(root, recipe, seed, *, tag=None, checkpoint_step=None, provenance_override=None):
    tag = tag or f"{recipe['meta']['name']}_seed{seed}-40M"
    directory = root / "mappo_jax" / tag
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"checkpoint_{checkpoint_step}" if checkpoint_step else f"model_{tag}"
    path = directory / f"{stem}.safetensors"
    saved = copy.deepcopy(recipe)
    batch = saved["jax"]["num_envs"] * saved["train"]["episode_length"]
    steps = checkpoint_step or saved["train"]["total_timesteps"] // batch * batch
    saved["run"] = {"seed": seed, "total_steps": steps, "train_run_id": f"run-{tag}", "backend": "jax"}
    provenance = {"recipe": recipe["meta"]["name"], "seed": seed, "total_steps": steps, "train_run_id": f"run-{tag}"}
    provenance.update(provenance_override or {})
    metadata = {
        "backend": "jax",
        "schema_version": 1,
        "provenance": provenance,
        "policies": {team: {"trainable": True} for team in ("blue", "red")},
    }
    save_file({"dummy": np.zeros(1, dtype=np.float32)}, str(path), metadata={"jaxborg_bundle": json.dumps(metadata)})
    sidecar_stem = stem if checkpoint_step else tag
    path.with_name(f"recipe_{sidecar_stem}.yaml").write_text(yaml.safe_dump(saved))
    return path


def populate(root, recipes):
    for recipe in recipes:
        for seed in (42, 200):
            write_model(root, recipe, seed)


def test_plan_matches_common_opponents_and_uses_provenance_not_tag_budget(tmp_path, recipes):
    a, b = recipes
    a["jax"]["num_envs"] = 48
    populate(tmp_path, recipes)
    write_model(tmp_path, b, 100)
    plan = comparison.build_plan(a, b, tmp_path, seeds=(1000,))
    assert plan["train_seeds"] == [42, 200]
    assert len(plan["matchups"]) == 8
    assert {m["steps"] for m in plan["models"]} == {69_984_000}
    assert all("-40M" in m["path"] for m in plan["models"])
    assert any("Excluded diverse seeds [100]" in note for note in plan["notes"])
    assert any("baseline num_envs=48, diverse num_envs=96" in note for note in plan["notes"])
    for seed in (42, 200):
        opponents = {}
        for condition in comparison.CONDITIONS:
            cells = [m for m in plan["matchups"] if m["blue_seed"] == seed and m["blue_condition"] == condition]
            assert all(m["red_seed"] != seed for m in cells)
            opponents[condition] = {m["red_path"] for m in cells}
        assert opponents["baseline"] == opponents["diverse"]
        assert len(opponents["baseline"]) == 2


def test_missing_requested_seed_and_ambiguous_reruns_fail(tmp_path, recipes):
    populate(tmp_path, recipes)
    with pytest.raises(ValueError, match="lack models.*100"):
        comparison.build_plan(*recipes, tmp_path, train_seeds=(42, 100))
    write_model(tmp_path, recipes[0], 42, tag="repeat_seed42")
    with pytest.raises(ValueError, match="Ambiguous baseline seed 42"):
        comparison.build_plan(*recipes, tmp_path)
    plan = comparison.build_plan(*recipes, tmp_path, baseline_tag="cotraining*")
    assert len(plan["matchups"]) == 8


def test_old_final_models_and_different_training_topologies_are_excluded(tmp_path, recipes):
    populate(tmp_path, recipes)
    old = copy.deepcopy(recipes[0])
    old["train"]["total_timesteps"] = 20_000_000
    write_model(tmp_path, old, 42, tag="old_seed42")
    old["train"]["total_timesteps"] = recipes[0]["train"]["total_timesteps"]
    old["train"]["topology_generation"]["count"] = 5
    write_model(tmp_path, old, 42, tag="five_topologies")
    plan = comparison.build_plan(*recipes, tmp_path)
    assert len(plan["models"]) == 4
    assert len([n for n in plan["notes"] if n.startswith("Skipped")]) == 2


@pytest.mark.parametrize("key,value", [("seed", 100), ("total_steps", 20_000_000), ("train_run_id", "other")])
def test_stale_sidecar_or_weights_provenance_is_rejected(tmp_path, recipes, key, value):
    populate(tmp_path, recipes)
    write_model(tmp_path, recipes[0], 42, provenance_override={key: value})
    with pytest.raises(ValueError, match="provenance"):
        comparison.build_plan(*recipes, tmp_path)


def test_exact_periodic_checkpoint_selection_needs_no_final_model(tmp_path, recipes):
    for recipe in recipes:
        for seed in (42, 200):
            write_model(tmp_path, recipe, seed, checkpoint_step=960_000)
    plan = comparison.build_plan(*recipes, tmp_path, checkpoint_step=960_000)
    assert {m["steps"] for m in plan["models"]} == {960_000}
    with pytest.raises(ValueError, match="at least two"):
        comparison.build_plan(*recipes, tmp_path)


def test_eval_mismatch_and_leakage_into_diverse_training_are_rejected(tmp_path, recipes):
    populate(tmp_path, recipes)
    a, b = recipes
    b["eval"]["topology_generation"]["count"] = 2
    with pytest.raises(ValueError, match="share evaluation"):
        comparison.build_plan(a, b, tmp_path)
    # Seed 50 is held out for baseline (train seed 0), but not for diversity
    # (train seeds 0..99). Validate the common eval recipe against both.
    common = copy.deepcopy(a)
    common["eval"]["topology_generation"]["seed_start"] = 50
    with pytest.raises(ValueError, match="overlap"):
        comparison.build_plan(a, b, tmp_path, eval_recipe=common)


def fake_result(plan, matchup, reward):
    count = plan["episodes_per_matchup"]
    return SimpleNamespace(
        blue_returns=[reward] * count,
        red_returns=[-reward] * count,
        episode_seeds=list(plan["seeds"]),
        policies={},
        cia_summary={},
        per_episode_cia=[{"c": reward / 10, "i": reward / 20, "a": 0.0}] * count,
        episode_topology_fingerprints=["same-topology"] * count,
        episode_role_map_ids=["same-roles"] * count,
        episode_topology_paths=["same-path"] * count,
        topology_role_maps=[],
    )


def test_execute_paired_summary_and_resume_without_replaying(tmp_path, recipes, monkeypatch):
    populate(tmp_path, recipes)
    plan = comparison.build_plan(*recipes, tmp_path, seeds=(1000, 1001))
    output = tmp_path / "comparison"
    original_project = comparison.project_eval
    monkeypatch.setattr(comparison, "project_eval", lambda r, **kwargs: original_project(r))
    calls = []

    def evaluate(blue, red, **kwargs):
        calls.append((blue, red, kwargs))
        matchup = next(m for m in plan["matchups"] if m["blue_path"] == blue and m["red_path"] == red)
        # Seed-level means are -100/-200 baseline and -50/-150 diverse.
        reward = -100.0 if matchup["blue_seed"] == 42 else -200.0
        if matchup["blue_condition"] == "diverse":
            reward += 50
        return fake_result(plan, matchup, reward)

    summary = comparison.run_comparison(plan, output, evaluate_fn=evaluate)
    assert len(calls) == 8
    assert all(call[2]["cia"]["enabled"] and call[2]["topology_sampling"] == "exhaustive" for call in calls)
    reward = summary["overall"]["metrics"]["reward"]
    assert reward["baseline"] == -150
    assert reward["diverse"] == -100
    assert reward["delta"] == 50
    assert reward["penalty_reduction_percent"] == pytest.approx(100 / 3)
    assert summary["overall"]["metrics"]["a"]["penalty_reduction_percent"] is None
    assert set(summary["by_red_condition"]) == {"baseline", "diverse"}
    assert (output / "summary.md").exists()
    assert json.loads((output / "manifest.json").read_text())["models"][0]["sha256"]
    comparison.run_comparison(plan, output, resume=True, evaluate_fn=evaluate)
    assert len(calls) == 8
    changed = copy.deepcopy(plan)
    changed["models"][0]["sha256"] = "changed-checkpoint"
    with pytest.raises(ValueError, match="identical manifest"):
        comparison.run_comparison(changed, output, resume=True, evaluate_fn=evaluate)
    with pytest.raises(FileExistsError):
        comparison.run_comparison(plan, output, evaluate_fn=evaluate)


def test_summary_rejects_unpaired_case_provenance_and_incomplete_matrix(tmp_path, recipes):
    populate(tmp_path, recipes)
    plan = comparison.build_plan(*recipes, tmp_path, seeds=(1000,))
    rows = [{**m, **vars(fake_result(plan, m, -100))} for m in plan["matchups"]]
    with pytest.raises(ValueError, match="every planned matchup"):
        comparison.summarize_comparison(plan, rows[:-1])
    rows[1]["episode_role_map_ids"] = ["different-roles"]
    with pytest.raises(ValueError, match="Unpaired evaluation cases"):
        comparison.summarize_comparison(plan, rows)


def test_cli_dry_run_does_not_roll_out_or_write_results(tmp_path, recipes, monkeypatch, capsys):
    populate(tmp_path, recipes)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_env_diversity_cli", root / "scripts/eval/eval_env_diversity.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    loaded = {r["meta"]["name"]: r for r in recipes}
    monkeypatch.setattr(cli, "load", loaded.__getitem__)
    monkeypatch.setattr(cli, "run_comparison", lambda *_a, **_k: pytest.fail("dry-run started evaluation"))
    output = tmp_path / "dry_results"
    cli.main([*loaded, "--exp-dir", str(tmp_path), "--output-dir", str(output), "--dry-run"])
    assert not output.exists()
    assert "8 matchups x 10 episodes" in capsys.readouterr().out
