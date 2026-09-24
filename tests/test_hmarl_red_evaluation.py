"""Exercise the H-MARL Red CLI through real case iteration and JSONL reporting."""

import json
from pathlib import Path

import pytest

from jaxborg.evaluation import hmarl_reds, jax_scripted_red
from jaxborg.evaluation.cia.fixed_topology import EvaluationCase
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
from jaxborg.evaluation.post_training import PostTrainingEvalSettings
from jaxborg.evaluation.scripted_red import ScriptedRedEvalSettings
from jaxborg.pretrained import hmarl
from jaxborg.recipe import load
from jaxborg.scenarios.cc4.hmarl_reds import HMARL_REDS


@pytest.mark.parametrize(
    "recipe_name", ["hmarl_expert", "hmarl_meta", "cotraining_lstm", "cotraining_lstm_env_diversity"]
)
def test_cli_reuses_cases_and_loads_correct_blue_policy(monkeypatch, tmp_path, recipe_name):
    import jaxborg.recipe as recipes

    monkeypatch.delenv("JAXBORG_EVAL_BATCH_SIZE", raising=False)
    monkeypatch.delenv("JAXBORG_EVAL_NAME", raising=False)
    model = tmp_path / "model.safetensors"
    model.touch()
    topology = tmp_path / "held_out.snapshot.npz"
    topology.touch()
    output = tmp_path / "hmarl_reds.jsonl"
    policy = LoadedMatchupPolicy("blue", "jax", object(), object(), {"bundle_trainable": True})
    loaded = []

    def load_hmarl(path, *, variant):
        loaded.append((path, variant))
        return policy

    def load_trained(path, *, team, backend):
        loaded.append((path, team, backend))
        return policy

    monkeypatch.setattr(hmarl, "load_policy", load_hmarl)
    monkeypatch.setattr(jax_scripted_red, "load_matchup_policy", load_trained)
    original_project = recipes.project_eval

    def project(recipe, **kwargs):
        return {**original_project(recipe), "TOPOLOGY_BANK": (topology,)}

    monkeypatch.setattr(recipes, "project_eval", project)
    case_calls = []

    def cases(paths, seeds, episodes):
        case_calls.append((paths, seeds, episodes))
        return [EvaluationCase(0, topology, "fp", 1000, 0, 1000, (0, 1, 2, 3), "roles")]

    monkeypatch.setattr(jax_scripted_red, "build_evaluation_cases", cases)
    env_calls = []

    def env_factory(variant, **kwargs):
        env_calls.append((variant, kwargs))
        return object()

    monkeypatch.setattr(jax_scripted_red, "make_jax_env", env_factory)
    rollout_calls = []

    def episode(loaded_policy, *, env, variant, case, deterministic):
        assert loaded_policy is policy
        rollout_calls.append((variant.red_agent, case.episode_seed, deterministic))
        return jax_scripted_red.JaxScriptedRedEpisode(-12.0, (-1.0, -2.0, -3.0))

    monkeypatch.setattr(jax_scripted_red, "run_jax_scripted_red_episode", episode)
    hmarl_reds.main(
        [
            "--recipe",
            recipe_name,
            "--model",
            str(model),
            "--seeds",
            "1000",
            "--episodes-per-seed",
            "1",
            "--episode-length",
            "4",
            "--deterministic",
            "--output",
            str(output),
            "--no-mlflow",
            "--no-progress",
        ]
    )
    assert len(loaded) == 1
    if recipe_name.startswith("hmarl_"):
        assert loaded == [(model, recipe_name.removeprefix("hmarl_"))]
    else:
        assert loaded == [(model, "blue", "jax")]
    assert case_calls == [((topology,), (1000,), 1)]
    assert [call[0] for call in rollout_calls] == ["finite_state", "aggressive", "stealthy", "impact"]
    assert all(call[1:] == (1000, True) for call in rollout_calls)
    assert all(variant.num_steps == 4 and variant.cage4_enhanced_obs for variant, _ in env_calls)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["eval_red"] for row in rows] == list(HMARL_REDS)
    for row in rows:
        assert row["suite"] == "hmarl_reds" and row["eval_name"] == "hmarl-reds"
        assert row["episode_length"] == 4
        assert row["per_episode_seeds"] == [1000]
        assert row["episode_role_map_ids"] == ["roles"]
        assert row["mean_reward"] == -12.0 and row["n_episodes"] == 1
        assert row["cia_summary"]["c"]["mean"] == -1.0
        if recipe_name.startswith("hmarl_"):
            assert row["trained_backend"] == "rllib_torch"
            assert row["pretrained"]["variant"] == recipe_name.removeprefix("hmarl_")


@pytest.mark.parametrize("recipe_name", ["cotraining_lstm", "cotraining_lstm_env_diversity"])
def test_lstm_recipes_schedule_hmarl_suite(recipe_name):
    recipe = load(recipe_name)
    jobs = PostTrainingEvalSettings.from_recipe(recipe).evaluations
    job = next(job for job in jobs if job.name == "hmarl-reds")
    assert job.resolve_script() == Path("scripts/eval/eval_hmarl_reds.py").resolve()
    assert job.model_arg == "--model"
    assert job.args == ("--recipe", "{recipe}", "--seeds", "1000-1009", "--episodes-per-seed", "6", "--progress")
    assert ScriptedRedEvalSettings(reds=HMARL_REDS).reds == HMARL_REDS


@pytest.mark.parametrize("args", [[], ["--reds", "cia_c"], ["--model", "x", "--episode-length", "0"]])
def test_cli_rejects_missing_policy_or_invalid_hmarl_options(args):
    with pytest.raises(SystemExit) as exc:
        hmarl_reds.main(args)
    assert exc.value.code == 2
