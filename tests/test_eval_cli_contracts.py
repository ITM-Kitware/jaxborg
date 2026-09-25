"""Check real argument parsers without launching expensive evaluation jobs."""

import argparse
import runpy
import sys
from pathlib import Path

import pytest

from jaxborg.evaluation.post_training import PostTrainingEval, PostTrainingEvalSettings
from jaxborg.recipe import RECIPES_DIR, load

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", sorted((ROOT / "scripts/eval").glob("*.py")), ids=lambda path: path.name)
def test_evaluation_entry_points_can_show_help(monkeypatch, script):
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 0


def _recipe_commands():
    for path in sorted((RECIPES_DIR / "cotraining").glob("cotraining*.yaml")):
        for evaluation in PostTrainingEvalSettings.from_recipe(load(str(path))).evaluations:
            yield pytest.param(evaluation, path, id=f"{path.stem}-{evaluation.name}")
    for name in ("cross_play", "play_priors", "checkpoint_scripted_reds"):
        yield pytest.param(
            PostTrainingEval(name, f"scripts/eval/eval_{name}.py", args=("--recipe", "{recipe}")),
            RECIPES_DIR / "cotraining/cotraining_lstm.yaml",
            id=f"built-in-{name}",
        )


@pytest.mark.parametrize("evaluation,recipe", list(_recipe_commands()))
def test_recipe_commands_match_real_evaluation_parsers(monkeypatch, evaluation, recipe):
    replacements = {
        "model": "/tmp/model.safetensors",
        "recipe": str(recipe),
        "backend": "jax",
        "name": evaluation.name,
        "exp_dir": "/tmp/exp",
        "eval_dir": "/tmp/exp/eval",
        "nondiverse_red": "/tmp/nondiverse_red.safetensors",
    }
    args = [evaluation.script]
    if evaluation.model_arg:
        args += [evaluation.model_arg, replacements["model"]]
    args += [arg.format_map(replacements) for arg in evaluation.args]
    parse_args = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse_and_stop(parser, *args, **kwargs):
        parse_args(parser, *args, **kwargs)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(Parsed):
        runpy.run_path(str(ROOT / evaluation.script), run_name="__main__")
