"""Tests for the ``eval.red`` override hook (Phase 6, Stream S3).

Covers:
- ``eval.red`` override in :func:`jaxborg.recipe.eval_variant`.
- Resilience-role coupling: setting ``eval.red: cia_a`` on a ``cc4_stock``
  recipe must force ``resilience_roles=True`` because the cia_a selector
  requires role tags.
- Falls back to ``eval.variant`` when ``eval.red`` is null/missing.
- ``--eval-red`` CLI flag on ``scripts/eval/eval_recipe.py`` overrides
  whatever the loaded recipe says (CLI > recipe).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from jaxborg.recipe import eval_variant


def _make_recipe(*, eval_section: dict | None = None, train_variant: str = "cc4_stock") -> dict:
    """Build a minimal recipe dict that satisfies eval_variant."""
    recipe: dict = {
        "meta": {"name": "test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "mlp"},
        "train": {
            "variant": train_variant,
            "episode_length": 500,
            "total_timesteps": 1000,
        },
    }
    if eval_section is not None:
        recipe["eval"] = eval_section
    return recipe


def test_eval_red_cia_c_with_cia_resilience_variant():
    """Override to cia_c on a cia_resilience base preserves resilience_roles."""
    recipe = _make_recipe(eval_section={"variant": "cia_resilience", "red": "cia_c"})
    v = eval_variant(recipe)
    assert v.red_agent == "c"
    assert v.resilience_roles is True


def test_eval_red_fsm_on_cc4_stock_drops_resilience_roles():
    """Override to fsm on a cc4_stock base keeps resilience_roles off."""
    recipe = _make_recipe(eval_section={"variant": "cc4_stock", "red": "fsm"})
    v = eval_variant(recipe)
    assert v.red_agent == "finite_state"
    assert v.resilience_roles is False


def test_eval_red_cia_a_on_cc4_stock_forces_resilience_roles():
    """Override to cia_a on a cc4_stock base must force resilience_roles=True.

    The cia_a selector requires per-host role assignments (AUTH/DB/WEB), so
    the variant_for_red("cia_a", ...) helper always returns CIA_A which has
    resilience_roles=True regardless of the base. This is the deliberate
    corner-case behavior documented on eval_variant().
    """
    recipe = _make_recipe(eval_section={"variant": "cc4_stock", "red": "cia_a"})
    v = eval_variant(recipe)
    assert v.red_agent == "a"
    assert v.resilience_roles is True


def test_eval_red_missing_falls_back_to_variant():
    """No eval.red → existing eval.variant resolution (regression)."""
    recipe = _make_recipe(eval_section={"variant": "cia_resilience"})
    v = eval_variant(recipe)
    assert v.name == "cia_resilience"
    assert v.red_agent == "resilience"
    assert v.resilience_roles is True


def test_eval_red_null_falls_back_to_variant():
    """Explicit null eval.red is treated as 'unset'."""
    recipe = _make_recipe(eval_section={"variant": "cc4_stock", "red": None})
    v = eval_variant(recipe)
    assert v.name == "cc4_stock"
    assert v.red_agent == "finite_state"


def test_eval_section_missing_falls_back_to_train_variant():
    """No eval section → train.variant drives the eval variant."""
    recipe = _make_recipe(eval_section=None, train_variant="cia_resilience")
    v = eval_variant(recipe)
    assert v.name == "cia_resilience"


def test_eval_red_resilience_keeps_resilience_roles():
    """Override to 'resilience' returns CIA_RESILIENCE."""
    recipe = _make_recipe(eval_section={"variant": "cc4_stock", "red": "resilience"})
    v = eval_variant(recipe)
    assert v.red_agent == "resilience"
    assert v.resilience_roles is True


def test_eval_red_sleep_on_cc4_stock():
    """sleep red is a thin variant on top of cc4_stock."""
    recipe = _make_recipe(eval_section={"variant": "cc4_stock", "red": "sleep"})
    v = eval_variant(recipe)
    assert v.red_agent == "sleep"
    assert v.resilience_roles is False


def test_eval_red_cli_flag_overrides_recipe(monkeypatch, tmp_path):
    """--eval-red CLI flag must override whatever the loaded recipe sidecar says.

    We stub out the heavy CybORG/JAX runner imports and capture the variant
    that ``eval_recipe.main`` actually resolves, exercising the CLI > recipe
    precedence end to end.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval" / "eval_recipe.py"
    spec_root = str(repo_root / "scripts" / "eval")
    if spec_root not in sys.path:
        sys.path.insert(0, spec_root)

    # Recipe sidecar says cia_c; CLI says cia_a. CLI must win.
    sidecar_recipe = _make_recipe(eval_section={"variant": "cia_resilience", "red": "cia_c"})
    sidecar_recipe["meta"]["name"] = "test_cli_override"

    captured: dict = {}

    def fake_evaluate_on_cyborg(model_path, *, variant, seeds, episodes_per_seed, deterministic, workers):
        captured["variant"] = variant
        return ([0.0], [0])

    fake_runner = types.ModuleType("jaxborg.evaluation.cyborg_runner")
    fake_runner.evaluate_on_cyborg = fake_evaluate_on_cyborg

    fake_model = tmp_path / "model.pt"
    fake_model.write_bytes(b"")

    # Patch read_sidecar to return our test recipe.
    monkeypatch.setattr("jaxborg.checkpoint.read_sidecar", lambda p: dict(sidecar_recipe))
    # Stub the CybORG runner module entirely so import doesn't pull CybORG.
    monkeypatch.setitem(sys.modules, "jaxborg.evaluation.cyborg_runner", fake_runner)
    # Skip MLflow attach.
    monkeypatch.setattr("jaxborg.mlflow_setup.attach_eval_metrics", lambda *a, **kw: None)

    # Run the CLI with --eval-red cia_a.
    argv = [
        "eval_recipe.py",
        "--model",
        str(fake_model),
        "--episodes",
        "1",
        "--seeds",
        "0",
        "--eval-red",
        "cia_a",
        "--output",
        str(tmp_path / "out.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Load the script as a module so `main()` is callable.
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_recipe_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    variant = captured["variant"]
    assert variant.red_agent == "a", f"expected red_agent='a' from CLI override, got {variant.red_agent}"
    assert variant.resilience_roles is True


def test_eval_red_cli_flag_unset_uses_recipe_red(monkeypatch, tmp_path):
    """When --eval-red is not passed, recipe eval.red still applies."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval" / "eval_recipe.py"

    sidecar_recipe = _make_recipe(eval_section={"variant": "cia_resilience", "red": "cia_i"})
    sidecar_recipe["meta"]["name"] = "test_no_cli"

    captured: dict = {}

    def fake_evaluate_on_cyborg(model_path, *, variant, seeds, episodes_per_seed, deterministic, workers):
        captured["variant"] = variant
        return ([0.0], [0])

    fake_runner = types.ModuleType("jaxborg.evaluation.cyborg_runner")
    fake_runner.evaluate_on_cyborg = fake_evaluate_on_cyborg

    fake_model = tmp_path / "model.pt"
    fake_model.write_bytes(b"")

    monkeypatch.setattr("jaxborg.checkpoint.read_sidecar", lambda p: dict(sidecar_recipe))
    monkeypatch.setitem(sys.modules, "jaxborg.evaluation.cyborg_runner", fake_runner)
    monkeypatch.setattr("jaxborg.mlflow_setup.attach_eval_metrics", lambda *a, **kw: None)

    argv = [
        "eval_recipe.py",
        "--model",
        str(fake_model),
        "--episodes",
        "1",
        "--seeds",
        "0",
        "--output",
        str(tmp_path / "out.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_recipe_under_test_2", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    assert captured["variant"].red_agent == "i"
