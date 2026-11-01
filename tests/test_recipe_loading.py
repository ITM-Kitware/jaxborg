from pathlib import Path

import pytest
import yaml

from jaxborg import recipe as recipe_module
from jaxborg.recipe import load

COTRAINING_RECIPES = sorted((recipe_module.RECIPES_DIR / "cotraining").glob("*.yaml"))
assert COTRAINING_RECIPES


@pytest.mark.parametrize("path", COTRAINING_RECIPES, ids=lambda path: path.stem)
def test_grouped_recipes_load_by_short_name_qualified_name_and_path(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = load(str(path))
    for name in (path.stem, path.name, f"cotraining/{path.stem}", f"cotraining/{path.name}"):
        assert load(name) == expected
    assert Path(expected["__source_path__"]) == path


def test_existing_directory_does_not_hide_a_recipe_name(tmp_path, monkeypatch):
    (tmp_path / "cotraining").mkdir()
    monkeypatch.chdir(tmp_path)
    assert load("cotraining")["meta"]["name"] == "cotraining"


def test_missing_explicit_paths_do_not_fall_back_to_short_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in (tmp_path / "cotraining.yaml", Path("missing/cotraining.yaml")):
        with pytest.raises(FileNotFoundError, match="Recipe not found"):
            load(str(path))


def test_ambiguous_short_names_require_a_qualified_name(tmp_path, monkeypatch):
    source = load("default")
    recipes = tmp_path / "recipes"
    for group in ("one", "two"):
        directory = recipes / group
        directory.mkdir(parents=True)
        source["meta"]["name"] = group
        (directory / "shared.yaml").write_text(yaml.safe_dump(source))
    monkeypatch.setattr(recipe_module, "RECIPES_DIR", recipes)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Ambiguous recipe name.*one/shared.yaml, two/shared.yaml"):
        load("shared")
    assert load("one/shared")["meta"]["name"] == "one"
    assert load("two/shared.yaml")["meta"]["name"] == "two"

    source["meta"]["name"] = "top-level"
    (recipes / "shared.yaml").write_text(yaml.safe_dump(source))
    assert load("shared")["meta"]["name"] == "top-level"

    source["meta"]["name"] = "local-file"
    (tmp_path / "shared.yaml").write_text(yaml.safe_dump(source))
    assert load("shared.yaml")["meta"]["name"] == "local-file"
