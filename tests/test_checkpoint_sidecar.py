"""Read adjacent checkpoint recipes using the maintained naming and YAML behavior."""

import pytest
import yaml

from jaxborg.checkpoint import read_sidecar, sidecar_path


@pytest.mark.parametrize(
    "model_name,recipe_name",
    [("model_demo.pt", "recipe_demo"), ("model_demo.safetensors", "recipe_demo"), ("policy.pt", "recipe_policy")],
)
@pytest.mark.parametrize("suffix", ["yaml", "yml"])
def test_read_adjacent_recipe_without_requiring_model_bytes(tmp_path, model_name, recipe_name, suffix):
    model = tmp_path / model_name
    recipe = tmp_path / f"{recipe_name}.{suffix}"
    recipe.write_text("arch: {name: shared}\nrun: {seed: 17}\n")
    assert read_sidecar(model) == {"arch": {"name": "shared"}, "run": {"seed": 17}}
    assert sidecar_path(model) == recipe
    assert not model.exists()


def test_yaml_precedes_yml_and_malformed_yaml_does_not_fall_through(tmp_path):
    model = tmp_path / "model_demo.pt"
    primary = tmp_path / "recipe_demo.yaml"
    alternate = tmp_path / "recipe_demo.yml"
    primary.write_text("arch: {name: shared}\n")
    alternate.write_text("arch: {name: separate}\n")
    assert read_sidecar(model) == {"arch": {"name": "shared"}}
    primary.write_text("arch: [")
    with pytest.raises(yaml.YAMLError):
        read_sidecar(model)
    assert sidecar_path(model) == primary


def test_missing_recipe_diagnostic_lists_both_candidates(tmp_path):
    model = tmp_path / "model_demo.pt"
    with pytest.raises(FileNotFoundError) as failure:
        read_sidecar(model)
    message = str(failure.value)
    assert f"No recipe sidecar found next to {model}" in message
    for suffix in ("yaml", "yml"):
        assert str(tmp_path / f"recipe_demo.{suffix}") in message


@pytest.mark.parametrize("payload,expected", [("", None), ("[one, two]", ["one", "two"])])
def test_reader_preserves_yaml_values_without_capture_specific_schema_validation(tmp_path, payload, expected):
    model = tmp_path / "model_demo.pt"
    (tmp_path / "recipe_demo.yaml").write_text(payload)
    assert read_sidecar(model) == expected
