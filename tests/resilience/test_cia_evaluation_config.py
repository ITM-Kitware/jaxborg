from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from jaxborg.constants import SUBNET_IDS
from jaxborg.evaluation.cia.config import CIAEvalSettings
from jaxborg.recipe import load, project_eval
from jaxborg.scenarios.cc4.topology import build_topology, save_topology
from jaxborg.scenarios.cc4.topology_roles import count_resilience_candidates


def _recipe(
    tmp_path: Path,
    *,
    variant: str = "cia_resilience",
    sampling: str = "exhaustive",
    topology: bool = True,
) -> dict:
    evaluation = {
        "variant": variant,
        "topology_sampling": sampling,
        "cia": {
            "enabled": True,
            "metric": "resilience",
            "role_assignment": "fixed_per_topology",
        },
    }
    if topology:
        evaluation["topology_generation"] = {
            "generator": "jax",
            "seed_start": 10,
            "count": 1,
            "op_zone_servers": 3,
            "cache_dir": str(tmp_path / "bank"),
        }
    return {
        "meta": {"name": "cia-config-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": {
            "teams": "both",
            "episode_length": 10,
            "buffer_size": 20,
            "total_timesteps": 100,
            "variant": "cc4_stock",
        },
        "eval": evaluation,
    }


def _load_yaml(tmp_path: Path, recipe: dict) -> dict:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return load(str(path))


def test_cia_defaults_disabled_and_legacy_metric_is_selection_only():
    default = CIAEvalSettings.from_recipe({"eval": {}})
    legacy = CIAEvalSettings.from_recipe({"eval": {"cia_metric": "resilience"}})

    assert default.as_dict() == {
        "enabled": False,
        "metric": "resilience",
        "role_assignment": "fixed_per_topology",
    }
    assert legacy.metric == "resilience"
    assert not legacy.enabled


@pytest.mark.parametrize("value", [False, 0, ""])
def test_falsey_legacy_metric_alias_is_not_silently_reinterpreted(value):
    with pytest.raises(ValueError, match="eval.cia.metric must be 'resilience'"):
        CIAEvalSettings.from_recipe({"eval": {"cia_metric": value}})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"variant": "cc4_stock"}, "resilience_roles=True"),
        ({"sampling": "random"}, "topology_sampling: exhaustive"),
        ({"topology": False}, "non-empty eval topology bank"),
    ],
)
def test_invalid_cia_contract_fails_at_recipe_load(tmp_path, overrides, message):
    recipe = _recipe(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        _load_yaml(tmp_path, recipe)


def test_insufficient_snapshot_fails_during_projection_before_rollout(tmp_path):
    path = tmp_path / "insufficient.snapshot.npz"
    const = build_topology(jax.random.PRNGKey(4), op_zone_min_servers=3)
    candidates = np.flatnonzero(
        np.asarray(const.host_active)
        & np.asarray(const.host_is_server)
        & np.isin(
            np.asarray(const.host_subnet),
            (SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_B"]),
        )
    )
    const = const.replace(host_is_server=const.host_is_server.at[candidates[2:]].set(False))
    assert count_resilience_candidates(const) == 2
    save_topology(const, path)
    recipe = _recipe(tmp_path)
    recipe["eval"].pop("topology_generation")
    recipe["eval"]["topology_bank"] = [str(path)]

    with pytest.raises(ValueError, match="only 2 eligible operational servers"):
        project_eval(recipe, materialize_topologies=True)


def test_enabled_projection_exposes_nested_contract_and_safe_bank(tmp_path):
    recipe = _load_yaml(tmp_path, _recipe(tmp_path))

    projected = project_eval(recipe, materialize_topologies=True)

    assert projected["CIA"] == recipe["eval"]["cia"]
    assert projected["cia_metric"] == "resilience"
    assert projected["TOPOLOGY_SAMPLING"] == "exhaustive"
    assert len(projected["TOPOLOGY_BANK"]) == 1
