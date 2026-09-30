import copy
from pathlib import Path

import pytest

from jaxborg.oracle_stage_b import (
    TRAIN_SEEDS,
    assert_recipe_contract,
    budget,
    paired_gap,
    seed_protocol,
    select_candidate,
    source_specific_recipe,
)
from jaxborg.recipe import load
from jaxborg.topology_banks import validate_topology_split


@pytest.fixture
def recipe(tmp_path):
    source = copy.deepcopy(load("cotraining/cotraining"))
    source["cage4_enhanced_obs"] = True
    source["run"] = dict(total_steps=9600000, seed=42, blue_observation_version=2)
    source["train"]["topology_generation"] = dict(generator="jax", seed_start=0, count=1)
    source["jax"]["num_envs"] = 96
    return source_specific_recipe(source, tmp_path / "source.safetensors", tmp_path / "topology.npz")


def test_source_recipe_keeps_stock_contract_and_actual_rounded_budget(recipe):
    assert_recipe_contract(recipe)
    assert budget(recipe) == dict(
        requested_steps=10000000, completed_steps=9984000, updates=208, steps_per_update=48000
    )
    assert recipe["train"]["opponents"]["blue"]["path"].startswith("/")
    assert recipe["arch"]["name"] == "shared"
    assert recipe["eval"]["allow_training_topologies"] is True


@pytest.mark.parametrize("mutation", ["cia", "topology", "suite"])
def test_rejects_protocol_drift(recipe, mutation):
    if mutation == "cia":
        recipe["eval"]["cia"]["enabled"] = True
    elif mutation == "topology":
        recipe["eval"]["topology_bank"] = ["different.npz"]
    else:
        recipe["eval"]["cross_play"]["enabled"] = True
    with pytest.raises(ValueError):
        assert_recipe_contract(recipe)


def test_layout_overlap_requires_explicit_pilot_opt_in(recipe):
    validate_topology_split(recipe, repo_root=Path.cwd())
    recipe["eval"].pop("allow_training_topologies")
    with pytest.raises(ValueError, match="overlap"):
        validate_topology_split(recipe, repo_root=Path.cwd())
    recipe["eval"]["allow_training_topologies"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        validate_topology_split(recipe, repo_root=Path.cwd())


def test_seed_domains_are_disjoint_including_policy_and_environment_roots():
    protocol = seed_protocol()
    roots = [v for key, values in protocol.items() if key.endswith("_roots") for v in values]
    assert len(set(roots)) == len(roots)
    assert len(protocol["final_test_episode_roots"]) == 600
    assert len(protocol["validation_episode_roots"]) == 100


def test_candidate_selection_includes_original_fallback_and_fixed_tie_break():
    scores = {"original": -10, **{f"seed-{seed}": -9 for seed in TRAIN_SEEDS}}
    assert select_candidate(scores) == "original"
    scores["seed-11001"] = -10
    assert select_candidate(scores) == "original"
    scores["seed-22001"] = -11
    assert select_candidate(scores) == "seed-22001"
    del scores["seed-33001"]
    with pytest.raises(ValueError):
        select_candidate(scores)


def test_paired_bootstrap_retains_negative_gain_and_checks_alignment():
    gap = paired_gap([-10, -20, -30], [-5, -15, -25], [1, 2, 3], [1, 2, 3])
    assert gap["red_improvement"] == -5
    assert gap["ci95"] == [-5, -5]
    with pytest.raises(ValueError, match="aligned"):
        paired_gap([-10, -20], [-5, -15], [1, 2], [2, 1])
