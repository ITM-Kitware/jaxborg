import copy
import subprocess
from pathlib import Path

import pytest
import yaml

from jaxborg.recipe import load, team_recipe
from jaxborg.response_campaign import load_campaign
from jaxborg.response_oracle import (
    TRAIN_SEEDS,
    assert_recipe_contract,
    budget,
    gpu_lock_provenance,
    paired_gap,
    seed_protocol,
    select_candidate,
    source_specific_recipe,
)
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


def test_generated_recipe_loads_through_production_parser(recipe, tmp_path):
    path = tmp_path / "generated.yaml"
    path.write_text(yaml.safe_dump(recipe))
    resolved = load(path)
    assert_recipe_contract(resolved)
    resolved["eval"]["scripted_red"]["after_training"] = True
    with pytest.raises(ValueError, match="scripted evaluation"):
        assert_recipe_contract(resolved)


def test_mappo_source_requires_explicit_checkpoint_and_fixed_ippo_challenger(recipe, tmp_path):
    template = copy.deepcopy(recipe)
    template["run"] = dict(total_steps=9600000, seed=42, blue_observation_version=2)
    template["train"]["topology_generation"] = dict(generator="jax", seed_start=0, count=1)
    source = copy.deepcopy(template)
    source["algorithm"] = "mappo"
    source["run"]["total_steps"] = 49968000
    source["train"]["team_overrides"] = {
        team: {
            "arch": {
                "name": "mappo",
                "hidden_dim": 256,
                "hidden_layers": 2,
                "activation": "tanh",
                "critic_input": "global_state",
            }
        }
        for team in ("blue", "red")
    }
    with pytest.raises(ValueError, match="specified steps"):
        source_specific_recipe(source, "blue", "topology", challenger_source=template)
    with pytest.raises(ValueError, match="explicit fixed IPPO"):
        source_specific_recipe(source, "blue", "topology", expected_source_steps=49968000)
    generated = source_specific_recipe(
        source, "blue", "topology", expected_source_steps=49968000, challenger_source=template
    )
    assert generated["algorithm"] == "ippo"
    assert team_recipe(generated, "red")["arch"] == template["arch"]
    assert "team_overrides" not in generated["train"]
    assert generated["core"] == template["core"]
    assert budget(generated) == budget(recipe)
    shorter = source_specific_recipe(
        source, "blue", "topology", expected_source_steps=49968000, challenger_source=template, requested_steps=96000
    )
    assert budget(shorter)["completed_steps"] == 96000
    assert budget(shorter)["updates"] == 2


def test_campaign_yaml_drives_seeds_budget_and_rejects_leakage(tmp_path):
    path = Path("campaigns/response-oracles/mappo-seed42.yaml")
    config, defender, randomness = load_campaign(path, "mappo-49968000")
    assert defender["steps"] == 49968000
    assert config["training"]["requested_steps"] == 10000000
    assert randomness["training_environment_roots"] == [11001, 22001, 33001]
    assert len(randomness["final_test_episode_roots"]) == 600
    config["training"]["seeds"] = [17, 29]
    config["evaluation"]["test"] = dict(seed_start=77, episodes=5)
    changed = tmp_path / "campaign.yaml"
    changed.write_text(yaml.safe_dump(config))
    _, _, randomness = load_campaign(changed, "mappo-9600000")
    assert randomness["training_environment_roots"] == [17, 29]
    assert randomness["final_test_episode_roots"] == [77, 78, 79, 80, 81]
    config["evaluation"]["validation"]["seed_start"] = 17
    changed.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="overlap"):
        load_campaign(changed, "mappo-9600000")


def test_selection_respects_configured_attempts_and_tie_order():
    assert select_candidate({"original": -3, "seed-29": -4, "seed-17": -4}, [29, 17]) == "seed-29"


def test_overnight_curve_keeps_red_protocol_and_blue_smoke_uses_original_red():
    path = Path("campaigns/response-oracles/ippo-seed42-red-curve.yaml")
    config = yaml.safe_load(path.read_text())
    assert sorted(entry["steps"] for entry in config["defenders"]) == [20160000, 29760000, 40320000, 49968000]
    assert config["existing_points"][0]["source_steps"] == 9600000
    reference, _, reference_roots = load_campaign("campaigns/response-oracles/mappo-seed42.yaml", "mappo-49968000")
    for entry in config["defenders"]:
        _, _, roots = load_campaign(path, entry["name"])
        assert roots == reference_roots
    assert config["training"] == reference["training"]
    smoke = load("verification/stage_c_blue_smoke")
    assert smoke["train"]["teams"] == "blue"
    assert set(smoke["train"]["opponents"]) == {"red"}
    assert smoke["train"]["opponents"]["red"]["path"].endswith("checkpoint_9600000.safetensors")
    assert smoke["train"]["total_timesteps"] == 4000
    assert smoke["eval"]["after_training"][0]["required"] is True
    assert "9100000-9100003" in smoke["eval"]["after_training"][0]["args"]


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


def test_gpu_lock_repair_preserves_source_versions_and_backend_matches():
    original = subprocess.check_output(["git", "show", "5c44cd7a69e91293ed3203ef82a9eb53bbf09749:uv.lock"])
    current = Path("uv.lock").read_bytes()
    proof = gpu_lock_provenance(original, current)
    assert proof["existing_distribution_versions_and_sources_unchanged"]
    assert "jax-cuda12-plugin" in proof["added_gpu_distributions"]
    with pytest.raises(ValueError, match="distributions changed"):
        gpu_lock_provenance(original, current.replace(b'version = "0.10.2"', b'version = "0.10.3"'))
