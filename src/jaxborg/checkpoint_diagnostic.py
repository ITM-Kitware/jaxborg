"""Prespecified checkpoint candidates and independent confirmation for response searches."""

import copy

import yaml


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic schema")
    if config["resources"]["partition"] != "community" or config["resources"]["gpus_per_job"] != 1:
        raise ValueError("diagnostics require one community GPU per point")
    if config["checkpoint_steps"] != [1920000, 4800000, "final"]:
        raise ValueError("this diagnostic prespecifies early, midpoint and final checkpoints")
    domains = []
    for split in ("validation", "confirmation"):
        spec = config["episodes"][split]
        if not isinstance(spec["count"], int) or spec["count"] < 2:
            raise ValueError("episode count must be at least two")
        seeds = list(range(spec["seed_start"], spec["seed_start"] + spec["count"]))
        domains.append(set(seeds))
    if domains[0] & domains[1]:
        raise ValueError("validation and confirmation episode roots overlap")
    return copy.deepcopy(config)


def candidates(protocol, state, checkpoint_steps):
    if protocol.get("trainable_team") != "blue":
        raise ValueError("this diagnostic requires completed Blue response searches")
    expected = {f"seed-{seed}" for seed in protocol["training_seeds"]}
    if set(state["training"]) != expected:
        raise ValueError("missing prescribed completed training attempts")
    result = [
        {
            "name": "original",
            "checkpoint": protocol["source"]["checkpoint"],
            "training_seed": None,
            "steps": protocol["source"]["original_training_steps"],
        }
    ]
    for seed in protocol["training_seeds"]:
        attempt = state["training"][f"seed-{seed}"]
        if attempt["actual_steps"] != protocol["oracle_budget_per_attempt"]["completed_steps"]:
            raise ValueError("training attempt budget differs")
        for step in checkpoint_steps:
            reference = (
                attempt["final_checkpoint"]
                if step == "final"
                else f"runs:/{attempt['run_id']}/checkpoints/checkpoint_{step}.safetensors"
            )
            result.append(
                {
                    "name": f"seed-{seed}-step-{step}",
                    "checkpoint": reference,
                    "training_seed": seed,
                    "steps": attempt["actual_steps"] if step == "final" else step,
                }
            )
    return result


def select_checkpoint(scores, candidate_order):
    if set(scores) != set(candidate_order):
        raise ValueError("validation must include every checkpoint and the original")
    # Maximum Blue return; exact ties retain original then prescribed seed/time order.
    return max(candidate_order, key=lambda name: scores[name])


def confirmation_candidates(candidate_list, selected):
    # Prespecified early-versus-final comparisons for every training seed, plus
    # original and the validation-selected checkpoint. Selection never sees test.
    names = [
        c["name"]
        for c in candidate_list
        if c["name"] == "original" or c["steps"] == 1920000 or c["name"].endswith("-final")
    ]
    if selected not in names:
        names.append(selected)
    return names
