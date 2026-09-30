"""Declarative campaign settings for bounded frozen-defender response searches."""

from pathlib import Path

import yaml

from jaxborg.response_oracle import seed_protocol


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_campaign(path, defender_name):
    config = yaml.safe_load(Path(path).read_text())
    if config["schema_version"] != 1:
        raise ValueError("unsupported response campaign schema")
    defenders = config["defenders"]
    names = [entry["name"] for entry in defenders]
    if len(set(names)) != len(names) or defender_name not in names:
        raise ValueError("defender must uniquely name a configured checkpoint")
    defender = defenders[names.index(defender_name)]
    for name, value in (
        ("tracking.root", config["tracking"]["root"]),
        ("collection_readme", config["collection_readme"]),
        ("defender.checkpoint", defender["checkpoint"]),
        ("defender.report_dir", defender["report_dir"]),
        ("challenger.template_model", config["challenger"]["template_model"]),
    ):
        if not Path(value).is_absolute():
            raise ValueError(f"{name} must be absolute")
    if config["challenger"]["algorithm"] != "ippo" or config["challenger"]["architecture"] != "shared":
        raise ValueError("supported challengers are shared IPPO Reds")
    if config["game"] != {
        "rules": "cc4_stock",
        "episode_length": 500,
        "enhanced_observations": True,
        "blue_observation_version": 2,
        "reward": "zero_sum",
        "topology": {"generator": "jax", "seed": 0, "count": 1},
    }:
        raise ValueError("supported target is enhanced-v2 singleton stock CC4 with zero-sum rewards")
    if config["selection"] != {
        "rule": "lowest_validation_blue_mean",
        "checkpoint": "final",
        "original_fallback": True,
        "tie_break": "original_then_training_seed_order",
    }:
        raise ValueError("unsupported selection rule")
    if config["evaluation"]["stochastic"] is not True:
        raise ValueError("response evaluation requires stochastic actions")
    seeds = config["training"]["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training seeds must be nonnegative integers")
    splits = {}
    for split in ("validation", "test", "smoke"):
        spec = config["evaluation"][split]
        count = positive_integer(spec["episodes"], f"{split}.episodes")
        start = spec["seed_start"]
        if type(start) is not int or start < 0:
            raise ValueError(f"{split}.seed_start must be a nonnegative integer")
        splits[split] = list(range(start, start + count))
    if len(splits["test"]) < 2:
        raise ValueError("paired confidence interval needs at least two test episodes")
    randomness = seed_protocol(
        seeds, splits["validation"], splits["test"], splits["smoke"], config["smoke"]["training_seed"]
    )
    positive_integer(config["training"]["requested_steps"], "training.requested_steps")
    positive_integer(config["bootstrap"]["resamples"], "bootstrap.resamples")
    if config["bootstrap"]["method"] != "paired_percentile_95":
        raise ValueError("unsupported paired bootstrap method")
    if config["bootstrap"]["seed"] in [
        root for key, values in randomness.items() if key.endswith("_roots") for root in values
    ]:
        raise ValueError("bootstrap and rollout roots overlap")
    resources = config["resources"]
    if resources["partition"] != "community" or resources["gpus_per_job"] != 1:
        raise ValueError("response campaigns require one GPU per community job")
    for key in ("cpus_per_task", "memory_gb", "max_concurrent_jobs"):
        positive_integer(resources[key], f"resources.{key}")
    for key in ("requested_steps", "num_envs", "num_minibatches", "update_epochs"):
        positive_integer(config["smoke"][key], f"smoke.{key}")
    return config, defender, randomness
