"""The prespecified singleton frozen-Blue response-oracle protocol."""

import copy
import hashlib
import tomllib

import numpy as np

from jaxborg.blue_observation_contract import recipe_blue_obs_size
from jaxborg.recipe import eval_variant, project_jax, train_variant

TRAIN_SEEDS = (11001, 22001, 33001)
VALIDATION_SEEDS = tuple(range(1000000, 1000100))
TEST_SEEDS = tuple(range(2000000, 2000600))
SMOKE_SEEDS = tuple(range(9000000, 9000004))
REQUESTED_STEPS = 10000000
BOOTSTRAP_SEED = 3000001
BOOTSTRAP_SAMPLES = 10000


def gpu_lock_provenance(original_bytes, current_bytes):
    """Permit missing CUDA backend additions, with every existing distribution unchanged."""

    def packages(raw):
        result = {}
        for package in tomllib.loads(raw.decode())["package"]:
            identity = (package["version"], tuple(sorted(package["source"].items())))
            result.setdefault(package["name"], set()).add(identity)
        return result

    original, current = packages(original_bytes), packages(current_bytes)
    changed = [name for name, identities in original.items() if current.get(name) != identities]
    if changed:
        raise ValueError(f"source environment distributions changed: {sorted(changed)}")
    added = sorted(set(current) - set(original))
    if any(
        not (name.startswith("jax-cuda12-") or (name.startswith("nvidia-") and name.endswith("-cu12")))
        for name in added
    ):
        raise ValueError(f"unexpected environment additions: {added}")
    for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
        if name in current and {version for version, _ in current[name]} != {version for version, _ in current["jax"]}:
            raise ValueError(f"{name} versions must match the locked JAX versions")
    return {
        "source_lockfile_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "current_lockfile_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "identical_lockfile": original_bytes == current_bytes,
        "existing_distribution_versions_and_sources_unchanged": True,
        "added_gpu_distributions": added,
    }


def seed_protocol():
    domains = {
        "training_environment_roots": list(TRAIN_SEEDS),
        "training_policy_rollout_roots": [seed + 1 for seed in TRAIN_SEEDS],
        "validation_episode_roots": list(VALIDATION_SEEDS),
        "final_test_episode_roots": list(TEST_SEEDS),
        "smoke_episode_roots": list(SMOKE_SEEDS),
        "smoke_training_roots": [880001, 880002],
    }
    roots = [seed for values in domains.values() for seed in values]
    if len(roots) != len(set(roots)):
        raise ValueError("randomness root domains overlap")
    return {
        **domains,
        "training_descendants": "jax.random.split from distinct environment and policy/rollout roots",
        "evaluation_descendants": "PRNGKey(episode_seed) split for reset, stochastic policies and environment steps",
        "episodes_per_seed": 1,
        "episode_seed_scheme": "base_times_count_plus_replica_v1",
        "separation_claim": "distinct PRNG roots; layout generator seed 0 intentionally shared",
    }


def source_specific_recipe(source, source_path, topology_path, *, name="stage-b-red-oracle"):
    if source.get("run", {}).get("total_steps") != 9600000 or source["run"].get("seed") != 42:
        raise ValueError("Stage B requires the specified seed-42 9,600,000-step source")
    if recipe_blue_obs_size(source) != 450 or source["run"].get("blue_observation_version") != 2:
        raise ValueError("source requires enhanced-v2 Blue observations")
    if source["train"].get("topology_generation", {}).get("seed_start") != 0:
        raise ValueError("source topology generator seed is not 0")
    generation = source["train"]["topology_generation"]
    if generation.get("generator") != "jax" or generation.get("count") != 1:
        raise ValueError("source is not the singleton JAX-generator cohort")
    recipe = copy.deepcopy(source)
    recipe.pop("run", None)
    recipe.pop("__source_path__", None)
    recipe["meta"] = {"name": name, "source": "Stage B: fresh Red against original frozen Blue"}
    recipe["train"].update(
        teams="red",
        opponents={"blue": {"path": str(source_path)}},
        total_timesteps=REQUESTED_STEPS,
        topology_bank=[str(topology_path)],
    )
    recipe["train"].pop("topology_generation", None)
    recipe["eval"] = {
        "variant": "cc4_stock",
        "cia": {"enabled": False},
        "topology_bank": [str(topology_path)],
        "topology_sampling": "exhaustive",
        "allow_training_topologies": True,
        "policy_backend": "jax",
        "after_training": [],
        "scripted_red": {"after_training": False},
        **{
            suite: {"enabled": False}
            for suite in ("play_priors", "cross_play", "cross_seed_play", "checkpoint_scripted_reds")
        },
    }
    recipe["mlflow"] = {"checkpoint_eval": {"every_steps": 0}}
    assert_recipe_contract(recipe)
    return recipe


def assert_recipe_contract(recipe):
    if recipe["train"]["teams"] != "red" or set(recipe["train"].get("opponents", {})) != {"blue"}:
        raise ValueError("fresh Red requires exactly one frozen Blue opponent")
    if recipe_blue_obs_size(recipe) != 450:
        raise ValueError("Stage B requires 450-wide enhanced Blue observations")
    training, evaluation = train_variant(recipe), eval_variant(recipe)
    if training != evaluation or training.name != "cc4_stock" or training.num_steps != 500:
        raise ValueError("training and evaluation must use the same 500-step stock game")
    if getattr(training, "red_reward", "zero_sum") != "zero_sum":
        raise ValueError("Stage B requires zero-sum rewards")
    if recipe["train"].get("topology_bank") != recipe["eval"].get("topology_bank"):
        raise ValueError("training and evaluation must use the same singleton topology")
    if len(recipe["train"]["topology_bank"]) != 1:
        raise ValueError("Stage B requires one topology")
    if recipe["eval"].get("cia", {}).get("enabled"):
        raise ValueError("CIA evaluation is outside the Stage B target game")
    if recipe["eval"].get("after_training"):
        raise ValueError("Stage B evaluations must run independently")
    if recipe["eval"].get("scripted_red", {}).get("after_training"):
        raise ValueError("unplanned scripted evaluation suite")
    for suite in ("play_priors", "cross_play", "cross_seed_play", "checkpoint_scripted_reds"):
        if recipe["eval"].get(suite, {}).get("enabled"):
            raise ValueError(f"unplanned evaluation suite {suite}")


def budget(recipe):
    config = project_jax(recipe, team="red")
    stride = config["NUM_STEPS"] * config["NUM_ENVS"]
    updates = config["TOTAL_TIMESTEPS"] // stride
    return {
        "requested_steps": config["TOTAL_TIMESTEPS"],
        "completed_steps": updates * stride,
        "updates": updates,
        "steps_per_update": stride,
    }


def select_candidate(scores):
    if "original" not in scores or len(scores) != 4:
        raise ValueError("selection requires Original Red and all three oracle attempts")
    if any(not np.isfinite(value) for value in scores.values()):
        raise ValueError("validation scores must be finite")
    # Exact ties favor Original Red, then the prespecified seed order.
    order = ["original", *[f"seed-{seed}" for seed in TRAIN_SEEDS]]
    if set(scores) != set(order):
        raise ValueError("unexpected candidate identities")
    return min(order, key=lambda name: (scores[name], order.index(name)))


def paired_gap(baseline, challenger, baseline_seeds, challenger_seeds, *, samples=BOOTSTRAP_SAMPLES):
    if list(baseline_seeds) != list(challenger_seeds) or len(set(baseline_seeds)) != len(baseline_seeds):
        raise ValueError("paired evaluation requires aligned distinct episode seeds")
    baseline, challenger = np.asarray(baseline, dtype=float), np.asarray(challenger, dtype=float)
    if baseline.shape != challenger.shape or baseline.ndim != 1 or len(baseline) != len(baseline_seeds):
        raise ValueError("paired evaluation lengths differ")
    if len(baseline) < 2 or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(challenger)):
        raise ValueError("paired evaluation requires at least two finite episode returns")
    differences = baseline - challenger
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = differences[rng.integers(len(differences), size=(samples, len(differences)))].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "baseline_blue_mean": float(baseline.mean()),
        "selected_blue_mean": float(challenger.mean()),
        "red_improvement": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "episodes": len(differences),
        "method": "paired episode bootstrap, percentile 95% interval",
        "resampling_unit": "aligned test episode",
        "bootstrap_samples": samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation": "evaluation uncertainty conditional on selected policies from one source training run",
    }
