"""Recipe smoke tests.

Loads every YAML under `recipes/`, projects it for both backends, and
asserts the resulting config is sane. Cheap (~milliseconds) — meant to
catch the kind of bug that would otherwise only surface 30 minutes into
a training run (e.g. the matched-v2 incident where `num_rollouts_per_update`
was derived from a JAX-shaped `buffer_size` and gave 214 rollouts/update
on the CybORG side, or where `num_envs: 1024` collapsed JAX to 5 update
cycles instead of 125).
"""

from pathlib import Path

import pytest

from jaxborg.policies import POLICY_REGISTRY
from jaxborg.recipe import load, project_cleanrl, project_jax

RECIPES_DIR = Path(__file__).resolve().parents[1] / "recipes"
RECIPE_NAMES = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))

assert RECIPE_NAMES, f"No recipes found in {RECIPES_DIR}"


@pytest.fixture(scope="module", params=RECIPE_NAMES)
def recipe(request):
    return load(request.param)


def test_required_sections(recipe):
    for section in ("meta", "algorithm", "core", "arch", "train"):
        assert section in recipe, f"missing required section: {section}"


def test_arch_name_in_registry(recipe):
    assert recipe["arch"]["name"] in POLICY_REGISTRY


def test_core_values_sane(recipe):
    core = recipe["core"]
    assert core["lr"] > 0
    assert 0 < core["gamma"] <= 1
    assert 0 < core["gae_lambda"] <= 1


def test_train_values_sane(recipe):
    train = recipe["train"]
    assert train["episode_length"] > 0
    assert train["total_timesteps"] > 0


def test_jax_projection(recipe):
    cfg = project_jax(recipe)
    for key in (
        "LR",
        "NUM_ENVS",
        "NUM_STEPS",
        "TOTAL_TIMESTEPS",
        "UPDATE_EPOCHS",
        "NUM_MINIBATCHES",
        "GAMMA",
        "GAE_LAMBDA",
    ):
        assert key in cfg, f"project_jax missing key: {key}"
    assert cfg["NUM_ENVS"] > 0
    assert cfg["NUM_STEPS"] > 0
    steps_per_update = cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
    updates = cfg["TOTAL_TIMESTEPS"] // steps_per_update
    assert updates >= 1, (
        f"JAX projection yields {updates} updates "
        f"({cfg['TOTAL_TIMESTEPS']} / ({cfg['NUM_ENVS']}*{cfg['NUM_STEPS']})) — "
        f"too few to train"
    )


def test_cleanrl_projection(recipe):
    cfg = project_cleanrl(recipe)
    for key in (
        "lr",
        "num_envs",
        "rollout_length",
        "num_rollouts_per_update",
        "total_timesteps",
        "num_epochs",
        "num_minibatches",
    ):
        assert key in cfg, f"project_cleanrl missing key: {key}"
    assert cfg["num_envs"] > 0
    assert cfg["rollout_length"] > 0
    assert cfg["num_rollouts_per_update"] >= 1
    steps_per_update = cfg["num_envs"] * cfg["rollout_length"] * cfg["num_rollouts_per_update"]
    updates = cfg["total_timesteps"] // steps_per_update
    assert updates >= 1, (
        f"CleanRL projection yields {updates} updates "
        f"({cfg['total_timesteps']} / "
        f"({cfg['num_envs']}*{cfg['rollout_length']}*{cfg['num_rollouts_per_update']})) — "
        f"too few to train"
    )


def test_minibatch_divides_batch(recipe):
    """num_minibatches must divide the rollout batch evenly on both backends."""
    j = project_jax(recipe)
    assert (j["NUM_ENVS"] * j["NUM_STEPS"]) % j["NUM_MINIBATCHES"] == 0, (
        "JAX: num_envs * num_steps not divisible by num_minibatches"
    )
    c = project_cleanrl(recipe)
    batch = c["num_envs"] * c["rollout_length"] * c["num_rollouts_per_update"]
    assert batch % c["num_minibatches"] == 0, (
        "CleanRL: num_envs * rollout_length * num_rollouts_per_update not divisible by num_minibatches"
    )
