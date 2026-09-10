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
from jaxborg.recipe import eval_variant, load, project_cleanrl, project_jax, train_variant
from jaxborg.scenarios.cc4.game_variant import GameVariant

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


def test_train_variant_resolves(recipe):
    v = train_variant(recipe)
    assert isinstance(v, GameVariant)


def test_eval_variant_resolves(recipe):
    v = eval_variant(recipe)
    assert isinstance(v, GameVariant)


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


def test_stock_cotraining_recipes_stay_feedforward():
    """The MLP co-training recipes are the control arm and the published runs.

    Every number in docs/cotraining_collapse.md was measured with
    `arch.name: shared`. A recurrent architecture belongs in its own recipe,
    never as an edit to these.
    """
    for name in ("cotraining", "cotraining_env_diversity", "cotraining_test_rule_change"):
        arch = load(name)["arch"]
        assert arch["name"] == "shared", f"{name} must stay feedforward"
        assert arch["hidden_layers"] == 2
        assert "cell" not in arch and "trunk" not in arch


def test_recurrent_cotraining_arms_differ_only_in_the_cell():
    """`cotraining_rnn` and `cotraining_lstm` are a controlled A/B on the cell.

    If they drift apart on anything else — budget, minibatches, topologies —
    a gap between the two runs stops being attributable to GRU vs LSTM.
    """
    gru = load("cotraining_rnn")
    lstm = load("cotraining_lstm")
    for recipe in (gru, lstm):
        # Prose and file path are expected to differ; nothing else is.
        del recipe["meta"], recipe["__source_path__"]

    assert gru["arch"].pop("cell") == "gru"
    assert lstm["arch"].pop("cell") == "lstm"
    assert gru == lstm


# The only recipe allowed to run non-stock CC4 rules. It exists to test the
# knobs, and it ships in the `both` arm rather than the control.
RULE_KNOB_HARNESS = "cotraining_test_rule_change"


def test_rule_knobs_are_off_everywhere_except_the_harness(recipe):
    """`red_reward` and `blue_block_policy` default to stock CC4.

    Both change the game, not just the optimizer: `damage` alters Red's payoff
    and `mission_safe` removes Blue actions outright, so a recipe that picks
    one up by accident produces numbers that cannot be compared to any
    published CC4 result. They also propagate to evaluation through the
    resolved sidecar, so the mistake would follow the policy out of training.
    """
    expected = ("damage", "mission_safe") if recipe["meta"]["name"] == RULE_KNOB_HARNESS else ("zero_sum", "cc4")
    for resolved, where in ((train_variant(recipe), "train"), (eval_variant(recipe), "eval")):
        assert (resolved.red_reward, resolved.blue_block_policy) == expected, (
            f"{recipe['meta']['name']} {where} variant is not {expected}"
        )


def test_the_rule_knob_harness_is_the_only_recipe_declaring_overrides():
    """Names the one file to look at when a run's rules are in question."""
    declared = sorted(name for name in RECIPE_NAMES if (load(name).get("train") or {}).get("variant_overrides"))
    assert declared == [RULE_KNOB_HARNESS]
