"""Recipe loader — single YAML, two backends.

A recipe lives at `recipes/<name>.yaml` (repo root) and declares both the
backend-agnostic training contract (algorithm, arch, core hyperparameters,
buffer/minibatch targets) and the backend-specific knobs needed to realize
that contract on JAX vs CybORG-CleanRL.

Use:
    >>> from jaxborg.recipe import load, project_jax, project_cleanrl
    >>> recipe = load("singh")              # by name
    >>> recipe = load("/abs/path/to.yaml")    # or absolute path
    >>> jax_cfg = project_jax(recipe)         # flat dict for ippo_jax.py
    >>> cr_cfg = project_cleanrl(recipe)      # flat dict for ippo_cyborg.py

Projection is one-way: it flattens a structured recipe into the dict-of-
upper-case-keys (jax) or dict-of-snake-case-keys (cleanrl) that each trainer
already consumes. The reverse direction is not needed — we never reconstruct
a recipe from a trainer config.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.game_variants import VARIANTS, variant_for_red

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / "recipes"

REQUIRED_SECTIONS = ("meta", "algorithm", "core", "arch", "train")


def load(name_or_path: str) -> dict[str, Any]:
    """Resolve a recipe name (e.g. 'singh') or absolute path; return parsed dict.

    Raises FileNotFoundError if the recipe doesn't exist, ValueError if a
    required section is missing.
    """
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        p = RECIPES_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Recipe not found: {name_or_path} (looked at {p})")
    raw = yaml.safe_load(p.read_text())
    _validate(raw, source=str(p))
    raw["__source_path__"] = str(p)
    return raw


def _validate(recipe: dict[str, Any], *, source: str) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in recipe:
            raise ValueError(f"{source}: missing required section '{section}'")
    if "name" not in recipe["arch"]:
        raise ValueError(f"{source}: arch.name is required")
    if "lr" not in recipe["core"]:
        raise ValueError(f"{source}: core.lr is required")


def train_variant(recipe: dict[str, Any]) -> GameVariant:
    name = recipe.get("train", {}).get("variant", "cc4_stock")
    return VARIANTS[name]


def eval_variant(recipe: dict[str, Any]) -> GameVariant:
    """Resolve the eval-time GameVariant.

    Precedence:
      1. ``eval.red`` (if set) — overrides the variant's red selector. The
         base variant (``eval.variant`` or ``train.variant``) is used only to
         decide ``resilience_roles`` for the fsm path; CIA-biased reds
         (``cia_c`` / ``cia_i`` / ``cia_a`` / ``resilience``) carry their
         own resilience_roles=True since their selectors require role tags.
         This means setting ``eval.red: cia_a`` on a ``cc4_stock`` recipe
         forces ``resilience_roles=True`` to keep the selector consistent.
      2. ``eval.variant`` — full variant name in ``VARIANTS``.
      3. ``train.variant`` — fallback if no eval section is configured.
    """
    eval_cfg = recipe.get("eval") or {}
    base_name = eval_cfg.get("variant") or recipe.get("train", {}).get("variant", "cc4_stock")
    base = VARIANTS[base_name]
    red = eval_cfg.get("red")
    if red is None:
        return base
    return variant_for_red(red, resilience_roles=base.resilience_roles)


def resolve_eval_variant(
    *,
    recipe_name: str | None = None,
    checkpoint: str | Path | None = None,
    default: GameVariant | None = None,
) -> GameVariant:
    """Resolve the eval variant by precedence: explicit recipe → checkpoint sidecar → default.

    One canonical helper for every entry-point script. ``recipe_name`` accepts
    either a recipe name (``"singh"``) or an absolute path. ``checkpoint`` is
    a ``.safetensors`` path whose paired ``recipe_*.yaml`` sidecar is read
    when ``recipe_name`` is unset. If both are unset, returns ``default``
    (or ``CC4_STOCK`` if ``default`` is None).
    """
    from jaxborg.scenarios.cc4.game_variants import CC4_STOCK

    if recipe_name is not None:
        return eval_variant(load(recipe_name))
    if checkpoint is not None:
        from jaxborg.checkpoint import read_sidecar

        return eval_variant(read_sidecar(checkpoint))
    return default if default is not None else CC4_STOCK


def project_jax(recipe: dict[str, Any]) -> dict[str, Any]:
    """Flatten recipe into the dict shape ippo_jax.py's config expects."""
    core = recipe["core"]
    arch = recipe["arch"]
    train = recipe["train"]
    jax_ = recipe.get("jax", {})
    return {
        "LR": float(core["lr"]),
        "GAMMA": float(core["gamma"]),
        "GAE_LAMBDA": float(core["gae_lambda"]),
        "CLIP_EPS": float(core.get("clip_eps", 0.2)),
        "VF_COEF": float(core.get("vf_coef", 0.5)),
        "MAX_GRAD_NORM": float(core.get("max_grad_norm", 0.5)),
        "ENT_COEF": float(core.get("ent_coef", 0.0)),
        "NORM_REWARDS": bool(core.get("norm_rewards", False)),
        "REWARD_SCALE": float(core.get("reward_scale", 1.0)),
        "ANNEAL_LR": bool(core.get("anneal_lr", False)),
        "NETWORK_TYPE": arch["name"],
        "HIDDEN_DIM": int(arch.get("hidden_dim", 256)),
        "HIDDEN_LAYERS": int(arch.get("hidden_layers", 2)),
        "ACTIVATION": arch.get("activation", "tanh"),
        "NUM_ENVS": int(jax_.get("num_envs", 1024)),
        "NUM_STEPS": int(train["episode_length"]),
        "NUM_MINIBATCHES": int(jax_.get("num_minibatches", 16)),
        "UPDATE_EPOCHS": int(jax_.get("update_epochs", 4)),
        "TOTAL_TIMESTEPS": int(train["total_timesteps"]),
        "CHECKPOINT_EVERY_UPDATES": int(jax_.get("checkpoint_every_updates", 50)),
        "BUSY_MASKING": bool(jax_.get("busy_masking", False)),
        "GRAD_CLIP_MODE": jax_.get("grad_clip_mode", "global"),
        "TRAIN_VARIANT": train_variant(recipe),
        "EVAL_VARIANT": eval_variant(recipe),
        "TRAINING_MODE": True,
        "MLFLOW_ENABLED": True,
    }


def project_cleanrl(recipe: dict[str, Any]) -> dict[str, Any]:
    """Flatten recipe into the dict that ippo_cyborg.py CLI args populate."""
    core = recipe["core"]
    arch = recipe["arch"]
    train = recipe["train"]
    cr = recipe.get("cleanrl", {})

    num_envs = int(cr.get("num_envs", 48))
    rollout_length = int(cr.get("rollout_length", train["episode_length"]))
    per_rollout = num_envs * rollout_length
    if "num_rollouts_per_update" in cr:
        rollouts_per_update = int(cr["num_rollouts_per_update"])
    else:
        rollouts_per_update = max(1, math.ceil(int(train["buffer_size"]) / per_rollout))

    return {
        "lr": float(core["lr"]),
        "gamma": float(core["gamma"]),
        "gae_lambda": float(core["gae_lambda"]),
        "clip_coef": float(core.get("clip_eps", 0.2)),
        "vf_coef": float(core.get("vf_coef", 0.5)),
        "ent_coef": float(core.get("ent_coef", 0.0)),
        "max_grad_norm": float(core.get("max_grad_norm", 0.5)),
        "norm_rewards": bool(core.get("norm_rewards", False)),
        "anneal_lr": bool(core.get("anneal_lr", False)),
        "arch_name": arch["name"],
        "hidden_dim": int(arch.get("hidden_dim", 256)),
        "hidden_layers": int(arch.get("hidden_layers", 2)),
        "activation": arch.get("activation", "tanh"),
        "num_envs": num_envs,
        "rollout_length": rollout_length,
        "num_rollouts_per_update": rollouts_per_update,
        "num_epochs": int(cr.get("num_epochs", 4)),
        "num_minibatches": int(cr.get("num_minibatches", 16)),
        "total_timesteps": int(train["total_timesteps"]),
        "TRAIN_VARIANT": train_variant(recipe),
        "EVAL_VARIANT": eval_variant(recipe),
    }


def project_eval(recipe: dict[str, Any]) -> dict[str, Any]:
    """Flatten the eval section of a recipe into a config dict.

    Keys returned:
        cia_metric    — only "resilience" today; default if unset
        EVAL_VARIANT  — resolved GameVariant
    """
    ev = recipe.get("eval") or {}
    return {
        "cia_metric": ev.get("cia_metric", "resilience"),
        "EVAL_VARIANT": eval_variant(recipe),
    }


def flatten_for_logging(recipe: dict[str, Any]) -> dict[str, Any]:
    """Flatten the recipe to dotted-key form for MLflow params logging.

    Skips internal keys (underscore-prefixed) and non-scalar values.
    """
    out: dict[str, Any] = {}

    def _walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("__"):
                    continue
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(node, (list, tuple)):
            out[prefix] = ",".join(str(x) for x in node)
        else:
            out[prefix] = node

    _walk("", recipe)
    return out
