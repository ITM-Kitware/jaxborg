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

import copy
import math
import os
from pathlib import Path
from typing import Any

import yaml

from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.game_variants import VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / "recipes"

REQUIRED_SECTIONS = ("meta", "algorithm", "core", "arch", "train")
TEAMS = ("blue", "red")
TRAIN_TEAM_MODES = (*TEAMS, "both")
POLICY_BACKENDS = ("jax", "cyborg")


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
    if not isinstance(recipe, dict):
        raise ValueError(f"{source}: recipe must be a YAML mapping")
    for section in REQUIRED_SECTIONS:
        if section not in recipe:
            raise ValueError(f"{source}: missing required section '{section}'")
        if section != "algorithm" and not isinstance(recipe[section], dict):
            raise ValueError(f"{source}: section '{section}' must be a mapping")
    if "name" not in recipe["arch"]:
        raise ValueError(f"{source}: arch.name is required")
    if "lr" not in recipe["core"]:
        raise ValueError(f"{source}: core.lr is required")

    train = recipe["train"]
    mode = train.get("teams", "blue")
    if mode not in TRAIN_TEAM_MODES:
        raise ValueError(f"{source}: train.teams must be one of {TRAIN_TEAM_MODES}, got {mode!r}")

    opponents = train.get("opponents") or {}
    if not isinstance(opponents, dict):
        raise ValueError(f"{source}: train.opponents must be a mapping")
    unknown_opponents = set(opponents) - set(TEAMS)
    if unknown_opponents:
        raise ValueError(f"{source}: unknown opponent teams: {sorted(unknown_opponents)}")
    for team, ref in opponents.items():
        _validate_model_ref(ref, source=f"{source}: train.opponents.{team}")

    if mode == "red" and not opponents.get("blue"):
        raise ValueError(f"{source}: red-only training requires train.opponents.blue")
    expected_opponent = {"blue": "red", "red": "blue"}.get(mode)
    if expected_opponent is not None:
        wrong_opponents = set(opponents) - {expected_opponent}
        if wrong_opponents:
            raise ValueError(
                f"{source}: {mode}-only training may only configure train.opponents.{expected_opponent}; "
                f"got {sorted(wrong_opponents)}"
            )
    if mode == "both" and opponents:
        raise ValueError(f"{source}: train.opponents is not allowed when train.teams is 'both'")
    if mode == "both" and train.get("initial_weights"):
        raise ValueError(f"{source}: train.initial_weights is not allowed when train.teams is 'both'")

    overrides = train.get("team_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{source}: train.team_overrides must be a mapping")
    unknown_teams = set(overrides) - set(TEAMS)
    if unknown_teams:
        raise ValueError(f"{source}: unknown teams in train.team_overrides: {sorted(unknown_teams)}")
    for team, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"{source}: train.team_overrides.{team} must be a mapping")
        unknown_sections = set(override) - {"core", "arch"}
        if unknown_sections:
            raise ValueError(
                f"{source}: train.team_overrides.{team} may only contain core and arch; got {sorted(unknown_sections)}"
            )
        for section, values in override.items():
            if not isinstance(values, dict):
                raise ValueError(f"{source}: train.team_overrides.{team}.{section} must be a mapping")

    ev = recipe.get("eval") or {}
    if not isinstance(ev, dict):
        raise ValueError(f"{source}: eval must be a mapping")
    backend = ev.get("policy_backend")
    if backend is not None and backend not in POLICY_BACKENDS:
        raise ValueError(f"{source}: eval.policy_backend must be one of {POLICY_BACKENDS}, got {backend!r}")
    policies = ev.get("policies") or {}
    if not isinstance(policies, dict):
        raise ValueError(f"{source}: eval.policies must be a mapping")
    if policies:
        if backend is None:
            raise ValueError(f"{source}: eval.policy_backend is required when eval.policies is set")
        missing = set(TEAMS) - set(policies)
        unknown = set(policies) - set(TEAMS)
        if missing:
            raise ValueError(f"{source}: eval.policies requires both blue and red; missing {sorted(missing)}")
        if unknown:
            raise ValueError(f"{source}: unknown teams in eval.policies: {sorted(unknown)}")
        for team, ref in policies.items():
            _validate_model_ref(ref, source=f"{source}: eval.policies.{team}", expected_backend=backend)


def _validate_model_ref(ref: Any, *, source: str, expected_backend: str | None = None) -> None:
    """Validate a path/experiment reference without touching the filesystem."""
    if isinstance(ref, (str, Path)):
        if not str(ref):
            raise ValueError(f"{source}: model path cannot be empty")
        return
    if not isinstance(ref, dict):
        raise ValueError(f"{source}: model reference must be a path string or mapping")
    if not ref.get("path") and not ref.get("experiment"):
        raise ValueError(f"{source}: model reference requires 'path' or 'experiment'")
    backend = ref.get("backend")
    if backend == "torch":
        backend = "cyborg"
    if backend is not None and backend not in POLICY_BACKENDS:
        raise ValueError(f"{source}: backend must be one of {POLICY_BACKENDS}, got {backend!r}")
    if expected_backend is not None and backend is not None and backend != expected_backend:
        raise ValueError(f"{source}: backend {backend!r} does not match selected backend {expected_backend!r}")


def training_teams(recipe: dict[str, Any]) -> tuple[str, ...]:
    """Return concrete trainable teams, preserving the legacy Blue default."""
    mode = recipe.get("train", {}).get("teams", "blue")
    if mode == "both":
        return TEAMS
    if mode not in TEAMS:
        raise ValueError(f"train.teams must be one of {TRAIN_TEAM_MODES}, got {mode!r}")
    return (mode,)


def team_recipe(recipe: dict[str, Any], team: str) -> dict[str, Any]:
    """Return a copy with ``team_overrides.<team>`` deep-merged into core/arch."""
    if team not in TEAMS:
        raise ValueError(f"team must be one of {TEAMS}, got {team!r}")
    projected = copy.deepcopy(recipe)
    override = projected.get("train", {}).get("team_overrides", {}).get(team, {})
    for section in ("core", "arch"):
        _deep_merge(projected[section], override.get(section, {}))
    return projected


def _deep_merge(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _normalise_backend(backend: str) -> str:
    backend = "cyborg" if backend == "torch" else backend
    if backend not in POLICY_BACKENDS:
        raise ValueError(f"backend must be one of {POLICY_BACKENDS} (or 'torch'), got {backend!r}")
    return backend


def resolve_model_ref(
    ref: str | Path | dict[str, Any],
    *,
    backend: str,
    recipe: dict[str, Any] | None = None,
    exp_dir: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a model path or experiment reference for one policy backend.

    A mapping's ``path`` takes precedence over ``experiment``. Relative paths
    are interpreted relative to the source recipe, when known, rather than the
    process working directory. Experiment references use the canonical
    ``$JAXBORG_EXP_DIR/<algo>_<backend>/<experiment>/model_<experiment>`` layout,
    where ``<algo>`` comes from the recipe. Calls without a recipe retain the
    legacy ``ippo`` default.
    """
    backend = _normalise_backend(backend)
    _validate_model_ref(ref, source="model reference", expected_backend=backend)
    raw_ref: dict[str, Any] = {"path": str(ref)} if isinstance(ref, (str, Path)) else ref

    if raw_ref.get("path"):
        path = Path(raw_ref["path"]).expanduser()
        if not path.is_absolute():
            source_path = (recipe or {}).get("__source_path__") or (recipe or {}).get("meta", {}).get("source_path")
            base = Path(source_path).expanduser().resolve().parent if source_path else Path.cwd()
            path = base / path
    else:
        experiment = str(raw_ref["experiment"])
        root = Path(exp_dir or os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).expanduser()
        suffix = ".safetensors" if backend == "jax" else ".pt"
        algorithm = "ippo" if recipe is None else recipe["algorithm"]
        path = root / f"{algorithm}_{backend}" / experiment / f"model_{experiment}{suffix}"

    path = path.resolve()
    incompatible_suffixes = {"jax": {".pt"}, "cyborg": {".safetensors", ".flax", ".orbax"}}
    if path.suffix.lower() in incompatible_suffixes[backend]:
        raise ValueError(f"Model {path} is not compatible with the {backend!r} policy backend")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


def resolve_train_opponents(
    recipe: dict[str, Any],
    *,
    backend: str,
    exp_dir: str | Path | None = None,
    must_exist: bool = True,
) -> dict[str, Path]:
    """Resolve all configured frozen opponents for a trainer."""
    backend = _normalise_backend(backend)
    refs = recipe.get("train", {}).get("opponents") or {}
    return {
        team: resolve_model_ref(
            ref,
            backend=backend,
            recipe=recipe,
            exp_dir=exp_dir,
            must_exist=must_exist,
        )
        for team, ref in refs.items()
    }


def resolve_eval_policies(
    recipe: dict[str, Any],
    *,
    exp_dir: str | Path | None = None,
    must_exist: bool = True,
) -> dict[str, Path]:
    """Resolve the independently configured Blue and Red evaluation models."""
    ev = recipe.get("eval") or {}
    refs = ev.get("policies") or {}
    if not refs:
        return {}
    backend = _normalise_backend(ev["policy_backend"])
    return {
        team: resolve_model_ref(
            ref,
            backend=backend,
            recipe=recipe,
            exp_dir=exp_dir,
            must_exist=must_exist,
        )
        for team, ref in refs.items()
    }


def train_variant(recipe: dict[str, Any]) -> GameVariant:
    name = recipe.get("train", {}).get("variant", "cc4_stock")
    return VARIANTS[name]


def eval_variant(recipe: dict[str, Any]) -> GameVariant:
    eval_cfg = recipe.get("eval") or {}
    name = eval_cfg.get("variant") or recipe.get("train", {}).get("variant", "cc4_stock")
    return VARIANTS[name]


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


def project_jax(recipe: dict[str, Any], *, team: str | None = None) -> dict[str, Any]:
    """Flatten a team view of a recipe into the JAX trainer config shape.

    Calling this without ``team`` retains the historical behavior for legacy
    Blue recipes. A single-team Red recipe naturally selects Red; joint recipes
    select Blue for this one-policy view and should use ``project_team_configs``
    to obtain both views.
    """
    teams = training_teams(recipe)
    selected_team = team or (teams[0] if len(teams) == 1 else "blue")
    resolved = team_recipe(recipe, selected_team)
    core = resolved["core"]
    arch = resolved["arch"]
    train = resolved["train"]
    jax_ = resolved.get("jax", {})
    opponents = copy.deepcopy(train.get("opponents") or {})
    for ref in opponents.values():
        if isinstance(ref, dict):
            _validate_model_ref(ref, source="train opponent", expected_backend="jax")
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
        "TRAIN_TEAMS": teams,
        "TRAIN_TEAM": selected_team,
        "OPPONENTS": opponents,
        "TRAINING_MODE": True,
        "MLFLOW_ENABLED": True,
    }


def project_cleanrl(recipe: dict[str, Any], *, team: str | None = None) -> dict[str, Any]:
    """Flatten a team view into the dict that ippo_cyborg.py consumes."""
    teams = training_teams(recipe)
    selected_team = team or (teams[0] if len(teams) == 1 else "blue")
    resolved = team_recipe(recipe, selected_team)
    core = resolved["core"]
    arch = resolved["arch"]
    train = resolved["train"]
    cr = resolved.get("cleanrl", {})
    opponents = copy.deepcopy(train.get("opponents") or {})
    for ref in opponents.values():
        if isinstance(ref, dict):
            _validate_model_ref(ref, source="train opponent", expected_backend="cyborg")

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
        "train_teams": teams,
        "train_team": selected_team,
        "opponents": opponents,
    }


def project_team_configs(recipe: dict[str, Any], backend: str) -> dict[str, dict[str, Any]]:
    """Project one independent optimizer/policy config for every trainable team."""
    backend = _normalise_backend(backend)
    projector = project_jax if backend == "jax" else project_cleanrl
    return {team: projector(recipe, team=team) for team in training_teams(recipe)}


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
        "policy_backend": ev.get("policy_backend"),
        "policies": copy.deepcopy(ev.get("policies") or {}),
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
