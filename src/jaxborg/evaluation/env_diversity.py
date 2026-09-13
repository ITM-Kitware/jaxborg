"""Paired recipe comparison against a common pool of cross-seed learned Reds.

Only training seeds with a checkpoint in both conditions enter either pool.
For each Blue seed, both conditions face both conditions' Reds from every
other seed. Training, checkpoint selection by score, and MLflow mutation are
deliberately outside this evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from safetensors import safe_open

from jaxborg.checkpoint import read_sidecar
from jaxborg.evaluation.play_priors import _git_commit
from jaxborg.recipe import REPO_ROOT, project_eval, train_variant
from jaxborg.topology_banks import validate_topology_split

CONDITIONS = ("baseline", "diverse")
METRICS = ("reward", "c", "i", "a")


@dataclass(frozen=True)
class ComparisonModel:
    condition: str
    seed: int
    steps: int
    path: str
    sha256: str
    recipe: dict[str, Any]


def _training_topologies(recipe):
    train = recipe["train"]
    generated = train.get("topology_generation")
    if generated:
        return {k: v for k, v in generated.items() if k not in ("cache_dir", "output_dir")}
    return train.get("topology_bank")


def discover_models(recipe, condition, exp_dir, *, checkpoint_step=None, tag="*"):
    """Discover by saved recipe/provenance, never by a budget suffix in a tag."""
    name = recipe["meta"]["name"]
    root = Path(exp_dir).expanduser().resolve() / f"{recipe['algorithm']}_jax"
    pattern = "model_*.safetensors" if checkpoint_step is None else f"checkpoint_{checkpoint_step}.safetensors"
    models, notes = {}, []
    for directory in sorted(root.glob(tag)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern)):
            saved = read_sidecar(path)
            if saved.get("meta", {}).get("name") != name:
                continue
            reason = None
            if saved.get("algorithm") != recipe["algorithm"] or saved.get("train", {}).get("teams") != "both":
                reason = "not a matching cotraining recipe"
            elif _training_topologies(saved) != _training_topologies(recipe):
                reason = "training topology declaration differs from the input recipe"
            elif train_variant(saved) != train_variant(recipe):
                reason = "training game rules differ from the input recipe"
            elif checkpoint_step is None and saved["train"]["total_timesteps"] != recipe["train"]["total_timesteps"]:
                reason = "training budget differs from the input recipe"
            if reason:
                notes.append(f"Skipped {path}: {reason}")
                continue

            with safe_open(str(path), framework="np") as weights:
                manifest = json.loads((weights.metadata() or {}).get("jaxborg_bundle", "{}"))
            provenance = manifest.get("provenance", {})
            run = saved.get("run", {})
            for key in ("seed", "total_steps"):
                value = provenance.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value != run.get(key):
                    raise ValueError(f"{path}: bundle/sidecar {key} provenance is missing or inconsistent")
            if provenance.get("recipe") != name or provenance.get("train_run_id") != run.get("train_run_id"):
                raise ValueError(f"{path}: bundle/sidecar run provenance differs")
            policies = manifest.get("policies", {})
            if manifest.get("backend") != "jax" or any(
                not policies.get(team, {}).get("trainable") for team in ("blue", "red")
            ):
                raise ValueError(f"{path}: expected a JAX bundle with both teams trained")
            steps = provenance["total_steps"]
            batch = saved["jax"]["num_envs"] * saved["train"]["episode_length"]
            expected = checkpoint_step or (recipe["train"]["total_timesteps"] // batch * batch)
            if steps != expected:
                notes.append(f"Skipped {path}: contains {steps} steps, expected {expected}")
                continue
            seed = provenance["seed"]
            if seed in models:
                raise ValueError(
                    f"Ambiguous {condition} seed {seed}: {models[seed].path} and {path}. "
                    f"Use --{condition}-tag to select one run family."
                )
            models[seed] = ComparisonModel(
                condition, seed, steps, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), saved
            )
    return models, notes


def _learning_settings(recipe):
    return {
        "core": recipe.get("core"),
        "arch": recipe.get("arch"),
        "team_overrides": recipe["train"].get("team_overrides"),
        "episode_length": recipe["train"]["episode_length"],
        "jax": {k: v for k, v in recipe.get("jax", {}).items() if k != "checkpoint_every_updates"},
    }


def build_plan(
    baseline_recipe,
    diverse_recipe,
    exp_dir,
    *,
    train_seeds=None,
    seeds=(1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009),
    episodes_per_seed=1,
    deterministic=False,
    checkpoint_step=None,
    baseline_tag="*",
    diverse_tag="*",
    eval_recipe=None,
):
    """Read-only preflight: select checkpoints and enumerate paired matchups."""
    if baseline_recipe["meta"]["name"] == diverse_recipe["meta"]["name"]:
        raise ValueError("Choose two different recipe names")
    if checkpoint_step is not None and checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    if episodes_per_seed < 1 or not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive episodes_per_seed and distinct non-negative evaluation seeds")
    recipes = dict(zip(CONDITIONS, (baseline_recipe, diverse_recipe), strict=True))
    reference = copy.deepcopy(eval_recipe or baseline_recipe)
    settings = project_eval(reference)
    compare_keys = ("EVAL_VARIANT", "CIA", "TOPOLOGY_BANK", "TOPOLOGY_SAMPLING")
    if eval_recipe is None:
        other = project_eval(diverse_recipe)
        if any(settings[key] != other[key] for key in compare_keys):
            raise ValueError("Recipes must share evaluation rules, CIA settings and topology bank; use --eval-recipe")
    if (
        not settings["CIA"]["enabled"]
        or settings["CIA"]["role_assignment"] != "fixed_per_topology"
        or settings["TOPOLOGY_SAMPLING"] != "exhaustive"
        or not settings["TOPOLOGY_BANK"]
    ):
        raise ValueError("Comparison requires enabled CIA, fixed_per_topology roles and an exhaustive held-out bank")

    pools, notes = {}, []
    for condition, tag in zip(CONDITIONS, (baseline_tag, diverse_tag), strict=True):
        pools[condition], skipped = discover_models(
            recipes[condition], condition, exp_dir, checkpoint_step=checkpoint_step, tag=tag
        )
        notes.extend(skipped)
    common = set(pools["baseline"]) & set(pools["diverse"])
    selected = sorted(common if train_seeds is None else set(train_seeds))
    if not set(selected) <= common:
        raise ValueError(f"Requested training seeds lack models in both conditions: {sorted(set(selected) - common)}")
    if len(selected) < 2:
        raise ValueError(
            f"Need at least two training seeds present in both conditions; found {selected}. {'; '.join(notes)}"
        )
    for condition in CONDITIONS:
        excluded = sorted(set(pools[condition]) - set(selected))
        if excluded:
            notes.append(f"Excluded {condition} seeds {excluded} from both pools to keep the comparison paired")
    models = [pools[c][s] for c in CONDITIONS for s in selected]
    if len({m.steps for m in models}) != 1:
        raise ValueError("Selected models have different training steps; use --checkpoint-step for a common saved step")
    for model in models:
        # Check held-out provenance against the actual training recipe of every
        # selected Blue AND Red, including when --eval-recipe overrides defaults.
        candidate = copy.deepcopy(model.recipe)
        candidate["eval"] = copy.deepcopy(reference["eval"])
        if project_eval(candidate)["EVAL_VARIANT"] != settings["EVAL_VARIANT"]:
            raise ValueError(f"{model.path}: training action/reward rules conflict with the evaluation rules")
        validate_topology_split(candidate, repo_root=REPO_ROOT)
    for seed in selected:
        a, b = (pools[c][seed] for c in CONDITIONS)
        if _learning_settings(a.recipe) != _learning_settings(b.recipe):
            notes.append(
                f"Training settings differ for seed {seed} (baseline num_envs={a.recipe['jax']['num_envs']}, "
                f"diverse num_envs={b.recipe['jax']['num_envs']}); interpret as a configuration comparison. "
                "Full saved settings are recorded in the manifest."
            )
    matchups = []
    for blue_seed in selected:
        for red_condition in CONDITIONS:
            for red_seed in selected:
                if red_seed == blue_seed:
                    continue
                for blue_condition in CONDITIONS:
                    matchups.append(
                        {
                            "id": f"{blue_condition}_{blue_seed}_vs_{red_condition}_{red_seed}",
                            "blue_condition": blue_condition,
                            "blue_seed": blue_seed,
                            "red_condition": red_condition,
                            "red_seed": red_seed,
                            "blue_path": pools[blue_condition][blue_seed].path,
                            "red_path": pools[red_condition][red_seed].path,
                        }
                    )
    return {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "recipes": recipes,
        "eval_recipe": reference,
        "train_seeds": selected,
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "deterministic": deterministic,
        "episodes_per_matchup": len(settings["TOPOLOGY_BANK"]) * len(seeds) * episodes_per_seed,
        "models": [asdict(m) for m in models],
        "matchups": matchups,
        "notes": notes,
    }


def _scores(row):
    return {
        "reward": mean(row["blue_returns"]),
        **{axis: mean(ep[axis] for ep in row["per_episode_cia"]) for axis in ("c", "i", "a")},
    }


def _difference(baseline, diverse):
    return {
        "baseline": baseline,
        "diverse": diverse,
        "delta": diverse - baseline,
        "penalty_reduction_percent": 100 * (diverse - baseline) / abs(baseline) if baseline < -1e-8 else None,
    }


def summarize_comparison(plan, rows):
    """Pair identical opponents/cases, then average equally over training seeds."""
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {m["id"] for m in plan["matchups"]}:
        raise ValueError("Summary needs exactly one result for every planned matchup")
    pairs = []
    for matchup in plan["matchups"]:
        if matchup["blue_condition"] != "baseline":
            continue
        a = indexed[matchup["id"]]
        b = indexed[matchup["id"].replace("baseline_", "diverse_", 1)]
        for key in ("episode_seeds", "episode_topology_fingerprints", "episode_role_map_ids"):
            if a[key] != b[key]:
                raise ValueError(f"Unpaired evaluation cases ({key}) for {matchup['id']}")
        av, bv = _scores(a), _scores(b)
        pairs.append(
            {
                "blue_seed": matchup["blue_seed"],
                "red_condition": matchup["red_condition"],
                "red_seed": matchup["red_seed"],
                "metrics": {key: _difference(av[key], bv[key]) for key in METRICS},
            }
        )

    def aggregate(selected_pairs):
        per_seed = {}
        for seed in plan["train_seeds"]:
            subset = [p for p in selected_pairs if p["blue_seed"] == seed]
            per_seed[str(seed)] = {
                key: _difference(*(mean(p["metrics"][key][c] for p in subset) for c in CONDITIONS)) for key in METRICS
            }
        metrics = {}
        for key in METRICS:
            values = {c: [s[key][c] for s in per_seed.values()] for c in CONDITIONS}
            metrics[key] = {
                **_difference(*(mean(values[c]) for c in CONDITIONS)),
                **{f"{c}_seed_std": stdev(values[c]) for c in CONDITIONS},
            }
        return {"metrics": metrics, "per_blue_seed": per_seed}

    return {
        "metric_direction": "Higher is better. Percent = 100 * (diverse - baseline) / abs(baseline).",
        "aggregation": "Equal weight per held-out Red within a Blue seed, then equal weight per Blue training seed.",
        "uncertainty": "Seed deviations are descriptive; evaluation episodes are not independent training runs.",
        "train_seeds": plan["train_seeds"],
        "overall": aggregate(pairs),
        "by_red_condition": {c: aggregate([p for p in pairs if p["red_condition"] == c]) for c in CONDITIONS},
        "paired_matchups": pairs,
        "notes": plan["notes"],
    }


def run_comparison(plan, output_dir, *, resume=False, evaluate_fn=None):
    """Write a manifest and each matchup immediately; safely resume completed cells."""
    output = Path(output_dir).expanduser().resolve()
    canonical = json.loads(json.dumps(plan, default=str))
    manifest_path = output / "manifest.json"
    if resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != canonical:
            raise ValueError(
                "Resume requires an identical manifest, including checkpoint hashes and evaluation settings"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(canonical, indent=2) + "\n")
    settings = project_eval(plan["eval_recipe"], materialize_topologies=True)
    if evaluate_fn is None:
        from functools import partial

        from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, evaluate_matchup

        evaluate_fn = partial(evaluate_matchup, context=MatchupEvaluationContext())
    rows = []
    for index, matchup in enumerate(plan["matchups"], 1):
        path = output / f"{matchup['id']}.json"
        if resume and path.is_file():
            row = json.loads(path.read_text())
        else:
            print(f"[{index}/{len(plan['matchups'])}] {matchup['id']}", flush=True)
            result = evaluate_fn(
                matchup["blue_path"],
                matchup["red_path"],
                backend="jax",
                variant=settings["EVAL_VARIANT"],
                seeds=plan["seeds"],
                episodes_per_seed=plan["episodes_per_seed"],
                deterministic=plan["deterministic"],
                topology_path=list(settings["TOPOLOGY_BANK"]),
                topology_sampling="exhaustive",
                cia=settings["CIA"],
            )
            row = {
                **matchup,
                **{
                    key: getattr(result, key)
                    for key in (
                        "blue_returns",
                        "red_returns",
                        "episode_seeds",
                        "policies",
                        "cia_summary",
                        "per_episode_cia",
                        "episode_topology_fingerprints",
                        "episode_role_map_ids",
                        "episode_topology_paths",
                        "topology_role_maps",
                    )
                },
            }
        if any(row.get(key) != value for key, value in matchup.items()):
            raise ValueError(f"Result provenance does not match the plan: {path}")
        expected = plan["episodes_per_matchup"]
        for key in (
            "blue_returns",
            "red_returns",
            "per_episode_cia",
            "episode_seeds",
            "episode_topology_fingerprints",
            "episode_role_map_ids",
        ):
            if len(row[key]) != expected:
                raise ValueError(f"{matchup['id']}: expected {expected} entries in {key}")
        values = row["blue_returns"] + row["red_returns"] + [v for ep in row["per_episode_cia"] for v in ep.values()]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite evaluation metrics for {matchup['id']}")
        row["metrics"] = _scores(row)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2, default=str, allow_nan=False) + "\n")
            temporary.replace(path)
        rows.append(row)
        print("  " + " ".join(f"{k}={v:.3f}" for k, v in row["metrics"].items()), flush=True)
    summary = summarize_comparison(plan, rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Environment diversity: held-out learned Reds",
        "",
        "Higher is better; percentages describe penalty reduction.",
        "",
        "| Metric | Baseline | Diverse | Difference | Penalty reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in summary["overall"]["metrics"].items():
        pct = values["penalty_reduction_percent"]
        label = "n/a (baseline near zero)" if pct is None else f"{pct:.2f}%"
        lines.append(
            f"| {metric} | {values['baseline']:.3f} | {values['diverse']:.3f} | {values['delta']:.3f} | {label} |"
        )
    lines.extend(["", summary["aggregation"], "", summary["uncertainty"], ""])
    lines.extend(f"- {note}" for note in plan["notes"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
