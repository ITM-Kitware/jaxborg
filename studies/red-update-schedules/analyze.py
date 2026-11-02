"""Explicit, source-bound comparison of the three Red update recipes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import shlex
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPES = {
    "baseline": "cotraining_separate.yaml",
    "quarter": "cotraining_red_quarter_rate_separate.yaml",
    "adaptive": "cotraining_adaptive_updates_separate.yaml",
}
OPPONENTS = ("fsm", "cia_c", "cia_i", "cia_a")
METRICS = ("reward", "c", "i", "a")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(encoded(value))


def fact(record, name):
    value = record["facts"].get(name, {}).get("value")
    if value is None:
        raise ValueError(f"Missing {name}: {record['id']}")
    return value


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def prepare(args):
    """Write ordinary recipes and commands, without launching a workload."""
    import yaml

    from jaxborg.evaluation.capture_store import json_configuration
    from jaxborg.evaluation.episode_seeds import expand_episode_seeds
    from jaxborg.recipe import load

    output = args.output.resolve()
    if output.is_relative_to(HERE.parents[1]):
        raise ValueError("Evidence and prepared settings must be outside the checkout")
    if len(args.expected_revision) != 40 or any(c not in "0123456789abcdef" for c in args.expected_revision):
        raise ValueError("Preparation requires an exact lowercase source revision")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Training seeds must be non-negative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("Seed lists must contain unique identities")
    if args.mode == "substantive" and any(
        value is None
        for value in (args.steps, args.num_envs, args.topologies, args.horizon, args.resources, args.wall_time)
    ):
        raise ValueError(
            "Substantive preparation requires steps, num-envs, topologies, horizon, resources and wall-time"
        )
    steps = args.steps if args.steps is not None else 32
    envs = args.num_envs if args.num_envs is not None else 1
    count = args.topologies if args.topologies is not None else 1
    horizon = args.horizon if args.horizon is not None else 8
    if horizon > 500:
        raise ValueError("Horizon cannot exceed the existing 500-step procedure")
    output.mkdir(parents=True, exist_ok=False)
    specification = {
        "mode": args.mode,
        "source_revision": args.expected_revision,
        "training_seeds": args.seeds,
        "requested_steps": steps,
        "num_envs": envs,
        "eval_seeds": args.eval_seeds,
        "episode_seeds": expand_episode_seeds(args.eval_seeds, args.episodes_per_seed),
        "episodes_per_seed": args.episodes_per_seed,
        "topology_count": count,
        "horizon": horizon,
        "resources": args.resources or "integration only; resource allocation not specified",
        "wall_time_limit": args.wall_time or "integration only; scheduler limit not specified",
        "checkpoint_rule": "final completed checkpoint at floor(requested_steps / rollout_steps) * rollout_steps",
        "opponents": list(OPPONENTS),
        "weights": "equal episodes within topology, equal topologies within opponent, equal training seeds",
        "missingness": (
            "retain expected cells; no primary estimate unless every cell for that condition/opponent is complete"
        ),
        "uncertainty": "show each training seed and their sample SD/range; no confidence interval or paired claim",
        "recipes": {},
    }
    commands = []
    for condition, name in RECIPES.items():
        recipe = json_configuration(load(str(HERE.parents[1] / "recipes/cotraining/separate" / name)))
        recipe.pop("__source_path__", None)
        recipe["train"]["total_timesteps"] = steps
        recipe["jax"]["num_envs"] = envs
        if args.mode == "smoke":
            recipe["train"]["episode_length"] = 8
            recipe["jax"].update(num_minibatches=1, update_epochs=1, checkpoint_every_updates=2)
            if condition == "adaptive":
                recipe["train"]["adaptive_updates"].update(
                    phase_switch_blue_updates=1, window_blue_updates=1, max_frozen_blue_rollouts=1
                )
        rollout = envs * recipe["train"]["episode_length"]
        if steps < rollout:
            raise ValueError(f"Budget must include a full {rollout}-transition rollout")
        specification["rollout_steps"] = rollout
        specification["completed_steps"] = steps // rollout * rollout
        recipe["train"]["topology_generation"]["cache_dir"] = str(output / "topologies/train")
        evaluation = recipe["eval"]
        for suite in ("cross_seed_play", "cross_play", "play_priors", "checkpoint_scripted_reds", "scripted_red"):
            evaluation.pop(suite, None)
        evaluation["topology_generation"]["count"] = count
        evaluation["topology_generation"]["cache_dir"] = str(output / "topologies/eval")
        evaluation["after_training"] = [
            {
                "name": "red-update-primary",
                "script": "scripts/eval/eval_scripted_reds_jax.py",
                "args": [
                    "--recipe",
                    "{recipe}",
                    "--reds",
                    *OPPONENTS,
                    "--seeds",
                    ",".join(map(str, args.eval_seeds)),
                    "--episodes-per-seed",
                    str(args.episodes_per_seed),
                    "--horizon",
                    str(horizon),
                ],
            }
        ]
        path = output / f"{condition}.yaml"
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        specification["recipes"][condition] = recipe
        for seed in args.seeds:
            command = [
                "uv",
                "run",
                "python",
                "scripts/train/algorithms/ippo_jax.py",
                "--recipe",
                str(path),
                "--seed",
                str(seed),
                "--tag",
                f"red-update-{args.mode}-{condition}-{seed}",
                "--evidence-root",
                str(output / "evidence"),
                "--protocol",
                str(output / "protocol.json"),
                "--trial",
                f"{condition}:{seed}",
                "--expected-revision",
                args.expected_revision,
            ]
            commands.append(shlex.join(command))
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": f"red-update:{digest(encoded(specification))}",
        "study_id": "red-update-schedules",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config", "source_identity", "source_revision"], "evaluation": []},
        "trials": [
            {
                "id": f"{condition}:{seed}",
                "condition_id": condition,
                "replicate_id": f"training-{seed}",
                "matched_unit_id": None,
            }
            for condition in RECIPES
            for seed in args.seeds
        ],
        "metric_definitions": [
            {
                "name": metric,
                "unit": "score",
                "direction": "higher",
                "aggregation": specification["weights"],
                "sampling_unit": "trained policy",
                "uncertainty": specification["uncertainty"],
            }
            for metric in METRICS
        ],
        "extensions": {"red_update": {"schema_version": 1, "source": "analyze.py prepare", "values": specification}},
    }
    (output / "prepare-analyzer.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "selection-template.json",
        {
            "protocol_id": protocol["id"],
            "selected": [],
            "exploratory": False,
            "revision_of": None,
            "rationale": "Initial declared analysis",
        },
    )
    (output / "commands.sh").write_text("\n".join(commands) + "\n")
    print(output / "commands.sh")


def scientific_recipe(recipe):
    value = copy.deepcopy(recipe)
    for key in ("__source_path__", "meta", "run"):
        value.pop(key, None)
    # Materialization retains these paths separately; generation is checked from provenance.
    value["train"].pop("topology_bank", None)
    value["train"].pop("topology_generation", None)
    return value


def read_artifact(root, artifact):
    path = (root / artifact["uri"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes retained store")
    data = path.read_bytes()
    if len(data) != artifact["size_bytes"] or digest(data) != artifact["sha256"]:
        raise ValueError(f"Artifact bytes differ: {artifact['id']}")
    return data


def check_primary_evaluation(root, index, evaluation, row, spec, condition):
    """Bind the declared suite to the effective settings and retained input bank."""
    from dataclasses import replace

    from jaxborg.evaluation.capture_store import json_configuration
    from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint
    from jaxborg.recipe import eval_variant
    from jaxborg.scenarios.cc4.topology import load_topology, load_topology_metadata
    from jaxborg.topology_banks import parse_topology_generation

    declared = spec["recipes"][condition]
    config = fact(evaluation, "resolved_config")
    expected_variant = json_configuration(replace(eval_variant(declared), num_steps=spec["horizon"]))
    if (
        scientific_recipe(config["recipe"]) != scientific_recipe(declared)
        or config["base_variant"] != expected_variant
        or config["seeds"] != spec["eval_seeds"]
        or config["episodes_per_seed"] != spec["episodes_per_seed"]
        or config["deterministic"] is not False
        or config["reds"] != list(OPPONENTS)
    ):
        raise ValueError("Effective primary evaluation settings differ from declaration")
    generated = parse_topology_generation(declared["eval"], scope="eval", repo_root=HERE.parents[1])
    roles = fact(evaluation, "input_artifacts")
    if generated is None or {name for name in roles if name.startswith("topology ")} != {
        f"topology {i}" for i in range(len(generated.seeds))
    }:
        raise ValueError("Retained evaluation topology bank differs from declared generator suite")
    topologies = []
    for i, seed in enumerate(generated.seeds):
        artifact = index[roles[f"topology {i}"]]
        if artifact["id"] not in evaluation["inputs"]:
            raise ValueError("Evaluation topology is not a retained input")
        read_artifact(root, artifact)
        path = root / artifact["uri"]
        metadata = load_topology_metadata(path)
        if (metadata.get("source"), metadata.get("source_seed"), metadata.get("op_zone_servers")) != (
            generated.source,
            seed,
            generated.op_zone_servers,
        ):
            raise ValueError("Retained evaluation topology provenance differs from declared generator suite")
        topologies.append(
            {
                "artifact": artifact,
                "metadata": metadata,
                "fingerprint": canonical_topology_fingerprint(load_topology(path)),
            }
        )
    case_count = len(spec["episode_seeds"])
    expected_fingerprints = [topology["fingerprint"] for topology in topologies for _ in range(case_count)]
    if row["per_episode_topology_fingerprints"] != expected_fingerprints:
        raise ValueError("Episode topology fingerprints differ from retained evaluation inputs")
    return topologies


def collect(root, selection):
    """Resolve only explicit IDs against the verified existing capture store."""
    from jaxborg.evaluation.capture_store import load_records

    records = load_records(root)
    index = {record["id"]: record for record in records}
    protocol = index[selection["protocol_id"]]
    spec = protocol["extensions"]["red_update"]["values"]
    if protocol["id"] != f"red-update:{digest(encoded(spec))}":
        raise ValueError("Protocol identity differs from its declared study settings")
    expected = {(condition, seed) for condition in RECIPES for seed in spec["training_seeds"]}
    selected, used = [], set()
    training_contract = None
    inventory_ids = set()
    for entry in selection["selected"]:
        cell = (entry["condition"], entry["seed"])
        if cell not in expected or cell in used:
            raise ValueError("Unknown or duplicate training cell")
        used.add(cell)
        attempt = index[entry["attempt_id"]]
        if (
            attempt["execution_kind"] != "training"
            or attempt["protocol_id"] != protocol["id"]
            or attempt["trial_id"] != f"{cell[0]}:{cell[1]}"
        ):
            raise ValueError("Training identity does not match declared trial")
        if fact(attempt, "source_revision") != spec["source_revision"]:
            raise ValueError("Training source revision differs from declared source pin")
        inventory_ids.add(attempt["id"])
        config = fact(attempt, "resolved_config")
        if config["seed_roles"]["environment_reset_root"] != cell[1]:
            raise ValueError("Training seed differs from selected replicate")
        if scientific_recipe(config["recipe"]) != scientific_recipe(spec["recipes"][cell[0]]):
            raise ValueError("Training recipe differs from the declared condition")
        contract = {
            name: fact(attempt, name)
            for name in ("source_identity", "source_revision", "policy_contract", "game_contract", "topology")
        }
        if training_contract is not None and contract != training_contract:
            raise ValueError("Training source, topology or policy/game contract differs between cells")
        training_contract = contract
        checkpoint = index[entry["checkpoint_id"]]
        final = fact(attempt, f"checkpoint_{spec['completed_steps']}")
        if checkpoint["producer_id"] != attempt["id"] or final["model"] != checkpoint["id"]:
            raise ValueError("Select the declared final training-owned checkpoint")
        if attempt["status"] != "succeeded":
            raise ValueError("Failed training remains missing; do not select a partial policy as final")
        evals = []
        for pick in entry["evaluations"]:
            artifact = index[pick["artifact_id"]]
            evaluation = index[artifact["producer_id"]]
            if (
                evaluation["kind"] != "evaluation"
                or evaluation["status"] != "succeeded"
                or checkpoint["id"] not in evaluation["inputs"]
            ):
                raise ValueError("Evaluation must consume the selected training-owned checkpoint")
            if fact(evaluation, "source_revision") != fact(attempt, "source_revision"):
                raise ValueError("Training and evaluation source revisions differ")
            train_sources, eval_sources = (fact(record, "source_files") for record in (attempt, evaluation))
            overlap = train_sources.keys() & eval_sources.keys()
            if not overlap or any(train_sources[name] != eval_sources[name] for name in overlap):
                raise ValueError("Shared training and evaluation source bytes differ")
            if isinstance(evaluation.get("attempt_id"), str):
                inventory_ids.add(evaluation["attempt_id"])
            rows = [json.loads(line) for line in read_artifact(root, artifact).splitlines() if line.strip()]
            matches = [row for row in rows if row["eval_id"] == pick["eval_id"]]
            if len(matches) != 1:
                raise ValueError("Evaluation ID is missing or ambiguous in selected artifact")
            row = matches[0]
            if row["evaluation_id"] != evaluation["id"]:
                raise ValueError("Evaluation row identity differs from its producing record")
            if (
                row["training_attempt_id"] != attempt["id"]
                or row["training_checkpoint_artifact_id"] != checkpoint["id"]
            ):
                raise ValueError("Evaluation row training ancestry differs")
            if row["model_sha256"] != checkpoint["sha256"]:
                raise ValueError("Evaluation did not consume the selected checkpoint bytes")
            topologies = check_primary_evaluation(root, index, evaluation, row, spec, cell[0])
            evals.append({"artifact": artifact, "record": evaluation, "row": row, "topologies": topologies})
        updates = [
            json.loads(read_artifact(root, record))
            for record in records
            if record["kind"] == "artifact"
            and record["producer_id"] == attempt["id"]
            and record["role"] == "training update observation"
        ]
        selected.append(
            {
                "condition": cell[0],
                "seed": cell[1],
                "attempt": attempt,
                "checkpoint": checkpoint,
                "evaluations": evals,
                "updates": sorted(updates, key=lambda row: row["update_idx"]),
            }
        )
    pending = list(inventory_ids)
    while pending:
        ancestor = index[pending.pop()].get("retry_of")
        if isinstance(ancestor, str) and ancestor not in inventory_ids:
            inventory_ids.add(ancestor)
            pending.append(ancestor)
    return {
        "protocol": protocol,
        "selection": selection,
        "selected": selected,
        "attempt_inventory": [
            {key: record[key] for key in ("id", "trial_id", "status", "retry_of", "budgets")}
            for record in sorted((index[identity] for identity in inventory_ids), key=lambda record: record["id"])
        ],
    }


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Metrics must be finite numbers")
    return value


def tables(inputs):
    """Study-local estimators; raw observations remain available for revisions."""
    spec = inputs["protocol"]["extensions"]["red_update"]["values"]
    episodes, cases, seeds, costs, updates, missing = [], [], [], [], [], []
    available = {}
    common_contract = None
    common_cases = None
    for item in inputs["selected"]:
        condition, seed = item["condition"], item["seed"]
        attempt = item["attempt"]
        identity = {"condition": condition, "training_seed": seed, "attempt_id": attempt["id"]}
        budgets = {budget["unit"]: budget["completed"] for budget in attempt["budgets"]}
        cost = attempt["facts"].get("training_cost", {}).get("value", {})
        costs.append(
            {
                **identity,
                **budgets,
                **cost,
                "evaluation_wall_seconds": sum(finite(e["row"]["wall_time_s"]) for e in item["evaluations"]),
                "evaluation_cost_scope": "selected opponent rows only; excludes process startup and artifact delivery",
            }
        )
        updates.extend({**identity, **row} for row in item["updates"])
        seen = set()
        for evaluation in item["evaluations"]:
            row = evaluation["row"]
            opponent = row["eval_red"]
            if opponent not in OPPONENTS or opponent in seen:
                raise ValueError("Unknown or duplicate opponent cell")
            seen.add(opponent)
            if row["episode_horizon"] != spec["horizon"]:
                raise ValueError("Evaluation horizon differs from declared protocol")
            if row.get("status", "succeeded") != "succeeded":
                raise ValueError("Select complete evaluation rows only")
            contract = {
                key: row[key]
                for key in (
                    "eval_env",
                    "policy_backend",
                    "base_variant",
                    "cia_metric",
                    "cia_config",
                    "stochastic",
                    "blue_busy_action_masking",
                    "topology_sampling",
                )
            }
            if (
                contract["eval_env"] != "jax_fsm"
                or contract["policy_backend"] != "jax"
                or not contract["stochastic"]
                or contract["cia_metric"] != "resilience"
                or contract["topology_sampling"] != "exhaustive"
            ):
                raise ValueError("Evaluation is not the declared stochastic JAX resilience suite")
            if common_contract is not None and contract != common_contract:
                raise ValueError("Evaluation procedure differs between selected cells")
            common_contract = contract
            roles = {(entry["topology_fingerprint"], entry["role_map_id"]) for entry in row["topology_role_maps"]}
            columns = [
                row[name]
                for name in (
                    "per_episode",
                    "per_episode_cia",
                    "per_episode_seeds",
                    "per_episode_topology_fingerprints",
                    "episode_role_map_ids",
                )
            ]
            expected_count = spec["topology_count"] * len(spec["eval_seeds"]) * spec["episodes_per_seed"]
            if any(len(column) != expected_count for column in columns) or row["n_episodes"] != expected_count:
                raise ValueError("Incomplete or inconsistent episode columns")
            local = defaultdict(list)
            case_ids = []
            occurrence = defaultdict(int)
            for reward, cia, episode_seed, topology, role in zip(*columns):
                if not topology or not role or (topology, role) not in roles:
                    raise ValueError("Each episode needs topology and role identity")
                key = (topology, role, episode_seed)
                repeat = occurrence[key]
                occurrence[key] += 1
                case_ids.append((*key, repeat))
                ep = {
                    **identity,
                    "opponent": opponent,
                    "eval_id": row["eval_id"],
                    "topology_fingerprint": topology,
                    "role_map_id": role,
                    "episode_seed": episode_seed,
                    "repeat": repeat,
                    "reward": finite(reward),
                    **{metric: finite(cia[metric]) for metric in ("c", "i", "a")},
                }
                episodes.append(ep)
                local[(topology, role)].append(ep)
            if len(local) != spec["topology_count"] or any(
                sorted(ep["episode_seed"] for ep in values) != sorted(spec["episode_seeds"])
                for values in local.values()
            ):
                raise ValueError("Evaluation topology/seed expansion differs from protocol")
            if common_cases is not None and sorted(case_ids) != common_cases:
                raise ValueError("Topologies, role assignments or episode cases differ between cells")
            common_cases = sorted(case_ids)
            per_case = []
            for (topology, role), values in sorted(local.items()):
                case = {
                    **identity,
                    "opponent": opponent,
                    "topology_fingerprint": topology,
                    "role_map_id": role,
                    "n_episodes": len(values),
                    **{metric: statistics.mean(ep[metric] for ep in values) for metric in METRICS},
                }
                cases.append(case)
                per_case.append(case)
            summary = {
                **identity,
                "opponent": opponent,
                "n_topologies": len(per_case),
                "n_episodes": expected_count,
                **{metric: statistics.mean(case[metric] for case in per_case) for metric in METRICS},
                **{
                    f"{metric}_episode_sd_conditional": statistics.stdev(
                        ep[metric] for values in local.values() for ep in values
                    )
                    if expected_count > 1
                    else None
                    for metric in METRICS
                },
            }
            seeds.append(summary)
            available[(condition, seed, opponent)] = summary
    comparison = []
    for condition in RECIPES:
        for opponent in OPPONENTS:
            selected = [
                available[(condition, seed, opponent)]
                for seed in spec["training_seeds"]
                if (condition, seed, opponent) in available
            ]
            complete = len(selected) == len(spec["training_seeds"])
            for seed in spec["training_seeds"]:
                if (condition, seed, opponent) not in available:
                    missing.append(
                        {
                            "condition": condition,
                            "training_seed": seed,
                            "opponent": opponent,
                            "reason": "no explicit complete selection",
                        }
                    )
            result = {
                "condition": condition,
                "opponent": opponent,
                "complete": complete,
                "n_training_seeds": len(selected),
                "expected_training_seeds": len(spec["training_seeds"]),
            }
            for metric in METRICS:
                values = [row[metric] for row in selected]
                result.update(
                    {
                        f"{metric}_mean": statistics.mean(values) if complete else None,
                        f"{metric}_training_seed_sd": statistics.stdev(values)
                        if complete and len(values) > 1
                        else None,
                        f"{metric}_min": min(values) if values else None,
                        f"{metric}_max": max(values) if values else None,
                    }
                )
            comparison.append(result)
    baseline = {row["opponent"]: row for row in comparison if row["condition"] == "baseline"}
    for row in comparison:
        for metric in METRICS:
            value, reference = row[f"{metric}_mean"], baseline[row["opponent"]][f"{metric}_mean"]
            row[f"{metric}_difference_from_baseline"] = (
                value - reference if value is not None and reference is not None else None
            )
    return {
        "episodes": episodes,
        "cases": cases,
        "training-seeds": seeds,
        "comparison": comparison,
        "costs": costs,
        "training-updates": updates,
        "missing": missing,
    }


def csv_table(path, rows):
    fields = sorted({key for row in rows for key in row}) or ["no_rows"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def render(inputs, output):
    if output.resolve().is_relative_to(HERE.parents[1]) and (HERE.parents[1] / "pyproject.toml").exists():
        raise ValueError("Analysis output must be outside the checkout")
    inputs = copy.deepcopy(inputs)
    inputs.setdefault(
        "analysis_environment", {"python": platform.python_version(), "dependencies": "Python standard library"}
    )
    outputs = tables(inputs)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "analysis-inputs.json", inputs)
    (output / "analyze.py").write_bytes(Path(__file__).read_bytes())
    for name, rows in outputs.items():
        csv_table(output / f"{name}.csv", rows)
    selection = inputs["selection"]
    if selection.get("revision_of") and (not selection.get("exploratory") or not selection.get("rationale")):
        raise ValueError("Revisions require exploratory=true and a rationale; preserve the prior bundle")
    write_json(output / "selection.json", selection)
    spec = inputs["protocol"]["extensions"]["red_update"]["values"]
    evidence_status = (
        "integration check only; no learned-performance conclusion"
        if spec["mode"] == "smoke"
        else "prospective pilot; exploratory configuration comparison"
    )
    report = f"""# Red update schedules: {spec["mode"]} comparison

Evidence status: {evidence_status}.

Selection: `selection.json`; exact protocol, retained record facts and raw observations:
`analysis-inputs.json`. Analyzer: `analyze.py`. Byte identities: `manifest.json`.
Revision of: {selection.get("revision_of") or "initial comparison"}. Rationale: {selection.get("rationale", "")}.

`comparison.csv` reports each opponent separately. Primary means require all declared
training seeds. `missing.csv` lists {len(outputs["missing"])} absent cells. `training-seeds.csv`
shows every selected policy, and conditional episode SD is separate from training-seed SD.
`cases.csv` gives topology means; `episodes.csv` retains the exact episode observations.
Weights: {spec["weights"]}. Opponents are never pooled. Differences from baseline
are descriptive differences of means, without a paired-training interpretation. Matching seed numbers do not
establish paired training replicates. With two seeds, ranges and SD are descriptive;
rankings remain exploratory and evaluation episodes do not add training replication.

Adaptive is a combined schedule and optimizer treatment, including Blue LR 1e-4 and
no annealing. It does not isolate the adaptive gate. Training scheduler observations
are in `training-updates.csv`; counts are PPO rounds, not optimizer minibatch steps.

`costs.csv` reports actual completed transitions, team PPO rounds, retained training
timer scope and selected evaluation seconds alongside performance. JAX device counts
are not scheduler accelerator allocation. Accelerator-hours and queue delay are unknown
unless independently retained in scheduler evidence. Reduced Red rounds alone do not
establish runtime savings. Compare timing only with matching hardware and scope.

Decision: inspect per-seed performance and cost together before choosing a larger study.
This generated report makes no superiority claim. Missing cells or incomparable cost
can leave that decision unresolved.
"""
    (output / "report.md").write_text(report)
    manifest = {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())}
    write_json(output / "manifest.json", manifest)
    print(output)


def rebuild(bundle, output):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or digest((bundle / name).read_bytes()) != expected:
            raise ValueError(f"Bundle byte identity differs: {name}")
    if digest(Path(__file__).read_bytes()) != manifest["analyze.py"]:
        raise ValueError("Reconstruction requires the exact retained analyzer source")
    render(json.loads((bundle / "analysis-inputs.json").read_text()), output)
    for name in manifest:
        if digest((output / name).read_bytes()) != manifest[name]:
            raise ValueError(f"Reconstructed output differs: {name}")


def inventory(root):
    """Display retained IDs without choosing attempts or evaluations."""
    from jaxborg.evaluation.capture_store import load_records

    records = load_records(root)
    values = []
    for record in records:
        if record["kind"] == "attempt":
            values.append({key: record[key] for key in ("id", "trial_id", "status", "retry_of", "execution_kind")})
            values[-1]["checkpoints"] = {
                name: value["value"] for name, value in record["facts"].items() if name.startswith("checkpoint_")
            }
        elif record["kind"] == "artifact" and record["role"] == "scripted opponent episode results":
            for line in read_artifact(root, record).splitlines():
                row = json.loads(line)
                values.append(
                    {
                        "artifact_id": record["id"],
                        **{
                            key: row[key]
                            for key in ("eval_id", "eval_red", "training_attempt_id", "training_checkpoint_artifact_id")
                        },
                    }
                )
    print(json.dumps(values, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--mode", choices=("smoke", "substantive"), required=True)
    prep.add_argument("--expected-revision", required=True)
    prep.add_argument("--seeds", type=int, nargs="+", required=True)
    prep.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    prep.add_argument("--steps", type=positive)
    prep.add_argument("--num-envs", type=positive)
    prep.add_argument("--topologies", type=positive)
    prep.add_argument("--horizon", type=positive)
    prep.add_argument("--episodes-per-seed", type=positive, default=1)
    prep.add_argument("--resources")
    prep.add_argument("--wall-time")
    inspect = commands.add_parser("inventory")
    inspect.add_argument("store", type=Path)
    analysis = commands.add_parser("compare")
    analysis.add_argument("store", type=Path)
    analysis.add_argument("selection", type=Path)
    analysis.add_argument("output", type=Path)
    reconstruction = commands.add_parser("rebuild")
    reconstruction.add_argument("bundle", type=Path)
    reconstruction.add_argument("output", type=Path)
    revision = commands.add_parser("revise")
    revision.add_argument("bundle", type=Path)
    revision.add_argument("output", type=Path)
    revision.add_argument("--rationale", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "inventory":
        inventory(args.store)
    elif args.command == "compare":
        render(collect(args.store, json.loads(args.selection.read_text())), args.output)
    elif args.command == "rebuild":
        rebuild(args.bundle, args.output)
    else:
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        for name, expected in manifest.items():
            if Path(name).name != name or digest((args.bundle / name).read_bytes()) != expected:
                raise ValueError(f"Bundle byte identity differs: {name}")
        inputs = json.loads((args.bundle / "analysis-inputs.json").read_text())
        inputs.pop("analysis_environment", None)
        inputs["selection"].update(exploratory=True, revision_of=digest(encoded(manifest)), rationale=args.rationale)
        render(inputs, args.output)


if __name__ == "__main__":
    main()
