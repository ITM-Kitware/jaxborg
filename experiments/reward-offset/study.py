"""Four supported payoff/blocking conditions and their scientific checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parents[1])]

CONDITIONS = {
    "baseline": {"offset": 0.0, "blue_block_policy": "cc4"},
    "offset": {"offset": 1.0, "blue_block_policy": "cc4"},
    "blocking": {"offset": 0.0, "blue_block_policy": "mission_safe"},
    "combined": {"offset": 1.0, "blue_block_policy": "mission_safe"},
}


def sources():
    return {f"experiments/reward-offset/{name}": HERE / name for name in ("study.py", "rules.py", "recipe.yaml")}


def bind_condition(condition_id, recipe_path=HERE / "recipe.yaml"):
    """Bind one existing recipe and the explicit argument to the local rule."""
    from jaxborg.recipe import load

    spec = importlib.util.spec_from_file_location("reward_offset_rules", HERE / "rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    settings = CONDITIONS[condition_id]
    recipe = copy.deepcopy(load(str(recipe_path)))
    recipe["train"]["variant_overrides"] = {
        "red_reward": "zero_sum",
        "blue_block_policy": settings["blue_block_policy"],
    }
    return recipe, partial(rules.build_env, offset=settings["offset"]), partial(check_condition, condition_id)


def check_condition(condition_id, resolved_config):
    """Reject a mislabeled condition against the final execution configuration."""
    settings = CONDITIONS[condition_id]
    variant, builder = resolved_config["variant"], resolved_config["builder"]
    if variant["red_reward"] != "zero_sum" or variant["blue_block_policy"] != settings["blue_block_policy"]:
        raise ValueError(f"Resolved game differs from condition {condition_id}")
    if (
        builder["module"] != "reward_offset_rules"
        or builder["function"] != "build_env"
        or builder["arguments"] != {"offset": settings["offset"]}
    ):
        raise ValueError(f"Resolved builder differs from condition {condition_id}")
    expected_hash = hashlib.sha256((HERE / "rules.py").read_bytes()).hexdigest()
    if builder["source_hashes"].get("experiments/reward-offset/rules.py") != expected_hash:
        raise ValueError("Resolved rule source differs from the maintained study")
    if variant.get("resilience_roles"):
        raise ValueError("This study does not support resilience-role variants")


def protocol():
    trials = [
        {
            "id": f"{condition}:{procedure}:1",
            "condition_id": condition,
            "replicate_id": "training-1" if procedure == "train" else "evaluation-cohort-1",
            "matched_unit_id": "initialization-and-cohort-1",
        }
        for condition in CONDITIONS
        for procedure in ("train", "evaluate")
    ]
    trials.append(
        {
            "id": "baseline:evaluate:unlaunched",
            "condition_id": "baseline",
            "replicate_id": "unlaunched",
            "matched_unit_id": None,
        }
    )
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": "payoff-blocking:protocol:1",
        "study_id": "payoff-blocking",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {
            "attempt": ["resolved_config", "source_identity", "source_revision", "source_files"],
            "evaluation": [
                "resolved_config",
                "source_identity",
                "source_revision",
                "source_files",
                "policies",
                "policy_contract",
                "topology",
                "seed_roles",
            ],
        },
        "trials": trials,
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policies",
                "uncertainty": "none",
            }
        ],
    }


def report_configuration():
    from scripts.eval.plot_results import REPORT_CONFIGURATION

    return {
        **REPORT_CONFIGURATION,
        "study": "payoff-blocking",
        "conditions": CONDITIONS,
        "intentional_differences": ["builder.arguments.offset", "variant.blue_block_policy", "trained policy weights"],
        "procedure_differences": {"training_topology_seed": 0, "evaluation_topology_seed": 1},
        "interpretation": "bounded construction demonstration; no independent training replication or effect claim",
    }


def _fact(record, key):
    value = record["facts"].get(key, {}).get("value")
    if value is None:
        raise ValueError(f"Missing known {key} on {record['id']}")
    return value


def _recipe_invariants(recipe):
    value = copy.deepcopy(recipe)
    for name in ("meta", "run", "__source_path__"):
        value.pop(name, None)
    value["train"]["variant_overrides"].pop("blue_block_policy")
    return value


def _team_invariants(configs):
    value = copy.deepcopy(configs)
    for config in value.values():
        for name in ("TRAIN_VARIANT", "EVAL_VARIANT"):
            if name in config:
                config[name].pop("blue_block_policy")
    return value


def check_comparison(collation, rows):
    """Check declared differences, common contracts, and exact policy ancestry."""
    index = {record["id"]: record for record in collation["records"]}
    if index[collation["protocol_id"]] != protocol():
        raise ValueError("Study protocol differs, including its unlaunched cell")
    if {row["condition_id"] for row in rows} != set(CONDITIONS) or len(rows) != 4:
        raise ValueError("Select exactly one evaluation for each declared condition")
    common = []
    for row in rows:
        condition = row["condition_id"]
        if row["trial_id"] != f"{condition}:evaluate:1":
            raise ValueError("Selected evaluation is not the declared comparison cell")
        evaluation = index[row["evaluation_id"]]
        training_id = collation["selected_attempts"].get(f"{condition}:train:1")
        if training_id is None:
            raise ValueError("Each condition needs an explicitly selected training attempt")
        training = index[training_id]
        if training["execution_kind"] != "training":
            raise ValueError("Selected training cell is not a training execution")
        train_config, config = (_fact(record, "resolved_config") for record in (training, evaluation))
        check_condition(condition, train_config)
        check_condition(condition, config)
        if _fact(training, "source_revision") != _fact(evaluation, "source_revision"):
            raise ValueError("Training/evaluation source revisions differ")
        training_sources, evaluation_sources = (_fact(record, "source_files") for record in (training, evaluation))
        overlap = training_sources.keys() & evaluation_sources.keys()
        if not overlap or any(training_sources[name] != evaluation_sources[name] for name in overlap):
            raise ValueError("Training/evaluation shared source files differ")
        for record in (training, evaluation):
            execution = _fact(record, "resolved_config")["execution"]
            if any(_fact(record, key) != execution[key] for key in ("dependencies", "runtime")):
                raise ValueError("Execution environment facts differ from retained configuration")
            topology = _fact(record, "topology")
            hashes = topology.get("sha256", [])
            if not hashes or any(
                not any(index[identity]["sha256"] == digest for identity in record["inputs"]) for digest in hashes
            ):
                raise ValueError("Study topology must reference exact retained snapshot inputs")
        policies = _fact(evaluation, "policies")
        if set(policies) != {"blue", "red"}:
            raise ValueError("Both exact policy inputs are required")
        for artifact_id in policies.values():
            artifact = index[artifact_id]
            if artifact["producer_id"] != training_id or artifact_id not in evaluation["inputs"]:
                raise ValueError("Policy is not an exact input owned by the selected training attempt")
        episodes = row["episodes"]
        if any(
            isinstance(ep.get("steps"), bool) or not isinstance(ep.get("steps"), int) or ep["steps"] <= 0
            for ep in episodes
        ):
            raise ValueError("Actual completed steps are required to separate payoff from original score")
        offset = CONDITIONS[condition]["offset"]
        original_scores = [ep["reward"] - offset * ep["steps"] for ep in episodes]
        row.update(
            offset=offset,
            blue_block_policy=CONDITIONS[condition]["blue_block_policy"],
            original_score_mean=statistics.mean(original_scores),
            original_score_episode_sd=statistics.stdev(original_scores) if len(original_scores) > 1 else None,
        )
        # Only the declared blocking rule and payoff offset may differ in these configurations.
        train_variant, eval_variant = (dict(c["variant"]) for c in (train_config, config))
        for variant in (train_variant, eval_variant):
            variant.pop("blue_block_policy")
        common.append(
            {
                "train_variant": train_variant,
                "eval_variant": eval_variant,
                "training_recipe": _recipe_invariants(train_config["recipe"]),
                "team_configs": _team_invariants(train_config["team_configs"]),
                "initialization": train_config["initialization"],
                "evaluation_recipe": _recipe_invariants(config["recipe"]),
                "deterministic": config["deterministic"],
                "horizon": config["horizon"],
                "episodes_per_seed": config["episodes_per_seed"],
                "training_source_identity": _fact(training, "source_identity"),
                "evaluation_source_identity": _fact(evaluation, "source_identity"),
                "source_revision": _fact(training, "source_revision"),
                "training_dependencies": _fact(training, "dependencies"),
                "evaluation_dependencies": _fact(evaluation, "dependencies"),
                "training_runtime": _fact(training, "runtime"),
                "evaluation_runtime": _fact(evaluation, "runtime"),
                "policy_contract": _fact(evaluation, "policy_contract"),
                "training_topology": _fact(training, "topology"),
                "topology": _fact(evaluation, "topology"),
                "train_seed_roles": _fact(training, "seed_roles"),
                "seed_roles": _fact(evaluation, "seed_roles"),
                "episode_seeds": [ep["seed"] for ep in episodes],
                "steps": [ep["steps"] for ep in episodes],
                "train_budgets": training["budgets"],
                "evaluation_budgets": evaluation["budgets"],
                "backend": _fact(evaluation, "backend"),
                "eval_env": _fact(evaluation, "eval_env"),
            }
        )
    if any(value != common[0] for value in common[1:]):
        raise ValueError(
            "Undeclared difference in sources, runtime, dependencies, budgets, seeds, "
            "backend, topology, or policy/game contracts"
        )


def topology_declarations(output):
    """Use the existing recipe topology generator with disjoint fixed seed banks."""
    return {
        scope: {
            "topology_generation": {
                "generator": "jax",
                "seed_start": seed,
                "count": 1,
                "cache_dir": str(Path(output) / "topologies" / scope),
            }
        }
        for scope, seed in (("train", 0), ("eval", 1))
    }


def _write_receipt(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _wait_condition(command):
    """Keep cancellation scoped to this coordinator's one child process."""
    import signal
    import subprocess

    child = subprocess.Popen(command, start_new_session=True)
    try:
        returncode = child.wait()
    except KeyboardInterrupt:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            child.wait()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _verify_inputs(recipe_sha256, train_topology, eval_topology):
    for role, (name, digest) in (
        ("recipe", (HERE / "recipe.yaml", recipe_sha256)),
        ("training topology", train_topology),
        ("evaluation topology", eval_topology),
    ):
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Supplied {role} bytes differ from the expected identity")


def run_condition(
    output,
    condition,
    *,
    expected_revision,
    recipe_sha256,
    train_topology,
    eval_topology,
    seed=42,
    evaluation_seeds=(17, 18),
    horizon=2,
    allow_dirty=False,
):
    """Execute one condition against the exact supplied topology snapshots."""
    import os
    from dataclasses import replace
    from types import SimpleNamespace

    from jaxborg.evaluation.capture_store import load_records
    from jaxborg.evaluation.joint_capture import run_matchup
    from jaxborg.recipe import eval_variant

    output = Path(output).resolve()
    receipt = output / f"{condition}.result.json"
    if condition not in CONDITIONS or receipt.exists():
        raise ValueError("Condition must be declared and have no existing result receipt")
    if not 1 <= horizon <= 500:
        raise ValueError("Evaluation horizon must be between 1 and 500")
    _verify_inputs(recipe_sha256, train_topology, eval_topology)
    banks = {scope: [Path(item[0]).resolve()] for scope, item in (("train", train_topology), ("eval", eval_topology))}
    declarations = topology_declarations(output)
    root, declared = output / "evidence", protocol()
    if any(
        record.get("trial_id") in {f"{condition}:train:1", f"{condition}:evaluate:1"}
        for record in load_records(root)
        if record["kind"] == "attempt"
    ):
        raise ValueError("Condition already has attempts; retries require explicit capture and selection")
    os.environ["JAXBORG_EXP_DIR"] = str(output / "runtime")
    from scripts.train.algorithms.ippo_jax import _run_joint_training

    recipe, builder, check = bind_condition(condition)
    for scope, paths in banks.items():
        recipe[scope]["topology_bank"] = [str(path) for path in paths]
    training_trial = f"{condition}:train:1"
    _verify_inputs(recipe_sha256, train_topology, eval_topology)
    training = _run_joint_training(
        SimpleNamespace(seed=seed),
        recipe,
        condition,
        output / "runtime" / condition,
        env_factory=builder,
        recording={
            "root": root,
            "protocol": declared,
            "trial_id": training_trial,
            "expected_revision": expected_revision,
            "source_roots": (HERE,),
            "condition_check": check,
            "allow_dirty": allow_dirty,
            "entrypoint": Path(__file__),
            "supplied_recipe": load_recipe_snapshot(),
            "overrides": {
                "train.variant_overrides": recipe["train"]["variant_overrides"],
                "builder.arguments": builder.keywords,
                "topology_generation": declarations,
            },
        },
    )
    records = {record["id"]: record for record in load_records(root)}
    checkpoint = _fact(records[training.attempt_id], f"checkpoint_{recipe['train']['total_timesteps']}")
    model_id, sidecar_id = checkpoint["model"], checkpoint["sidecar"]
    model = root / records[model_id]["uri"]
    evaluation_trial = f"{condition}:evaluate:1"
    _verify_inputs(recipe_sha256, train_topology, eval_topology)
    evaluation, _ = run_matchup(
        model,
        model,
        root=root,
        protocol=declared,
        trial_id=evaluation_trial,
        expected_revision=expected_revision,
        variant=replace(eval_variant(recipe), num_steps=horizon),
        seeds=list(evaluation_seeds),
        topology_path=banks["eval"],
        env_factory=builder,
        source_roots=(HERE,),
        condition_check=check,
        input_artifacts={
            f"{team} {kind}": identity
            for team in ("blue", "red")
            for kind, identity in (("checkpoint", model_id), ("sidecar", sidecar_id))
        },
        allow_dirty=allow_dirty,
        recipe=recipe,
        supplied_recipe=load_recipe_snapshot(),
        overrides={
            "horizon": horizon,
            "seeds": list(evaluation_seeds),
            "topology_generation": declarations,
        },
        entrypoint=Path(__file__),
    )
    _verify_inputs(recipe_sha256, train_topology, eval_topology)
    result = {
        "selected_attempts": {training_trial: training.attempt_id, evaluation_trial: evaluation.attempt_id},
        "selected_evaluations": [evaluation.evaluation_id],
    }
    _write_receipt(receipt, result)
    return receipt


def run_study(output, *, expected_revision, seed=42, evaluation_seeds=(17, 18), horizon=2, allow_dirty=False):
    """Run each condition in a fresh process, sharing verified prepared banks."""
    from jaxborg.evaluation.capture_store import declare_protocol, load_records
    from jaxborg.evaluation.provenance import collect_provenance
    from jaxborg.experiment_records import collate
    from jaxborg.topology_banks import materialize_topology_bank, validate_topology_split

    output = Path(output).expanduser().resolve()
    if output.is_relative_to(HERE.parents[1]):
        raise ValueError("Study output must be outside the checkout")
    if not 1 <= horizon <= 500:
        raise ValueError("Evaluation horizon must be between 1 and 500")
    provenance = collect_provenance(
        expected_revision=expected_revision,
        allow_dirty=allow_dirty,
        entrypoint=Path(__file__),
        source_roots=(HERE, HERE.parents[1] / "scripts/train/algorithms"),
    )
    recipe_sha256 = hashlib.sha256((HERE / "recipe.yaml").read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    declarations = topology_declarations(output)
    validate_topology_split(declarations, repo_root=HERE.parents[1])
    banks = {
        scope: materialize_topology_bank(section, scope=scope, repo_root=HERE.parents[1])
        for scope, section in declarations.items()
    }
    provenance.verify_sources()
    root, declared = output / "evidence", protocol()
    declare_protocol(root, declared)
    topologies = {
        scope: (str(paths[0]), hashlib.sha256(paths[0].read_bytes()).hexdigest()) for scope, paths in banks.items()
    }
    topology_arguments = [argument for scope, item in topologies.items() for argument in (f"--{scope}-topology", *item)]
    selected_attempts, selected_evaluations = {}, []
    for condition in CONDITIONS:
        provenance.verify_sources()
        _verify_inputs(recipe_sha256, topologies["train"], topologies["eval"])
        _wait_condition(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "condition",
                "--output",
                str(output),
                "--expected-revision",
                expected_revision,
                "--recipe-sha256",
                recipe_sha256,
                "--seed",
                str(seed),
                "--evaluation-seeds",
                *map(str, evaluation_seeds),
                "--horizon",
                str(horizon),
                *topology_arguments,
                *(["--allow-dirty"] if allow_dirty else []),
                "--condition",
                condition,
            ]
        )
        provenance.verify_sources()
        _verify_inputs(recipe_sha256, topologies["train"], topologies["eval"])
        result = json.loads((output / f"{condition}.result.json").read_text())
        if (
            set(result["selected_attempts"]) != {f"{condition}:train:1", f"{condition}:evaluate:1"}
            or len(result["selected_evaluations"]) != 1
        ):
            raise ValueError("Condition must return its exact training and evaluation selections")
        selected_attempts.update(result["selected_attempts"])
        selected_evaluations.extend(result["selected_evaluations"])
    from scripts.eval.plot_results import analysis_identity, write_collation_report

    selection = collate(
        load_records(root),
        protocol_id=declared["id"],
        selected_attempts=selected_attempts,
        selected_evaluations=selected_evaluations,
        analysis={"source": analysis_identity(sources()), "configuration": report_configuration()},
    )
    collation = output / "collation.json"
    collation.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    write_collation_report(
        collation,
        root,
        output / "report",
        configuration=report_configuration(),
        source_files=sources(),
        check=check_comparison,
    )
    return output


def load_recipe_snapshot():
    import yaml

    return yaml.safe_load((HERE / "recipe.yaml").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="Verify and plot an explicit saved comparison")
    analysis.add_argument("--collation", required=True, type=Path)
    analysis.add_argument("--evidence-root", required=True, type=Path)
    analysis.add_argument("--output", required=True, type=Path)
    execution = commands.add_parser("run", help="Run the bounded four-condition construction demonstration")
    condition = commands.add_parser("condition", help="Execute one condition with explicit topology inputs")
    for command in (execution, condition):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--expected-revision", required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--evaluation-seeds", type=int, nargs="+", default=[17, 18])
        command.add_argument("--horizon", type=int, default=2)
        command.add_argument("--allow-dirty", action="store_true")
    condition.add_argument("--train-topology", required=True, nargs=2, metavar=("PATH", "SHA256"))
    condition.add_argument("--eval-topology", required=True, nargs=2, metavar=("PATH", "SHA256"))
    condition.add_argument("--recipe-sha256", required=True)
    condition.add_argument("--condition", required=True, choices=CONDITIONS)
    args = parser.parse_args(argv)
    if args.command == "run":
        output = run_study(
            args.output,
            expected_revision=args.expected_revision,
            seed=args.seed,
            evaluation_seeds=args.evaluation_seeds,
            horizon=args.horizon,
            allow_dirty=args.allow_dirty,
        )
    elif args.command == "condition":
        output = run_condition(
            args.output,
            args.condition,
            expected_revision=args.expected_revision,
            recipe_sha256=args.recipe_sha256,
            train_topology=args.train_topology,
            eval_topology=args.eval_topology,
            seed=args.seed,
            evaluation_seeds=args.evaluation_seeds,
            horizon=args.horizon,
            allow_dirty=args.allow_dirty,
        )
    else:
        from scripts.eval.plot_results import write_collation_report

        output = write_collation_report(
            args.collation,
            args.evidence_root,
            args.output,
            configuration=report_configuration(),
            source_files=sources(),
            check=check_comparison,
        )
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
