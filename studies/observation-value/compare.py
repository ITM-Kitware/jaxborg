"""Run the frozen observation-value comparison and reconstruct its report."""

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from baselines import fit_baseline, summarize_comparison

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_cases(manifest, stage):
    roots = manifest["calibration_roots" if stage == "calibration" else "test_roots"]
    return {(p["sha256"], seed) for p in manifest["topologies"] for seed in roots}


def validate_cases(result, expected):
    cases = [(p["topology_sha256"], p["seed"]) for p in result["episodes"]]
    if len(cases) != len(set(cases)) or set(cases) != expected:
        raise ValueError(f"Incomplete or duplicate episode cases in {result['trial']}")
    if any(p["steps"] != 500 or not math.isfinite(p["blue_return"]) for p in result["episodes"]):
        raise ValueError("Unexpected episode horizon or nonfinite return")


def validate_identity(result, blue, red, mode, *, revision=None, baseline_sha=None):
    if (result["blue"], result["red"], result["mode"]) != (blue, red, mode):
        raise ValueError("Selected result belongs to a different policy or condition")
    config = result["configuration"]
    if config["red_selection"] != "sampled" or config["blue_selection"] != mode:
        raise ValueError("Executed action selection differs from the comparison")
    if revision and config["execution"]["source_revision"] != revision:
        raise ValueError("Executing source differs from selected source revision")
    if mode == "blind" and config["baseline_sha256"] != baseline_sha:
        raise ValueError("Executed baseline differs from its frozen identity")


def environment(manifest, replay=False):
    from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    topologies = manifest["topologies"][:1] if replay else manifest["topologies"]
    variant = GameVariant(name="cc4_stock", cage4_enhanced_obs=True, resilience_roles=False)
    env = make_joint_jax_env(variant, training_mode=False, topology_path=[p["path"] for p in topologies])
    return env, variant, topologies


def execute(args, blue, red, stage, env, variant, topologies, seeds, *, baseline=None, collect=False):
    from engine import evaluate

    trial = f"{blue['id']}__replay" if stage == "replay" else f"{blue['id']}__{red['id']}__{stage}"
    output = args.root / "results" / f"{trial}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retained result: {output}")
    print(f"START {trial}", flush=True)
    started = time.perf_counter()
    result = evaluate(
        root=args.root,
        protocol=read(args.root / "inputs/protocol.json"),
        trial=trial,
        revision=args.revision,
        blue=blue,
        red=red,
        topologies=topologies,
        seeds=seeds,
        mode="sampled" if stage in ("calibration", "replay") else stage,
        env=env,
        variant=variant,
        source_root=HERE,
        baseline_path=baseline,
        collect=collect,
        collection_stride=5,
        auxiliary_paths=(
            {
                "sensitivity summary": args.root / "sensitivity" / f"{blue['id']}.json",
                "sensitivity measurements": args.root / "sensitivity" / f"{blue['id']}.npz",
            }
            if stage == "blind"
            else None
        ),
    )
    result.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
    write(output, result)
    print(f"COMPLETE {trial} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def replay(args):
    manifest = read(args.root / "inputs/manifest.json")
    blue = next(p for p in manifest["blues"] if p["id"] == args.blue)
    selected = manifest["replay"][args.blue]
    red = next(p for p in manifest["reds"] if p["id"] == selected["red"])
    env, variant, topologies = environment(manifest, replay=True)
    result = execute(args, blue, red, "replay", env, variant, topologies, [p["seed"] for p in selected["episodes"]])
    validate_identity(result, blue["id"], red["id"], "sampled", revision=args.revision)
    actual = {(p["topology_sha256"], p["seed"]): p["blue_return"] for p in result["episodes"]}
    expected = {(p["topology_sha256"], p["seed"]): p["blue_return"] for p in selected["episodes"]}
    if actual != expected:
        raise ValueError(f"Historical replay differs: actual={actual}, expected={expected}")
    write(
        args.root / "checks" / f"{args.blue}-replay.json",
        {"exact_match": True, "episodes": result["episodes"], "source_revision": args.revision},
    )


def calibrate(args):
    manifest = read(args.root / "inputs/manifest.json")
    if not read(args.root / "checks" / f"{args.blue}-replay.json")["exact_match"]:
        raise ValueError("Require exact replay before calibration")
    blue = next(p for p in manifest["blues"] if p["id"] == args.blue)
    env, variant, topologies = environment(manifest)
    for red in manifest["reds"]:
        result = execute(
            args, blue, red, "calibration", env, variant, topologies, manifest["calibration_roots"], collect=True
        )
        validate_cases(result, expected_cases(manifest, "calibration"))


def fit(args):
    from sensitivity import analyze_inputs

    from jaxborg.evaluation.matchup_runner import load_matchup_policy

    manifest = read(args.root / "inputs/manifest.json")
    selected, collected = [], []
    for red in manifest["reds"]:
        path = args.root / "results" / f"{args.blue}__{red['id']}__calibration.json"
        result = read(path)
        validate_identity(result, args.blue, red["id"], "sampled", revision=args.revision)
        validate_cases(result, expected_cases(manifest, "calibration"))
        records_path = Path(result["records_path"])
        selected.append(
            {
                "trial": result["trial"],
                "evaluation_id": result["evaluation_id"],
                "result": str(path),
                "result_sha256": digest(path),
                "records": str(records_path),
                "records_sha256": digest(records_path),
            }
        )
        with np.load(records_path, allow_pickle=False) as data:
            collected.append({key: data[key] for key in data.files})
    keys = collected[0].keys()
    if any(item.keys() != keys for item in collected):
        raise ValueError("Calibration record schemas differ")
    records = {key: np.concatenate([item[key] for item in collected], axis=0) for key in keys}
    baseline = fit_baseline(records["fit_probabilities"], records["fit_masks"], records["fit_genuine"])
    baseline.update(
        blue=args.blue,
        source_revision=args.revision,
        calibration=selected,
        calibration_roots=manifest["calibration_roots"],
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        phase_conditioned=False,
        opponent_conditioned=False,
        calibration_stride=1,
        sensitivity_stride=manifest["collection_stride"],
    )
    destination = args.root / "baselines" / f"{args.blue}.json"
    write(destination, baseline)
    write(
        args.root / "baselines" / f"{args.blue}-frozen.json",
        {"path": str(destination), "sha256": digest(destination), "calibration": selected},
    )
    print(f"BASELINE FROZEN {args.blue} sha256={digest(destination)}", flush=True)
    blue = next(p for p in manifest["blues"] if p["id"] == args.blue)
    policy = load_matchup_policy(Path(blue["model"]["path"]), team="blue", backend="jax")
    indices = np.arange(0, len(records["genuine"]), max(1, math.ceil(len(records["genuine"]) / 1024)))
    summary, measurements = analyze_inputs(
        policy.module, policy.weights, {k: v[indices] for k, v in records.items() if not k.startswith("fit_")}
    )
    summary.update(blue=args.blue, calibration=selected, selected_row_indices=indices.tolist())
    write(args.root / "sensitivity" / f"{args.blue}.json", summary)
    with (args.root / "sensitivity" / f"{args.blue}.npz").open("xb") as stream:
        np.savez_compressed(stream, **measurements, selected_row_indices=indices)
    print(f"SENSITIVITY COMPLETE {args.blue}", flush=True)


def run(args):
    manifest = read(args.root / "inputs/manifest.json")
    blue = next(p for p in manifest["blues"] if p["id"] == args.blue)
    baseline = args.root / "baselines" / f"{args.blue}.json"
    frozen = read(args.root / "baselines" / f"{args.blue}-frozen.json")
    if digest(baseline) != frozen["sha256"]:
        raise ValueError("Frozen baseline changed")
    env, variant, topologies = environment(manifest)
    for red in manifest["reds"]:
        for mode in manifest["primary_modes"]:
            result = execute(
                args,
                blue,
                red,
                mode,
                env,
                variant,
                topologies,
                manifest["test_roots"],
                baseline=baseline if mode == "blind" else None,
            )
            validate_cases(result, expected_cases(manifest, "test"))
    if digest(baseline) != frozen["sha256"]:
        raise ValueError("Frozen baseline changed during evaluation")


def table(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    manifest = read(args.root / "inputs/manifest.json")
    output = args.root / "report"
    output.mkdir(exist_ok=False)
    summaries, opponents, episodes, selections = [], [], [], []
    for blue in manifest["blues"]:
        differences = np.empty((len(manifest["test_roots"]), len(manifest["reds"]), len(manifest["topologies"])))
        pooled = {"sampled": [], "blind": []}
        baseline = args.root / "baselines" / f"{blue['id']}.json"
        frozen = read(args.root / "baselines" / f"{blue['id']}-frozen.json")
        if digest(baseline) != frozen["sha256"]:
            raise ValueError("Baseline no longer matches its frozen identity")
        for red_index, red in enumerate(manifest["reds"]):
            values = {}
            for mode in manifest["primary_modes"]:
                path = args.root / "results" / f"{blue['id']}__{red['id']}__{mode}.json"
                result = read(path)
                validate_identity(
                    result, blue["id"], red["id"], mode, revision=args.revision, baseline_sha=frozen["sha256"]
                )
                validate_cases(result, expected_cases(manifest, "test"))
                values[mode] = {(p["topology_sha256"], p["seed"]): p["blue_return"] for p in result["episodes"]}
                pooled[mode].extend(values[mode].values())
                selections.append(
                    {"trial": result["trial"], "evaluation_id": result["evaluation_id"], "sha256": digest(path)}
                )
                episodes.extend(
                    {
                        "blue": blue["id"],
                        "red": red["id"],
                        "mode": mode,
                        "root": p["seed"],
                        "topology_sha256": p["topology_sha256"],
                        "blue_return": p["blue_return"],
                        "evaluation_id": result["evaluation_id"],
                    }
                    for p in result["episodes"]
                )
            for root_index, seed in enumerate(manifest["test_roots"]):
                for layout_index, topology in enumerate(manifest["topologies"]):
                    case = topology["sha256"], seed
                    differences[root_index, red_index, layout_index] = values["sampled"][case] - values["blind"][case]
            opponents.append(
                {
                    "blue": blue["id"],
                    "red": red["id"],
                    "full_mean": float(np.mean(list(values["sampled"].values()))),
                    "blind_mean": float(np.mean(list(values["blind"].values()))),
                    "full_minus_blind": float(differences[:, red_index, :].mean()),
                }
            )
        summary = summarize_comparison(differences, margin=manifest["practical_margin"])
        summary.update(
            blue=blue["id"],
            full_mean=float(np.mean(pooled["sampled"])),
            blind_mean=float(np.mean(pooled["blind"])),
            baseline_sha256=frozen["sha256"],
            confidence_lower=summary["confidence_interval"][0],
            confidence_upper=summary["confidence_interval"][1],
        )
        summaries.append(summary)
    report = {
        "study_id": manifest["study_id"],
        "summary": summaries,
        "sensitivity": {p["id"]: read(args.root / "sensitivity" / f"{p['id']}.json") for p in manifest["blues"]},
        "selection": selections,
        "scored_test_episodes": len(episodes),
        "limitation": (
            "Fixed trained policies and familiar layouts; intervals describe test-root variation, "
            "not training populations or calibration uncertainty. Distillation does not find "
            "the best return-optimized blind policy."
        ),
    }
    if len(episodes) != manifest["budget"]["test_episodes"]:
        raise ValueError("Test episode budget incomplete")
    write(output / "comparison.json", report)
    table(output / "summary.csv", [{k: v for k, v in p.items() if not isinstance(v, (list, dict))} for p in summaries])
    table(output / "opponents.csv", opponents)
    table(output / "episodes.csv", episodes)
    lines = ["# Observation value in the frozen Blue panel", ""]
    for summary in summaries:
        lower, upper = summary["confidence_interval"]
        lines.append(
            f"- {summary['blue']}: full {summary['full_mean']:.1f}, blind {summary['blind_mean']:.1f}; "
            f"difference {summary['mean_difference']:+.1f}, 95% interval [{lower:+.1f}, {upper:+.1f}]. "
            f"{summary['classification']}."
        )
    lines.extend(
        [
            "",
            report["limitation"],
            "",
            "Input sensitivity measures dependence; the matched returns test its functional value.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "calibrate", "fit", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--blue")
    parser.add_argument("--revision")
    arguments = parser.parse_args()
    {"replay": replay, "calibrate": calibrate, "fit": fit, "run": run, "analyze": analyze}[arguments.command](arguments)
