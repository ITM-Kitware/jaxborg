"""Regenerate a frozen-Blue response report from a verified offline bundle."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def reference_parts(reference):
    require(reference.startswith("runs:/"), f"Not a producing run reference: {reference}")
    run, name = reference[6:].split("/", 1)
    require(bool(run) and bool(name), "Empty artifact reference")
    return run, name


def verified_bundle(root):
    index = load_json(root / "index.json")
    require(index["format"] == "frozen-defender-response-v1", "Unsupported report bundle")
    references = {}
    for name, item in index["files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Invalid bundle path: {name}")
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Bundle file escapes input directory: {name}")
        require(path.is_file(), f"Missing bundle file: {name}")
        require(path.stat().st_size == item["size"] and digest(path) == item["sha256"], f"Hash mismatch: {name}")
        if item.get("reference"):
            ref = item["reference"]
            require(ref not in references, f"Duplicate artifact reference: {ref}")
            references[ref] = path
    for key in ("protocol", "state", "reference_summary"):
        require(index[key] in references, f"Missing {key} reference")
    require(index["configuration"] in index["files"], "Analysis configuration is not indexed")
    require(bool(index["source_store"]), "Missing original tracking-store identity")
    return index, references


def artifact(references, reference):
    require(reference in references, f"Missing retained artifact: {reference}")
    run, name = reference_parts(reference)
    manifest_ref = f"runs:/{run}/manifest.json"
    require(manifest_ref in references, f"Missing producer manifest: {run}")
    owner = load_json(references[manifest_ref])
    require(owner["run_id"] == run, f"Producer identity differs: {run}")
    record = owner["outputs"].get(name)
    require(record and record["complete"], f"Artifact was not completed: {reference}")
    path = references[reference]
    require(
        record["sha256"] == digest(path) and record["size"] == path.stat().st_size,
        f"Producer hash differs: {reference}",
    )
    return path, owner


def checked_owner(owner, protocol, source_owner, kind):
    require(
        owner["kind"] == kind and owner["status"] == "FINISHED" and owner.get("ended_utc"), "Producer is incomplete"
    )
    require(owner["source"]["git_commit"] == protocol["source_revision"], "Producer source revision differs")
    require(not owner["source"]["dirty"], "Recorded producer source is dirty")
    require(owner["source"]["files"] == source_owner["source"]["files"], "Producer source file identities differ")
    require(
        owner["environment"]["lockfile_hash"] == protocol["dependencies"]["lockfile_sha256"], "Producer lock differs"
    )
    require(
        owner["game_contract"]["implementation"] == source_owner["game_contract"]["implementation"],
        "Game implementation differs",
    )
    require(owner["game_contract"]["resolved"] == owner["effective_config"], "Recorded resolved configuration differs")


def checked_input(owner, role, reference, sha256):
    matches = [item for item in owner["inputs"] if item["role"] == role]
    require(len(matches) == 1, f"Expected one input with role {role}")
    item = matches[0]
    require(item["sha256"] == sha256 and item["canonical_reference"] == reference, f"Wrong {role} identity")
    return item


def checked_variant(variant, protocol):
    game = protocol["game"]
    expected = {
        "name": game["rules"],
        "num_steps": game["episode_length"],
        "red_reward": game["reward"],
        "blue_observation_version": game["blue_observation_version"],
        "cage4_enhanced_obs": game["enhanced_observations"],
    }
    require(
        all(variant.get(key) == value for key, value in expected.items()),
        "Horizon, reward, or observation contract differs",
    )


def checked_training(references, protocol, state, source_owner):
    seeds = protocol["training_seeds"]
    order = [f"seed-{seed}" for seed in seeds]
    require(set(state["training"]) == set(order), "Missing or unexpected training attempts")
    budget = protocol["oracle_budget_per_attempt"]
    candidates = {
        "original": {
            "reference": protocol["source"]["checkpoint"],
            "sha256": protocol["source"]["sha256"],
            "policies": {
                team: {"obs_dim": p["obs_dim"], "action_dim": p["action_dim"], "arch": p["architecture"]}
                for team, p in protocol["source"]["policies"].items()
            },
        }
    }
    curves, tails, seconds = [], [], 0.0
    for seed, name in zip(seeds, order):
        attempt = state["training"][name]
        run = attempt["run_id"]
        path, owner = artifact(references, f"runs:/{run}/logs/metrics.jsonl")
        checked_owner(owner, protocol, source_owner, "training")
        require(
            owner["seed"] == seed and owner["trainable_teams"] == ["red"] and owner["frozen_teams"] == ["blue"],
            "Training teams or seed differ",
        )
        require(
            owner["requested_steps"] == attempt["requested_steps"] == budget["requested_steps"],
            "Requested training budget differs",
        )
        require(
            owner["actual_steps"] == attempt["actual_steps"] == budget["completed_steps"],
            "Completed training budget differs",
        )
        require(owner["completed_updates"] == budget["updates"], "Training update count differs")
        checked_input(owner, "frozen blue policy", protocol["source"]["checkpoint"], protocol["source"]["sha256"])
        checked_input(owner, "training topology", protocol["game"]["topology"], protocol["game"]["topology_sha256"])
        for team in ("blue", "red"):
            checked_variant(owner["effective_config"][team]["TRAIN_VARIANT"], protocol)
            checked_variant(owner["effective_config"][team]["EVAL_VARIANT"], protocol)
        blue = owner["parameter_checks"]["blue"]
        require(
            not blue["changed"]
            and not blue["trainable"]
            and blue["initial_sha256"]
            == blue["current_sha256"]
            == protocol["source"]["policies"]["blue"]["parameter_sha256"],
            "Recorded Blue freeze evidence differs",
        )
        require(
            owner["parameter_checks"]["red"]["changed"] and owner["parameter_checks"]["red"]["trainable"],
            "Recorded Red training evidence differs",
        )
        require(owner["policy_sources"]["red"] == {"kind": "fresh", "seed": seed}, "Challenger initialization differs")
        require(
            owner["seed_contract"]["environment_reset_root"] == seed
            and owner["seed_contract"]["policy_and_rollout_root"] == seed + 1,
            "Training RNG roots differ",
        )
        require(
            owner["policy_contract"]["blue"] == candidates["original"]["policies"]["blue"],
            "Frozen policy contract differs",
        )
        expected_red = {
            **candidates["original"]["policies"]["red"],
            "arch": protocol["challenger"]["architecture"],
        }
        require(owner["policy_contract"]["red"] == expected_red, "Challenger policy contract differs")
        ref = attempt["final_checkpoint"]
        checkpoint_run, checkpoint_name = reference_parts(ref)
        require(checkpoint_run == run, "Final checkpoint belongs to another training run")
        checkpoint = owner["outputs"][checkpoint_name]
        require(
            checkpoint["complete"] and checkpoint["step"] == budget["completed_steps"], "Final checkpoint is incomplete"
        )
        sidecar, _ = artifact(references, f"runs:/{run}/{checkpoint['sidecar']}")
        require(sidecar.is_file(), "Missing final checkpoint sidecar")
        result_path, _ = artifact(references, f"runs:/{run}/training/result.json")
        result = load_json(result_path)
        require(
            all(
                result[key] == attempt[key] for key in ("run_id", "final_checkpoint", "actual_steps", "requested_steps")
            ),
            "Training result differs from selection record",
        )
        candidates[name] = {"reference": ref, "sha256": checkpoint["sha256"], "policies": owner["policy_contract"]}
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == budget["updates"], "Missing or duplicate training updates")
        previous_time = -1.0
        for update, row in enumerate(rows, 1):
            require(
                row["update_idx"] == update and row["env_steps"] == update * budget["steps_per_update"],
                "Training steps are missing or duplicated",
            )
            keys = ("wall_time_s", "team.red.return", "team.blue.return", "throughput_sps")
            require(all(math.isfinite(row[key]) for key in keys), "Nonfinite training measurement")
            require(row["wall_time_s"] >= previous_time, "Training time decreases")
            previous_time = row["wall_time_s"]
            curves.append({"candidate": name, **{key: row[key] for key in ("env_steps", *keys)}})
        seconds += rows[-1]["wall_time_s"]
        tails.append((name, rows))
    return candidates, curves, tails, seconds


def checked_evaluation(references, reference, seeds, candidate, protocol, source_owner):
    path, owner = artifact(references, reference)
    checked_owner(owner, protocol, source_owner, "evaluation")
    row = load_json(path)
    cfg = owner["effective_config"]
    require(
        row["eval_id"] == owner["run_id"] and row["git_commit"] == protocol["source_revision"],
        "Evaluation identity differs",
    )
    checked_variant(cfg["variant"], protocol)
    require(
        cfg["episodes_per_seed"] == row["episodes_per_seed"] == 1 and not cfg["deterministic"] and row["stochastic"],
        "Evaluation randomness contract differs",
    )
    require(
        cfg["policy_backend"] == row["policy_backend"] == "jax" and not cfg["cia"]["enabled"],
        "Evaluation backend or reward mode differs",
    )
    require(
        cfg["episode_seed_scheme"] == protocol["randomness"]["episode_seed_scheme"], "Episode seed derivation differs"
    )
    require(
        cfg["policy_rng_contract"] == "PRNGKey(episode_seed); split for reset, per-step policies and environment",
        "Policy RNG derivation differs",
    )
    require(
        all(values == seeds for values in (row["seeds"], row["per_episode_seeds"], cfg["seeds"], cfg["episode_seeds"])),
        "Missing, duplicate, or mismatched evaluation cases",
    )
    require(row["n_episodes"] == row["total_episodes"] == len(seeds), "Evaluation cohort count differs")
    checked_input(owner, "blue policy", protocol["source"]["checkpoint"], protocol["source"]["sha256"])
    checked_input(owner, "red policy", candidate["reference"], candidate["sha256"])
    topology = checked_input(
        owner, "evaluation topology", protocol["game"]["topology"], protocol["game"]["topology_sha256"]
    )
    require(cfg["topology_sampling"] == row["topology_sampling"] == "exhaustive", "Topology sampling differs")
    require(cfg["topology_paths"] == row["topology_paths"] == [topology["path"]], "Evaluation topology list differs")
    require(row["per_episode_topology_paths"] == [topology["path"]] * len(seeds), "Per-case topology differs")
    for team in ("blue", "red"):
        expected = candidate["policies"][team]
        observed = row["policies"][team]
        require(
            observed["observation_dim"] == expected["obs_dim"]
            and observed["action_dim"] == expected["action_dim"]
            and observed["arch"] == expected["arch"],
            f"{team} policy dimensions or architecture differ",
        )
    blue, red = (
        np.asarray(row["per_episode_blue_returns"], dtype=float),
        np.asarray(row["per_episode_red_returns"], dtype=float),
    )
    require(
        blue.shape == red.shape == (len(seeds),) and np.all(np.isfinite(blue)) and np.all(np.isfinite(red)),
        "Invalid episode returns",
    )
    require(np.array_equal(red, -blue), "Episode returns are not zero sum")
    require(
        math.isclose(float(blue.mean()), row["blue_mean_return"], abs_tol=1e-10, rel_tol=0)
        and math.isclose(float(red.mean()), row["red_mean_return"], abs_tol=1e-10, rel_tol=0),
        "Saved evaluation mean differs from raw episodes",
    )
    return row, owner


def validate(references, index):
    protocol_path, source_owner = artifact(references, index["protocol"])
    protocol = load_json(protocol_path)
    require(
        source_owner["run_id"] == protocol["protocol_owner_run_id"] and source_owner["status"] == "FINISHED",
        "Protocol owner differs or is incomplete",
    )
    state_path, report_owner = artifact(references, index["state"])
    state = load_json(state_path)
    require(report_owner["status"] == "FINISHED", "Original report is incomplete")
    require(protocol["campaign"] == state["campaign"], "Study identity differs")
    require(
        protocol["game"]["rules"] == "cc4_stock"
        and protocol["game"]["reward"] == "zero_sum"
        and protocol["game"]["layout_count"] == 1,
        "Unsupported response-study game",
    )
    validation = protocol["randomness"]["validation_episode_roots"]
    test = protocol["randomness"]["final_test_episode_roots"]
    require(
        len(validation) == protocol["validation_episodes"] and len(test) == protocol["final_test_episodes"],
        "Protocol cohort counts differ",
    )
    require(
        len(validation) == len(set(validation)) and len(test) == len(set(test)) and set(validation).isdisjoint(test),
        "Validation/test cohorts overlap or contain duplicates",
    )
    require(len(test) >= 2, "At least two paired test cases required")
    source_run = protocol["protocol_owner_run_id"]
    lock, _ = artifact(references, f"runs:/{source_run}/environment/uv.lock")
    require(digest(lock) == protocol["dependencies"]["lockfile_sha256"], "Retained producer lock differs")
    archive, _ = artifact(references, source_owner["source"]["archive_reference"])
    require(digest(archive) == source_owner["source"]["archive_sha256"], "Producer source archive differs")
    with tarfile.open(archive) as source:
        for name, sha256 in source_owner["source"]["files"].items():
            stream = source.extractfile(name)
            require(
                stream is not None and hashlib.sha256(stream.read()).hexdigest() == sha256,
                f"Producer source file differs: {name}",
            )
    sidecar, _ = artifact(references, protocol["source"]["sidecar_reference"])
    require(digest(sidecar) == protocol["source"]["sidecar_sha256"], "Source sidecar identity differs")
    candidates, curves, training_rows, training_seconds = checked_training(references, protocol, state, source_owner)
    order = ["original", *[f"seed-{seed}" for seed in protocol["training_seeds"]]]
    require(
        protocol["selection"]["candidates"] == protocol["selection"]["tie_break"] == order,
        "Selection candidates or tie rule differ",
    )
    require(
        protocol["selection"]["rule"] == "lowest validation mean Blue episode return; original Red is fallback",
        "Unsupported selection rule",
    )
    require(set(state["validation"]) == set(order), "Missing validation candidates")
    evaluations = {
        name: checked_evaluation(
            references, state["validation"][name], validation, candidates[name], protocol, source_owner
        )[0]
        for name in order
    }
    scores = {name: float(np.mean(row["per_episode_blue_returns"])) for name, row in evaluations.items()}
    selected = min(order, key=lambda name: (scores[name], order.index(name)))
    selection = state["selection"]
    require(
        selected == selection["candidate"] and scores == selection["validation_scores"],
        "Recorded selection differs from raw validation scores",
    )
    require(
        selection["checkpoint"] == candidates[selected]["reference"]
        and selection["checkpoint_sha256"] == candidates[selected]["sha256"],
        "Selected checkpoint identity differs",
    )
    require(set(state["test"]) == {"original", "selected"}, "Final test matchups differ")
    finals = {}
    for name, candidate in (("original", "original"), ("selected", selected)):
        row, owner = checked_evaluation(
            references, state["test"][name], test, candidates[candidate], protocol, source_owner
        )
        started = datetime.fromisoformat(owner["started_utc"].replace("Z", "+00:00")).timestamp()
        require(selection["frozen_before_test_epoch"] <= started, "Selection was recorded after final testing began")
        finals[name] = row
    return protocol, state, evaluations, finals, curves, training_rows, training_seconds


def resolve_config(protocol, supplied, overrides):
    config = {
        "bootstrap": {
            "resamples": protocol["confidence_interval"]["resamples"],
            "seed": protocol["confidence_interval"]["seed"],
        },
        "tail_updates": 20,
        "figure": {
            "width_inches": 6.4,
            "height_inches": 3.6,
            "dpi": 160,
            "colors": ["#577590", "#b64a3b"],
            "font_family": "DejaVu Sans",
        },
    }
    for values in (supplied, overrides):
        for key, value in values.items():
            require(key in config, f"Unknown analysis setting: {key}")
            if isinstance(config[key], dict):
                require(isinstance(value, dict) and set(value) <= set(config[key]), f"Unknown {key} setting")
                config[key].update(value)
            else:
                config[key] = value
    require(
        type(config["bootstrap"]["resamples"]) is int and config["bootstrap"]["resamples"] > 0,
        "Bootstrap resamples must be positive",
    )
    require(
        type(config["bootstrap"]["seed"]) is int and config["bootstrap"]["seed"] >= 0,
        "Bootstrap seed must be nonnegative",
    )
    require(type(config["tail_updates"]) is int and config["tail_updates"] > 0, "Tail window must be positive")
    return config


def summarize(finals, config):
    baseline = np.asarray(finals["original"]["per_episode_blue_returns"], dtype=float)
    selected = np.asarray(finals["selected"]["per_episode_blue_returns"], dtype=float)
    differences = baseline - selected
    samples, seed = config["bootstrap"]["resamples"], config["bootstrap"]["seed"]
    rng = np.random.default_rng(seed)
    bootstrap = differences[rng.integers(len(differences), size=(samples, len(differences)))].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "baseline_blue_mean": float(baseline.mean()),
        "selected_blue_mean": float(selected.mean()),
        "red_improvement": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "episodes": len(differences),
        "method": "paired episode bootstrap, percentile 95% interval",
        "resampling_unit": "aligned test episode",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "interpretation": "evaluation uncertainty conditional on selected policies from one source training run",
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(destination, summary, curves, order, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = config["figure"]
    plt.rcParams.update({"font.family": style["font_family"], "svg.hashsalt": "frozen-defender-response"})
    size = (style["width_inches"], style["height_inches"])
    fig, axis = plt.subplots(figsize=size)
    axis.bar(
        ["Original Red", "Selected Red"],
        [summary["baseline_blue_mean"], summary["selected_blue_mean"]],
        color=style["colors"],
    )
    axis.set_ylabel("Mean raw Blue episode return")
    axis.set_title(f"One frozen Blue, {summary['episodes']} paired test episodes")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(destination / f"comparison.{extension}", dpi=style["dpi"])
    plt.close(fig)
    fig, axis = plt.subplots(figsize=size)
    for name in order:
        rows = [row for row in curves if row["candidate"] == name]
        axis.plot([row["env_steps"] / 1e6 for row in rows], [row["team.red.return"] for row in rows], label=name)
    axis.set_xlabel("Extra Red training steps (millions)")
    axis.set_ylabel("Raw Red rollout return (500 steps)")
    axis.legend()
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(destination / f"learning-curves.{extension}", dpi=style["dpi"])
    plt.close(fig)


def source_record():
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(PROJECT), *args], text=True, capture_output=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    files = [PROJECT / name for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md")] + [
        HERE / name for name in ("analyze.py", "index_bundle.py", "test_analyze.py", "README.md")
    ]
    checkout = git("rev-parse", "--show-toplevel")
    is_checkout = checkout is not None and Path(checkout).resolve() == PROJECT
    status = git("status", "--porcelain", "--untracked-files=all", "--", ".") if is_checkout else None
    return {
        "git_commit": git("rev-parse", "HEAD") if is_checkout else None,
        "dirty": bool(status) if status is not None else None,
        "files": {path.relative_to(PROJECT).as_posix(): digest(path) for path in files},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    }


def run(args):
    root, output = args.input.resolve(), args.output.resolve()
    require(not output.exists(), "Output directory already exists; choose a new destination")
    require(
        not output.is_relative_to(root) and not root.is_relative_to(output),
        "Output must be separate from the input bundle",
    )
    require(not output.is_relative_to(PROJECT), "Generated reports must be outside the source checkout")
    index, references = verified_bundle(root)
    protocol, state, evaluations, finals, curves, training_rows, training_seconds = validate(references, index)
    config_path = args.config.resolve() if args.config else root / index["configuration"]
    supplied = load_json(config_path)
    overrides = {
        "bootstrap": {
            key: value
            for key, value in (("seed", args.bootstrap_seed), ("resamples", args.bootstrap_resamples))
            if value is not None
        }
    }
    config = resolve_config(protocol, supplied, overrides)
    summary = summarize(finals, config)
    reference_path, _ = artifact(references, index["reference_summary"])
    reference = load_json(reference_path)
    for key in ("baseline_blue_mean", "selected_blue_mean", "red_improvement", "episodes"):
        require(summary[key] == reference[key], f"Reference statistic differs: {key}")
    reference_interval = config["bootstrap"] == {
        "resamples": reference["bootstrap_samples"],
        "seed": reference["bootstrap_seed"],
    }
    if reference_interval:
        require(summary["ci95"] == reference["ci95"], "Reference bootstrap interval differs")
    tail_rows = []
    window = config["tail_updates"]
    for name, rows in training_rows:
        require(len(rows) >= 2 * window, "Insufficient training updates for tail comparison")
        previous = float(np.mean([row["team.red.return"] for row in rows[-2 * window : -window]]))
        final = float(np.mean([row["team.red.return"] for row in rows[-window:]]))
        tail_rows.append(
            {
                "candidate": name,
                "preceding_window_mean_red_return": previous,
                "final_window_mean_red_return": final,
                "change": final - previous,
                "window_updates": window,
            }
        )
    episodes = [
        {
            "episode_seed": seed,
            "baseline_blue_return": baseline,
            "selected_blue_return": selected,
            "red_improvement": baseline - selected,
        }
        for seed, baseline, selected in zip(
            protocol["randomness"]["final_test_episode_roots"],
            finals["original"]["per_episode_blue_returns"],
            finals["selected"]["per_episode_blue_returns"],
        )
    ]
    validation = [
        {
            "candidate": name,
            "mean_blue_return": row["blue_mean_return"],
            "episodes": row["n_episodes"],
            "evaluation": state["validation"][name],
            "selected": name == state["selection"]["candidate"],
        }
        for name, row in evaluations.items()
    ]
    source = source_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".response-report-", dir=output.parent) as temporary:
        destination = Path(temporary) / "report"
        destination.mkdir()
        write_json(destination / "summary.json", summary)
        write_json(destination / "resolved-config.json", config)
        write_json(destination / "selection.json", state["selection"])
        write_csv(destination / "per-episode.csv", episodes)
        write_csv(destination / "validation.csv", validation)
        write_csv(destination / "learning-curves.csv", curves)
        write_csv(destination / "learning-tail.csv", tail_rows)
        write_csv(
            destination / "summary.csv",
            [
                {
                    "matchup": "Original Blue vs Original Red",
                    "mean_blue_return": summary["baseline_blue_mean"],
                    "episodes": summary["episodes"],
                },
                {
                    "matchup": "Original Blue vs selected Red",
                    "mean_blue_return": summary["selected_blue_mean"],
                    "episodes": summary["episodes"],
                },
            ],
        )
        evaluation_seconds = sum(row["wall_time_s"] for row in [*evaluations.values(), *finals.values()])
        write_json(
            destination / "duration.json",
            {
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "pipeline_seconds": state["finished_epoch"] - state["started_epoch"],
            },
        )
        figures(destination, summary, curves, list(state["training"]), config)
        (destination / "analysis-source").mkdir()
        for name in source["files"]:
            copied = destination / "analysis-source" / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, copied)
        write_json(destination / "input-index.json", index)
        report = f"""# Frozen Blue response report

The selected Red is `{state["selection"]["candidate"]}`.
Additional Red return is **{summary["red_improvement"]:.6f}**,
with paired 95% interval **[{summary["ci95"][0]:.6f}, {summary["ci95"][1]:.6f}]**.

| Matchup | Mean Blue return | Paired episodes |
| --- | ---: | ---: |
| Original Red | {summary["baseline_blue_mean"]:.6f} | {summary["episodes"]} |
| Selected Red | {summary["selected_blue_mean"]:.6f} | {summary["episodes"]} |

![Score comparison](comparison.png)

Selection uses all four validation evaluations, with Original Red fallback and the recorded tie order.
Final test cases are disjoint from validation. The interval measures evaluation uncertainty for fixed
selected policies from one original training run, not training-population performance or equilibrium.

The game is stock CC4 with zero-sum reward, stochastic policies, 500 transitions,
enhanced-v2 Blue observations and one fixed JAX seed-0 topology.
Topology provenance: {protocol["game"]["topology_reconstruction"]}.

![Challenger learning curves](learning-curves.png)

Training duration from completed-run metric logs: {training_seconds / 3600:.6f} hours.
Evaluation duration: {evaluation_seconds / 3600:.6f} hours. The tail table uses {window} updates per window.
These descriptive curves do not establish convergence.

[Episode table](per-episode.csv), [validation and input links](validation.csv),
[learning curves](learning-curves.csv), [tail table](learning-tail.csv),
[resolved settings](resolved-config.json), [input index](input-index.json),
and [execution record](manifest.json).

Producer revision: `{protocol["source_revision"]}`. Analysis revision: `{source["git_commit"]}`.
The reference bootstrap interval was
{"verified exactly" if reference_interval else "not compared because bootstrap settings changed"}.
"""
        (destination / "README.md").write_text(report)
        outputs = {
            str(path.relative_to(destination)): {"sha256": digest(path), "size": path.stat().st_size}
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        write_json(
            destination / "manifest.json",
            {
                "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "producer_revision": protocol["source_revision"],
                "source_store": index["source_store"],
                "analysis_source": source,
                "command": [sys.executable, *sys.argv],
                "working_directory": str(Path.cwd()),
                "input_index_sha256": digest(root / "index.json"),
                "configuration_source": {"path": str(config_path), "sha256": digest(config_path), "values": supplied},
                "overrides": overrides,
                "resolved_configuration": config,
                "resolved_configuration_sha256": digest(destination / "resolved-config.json"),
                "reference_interval_verified": reference_interval,
                "inputs": index["files"],
                "outputs": outputs,
                "limitations": [
                    "Recorded policy parameter identities are checked without model tensor bytes",
                    protocol["game"]["topology_reconstruction"],
                ],
            },
        )
        os.rename(destination, output)
    print(
        json.dumps(
            {"output": str(output), "summary": summary, "reference_interval_verified": reference_interval}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Verified offline input bundle")
    parser.add_argument("--output", required=True, type=Path, help="New output directory outside the checkout")
    parser.add_argument("--config", type=Path, help="Optional external JSON analysis settings")
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Report rejected: {error}\n")


if __name__ == "__main__":
    main()
