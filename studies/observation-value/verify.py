"""Verify selected native evaluations and independently retrieve tracked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from mlflow import MlflowClient

from jaxborg.evaluation.capture_store import load_records
from jaxborg.evaluation.tracking import TAG_PREFIX


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_trials(manifest, pilot):
    for blue in manifest["blues"]:
        yield (
            blue,
            next(red for red in manifest["reds"] if red["id"] == manifest["replay"][blue["id"]]["red"]),
            "replay",
        )
        if not pilot:
            for red in manifest["reds"]:
                for stage in ("calibration", "sampled", "blind"):
                    yield blue, red, stage


def expected_cases(manifest, blue, stage):
    if stage == "replay":
        return {(row["topology_sha256"], row["seed"]) for row in manifest["replay"][blue["id"]]["episodes"]}
    roots = manifest["calibration_roots" if stage == "calibration" else "test_roots"]
    return {(topology["sha256"], seed) for topology in manifest["topologies"] for seed in roots}


def input_artifacts(index, attempt):
    roles = attempt["facts"]["input_artifacts"]["value"]
    return {role: index[identity] for role, identity in roles.items()}


def validate_result(root, manifest, index, blue, red, stage):
    trial = f"{blue['id']}__replay" if stage == "replay" else f"{blue['id']}__{red['id']}__{stage}"
    path = root / "results" / f"{trial}.json"
    result = read(path)
    expected_mode = "sampled" if stage in {"calibration", "replay"} else stage
    require(result["trial"] == trial, f"Selected result names another trial: {path}")
    require(result["blue"] == blue["id"] and result["red"] == red["id"], f"Selected policies differ: {trial}")
    require(result["stage"] == stage and result["mode"] == expected_mode, f"Selected condition differs: {trial}")
    attempt, evaluation = index[result["attempt_id"]], index[result["evaluation_id"]]
    require(attempt["kind"] == "attempt" and attempt["trial_id"] == trial, f"Attempt identity differs: {trial}")
    require(
        evaluation["kind"] == "evaluation" and evaluation["attempt_id"] == attempt["id"],
        f"Evaluation identity differs: {trial}",
    )
    require(attempt["status"] == evaluation["status"] == "succeeded", f"Selected execution did not succeed: {trial}")
    config = attempt["facts"]["resolved_config"]["value"]
    require(config == result["configuration"], f"Result configuration differs from native evidence: {trial}")
    require(
        config["red_selection"] == "sampled" and config["blue_selection"] == expected_mode,
        f"Selection rules differ: {trial}",
    )
    require(config["horizon"] == manifest["horizon"], f"Episode horizon differs: {trial}")
    expected = expected_cases(manifest, blue, stage)
    rows = result["episodes"]
    observed = [(row["topology_sha256"], row["seed"]) for row in rows]
    require(len(observed) == len(set(observed)) and set(observed) == expected, f"Episode panel incomplete: {trial}")
    for owner in (attempt, evaluation):
        budgets = [value for value in owner["budgets"] if value["unit"] == "episodes"]
        require(
            len(budgets) == 1 and budgets[0]["requested"] == budgets[0]["completed"] == len(expected),
            f"Native episode budget incomplete: {trial}",
        )
    episode_artifacts = [
        value
        for value in index.values()
        if value["kind"] == "artifact"
        and value["producer_id"] == evaluation["id"]
        and value["role"] == "evaluation episode observation"
    ]
    observations = [read(root / "evidence" / value["uri"]) for value in episode_artifacts]
    by_episode = {value["episode_index"]: value for value in observations}
    require(len(by_episode) == len(observations) == len(rows), f"Native episode observations differ: {trial}")
    component_checks = 0
    for number, row in enumerate(rows):
        observation = by_episode[number]
        require(
            row["steps"] == manifest["horizon"] and math.isfinite(row["blue_return"]),
            f"Incomplete or nonfinite scored episode: {trial}/{number}",
        )
        require(
            observation["seed"] == row["seed"]
            and observation["reward"] == row["blue_return"]
            and observation["steps"] == row["steps"],
            f"Result disagrees with native episode: {trial}/{number}",
        )
        components = row.get("behavior", {}).get("reward_components")
        if components is not None:
            require(
                math.isclose(sum(components.values()), row["blue_return"], abs_tol=1e-6),
                f"Reward components disagree with return: {trial}/{number}",
            )
            component_checks += 1
    inputs = input_artifacts(index, attempt)
    for team, policy in (("blue", blue), ("red", red)):
        for role in ("model", "sidecar"):
            require(
                inputs[f"{team} {role}"]["sha256"] == policy[role]["sha256"],
                f"Retained {team} {role} differs from frozen selection: {trial}",
            )
    topologies = manifest["topologies"][:1] if stage == "replay" else manifest["topologies"]
    for number, topology in enumerate(topologies):
        require(
            inputs[f"topology {number}"]["sha256"] == topology["sha256"],
            f"Retained layout differs from frozen selection: {trial}",
        )
    if stage == "blind":
        frozen = read(root / "baselines" / f"{blue['id']}-frozen.json")
        require(
            config["baseline_sha256"] == inputs["frozen baseline"]["sha256"] == frozen["sha256"],
            f"Executed baseline differs from frozen identity: {trial}",
        )
    else:
        require(config["baseline_sha256"] is None, f"Full policy unexpectedly receives baseline preferences: {trial}")
    return result, {
        "trial": trial,
        "stage": stage,
        "blue": blue["id"],
        "red": red["id"],
        "attempt_id": attempt["id"],
        "evaluation_id": evaluation["id"],
        "episodes": len(rows),
        "reward_component_checks": component_checks,
        "result_sha256": digest(path),
        "source_revision": config["execution"]["source_revision"],
        "blue_checkpoint_sha256": blue["model"]["sha256"],
        "red_checkpoint_sha256": red["model"]["sha256"],
    }


def validate_baselines(root, manifest):
    selected = []
    for blue in manifest["blues"]:
        identity = blue["id"]
        path = root / "baselines" / f"{identity}.json"
        baseline, frozen = read(path), read(path.with_name(f"{identity}-frozen.json"))
        require(digest(path) == frozen["sha256"], f"Frozen baseline changed: {identity}")
        require(baseline["blue"] == identity and baseline["converged"], f"Baseline identity/fit invalid: {identity}")
        require(
            baseline["calibration_roots"] == manifest["calibration_roots"], f"Calibration roots changed: {identity}"
        )
        require(
            not baseline["phase_conditioned"] and not baseline["opponent_conditioned"],
            f"Baseline information differs: {identity}",
        )
        require(
            not set(manifest["calibration_roots"]) & set(manifest["test_roots"]), "Calibration and test roots overlap"
        )
        expected = {f"{identity}__{red['id']}__calibration" for red in manifest["reds"]}
        retained = baseline["calibration"]
        require(
            len(retained) == len(expected) and {item["trial"] for item in retained} == expected,
            f"Baseline omitted calibration cells: {identity}",
        )
        require(frozen["calibration"] == retained, f"Frozen calibration selection differs: {identity}")
        for item in retained:
            result_path = root / "results" / f"{item['trial']}.json"
            result = read(result_path)
            require(
                digest(result_path) == item["result_sha256"] and result["evaluation_id"] == item["evaluation_id"],
                f"Selected calibration result changed: {item['trial']}",
            )
            require(
                digest(Path(result["records_path"])) == item["records_sha256"],
                f"Selected calibration decisions changed: {item['trial']}",
            )
        selected.append({"blue": identity, "sha256": frozen["sha256"], "frozen_at_utc": baseline["frozen_at_utc"]})
    return selected


def download_evidence(client, root, stage, selected):
    from jaxborg.evaluation.matchup_runner import load_matchup_policy

    destination = root / "checks" / f"{stage}-download"
    destination.mkdir(parents=True, exist_ok=False)
    downloaded = Path(client.download_artifacts(selected["run_id"], "evidence", str(destination)))
    records = load_records(downloaded, verify=True)
    index = {value["id"]: value for value in records}
    attempt, evaluation = index[selected["attempt_id"]], index[selected["evaluation_id"]]
    require(attempt["status"] == evaluation["status"] == "succeeded", "Downloaded execution did not succeed")
    inputs = input_artifacts(index, attempt)
    checkpoints = {}
    for team in ("blue", "red"):
        model = inputs[f"{team} model"]
        path = downloaded / model["uri"]
        require(
            digest(path) == model["sha256"] == selected[f"{team}_checkpoint_sha256"],
            f"Downloaded {team} checkpoint differs from frozen input",
        )
        policy = load_matchup_policy(path, team=team, backend="jax")
        checkpoints[team] = {"sha256": model["sha256"], "path": str(path), "policy_loadable": policy.module is not None}
    return {
        "run_id": selected["run_id"],
        "evaluation_id": selected["evaluation_id"],
        "destination": str(downloaded),
        "verified_records": len(records),
        "checkpoints": checkpoints,
        "scope": (
            "Fresh client retrieval and policy loading; this does not demonstrate another person's access permissions"
        ),
    }


def verify(root: Path, pilot: bool = False, download: bool = True) -> dict:
    """Require complete selected cases, native integrity and current service delivery."""
    root = Path(root).resolve()
    manifest = read(root / "inputs/manifest.json")
    protocol = read(root / "inputs/protocol.json")
    records = load_records(root / "evidence", verify=True)
    index = {value["id"]: value for value in records}
    require(index.get(protocol["id"]) == protocol, "Native protocol differs from frozen study protocol")
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    require(bool(uri) and urlparse(uri).scheme in {"http", "https"}, "Require the dedicated Jaxborg HTTP tracking URI")
    require(urlparse(uri).port != 5000, "Open World's MLflow port 5000 is not a Jaxborg tracking destination")
    client = MlflowClient(tracking_uri=uri)
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or f"jaxborg/{manifest['study_id']}"
    experiment = client.get_experiment_by_name(name)
    require(experiment is not None, f"Dedicated tracking experiment unavailable: {name}")
    require(
        experiment.artifact_location.startswith("mlflow-artifacts:/"),
        "Experiment does not use served Jaxborg artifacts",
    )
    baselines = [] if pilot else validate_baselines(root, manifest)
    freeze_times = {
        item["blue"]: datetime.fromisoformat(item["frozen_at_utc"]).timestamp() * 1000 for item in baselines
    }
    selections, episode_count, component_checks = [], 0, 0
    for blue, red, stage in selected_trials(manifest, pilot):
        _, selected = validate_result(root, manifest, index, blue, red, stage)
        matches = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.`{TAG_PREFIX}evaluation_id` = '{selected['evaluation_id']}'",
            max_results=2,
        )
        require(
            len(matches) == 1 and not getattr(matches, "token", None),
            f"Expected one exact tracking run: {selected['trial']}",
        )
        run = matches[0]
        tags = run.data.tags
        for field, expected in (
            ("evaluation_id", selected["evaluation_id"]),
            ("attempt_id", selected["attempt_id"]),
            ("trial_id", selected["trial"]),
            ("protocol_id", protocol["id"]),
            ("study_id", manifest["study_id"]),
            ("scientific_status", "succeeded"),
            ("delivery", "current"),
            ("data_incomplete", "false"),
            ("evidence_path", "evidence"),
        ):
            require(tags.get(TAG_PREFIX + field) == expected, f"Tracking {field} differs: {selected['trial']}")
        require(run.info.status == "FINISHED", f"Tracking execution is not finished: {selected['trial']}")
        require(
            run.info.artifact_uri.startswith("mlflow-artifacts:/"), "Run artifact destination bypasses shared service"
        )
        if stage in {"sampled", "blind"}:
            require(
                run.info.start_time >= freeze_times[blue["id"]],
                f"Fresh test run began before baseline freeze: {selected['trial']}",
            )
        selected.update(run_id=run.info.run_id, scientific_status="succeeded", delivery="current")
        selections.append(selected)
        episode_count += selected["episodes"]
        component_checks += selected["reward_component_checks"]
    expected_trials = 3 if pilot else 75
    expected_episodes = (
        manifest["budget"]["replay_episodes"]
        if pilot
        else sum(manifest["budget"][name] for name in ("replay_episodes", "calibration_episodes", "test_episodes"))
    )
    require(
        len(selections) == expected_trials and episode_count == expected_episodes, "Declared study budget is incomplete"
    )
    stage = "pilot" if pilot else "completion"
    representative = next((item for item in selections if item["stage"] == "blind"), selections[0])
    downloaded = download_evidence(client, root, stage, representative) if download else None
    return {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "study_id": manifest["study_id"],
        "manifest_sha256": digest(root / "inputs/manifest.json"),
        "protocol_sha256": digest(root / "inputs/protocol.json"),
        "tracking_uri": uri,
        "experiment_id": experiment.experiment_id,
        "experiment_name": name,
        "artifact_location": experiment.artifact_location,
        "validated_evaluations": len(selections),
        "scored_episodes": episode_count,
        "scored_environment_transitions": episode_count * manifest["horizon"],
        "verified_records": len(records),
        "reward_component_checks": component_checks,
        "baselines": baselines,
        "runs": selections,
        "representative_download": downloaded,
        "download_requested": download,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, pilot=args.pilot, download=not args.skip_download), indent=2, allow_nan=False))
