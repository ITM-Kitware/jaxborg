"""Prepare, capture and analyze the declared matched system benchmark; no training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root, item):
    path = (Path(root) / item["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or digest(path) != item["sha256"]:
        raise ValueError(f"Retained input differs from manifest: {item['path']}")
    return path


def retain(source, root, relative, expected=None):
    source = Path(source).expanduser().resolve()
    identity = digest(source)
    if expected is not None and identity != expected:
        raise ValueError(f"Source input hash differs from declaration: {source}")
    destination = Path(root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())
    if digest(destination) != identity:
        raise ValueError(f"Input changed while retaining {source}")
    return {"path": relative, "sha256": identity, "source": str(source)}


def assert_source(revision):
    actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if actual != revision or len(revision) != 40 or dirty:
        raise ValueError("Benchmark requires clean source at the full pinned Git revision")


def ledger_for(policies, cases, spec, stage):
    ids = spec["full_policy_ids"] if stage == "full" else spec["smoke_policy_ids"]
    indices = spec["full_case_indices"] if stage == "full" else [spec["smoke_case_index"]]
    rows = []
    for policy_id in ids:
        if sum(policy["id"] == policy_id for policy in policies) != 1:
            raise ValueError(f"Policy must identify exactly one fixed system: {policy_id}")
        for index in indices:
            case = cases[index]
            if any(type(case[key]) is not int for key in ("policy_seed", "dynamics_seed", "env_seed")):
                raise ValueError("Case seeds must be explicit integers")
            rows.append(
                {
                    "episode_id": f"{stage}:{policy_id}:{case['case_id']}",
                    "policy_id": policy_id,
                    "case_id": case["case_id"],
                    "case_index": index,
                    "policy_seed": case["policy_seed"],
                    "dynamics_seed": case["dynamics_seed"],
                    "env_seed": case["env_seed"],
                }
            )
    if len(rows) != spec[f"episodes_{stage}"] or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Episode ledger has missing or repeated cells")
    return rows


def prepare(study_root, transfer_manifest, native_root, hmarl_model, revision):
    assert_source(revision)
    study_root = Path(study_root).expanduser().resolve()
    native_root = Path(native_root).expanduser().resolve()
    if study_root.is_relative_to(REPO) or study_root.is_relative_to(native_root):
        raise ValueError("Study output must be separate from source and native inputs")
    inputs = study_root / "inputs"
    if inputs.exists() or (study_root / "evidence").exists() or (study_root / "smoke-evidence").exists():
        raise FileExistsError("Use a new study root; an existing benchmark is never silently resumed")
    spec = read_json(HERE / "benchmark-protocol.json")
    original = read_json(transfer_manifest)
    cases = read_json(native_root / "cases.json")
    if not isinstance(cases, list) or len(cases) != 100 or len({c["case_id"] for c in cases}) != 100:
        raise ValueError("Expected the complete 100-case native-matched ledger")
    inputs.mkdir(parents=True)
    sources = {
        "definition": retain(HERE / "benchmark-protocol.json", inputs, "benchmark-protocol.json"),
        "transfer_manifest": retain(transfer_manifest, inputs, "transfer-manifest.json"),
        "native_cases": retain(native_root / "cases.json", inputs, "native-cases.json"),
        "hmarl_manifest": retain(REPO / "src/jaxborg/pretrained/hmarl_manifest.json", inputs, "hmarl-manifest.json"),
    }
    policies = []
    for policy_id in spec["full_policy_ids"]:
        if policy_id.startswith("blue-"):
            matches = [p for p in original["policies"]["blue"] if p["id"] == policy_id and p["stage"] == "final"]
            if len(matches) != 1:
                raise ValueError(f"Missing declared final LSTM: {policy_id}")
            old = matches[0]
            if old["steps"] != 49968000:
                raise ValueError("The LSTM reference must be the final 49.968M checkpoint")
            policy = {
                "id": policy_id,
                "kind": "lstm",
                "condition": old["condition"],
                "training_seed": old["training_seed"],
                "model": retain(old["model"], inputs, f"policies/{policy_id}.safetensors", old["sha256"]),
                "sidecar": retain(old["sidecar"], inputs, f"policies/{policy_id}.yaml", old["sidecar_sha256"]),
                "source_record": old,
            }
        else:
            policy = {
                "id": policy_id,
                "kind": policy_id.replace("-", "_"),
                "condition": "imported system reference",
                "training_seed": None,
                "model": retain(
                    hmarl_model, inputs, f"policies/{policy_id}.safetensors", spec["hmarl_checkpoint_sha256"]
                ),
            }
        policies.append(policy)
    selected_cases = {}
    for index in [*spec["full_case_indices"], spec["smoke_case_index"]]:
        case = cases[index]
        source = (native_root / case["snapshot"]).resolve()
        if not source.is_relative_to(native_root):
            raise ValueError("Case snapshot must remain in its native attempt root")
        selected_cases[str(index)] = {
            **case,
            "snapshot": retain(source, inputs, f"topologies/{case['case_id']}.snapshot.npz"),
        }
    manifest = {
        "schema_version": 1,
        "source_revision": revision,
        "specification": spec,
        "source_inputs": sources,
        "policies": policies,
        "cases": selected_cases,
        "ledgers": {stage: ledger_for(policies, cases, spec, stage) for stage in ("smoke", "full")},
    }
    write_json(inputs / "manifest.json", manifest)
    for stage in ("smoke", "full"):
        write_json(inputs / f"protocol-{stage}.json", protocol_for(manifest, stage))
    subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar.gz", f"--output={study_root / 'source.tar.gz'}", revision],
        check=True,
    )
    return inputs / "manifest.json"


def protocol_for(manifest, stage):
    identity = hashlib.sha256(encoded(manifest)).hexdigest()
    ids = list(dict.fromkeys(row["policy_id"] for row in manifest["ledgers"][stage]))
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": f"matched-system:{stage}:{identity}",
        "study_id": manifest["specification"]["study_id"] + ":" + stage,
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {
            "attempt": ["resolved_config", "source_identity"],
            "evaluation": ["resolved_config", "source_identity"],
        },
        "trials": [
            {
                "id": f"{stage}:{policy_id}",
                "condition_id": policy_id,
                "replicate_id": policy_id,
                "matched_unit_id": "fixed-native-scenarios",
            }
            for policy_id in ids
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "fixed scenario conditional on saved system",
                "uncertainty": "Descriptive only; scenarios do not provide independent training replication",
            }
        ],
        "extensions": {
            "matched_system_benchmark": {
                "schema_version": 1,
                "source": "manifest.json",
                "values": {
                    "manifest_sha256": identity,
                    "stage": stage,
                    "comparison": manifest["specification"]["comparison"],
                },
            }
        },
    }


def verified_manifest(path, stage, revision):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest["source_revision"] != revision:
        raise ValueError("Manifest source differs from pinned execution")
    if read_json(path.parent / f"protocol-{stage}.json") != protocol_for(manifest, stage):
        raise ValueError("Manifest differs from immutable protocol")
    for item in manifest["source_inputs"].values():
        checked_path(path.parent, item)
    for policy in manifest["policies"]:
        checked_path(path.parent, policy["model"])
        if "sidecar" in policy:
            checked_path(path.parent, policy["sidecar"])
    for case in manifest["cases"].values():
        checked_path(path.parent, case["snapshot"])
    return manifest


def validate_episode(result, declared, snapshot, horizon):
    if result["steps"] != horizon or not math.isfinite(result["reward"]):
        raise ValueError("Episode did not complete the declared finite-return horizon")
    for key in ("policy_seed", "dynamics_seed"):
        if result[key] != declared[key]:
            raise ValueError("Episode seed differs from ledger")
    if result["snapshot_sha256"] != snapshot["sha256"]:
        raise ValueError("Episode used another scenario snapshot")
    for key in ("per_step_returns", "per_step_components", "per_step_actions", "per_step_done", "per_step_phases"):
        if len(result[key]) != horizon:
            raise ValueError(f"Incomplete trajectory column: {key}")
    if not math.isclose(sum(result["per_step_returns"]), result["reward"], rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError("Episode return differs from retained step rewards")
    if any(len(actions) != 5 for actions in result["per_step_actions"]):
        raise ValueError("Each trajectory step must retain all five Blue actions")


def run_policy(manifest_path, stage, policy_id, root, revision):
    from benchmark_rollout import load_policy, run_episode

    from jaxborg.evaluation.capture_store import (
        add_artifact,
        fail_evaluation,
        finish_evaluation,
        record_episode,
        start_evaluation,
    )
    from jaxborg.evaluation.provenance import collect_provenance
    from jaxborg.evaluation.tracking import destination_configured

    manifest_path = Path(manifest_path).resolve()
    manifest = verified_manifest(manifest_path, stage, revision)
    if not destination_configured():
        raise ValueError("Recorded benchmark execution requires the configured shared MLflow store")
    policy = next(p for p in manifest["policies"] if p["id"] == policy_id)
    ledger = [row for row in manifest["ledgers"][stage] if row["policy_id"] == policy_id]
    if not ledger:
        raise ValueError("Policy has no declared episodes in this stage")
    provenance = collect_provenance(expected_revision=revision, entrypoint=__file__, source_roots=(HERE,))
    inputs = {"manifest": manifest_path, "definition": HERE / "benchmark-protocol.json"}
    inputs.update(
        {f"source {name}": checked_path(manifest_path.parent, item) for name, item in manifest["source_inputs"].items()}
    )
    inputs["model"] = checked_path(manifest_path.parent, policy["model"])
    if "sidecar" in policy:
        inputs["sidecar"] = checked_path(manifest_path.parent, policy["sidecar"])
    for row in ledger:
        inputs[row["case_id"]] = checked_path(
            manifest_path.parent, manifest["cases"][str(row["case_index"])]["snapshot"]
        )
    configuration = {
        "policy": policy,
        "episode_ledger": ledger,
        "specification": manifest["specification"],
        "execution": provenance.configuration,
    }
    capture = start_evaluation(
        root,
        protocol=protocol_for(manifest, stage),
        trial_id=f"{stage}:{policy_id}",
        requested_episodes=len(ledger),
        resolved_config=configuration,
        facts=provenance.facts,
        inputs=inputs,
        source_files=provenance.source_files,
        track=destination_configured(),
    )
    try:
        provenance.verify_sources()
        hmarl_manifest = REPO / "src/jaxborg/pretrained/hmarl_manifest.json"
        if digest(hmarl_manifest) != manifest["source_inputs"]["hmarl_manifest"]["sha256"]:
            raise ValueError("H-MARL loader manifest changed before policy load")
        if digest(capture.inputs["model"]) != policy["model"]["sha256"]:
            raise ValueError("Checkpoint changed during capture")
        loaded = load_policy(capture.inputs["model"], kind=policy["kind"])
        for index, row in enumerate(ledger):
            case = manifest["cases"][str(row["case_index"])]
            result = run_episode(
                loaded,
                snapshot_path=capture.inputs[row["case_id"]],
                policy_seed=row["policy_seed"],
                dynamics_seed=row["dynamics_seed"],
                num_steps=manifest["specification"]["horizon"],
            )
            validate_episode(result, row, case["snapshot"], manifest["specification"]["horizon"])
            result.update({**row, "evaluation_id": capture.evaluation_id, "policy_kind": policy["kind"]})
            output = capture.directory / f"trajectory-{index:03}.json"
            write_json(output, result)
            add_artifact(capture, output, role="matched benchmark episode")
            record_episode(
                capture,
                episode_index=index,
                seed=row["dynamics_seed"],
                policy_seed=row["policy_seed"],
                reward=result["reward"],
                steps=result["steps"],
            )
        provenance.verify_sources()
        if digest(hmarl_manifest) != manifest["source_inputs"]["hmarl_manifest"]["sha256"]:
            raise ValueError("H-MARL loader manifest changed during evaluation")
        finish_evaluation(capture, status="succeeded")
        return {
            "trial_id": f"{stage}:{policy_id}",
            "evaluation_id": capture.evaluation_id,
            "attempt_id": capture.attempt_id,
            "directory": str(capture.directory),
        }
    except BaseException as error:
        fail_evaluation(capture, error)
        raise


def execute(args):
    import jax
    from execute import tracking_preflight

    if os.environ.get("SLURM_JOB_PARTITION") != "community" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use the community GPU launcher")
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise ValueError(f"GPU runtime preflight failed: {devices}")
    assert_source(args.expected_revision)
    manifest_path = prepare(
        args.study_root, args.transfer_manifest, args.native_root, args.hmarl_model, args.expected_revision
    )
    study_root = Path(args.study_root).expanduser().resolve()
    write_json(
        study_root / "runtime.json",
        {
            "source_revision": args.expected_revision,
            "source_archive_sha256": digest(study_root / "source.tar.gz"),
            "devices": [str(d) for d in devices],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tracking": tracking_preflight(),
            "command": sys.argv,
        },
    )
    manifest = read_json(manifest_path)
    for stage in ("smoke", "full"):
        evidence = study_root / ("smoke-evidence" if stage == "smoke" else "evidence")
        selection_path = study_root / f"selection-{stage}.json"
        selection = {}
        for policy_id in dict.fromkeys(row["policy_id"] for row in manifest["ledgers"][stage]):
            receipt_path = study_root / f"receipt-{stage}-{policy_id}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "benchmark.py"),
                    "run-policy",
                    "--manifest",
                    str(manifest_path),
                    "--stage",
                    stage,
                    "--policy-id",
                    policy_id,
                    "--root",
                    str(evidence),
                    "--expected-revision",
                    args.expected_revision,
                    "--receipt",
                    str(receipt_path),
                ],
                check=True,
            )
            receipt = read_json(receipt_path)
            selection[receipt["trial_id"]] = receipt["evaluation_id"]
            next_path = selection_path.with_suffix(".next")
            write_json(next_path, selection)
            next_path.replace(selection_path)
        analyze(manifest_path, evidence, selection_path, study_root / f"analysis-{stage}", stage)


def summarize(rows):
    by_policy = {}
    for row in rows:
        cases = by_policy.setdefault(row["policy_id"], {})
        if row["case_id"] in cases:
            raise ValueError("Duplicate selected policy/scenario row")
        cases[row["case_id"]] = row
    means = [
        {
            "policy_id": policy,
            "n_scenarios": len(cases),
            "mean_return": statistics.mean(row["reward"] for row in cases.values()),
        }
        for policy, cases in sorted(by_policy.items())
    ]
    gaps = []
    for left, right in itertools.combinations(sorted(by_policy), 2):
        if by_policy[left].keys() != by_policy[right].keys():
            raise ValueError("Paired policy gaps require identical scenario coverage")
        for case in sorted(by_policy[left]):
            a, b = by_policy[left][case], by_policy[right][case]
            if any(a[key] != b[key] for key in ("snapshot_sha256", "policy_seed", "dynamics_seed")):
                raise ValueError("Paired policy gaps require identical scenario and seed contracts")
            gaps.append(
                {"candidate": left, "reference": right, "case_id": case, "return_gap": a["reward"] - b["reward"]}
            )
    return means, gaps


def csv_table(path, rows, fields):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest_path, root, selection_path, output, stage):
    from jaxborg.evaluation.capture_store import load_records

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Reanalysis requires a new output directory")
    manifest = read_json(manifest_path)
    protocol = protocol_for(manifest, stage)
    if read_json(root / "protocol.json") != protocol:
        raise ValueError("Evidence store differs from manifest protocol")
    records = load_records(root, verify=True)
    by_id = {record["id"]: record for record in records}
    selection = read_json(selection_path)
    expected_trials = {trial["id"] for trial in protocol["trials"]}
    if set(selection) != expected_trials:
        raise ValueError("Analysis requires explicit complete selection for this stage")
    rows = []
    for trial, identity in selection.items():
        evaluation = by_id[identity]
        attempt = by_id[evaluation["attempt_id"]]
        if (
            evaluation["kind"] != "evaluation"
            or evaluation["status"] != "succeeded"
            or attempt["status"] != "succeeded"
            or attempt["trial_id"] != trial
            or attempt["protocol_id"] != protocol["id"]
        ):
            raise ValueError("Selection does not identify a successful declared trial")
        for artifact in records:
            if (
                artifact["kind"] == "artifact"
                and artifact["producer_id"] == identity
                and artifact["role"] == "matched benchmark episode"
            ):
                rows.append(read_json(root / artifact["uri"]))
    declared = {row["episode_id"]: row for row in manifest["ledgers"][stage]}
    if len(rows) != len(declared) or {row["episode_id"] for row in rows} != declared.keys():
        raise ValueError("Selected trajectories do not exactly cover declared episode ledger")
    for row in rows:
        plan = declared[row["episode_id"]]
        if any(row[key] != value for key, value in plan.items()):
            raise ValueError("Selected trajectory identity differs from declaration")
        validate_episode(
            row, plan, manifest["cases"][str(plan["case_index"])]["snapshot"], manifest["specification"]["horizon"]
        )
    means, gaps = summarize(rows)
    gap_means = [
        {"candidate": a, "reference": b, "n_scenarios": len(values), "mean_return_gap": statistics.mean(values)}
        for a, b in itertools.combinations(sorted(row["policy_id"] for row in means), 2)
        if (values := [row["return_gap"] for row in gaps if row["candidate"] == a and row["reference"] == b])
    ]
    output.mkdir(parents=True)
    write_json(output / "selection.json", selection)
    write_json(output / "manifest.json", manifest)
    write_json(output / "episodes.json", rows)
    write_json(
        output / "provenance.json",
        {
            "manifest_sha256": digest(manifest_path),
            "selection_sha256": digest(selection_path),
            "source_revision": manifest["source_revision"],
            "evidence_root": str(root),
            "verification": "capture_store.load_records verify=True",
            "interpretation": manifest["specification"]["interpretation"],
        },
    )
    csv_table(output / "means.csv", means, ("policy_id", "n_scenarios", "mean_return"))
    csv_table(output / "paired-scenario-gaps.csv", gaps, ("candidate", "reference", "case_id", "return_gap"))
    csv_table(output / "paired-mean-gaps.csv", gap_means, ("candidate", "reference", "n_scenarios", "mean_return_gap"))
    (output / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "hashes.json", {path.name: digest(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps({"stage": stage, "episodes": len(rows), "means": means, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute_parser = commands.add_parser("execute")
    for name in ("study-root", "transfer-manifest", "native-root", "hmarl-model"):
        execute_parser.add_argument(f"--{name}", type=Path, required=True)
    execute_parser.add_argument("--expected-revision", required=True)
    run_parser = commands.add_parser("run-policy")
    for name in ("manifest", "root", "receipt"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    run_parser.add_argument("--policy-id", required=True)
    run_parser.add_argument("--expected-revision", required=True)
    analysis_parser = commands.add_parser("analyze")
    for name in ("manifest", "root", "selection", "output"):
        analysis_parser.add_argument(f"--{name}", type=Path, required=True)
    analysis_parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    if args.command == "execute":
        execute(args)
    elif args.command == "run-policy":
        if args.receipt.exists():
            raise FileExistsError("Existing receipt requires a new explicit attempt; no silent rerun")
        receipt = run_policy(args.manifest, args.stage, args.policy_id, args.root, args.expected_revision)
        write_json(args.receipt, receipt)
        print(json.dumps(receipt), flush=True)
    else:
        analyze(args.manifest, args.root, args.selection, args.output, args.stage)


if __name__ == "__main__":
    main()
