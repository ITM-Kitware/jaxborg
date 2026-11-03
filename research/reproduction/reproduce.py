"""Offline artifact verification, restoration, statistics and rerun command generation."""

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
from pathlib import Path
from statistics import NormalDist, fmean, stdev

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def catalog():
    return read(ROOT / "artifact-catalog.json")


def verify():
    entries = {x["sha256"]: x for x in catalog()["files"]}
    for index, (sha, item) in enumerate(entries.items()):
        path = ROOT / "objects" / sha
        assert path.stat().st_size == item["bytes"] and digest(path) == sha, item["path"]
        if (index + 1) % 250 == 0:
            print(f"Verified {index + 1}/{len(entries)} objects", flush=True)
    result = {
        "passed": True,
        "unique_objects_verified": len(entries),
        "unique_checkpoint_files": len({x["sha256"] for x in catalog()["files"] if x["path"].endswith(".safetensors")}),
    }
    (ROOT / "OBJECT-VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def restore(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for item in catalog()["files"]:
        relative = Path(item["path"].lstrip("/"))
        assert ".." not in relative.parts
        destination = directory / relative
        source = ROOT / "objects" / item["sha256"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(destination) == item["sha256"], str(destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        mapping[item["path"]] = str(destination)
    (directory / "path-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Restored {len(mapping)} original paths below {directory}; retained records keep their original paths.")


def paired(a, b, family):
    assert len(a) == len(b)
    values = [x - y for x, y in zip(a, b)]
    mean, sd = fmean(values), stdev(values)
    se = sd / math.sqrt(len(values))
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * family))
    return mean, se, mean - critical * se, mean + critical * se


def assert_values(values, expected, mean_key):
    keys = [mean_key, "standard_error", "simultaneous_95_low", "simultaneous_95_high"]
    for value, key in zip(values, keys):
        assert math.isclose(value, expected[key], abs_tol=1e-8, rel_tol=1e-11), (key, value, expected[key])


def statistics(output):
    all_results = {}
    cpu = REPORTS / "overnight-experiments/cpu-crossplay-100m-125m-150m-20261102/evidence"
    report = read(cpu / "analysis/crossplay-report.json")
    cells = {(x["blue_steps"], x["red_steps"]): read(cpu / "eval" / (x["name"] + ".json")) for x in report["cells"]}
    results = []
    for row in report["contrasts"]:
        team, early, late = row["team"], row["earlier_steps"], row["later_steps"]
        opponents = report["steps"] if row["scope"] == "equal_three_opponent_panel" else [row["opponent_steps"]]
        differences = []
        for opponent in opponents:
            p, q = ((early, opponent), (late, opponent)) if team == "blue" else ((opponent, early), (opponent, late))
            a, b = cells[p], cells[q]
            assert a["per_episode_seeds"] == b["per_episode_seeds"] == list(range(51000000, 51004096))
            differences.append(
                [x - y for x, y in zip(a["per_episode_" + team + "_returns"], b["per_episode_" + team + "_returns"])]
            )
        within_root = [fmean(values) for values in zip(*differences)]
        values = paired(within_root, [0.0] * len(within_root), 24)
        assert_values(values, row, "mean_earlier_minus_later_own_return")
        results.append(
            {"team": team, "early": early, "late": late, "opponent": row["opponent_steps"], "mean_se_low_high": values}
        )
    all_results["cpu_crossplay_24"] = results
    response = REPORTS / "overnight-experiments/matched-red-response/evidence"
    selection, reported = read(response / "selection.json"), read(response / "analysis/response-report.json")
    results = []
    for row in reported["contrasts"]:
        name = row["contrast"]
        if name.startswith("selected_red_gain_at_blue"):
            target = name.split("blue")[-1]
            chosen = selection["selected"][target]["name"]
            a = read(response / "confirmation" / ("b" + target + "-" + chosen + ".json"))
            b = read(response / "confirmation" / ("b" + target + "-saved-r150000000.json"))
            team = "red"
        else:
            target = name.split("for")[-1]
            chosen = selection["selected"][target]["name"]
            a = read(response / "confirmation" / ("b150000000-" + chosen + ".json"))
            b = read(response / "confirmation" / ("b124800000-" + chosen + ".json"))
            team = "blue"
        assert a["per_episode_seeds"] == b["per_episode_seeds"] == list(range(53000000, 53004096))
        values = paired(a["per_episode_" + team + "_returns"], b["per_episode_" + team + "_returns"], 4)
        assert_values(values, row, "mean_gain")
        results.append({"contrast": name, "mean_se_low_high": values})
    all_results["fresh_response_4"] = results
    audit = REPORTS / "overnight-experiments/150m-checkpoint-audit/evidence"
    reported = read(audit / "analysis/audit-report.json")
    by_id = {d["eval_id"]: d for d in (read(p) for p in (audit / "eval").glob("*.json"))}
    results = []
    for row in reported["contrasts"]:
        a = by_id[row["challenger_reference"].split("/")[1]]
        b = by_id[row["incumbent_reference"].split("/")[1]]
        team = row["team"]
        assert a["per_episode_seeds"] == b["per_episode_seeds"] == list(range(51000000, 51004096))
        values = paired(a["per_episode_" + team + "_returns"], b["per_episode_" + team + "_returns"], 4)
        assert_values(values, row, "mean_own_return_gain")
        results.append({"team": team, "age": row["challenger_steps"], "mean_se_low_high": values})
    all_results["gpu_checkpoint_audit_4"] = results
    replication = REPORTS / "training-results/evidence/studies/training-replication-20261029"
    reported = read(replication / "analysis/replication-report.json")
    results = []
    for row in reported["primary_equal_panel_gains"] + reported["secondary_per_opponent_gains"]:
        team, age, base = row["team"], row["candidate_steps"], row["baseline_steps"]
        opponents = (
            reported["protocol"]["frozen_opponent_steps"]
            if row["scope"] == "equal_panel_primary"
            else [row["frozen_opponent_steps"]]
        )
        differences = []
        for opponent in opponents:
            a = read(replication / "evaluation/cells" / f"{team}-s{age}-o{opponent}.json")
            b = read(replication / "evaluation/cells" / f"{team}-s{base}-o{opponent}.json")
            assert a["episode_seeds"] == b["episode_seeds"] == list(range(49200000, 49201024))
            differences.append(
                [x - y for x, y in zip(a["per_episode_" + team + "_returns"], b["per_episode_" + team + "_returns"])]
            )
        within_root = [fmean(values) for values in zip(*differences)]
        values = paired(within_root, [0.0] * len(within_root), 28)
        assert_values(values, row, "mean_own_return_gain")
        results.append({"team": team, "age": age, "scope": row["scope"], "mean_se_low_high": values})
    all_results["replication_28"] = results
    assert [len(all_results[k]) for k in all_results] == [24, 4, 4, 28]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": True, "recomputed_contrasts": 60, "results": all_results}, indent=2) + "\n")
    print(f"All 60 primary contrasts reproduced from raw roots: {output}")


def eval_command(args):
    m = read(REPORTS / "overnight-experiments/cpu-crossplay-100m-125m-150m-20261102/evidence/manifest.json")
    restored, checkout = args.restored.resolve(), args.checkout.resolve()

    def relocated(path):
        return str(restored / path.lstrip("/"))

    output = args.output.resolve()
    command = [
        str(checkout / ".venv/bin/python"),
        str(checkout / "scripts/eval/eval_matchup.py"),
        "--recipe",
        relocated(m["recipe"]["path"]),
        "--policy-backend",
        "jax",
        "--blue-path",
        relocated(m["inputs"][str(args.blue_steps)]["source_path"]),
        "--red-path",
        relocated(m["inputs"][str(args.red_steps)]["source_path"]),
        "--episodes-per-seed",
        "1",
        "--seeds",
        "51000000-51004095",
        "--topology-path",
        relocated(m["topology"]["path"]),
        "--topology-sampling",
        "exhaustive",
        "--name",
        "reproduction-crossplay",
        "--output",
        str(output),
    ]
    assert not output.exists(), "Choose a new output path; retained evaluations must not be overwritten."
    print("set -euo pipefail")
    print("test ! -e " + shlex.quote(str(output)))
    for key, value in {
        "JAX_PLATFORMS": "cpu",
        "JAXBORG_ALLOW_CPU": "1",
        "JAXBORG_EVAL_BATCH_SIZE": "64",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
        "JAXBORG_EXP_DIR": str(ROOT / "new-runs"),
        "MLFLOW_TRACKING_URI": "sqlite:///" + str(ROOT / "new-runs/mlflow.db"),
        "JAXBORG_EXPECTED_SHA": m["source_revision"],
        "JAXBORG_MLFLOW_EXPERIMENT": "reproduction-crossplay",
        "PYTHONPATH": str(checkout / "src") + ":" + str(checkout),
    }.items():
        print("export " + key + "=" + shlex.quote(value))
    print("mkdir -p " + shlex.quote(str(ROOT / "new-runs")) + " " + shlex.quote(str(output.parent)))
    print("cd " + shlex.quote(str(checkout)))
    print(shlex.join(command))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
sub = parser.add_subparsers(dest="action", required=True)
sub.add_parser("check")
p = sub.add_parser("restore")
p.add_argument("--directory", type=Path)
p = sub.add_parser("statistics")
p.add_argument("--output", type=Path)
p = sub.add_parser("eval-command")
p.add_argument("--blue-steps", type=int, choices=[99840000, 124800000, 150000000], default=150000000)
p.add_argument("--red-steps", type=int, choices=[99840000, 124800000, 150000000], default=99840000)
p.add_argument("--restored", type=Path)
p.add_argument("--checkout", type=Path)
p.add_argument("--output", type=Path)
args = parser.parse_args()
ROOT = args.bundle.resolve()
REPORTS = ROOT / "reports"
if args.action == "eval-command":
    args.restored = args.restored or ROOT / "restored"
    args.checkout = args.checkout or ROOT / "code/stage-c-red-budget-check"
    args.output = args.output or ROOT / "new-runs/b150000000-r99840000.json"
if args.action == "check":
    verify()
elif args.action == "restore":
    restore(args.directory or ROOT / "restored")
elif args.action == "statistics":
    statistics(args.output or ROOT / "reproduced-statistics.json")
elif args.action == "eval-command":
    eval_command(args)
