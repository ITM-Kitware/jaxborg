#!/usr/bin/env python3
"""Consolidate saved Jaxborg studies and verify the 150M episode statistics."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, stdev


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--remote-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    verification = json.loads(args.remote_verification.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = bundle["source_root"].rstrip("/") + "/"
    entries = {entry["remote_path"]: entry for entry in bundle["files"]}
    index = []
    for remote, entry in entries.items():
        raw = entry["text"].encode()
        assert len(raw) == entry["bytes"] and digest(raw) == entry["sha256"]
        relative = (
            ("studies/" + remote[len(root) :]) if remote.startswith(root) else "study-context/" + Path(remote).name
        )
        path = output / "evidence" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        index.append(
            {
                "remote_path": remote,
                "local_path": "evidence/" + relative,
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
        )

    def entry(study, name):
        return entries[root + study + "/" + name]

    def read(study, name):
        return json.loads(entry(study, name)["text"])

    study = "training-replication-20261029"
    report = read(study, "analysis/replication-report.json")
    manifest = read(study, "manifest.json")
    inputs = read(study, "evaluation/input-manifest.json")
    assert report["study_manifest_sha256"] == entry(study, "manifest.json")["sha256"]
    assert report["evaluation_inputs_sha256"] == entry(study, "evaluation/input-manifest.json")["sha256"]
    assert verification["report_sha256"] == entry(study, "analysis/replication-report.json")["sha256"]
    assert inputs["training_result_sha256"] == entry(study, "training/result.json")["sha256"]
    expected_seeds = list(range(49_200_000, 49_201_024))
    cells = {}
    for row in report["cells"]:
        name = row["cell"]
        source = entry(study, "evaluation/cells/" + name + ".json")
        assert source["sha256"] == row["result_sha256"]
        cell = json.loads(source["text"])
        assert cell["episode_seeds"] == expected_seeds and cell["n_episodes"] == 1024
        assert cell["study_manifest_sha256"] == report["study_manifest_sha256"]
        assert cell["evaluation_inputs_sha256"] == report["evaluation_inputs_sha256"]
        assert cell["topology_sha256"] == manifest["topology"]["sha256"]
        assert cell["variant"] == "cc4_stock" and cell["episode_length"] == 500
        assert len(cell["per_episode_blue_returns"]) == len(cell["per_episode_red_returns"]) == 1024
        assert all(b == -r for b, r in zip(cell["per_episode_blue_returns"], cell["per_episode_red_returns"]))
        for team in ["blue", "red"]:
            values = cell["per_episode_" + team + "_returns"]
            close(cell[team + "_mean_return"], fmean(values))
            close(cell[team + "_sample_sd"], stdev(values))
        cells[name] = cell
    assert len(cells) == 37 and verification["cells"] == 37
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 28))
    close(report["protocol"]["normal_critical_value"], critical)
    primary = report["primary_equal_panel_gains"]
    secondary = report["secondary_per_opponent_gains"]
    for contrast in primary + secondary:
        team = contrast["team"]
        age, baseline = contrast["candidate_steps"], contrast["baseline_steps"]
        opponents = (
            report["protocol"]["frozen_opponent_steps"]
            if contrast["scope"] == "equal_panel_primary"
            else [contrast["frozen_opponent_steps"]]
        )
        differences = []
        for opponent in opponents:
            a = cells[f"{team}-s{age}-o{opponent}"][f"per_episode_{team}_returns"]
            b = cells[f"{team}-s{baseline}-o{opponent}"][f"per_episode_{team}_returns"]
            differences.append([x - y for x, y in zip(a, b)])
        paired = [fmean(values) for values in zip(*differences)]
        mean, sd = fmean(paired), stdev(paired)
        se = sd / math.sqrt(len(paired))
        for key, value in [
            ("mean_own_return_gain", mean),
            ("paired_sd", sd),
            ("standard_error", se),
            ("simultaneous_95_low", mean - critical * se),
            ("simultaneous_95_high", mean + critical * se),
        ]:
            close(contrast[key], value)
    assert len(primary) == 4 and len(secondary) == 24
    final = report["descriptive_final_co_pair"]
    close(final["blue_mean_return"], fmean(cells["final-co-pair"]["per_episode_blue_returns"]))
    (output / "evidence-index.json").write_text(
        json.dumps({"collected_utc": bundle["collected_utc"], "files": index}, indent=2) + "\n"
    )
    (output / "verification.json").write_text(
        json.dumps(
            {
                "remote": verification,
                "local": {
                    "copied_file_hashes_verified": len(index),
                    "cell_statistics_reproduced": 37,
                    "paired_contrast_statistics_reproduced": 28,
                    "paired_roots": 1024,
                    "all_zero_sum_episode_returns_verified": True,
                },
            },
            indent=2,
        )
        + "\n"
    )
    (output / "job-status.json").write_text(
        json.dumps({"collected_utc": bundle["collected_utc"], "commands": bundle["job_status"]}, indent=2) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "collected_utc": bundle["collected_utc"],
                "primary_replication_gains": primary,
                "secondary_per_opponent_gains": secondary,
                "final_copair": final,
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "primary-gains.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary[0]))
        writer.writeheader()
        writer.writerows(primary)
    (output / "evidence-bundle.json").write_bytes(args.bundle.read_bytes())
    (output / "remote-verification.json").write_bytes(args.remote_verification.read_bytes())
    (output / "consolidate.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "output": str(output),
                "files_verified": len(index),
                "cells_verified": 37,
                "contrasts_reproduced": 28,
                "primary_gains": primary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
