"""Rebuild the Nash archive's registered comparisons from retained observations."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
OVERNIGHT = Path("reports/overnight-experiments")
CPU = OVERNIGHT / "cpu-crossplay-100m-125m-150m-20261102"
GPU = OVERNIGHT / "crossplay-100m-125m-150m"
AUDIT = OVERNIGHT / "150m-checkpoint-audit"
RESPONSE = OVERNIGHT / "matched-red-response"
BUDGET = OVERNIGHT / "red-search-budget"
REPLICATION = Path("reports/training-results/evidence/studies/training-replication-20261029")
RESPONSE_LAUNCH = OVERNIGHT / "launch-records/response-125m-150m-20261101/manifest.json"
PRODUCERS = {
    "reproduce.py": "4278a12b6601e49144d1ecc64d2b803ea18956a278cc356c50eaea73a584dbbf",
    "verify_cpu.py": "437be90a7028f5952fc49d42ec074114e3f0dfe4b97d550c5b556db685873f0b",
    "verify_response.py": "708b412d2a3653fcc09dea9cbe842355c4eb0db84dc12ffbb43abe05588312cd",
}
INPUTS = (
    Path("PUBLICATION-RECORD.json"),
    CPU / "evidence",
    CPU / "launch-records",
    CPU / "registered-manifest-before-affinity.json",
    GPU / "evidence/eval",
    GPU / "evidence/manifest.json",
    GPU / "partial-report.json",
    AUDIT / "evidence/analysis/audit-report.json",
    AUDIT / "evidence/eval",
    AUDIT / "evidence/manifest.json",
    AUDIT / "job-status.txt",
    RESPONSE / "evidence",
    RESPONSE_LAUNCH,
    BUDGET / "launch-records/manifest.json",
    BUDGET / "terminal-summary.json",
    REPLICATION / "analysis/replication-report.json",
    REPLICATION / "evaluation/cells",
    REPLICATION / "manifest.json",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(bundle, output):
    source = output / "source"
    source.mkdir()
    shutil.copyfile(HERE, source / "analyze.py")
    for name, expected in PRODUCERS.items():
        archived = bundle / "published-code/research/reproduction" / name
        if digest(archived) != expected:
            raise ValueError(f"Unexpected archived producer source: {archived}")
        shutil.copyfile(archived, source / name)
    for relative in INPUTS:
        archived, target = bundle / relative, output / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if archived.is_dir():
            shutil.copytree(archived, target)
        else:
            shutil.copyfile(archived, target)


def run(source, *args):
    subprocess.run([sys.executable, "-E", str(source), *map(str, args)], check=True)


def replay(output):
    source, inputs = output / "source", output / "inputs"
    run(source / "reproduce.py", "--bundle", inputs, "statistics", "--output", output / "statistics.json")

    # The original verifiers write alongside the study root. Link only their
    # read-only inputs into scratch directories and retain the new outputs.
    with tempfile.TemporaryDirectory(prefix="jaxborg-nash-replay-") as directory:
        scratch = Path(directory)
        for relative in (GPU / "evidence/eval", RESPONSE_LAUNCH):
            target = scratch / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(inputs / relative)

        for relative, script, label in (
            (CPU, "verify_cpu.py", "cpu-crossplay"),
            (RESPONSE, "verify_response.py", "matched-red-response"),
        ):
            study = scratch / relative
            study.mkdir(parents=True)
            for name in ("evidence", "launch-records", "registered-manifest-before-affinity.json"):
                archived = inputs / relative / name
                if archived.exists():
                    (study / name).symlink_to(archived)
            run(source / script, "--study-root", study)
            target = output / "verification" / label
            target.mkdir(parents=True)
            for name in ("verification.json", "verified-contrasts.csv", "summary.json"):
                generated = study / name
                if generated.exists():
                    shutil.copyfile(generated, target / name)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contrast_rows(output):
    inputs = output / "inputs"
    summaries = (
        ("saved-policy-crossplay", "cpu_crossplay_24", CPU / "evidence/analysis/crossplay-report.json"),
        ("matched-red-response", "fresh_response_4", RESPONSE / "evidence/analysis/response-report.json"),
        ("saved-policy-deviations", "gpu_checkpoint_audit_4", AUDIT / "evidence/analysis/audit-report.json"),
        ("checkpoint-progress", "replication_28", REPLICATION / "analysis/replication-report.json"),
    )
    stats = read(output / "statistics.json")["results"]
    rows = []
    for study, family, path in summaries:
        summary = read(inputs / path)
        original = summary.get("contrasts") or (
            summary["primary_equal_panel_gains"] + summary["secondary_per_opponent_gains"]
        )
        for record, replayed in zip(original, stats[family], strict=True):
            team = record.get("team")
            if family == "fresh_response_4":
                name = record["contrast"]
                team = "red" if name.startswith("selected_red_gain") else "blue"
            else:
                name = None
            mean, se, low, high = replayed["mean_se_low_high"]
            rows.append(
                {
                    "study": study,
                    "family": family,
                    "family_size": summary["protocol"]["family_size"],
                    "contrast": name,
                    "team": team,
                    "scope": record.get("scope", "fixed_opponent"),
                    "first_policy_steps": record.get(
                        "earlier_steps", record.get("challenger_steps", record.get("candidate_steps"))
                    ),
                    "second_policy_steps": record.get(
                        "later_steps", record.get("incumbent_steps", record.get("baseline_steps"))
                    ),
                    "opponent_steps_or_panel": record.get("opponent_steps", record.get("frozen_opponent_steps")),
                    "first_eval_reference": record.get(
                        "candidate_eval_id", record.get("challenger_reference", record.get("candidate_reference"))
                    ),
                    "second_eval_reference": record.get(
                        "baseline_eval_id", record.get("incumbent_reference", record.get("baseline_reference"))
                    ),
                    "n_paired_roots": record.get("n_paired_roots", record.get("paired_episode_roots")),
                    "own_return_first_minus_second": mean,
                    "standard_error": se,
                    "simultaneous_95_low": low,
                    "simultaneous_95_high": high,
                    "archive_report": (Path("inputs") / path).as_posix(),
                }
            )
    return rows


def matchup_rows(inputs):
    groups = (
        ("checkpoint-progress", "fixed-panel", REPLICATION / "evaluation/cells", "complete"),
        ("saved-policy-deviations", "gpu-audit", AUDIT / "evidence/eval", "complete"),
        ("saved-policy-crossplay", "cpu-matrix", CPU / "evidence/eval", "complete"),
        ("saved-policy-crossplay", "gpu-matrix", GPU / "evidence/eval", "reused"),
        ("matched-red-response", "validation", RESPONSE / "evidence/validation", "complete"),
        ("matched-red-response", "confirmation", RESPONSE / "evidence/confirmation", "complete"),
    )
    rows = []
    for study, cohort, directory, status in groups:
        for path in sorted((inputs / directory).glob("*.json")):
            record = read(path)
            blue, red = (record["policies"][team]["bundle_provenance"] for team in ("blue", "red"))
            roots = record.get("per_episode_seeds", record.get("episode_seeds"))
            rows.append(
                {
                    "study": study,
                    "cohort": cohort,
                    "cell": path.stem,
                    "status": status,
                    "source_run_id": record.get("eval_id", record.get("tracking_run_id")),
                    "blue_training_run_id": blue.get("train_run_id"),
                    "red_training_run_id": red.get("train_run_id"),
                    "blue_steps": blue["total_steps"],
                    "red_steps": red["total_steps"],
                    "blue_seed": blue["seed"],
                    "red_seed": red["seed"],
                    "observed_episodes": record["n_episodes"],
                    "first_observed_root": min(roots),
                    "last_observed_root": max(roots),
                    "blue_mean_return": record["blue_mean_return"],
                    "red_mean_return": record["red_mean_return"],
                    "evidence": (Path("inputs") / path.relative_to(inputs)).as_posix(),
                }
            )

    for cell in read(inputs / GPU / "partial-report.json")["missing_cells"]:
        rows.append(
            {
                "study": "saved-policy-crossplay",
                "cohort": "gpu-matrix",
                "cell": cell["name"],
                "status": "missing",
                "blue_steps": cell["blue_steps"],
                "red_steps": cell["red_steps"],
                "evidence": (Path("inputs") / GPU / "partial-report.json").as_posix(),
            }
        )
    return rows


def render(output):
    inputs = output / "inputs"
    stats, matchups = contrast_rows(output), matchup_rows(inputs)
    write_csv(output / "contrasts.csv", stats)
    write_csv(output / "matchups.csv", matchups)
    selection = RESPONSE / "evidence/selection.json"
    shutil.copyfile(inputs / selection, output / "response-selection.json")

    gpu = read(inputs / GPU / "partial-report.json")
    cpu = read(output / "verification/cpu-crossplay/summary.json")
    budget = read(inputs / BUDGET / "terminal-summary.json")
    selected = read(inputs / selection)["selected"]
    response_rows = [row for row in stats if row["study"] == "matched-red-response"]
    lines = [
        "# Nash co-training replay",
        "",
        f"Reproduced {len(stats)} registered contrasts from retained episode returns. ",
        "The CSV gives signed own-return differences with their original multiplicity corrections. ",
        "The matchup table links each observation group to the archived result, training identities, ",
        "and root range. Reused GPU cells point to the original audit executions.",
        "",
        "## Study status",
        "",
        "| Study | Replayed contrasts | Recorded coverage |",
        "| --- | ---: | --- |",
        "| Fixed-panel checkpoint progress | 28 | 37 cells, including the descriptive final pair |",
        "| Saved-policy deviations | 4 | 5 GPU audit cells, with archive recovery after job failure |",
        "| Matched Red response | 4 | 6 trainings, 18 validation cells, 6 confirmation cells |",
        f"| Saved-policy cross-play | 24 | 9 CPU cells. Earlier GPU job {gpu['slurm_state']} "
        f"with {len(gpu['missing_cells'])} missing cells and {gpu['reused_cells_verified']} reused audit cells |",
        f"| 10M vs 20M Red search | 0 | Job {budget['slurm_state']} before training or evaluation |",
        "",
        "## Matched Red response",
        "",
        f"The frozen validation choices are seed {selected['124800000']['seed']} for Blue124.8M "
        f"and seed {selected['150000000']['seed']} for Blue150M. Both Red gains use Red150 as the baseline.",
        "",
        "| Confirmation contrast | Gain | Simultaneous approximate 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for row in response_rows:
        lines.append(
            f"| `{row['contrast']}` | {row['own_return_first_minus_second']:+.2f} | "
            f"[{row['simultaneous_95_low']:.2f}, {row['simultaneous_95_high']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "The Blue150 result supplies a profitable unilateral Red deviation for the named pair. "
            "The search budget leaves best-response saturation and global exploitability open.",
            "",
            "## Saved-policy cross-play",
            "",
            "The registered CPU analysis supports opponent-dependent Red rankings. "
            f"Its supported graph has {len(cpu['supported_better_response_edges'])} unilateral response edges "
            "and contains no cycle. Temporal cycling remains open.",
            "",
            "The CPU and response intervals describe episode-root variation conditional on selected policies "
            "from one defender trajectory on one topology. Equal-opponent panel contrasts average within "
            "each root before estimating uncertainty.",
            "",
            "## Artifacts",
            "",
            "- [Contrasts](contrasts.csv) and [matchups](matchups.csv)",
            "- [Frozen response selection](response-selection.json)",
            "- [CPU provenance verification](verification/cpu-crossplay/verification.json) and "
            "[response provenance verification](verification/matched-red-response/verification.json)",
            "- [Retained analyzer](source/analyze.py) and [file hashes](manifest.json)",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hashes(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="replay the unpacked archive")
    compare.add_argument("--bundle", required=True, type=Path)
    rebuild = commands.add_parser("rebuild", help="replay a retained comparison")
    rebuild.add_argument("report", type=Path)
    for command in (compare, rebuild):
        command.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    source = (args.bundle if args.command == "compare" else args.report).resolve()
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("Choose a new output directory outside the input directory.")
    if HERE.parent.parent.name == "studies" and output.is_relative_to(HERE.parents[2]):
        parser.error("Choose an output directory outside the source checkout.")
    if args.command == "rebuild":
        saved = read(source / "manifest.json")
        if hashes(source) != saved["files"] or digest(HERE) != saved["files"]["source/analyze.py"]:
            parser.error("The saved report or analyzer differs from its retained manifest.")
    output.mkdir(parents=True)
    if args.command == "compare":
        snapshot(source, output)
    else:
        for name in ("source", "inputs"):
            shutil.copytree(source / name, output / name)
    replay(output)
    render(output)
    (output / "manifest.json").write_text(
        json.dumps({"files": hashes(output)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
