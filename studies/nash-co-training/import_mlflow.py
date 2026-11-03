"""Mirror a retained Nash bundle into a new local MLflow store for inspection."""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.parse import unquote, urlparse

from mlflow import MlflowClient
from mlflow.entities import Metric, Param

CONTEXT = "historical-context"
STUDY_MAP = Path(__file__).with_name("archive-studies.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def artifact_root(run):
    uri = urlparse(run.info.artifact_uri)
    if uri.scheme != "file" or uri.netloc:
        raise ValueError(f"Expected a local file artifact root, got {run.info.artifact_uri!r}")
    return Path(unquote(uri.path))


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative archive path {value!r}")
    return path


def run_fields(record):
    """Read each export's native fields while retaining its original record."""
    if "info" in record:
        info = record["info"]
        histories = record["metric_histories"]
        points = (point for history in histories.values() for point in history)
        snapshots = {key: value for key, value in record["metrics"].items() if not histories.get(key)}
        return info, record["params"], record["tags"], points, snapshots

    info = record["run"]
    tables = record["tables"]
    params = {row["key"]: row["value"] for row in tables["params"]}
    tags = {row["key"]: row["value"] for row in tables["tags"]}
    points = iter(tables["metrics"])
    keys = {row["key"] for row in tables["metrics"]}
    snapshots = {row["key"]: row["value"] for row in tables["latest_metrics"] if row["key"] not in keys}
    return info, params, tags, points, snapshots


def place_run(name, tags, studies):
    campaign = tags.get("campaign")
    if campaign in studies:
        return campaign, "campaign-tag"
    # One analysis row lacks campaign metadata but retains the exact study prefix.
    for study_id in studies:
        if name.startswith(study_id + "-"):
            return study_id, "run-name-prefix"
    return None, "unassigned"


def import_run(client, record, original_id, source_store, studies, experiments):
    info, params, tags, points, snapshots = run_fields(record)
    name = info.get("run_name") or info.get("name") or original_id
    study_id, membership_source = place_run(name, tags, studies)
    group = studies[study_id]["group"] if study_id else CONTEXT
    run = client.create_run(
        experiments[group]["experiment_id"],
        start_time=info["start_time"],
        run_name=name,
        tags={
            **tags,
            "jaxborg.discovery.study_id": group,
            "nash.archive.record_kind": "historical-run",
            "nash.archive.original_run_id": original_id,
            "nash.archive.original_experiment_id": str(info["experiment_id"]),
            "nash.archive.source_store": source_store,
            "nash.archive.membership_source": membership_source,
            "nash.archive.study_id": study_id or "unassigned",
            "nash.archive.original_status": info["status"],
            "nash.archive.snapshot_only_count": str(len(snapshots)),
        },
    )
    run_id = run.info.run_id
    root = artifact_root(run)
    write_json(root / "archive/source-run.json", record)
    client.log_batch(run_id, params=[Param(key, value) for key, value in params.items()])
    count = 0
    while batch := list(islice(points, 1000)):
        metrics = [
            Metric(
                row["key"],
                float("nan") if row.get("is_nan") else row["value"],
                row["timestamp"],
                row["step"],
            )
            for row in batch
        ]
        client.log_batch(run_id, metrics=metrics)
        count += len(metrics)
    if snapshots:
        # The placeholder timestamp and step carry no historical observation meaning.
        timestamp = run.info.start_time
        client.log_batch(
            run_id,
            metrics=[Metric(f"snapshot.{key}", value, timestamp, 0) for key, value in snapshots.items()],
        )
    client.set_terminated(run_id, status=info["status"], end_time=info["end_time"])
    return {
        "run_id": run_id,
        "group": group,
        "study_id": study_id,
        "membership_source": membership_source,
        "source_store": source_store,
        "metric_history_points": count,
        "latest_only_metrics": len(snapshots),
    }, root


def link_catalog(bundle, store, catalog, studies, roots, archives):
    objects = store / "objects"
    objects.mkdir()
    copied = set()
    placements = {}
    per_run = Counter()
    catalog_run_root = Path(catalog["store"]) / "artifacts"
    for item in catalog["files"]:
        digest = item["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid object checksum {digest!r}")
        object_path = objects / digest
        if digest not in copied:
            shutil.copyfile(bundle / "objects" / digest, object_path)
            with object_path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Archived object hash mismatch for {item['path']}")
            object_path.chmod(0o444)
            copied.add(digest)
            if len(copied) % 500 == 0:
                print(f"Verified and staged {len(copied)} unique objects", flush=True)
        if object_path.stat().st_size != item["bytes"]:
            raise ValueError(f"Archived object size mismatch for {item['path']}")

        original = Path(item["path"])
        if not original.is_absolute() or item["path"] in placements:
            raise ValueError(f"Invalid or repeated original path {item['path']!r}")
        if original.is_relative_to(catalog_run_root):
            owner, container, *suffix = original.relative_to(catalog_run_root).parts
            if container != "artifacts" or owner not in roots:
                raise ValueError(f"Unknown original artifact owner {item['path']!r}")
            run_id, root = roots[owner]
            artifact = relative_path(Path(*suffix))
            per_run[owner] += 1
        else:
            study_id = next((role.removeprefix("study:") for role in item["roles"] if role.startswith("study:")), None)
            group = studies[study_id]["group"] if study_id else CONTEXT
            run_id, root = archives[group]
            artifact = Path("archived-paths") / relative_path(original.relative_to("/"))

        destination = root / artifact
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(object_path, destination)
        placements[item["path"]] = {
            "run_id": run_id,
            "artifact_path": artifact.as_posix(),
            "sha256": digest,
            "bytes": item["bytes"],
        }
    return placements, per_run, len(copied)


def copy_report_inputs(bundle, studies, archives, analysis):
    for group, (_, root) in archives.items():
        subset = [study for study in studies.values() if study["group"] == group]
        write_json(root / "archive/studies.json", subset)
        for source in sorted({relative_path(study["report"]).parent for study in subset}):
            shutil.copytree(bundle / source, root / "bundle" / source)
    _, root = archives[CONTEXT]
    # Retain the report tree and exact portable replay entry points in one location.
    shutil.copytree(bundle / "reports", root / "bundle/reports")
    for name in (
        "artifact-catalog.json",
        "checkpoint-index.json",
        "legacy-run-records.json",
        "PUBLICATION-RECORD.json",
        "REPRODUCIBILITY.md",
        "reproduce.py",
        "run-records.json",
        "SHA256SUMS",
        "jaxborg.bundle",
        "jaxborg-complete-history.bundle",
        "published-code.bundle",
    ):
        shutil.copyfile(bundle / name, root / "bundle" / name)
    shutil.copytree(
        bundle / "published-code/research/reproduction",
        root / "bundle/published-code/research/reproduction",
    )
    if analysis is not None:
        shutil.copytree(analysis, root / "replay")


def complete_portable_bundle(bundle, store, root):
    portable = root / "bundle"
    for name in (
        "checkpoint-index.csv",
        "dependency-sources.tar.gz",
        "installed-git-dependencies.tar.gz",
        "reproduced-statistics.json",
        *sorted(path.name for path in bundle.glob("source-*.tar.gz")),
    ):
        shutil.copyfile(bundle / name, portable / name)
    objects = portable / "objects"
    objects.mkdir(exist_ok=True)
    for source in (store / "objects").iterdir():
        target = objects / source.name
        if not target.exists():
            os.link(source, target)
        elif not target.samefile(source):
            raise ValueError(f"Conflicting portable object {target}")


def write_overview(client, root, experiments, replay_artifact):
    sections = [
        "# Jaxborg Nash retrospective\n",
        "Browse each study's archival run and its Artifacts tab. Historical runs retain their original IDs in tags.\n",
    ]
    for group, entry in experiments.items():
        if group != CONTEXT:
            link = f"/#/experiments/{entry['experiment_id']}/runs/{entry['archive_run_id']}"
            sections.append(f"- [{group}]({link})")
    sections.extend(
        [
            "\nOpen `bundle/reports/overnight-experiments/index.html` in the Artifacts tab for the archival findings.",
            "Download `bundle/` to use the packaged analysis, source, and object catalog locally.",
            "`archive/catalog-placement.json` resolves each original artifact path to a local run and path.",
        ]
    )
    if replay_artifact:
        sections.append("Read `replay/report.md`, `replay/contrasts.csv`, and `replay/matchups.csv` for the replay.")
    overview = "\n".join(sections) + "\n"
    (root / "archive/README.md").write_text(overview, encoding="utf-8")
    client.set_tag(experiments[CONTEXT]["archive_run_id"], "mlflow.note.content", overview)


def import_bundle(bundle, store, analysis):
    studies = {study["id"]: study for study in read_json(STUDY_MAP)}
    catalog = read_json(bundle / "artifact-catalog.json")
    current = read_json(bundle / "run-records.json")
    legacy = read_json(bundle / "legacy-run-records.json")["records"]
    if set(studies) != set(catalog["study_names"]) or set(current) & set(legacy):
        raise ValueError("Unexpected study coverage or colliding original run IDs")
    checkout = Path(__file__).resolve().parents[2]
    if store.is_relative_to(bundle) or store.is_relative_to(checkout):
        raise ValueError("Choose a fresh MLflow store outside the bundle and checkout")
    if analysis is not None:
        if not (analysis / "manifest.json").is_file() or store.is_relative_to(analysis):
            raise ValueError("Choose a completed analysis directory with manifest.json, separate from the new store")
    store.mkdir(parents=True, exist_ok=False)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    client = MlflowClient(tracking_uri=f"sqlite:///{store / 'mlflow.db'}")
    experiments = {}
    archives = {}
    for group in sorted({study["group"] for study in studies.values()} | {CONTEXT}):
        name = f"Jaxborg Nash / {group}"
        experiment_id = client.create_experiment(name, artifact_location=(store / "artifacts" / group).as_uri())
        run = client.create_run(
            experiment_id,
            run_name="Study archive and reports",
            tags={"nash.archive.record_kind": "study-archive", "jaxborg.discovery.study_id": group},
        )
        experiments[group] = {"name": name, "experiment_id": experiment_id, "archive_run_id": run.info.run_id}
        archives[group] = (run.info.run_id, artifact_root(run))

    runs, roots = {}, {}
    for records in (current, legacy):
        for original_id, record in records.items():
            imported, root = import_run(
                client,
                record,
                original_id,
                record.get("source_database", catalog["tracking_uri"]),
                studies,
                experiments,
            )
            runs[original_id] = imported
            roots[original_id] = (imported["run_id"], root)
            if len(runs) % 20 == 0:
                print(f"Imported {len(runs)} historical runs", flush=True)
    placements, per_run, unique_objects = link_catalog(bundle, store, catalog, studies, roots, archives)
    if unique_objects != catalog["unique_objects"]:
        raise ValueError("Staged object count differs from the archive catalog")
    copy_report_inputs(bundle, studies, archives, analysis)
    for original_id, record in runs.items():
        record["owned_catalog_paths"] = per_run[original_id]
        client.set_tag(record["run_id"], "nash.archive.owned_catalog_paths", str(per_run[original_id]))
    for run_id, _ in archives.values():
        client.set_terminated(run_id)
    summary = {
        "experiments": experiments,
        "historical_runs": runs,
        "archive_studies": list(studies.values()),
        "current_run_count": len(current),
        "legacy_run_count": len(legacy),
        "metric_history_points": sum(run["metric_history_points"] for run in runs.values()),
        "latest_only_metrics": sum(run["latest_only_metrics"] for run in runs.values()),
        "catalog_paths": len(placements),
        "unique_catalog_objects": unique_objects,
        "missing_referenced_paths": catalog["missing_referenced_paths"],
        "unresolved_source_run_ids": sorted(set(catalog["unresolved_run_ids"]) - set(runs)),
        "replay_artifact": (
            {"run_id": archives[CONTEXT][0], "artifact_path": "replay"} if analysis is not None else None
        ),
    }
    _, context_root = archives[CONTEXT]
    complete_portable_bundle(bundle, store, context_root)
    write_json(context_root / "archive/catalog-placement.json", placements)
    write_json(context_root / "archive/import-summary.json", summary)
    write_overview(client, context_root, experiments, summary["replay_artifact"])
    write_json(store / "import-summary.json", summary)
    print(f"Imported {len(runs)} historical runs and {len(placements)} artifact paths into {store}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Unpacked reproducibility bundle")
    parser.add_argument("--store", type=Path, required=True, help="Fresh local SQLite MLflow directory")
    parser.add_argument("--analysis", type=Path, help="Optional previously reconstructed results to mirror")
    args = parser.parse_args()
    import_bundle(args.bundle.resolve(), args.store.resolve(), args.analysis.resolve() if args.analysis else None)


if __name__ == "__main__":
    main()
