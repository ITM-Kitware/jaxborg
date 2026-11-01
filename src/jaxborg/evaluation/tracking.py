"""Best-effort native MLflow navigation over authoritative execution snapshots."""

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException

from jaxborg.mlflow_setup import tracking_client, training_run_metadata

TAG_PREFIX = "jaxborg.discovery."

_LOG = logging.getLogger(__name__)


def execution_tags(record, records):
    """Project scientific identities and observed status without claiming artifact availability."""
    index = {r["id"]: r for r in records}
    attempt_id = record.get("attempt_id")
    attempt = record if record["kind"] == "attempt" else index.get(attempt_id) if isinstance(attempt_id, str) else None
    protocol_id = attempt.get("protocol_id") if attempt else None
    protocol = index.get(protocol_id) if isinstance(protocol_id, str) else None
    tags = {
        TAG_PREFIX + "schema_version": "1",
        TAG_PREFIX + "record_id": record["id"],
        TAG_PREFIX + "record_kind": record["kind"],
        TAG_PREFIX + "scientific_status": record["status"],
    }
    for key, value in {
        "study_id": protocol["study_id"] if protocol else None,
        "protocol_id": protocol_id,
        "trial_id": attempt.get("trial_id") if attempt else None,
        "attempt_id": attempt["id"] if attempt else None,
        "retry_of": attempt.get("retry_of") if attempt else None,
        "execution_kind": attempt.get("execution_kind") if attempt else None,
    }.items():
        if isinstance(value, str):
            tags[TAG_PREFIX + key] = value
    facts = record["facts"]
    for key in ("policy_backend", "eval_env"):
        fact = facts.get(key)
        if fact and isinstance(fact.get("value"), (str, int, float, bool)):
            tags[TAG_PREFIX + key] = str(fact["value"])
    backend = facts.get("backend", facts.get("policy_backend", {})).get("value")
    source = facts.get("source", {}).get("value")
    revision = (
        facts["source_revision"].get("value")
        if "source_revision" in facts
        else source.get("git_commit")
        if isinstance(source, dict)
        else None
    )
    for key, value in (("backend", backend), ("source_revision", revision)):
        if isinstance(value, (str, int, float, bool)):
            tags[TAG_PREFIX + key] = str(value)
    for number, budget in enumerate(record["budgets"]):
        tags[f"{TAG_PREFIX}budget.{number}.unit"] = budget["unit"]
        for key in ("requested", "completed"):
            if isinstance(budget[key], (int, float)):
                tags[f"{TAG_PREFIX}budget.{number}.{key}"] = str(budget[key])
    correction = record.get("extensions", {}).get("evaluation_correction")
    if correction:
        tags[TAG_PREFIX + "corrects_evaluation_id"] = correction["values"]["evaluation_id"]
    if protocol and attempt and isinstance(attempt.get("trial_id"), str):
        cell = next(cell for cell in protocol["trials"] if cell["id"] == attempt["trial_id"])
        for key in ("condition_id", "replicate_id", "matched_unit_id"):
            if cell[key] is not None:
                tags[TAG_PREFIX + key] = cell[key]
    return tags


def destination_configured():
    """Evaluation indexing requires an explicitly chosen destination."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("JAXBORG_EXP_DIR"))


@dataclass
class ExecutionRun:
    client: object
    run_id: str
    protocol: dict
    configuration: dict | None
    root: Path
    data_incomplete: bool = False
    scientific_status: str = "running"

    def start_training(self, recipe, *, seed):
        """Label the owned run without activating a second fluent lifecycle."""
        try:
            tags, params = training_run_metadata(recipe, backend="jax", seed=seed)
            self.client.log_batch(
                self.run_id,
                tags=[RunTag(key, value) for key, value in tags.items()],
                params=[Param(key, value) for key, value in params.items()],
            )
            source = recipe.get("__source_path__")
            if source and Path(source).exists():
                self.log_artifact(source)
        except Exception as error:
            self.delivery_failed(error, data_incomplete=True)

    def log_metrics(self, values, *, step=0):
        try:
            timestamp = int(time.time() * 1000)
            self.client.log_batch(
                self.run_id, metrics=[Metric(key, float(value), timestamp, step) for key, value in values.items()]
            )
        except Exception as error:
            self.delivery_failed(error, data_incomplete=True)

    def log_artifact(self, path):
        try:
            self.client.log_artifact(self.run_id, str(path))
        except Exception as error:
            self.delivery_failed(error, data_incomplete=True)

    def update(self, records):
        """Project only retained observations; delivery cannot change scientific status."""
        try:
            attempt = records[0]
            self.scientific_status = attempt["status"]
            tags = execution_tags(attempt, [self.protocol, *records])
            if len(records) > 1 and records[1]["kind"] == "evaluation":
                tags[TAG_PREFIX + "evaluation_id"] = records[1]["id"]
            tags.update(
                {
                    TAG_PREFIX + "role": "execution",
                    TAG_PREFIX + "state": "unverified",
                    TAG_PREFIX + "delivery": "failed" if self.data_incomplete else "current",
                    TAG_PREFIX + "data_incomplete": str(self.data_incomplete).lower(),
                }
            )
            if self.configuration is not None:
                self.client.log_dict(self.run_id, self.configuration, "configuration.json")
            self.client.log_dict(self.run_id, {"protocol": self.protocol, "records": records}, "execution.json")
            if attempt["status"] != "running":
                self.upload_evidence(attempt["id"])
                tags[TAG_PREFIX + "evidence_path"] = "evidence"
            self.client.log_batch(self.run_id, tags=[RunTag(key, value) for key, value in tags.items()])
            if attempt["status"] != "running":
                status = {"succeeded": "FINISHED", "failed": "FAILED", "cancelled": "KILLED"}[attempt["status"]]
                self.client.set_terminated(self.run_id, status="FAILED" if self.data_incomplete else status)
            return True
        except KeyboardInterrupt as error:
            self.delivery_failed(error, records[0]["status"])
            raise
        except Exception as error:
            self.delivery_failed(error, records[0]["status"])
            return False

    def upload_evidence(self, attempt_id):
        """Upload a verified native store slice on this execution's existing run."""
        from jaxborg.evaluation.capture_store import execution_files, load_records

        with tempfile.TemporaryDirectory(prefix="execution-evidence-") as temporary:
            snapshot = Path(temporary)
            for source in execution_files(self.root, attempt_id):
                destination = snapshot / source.relative_to(self.root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            load_records(snapshot)
            self.client.log_artifacts(self.run_id, str(snapshot), artifact_path="evidence")

    def delivery_failed(self, error, scientific_status=None, *, data_incomplete=False):
        self.data_incomplete |= data_incomplete
        scientific_status = scientific_status or self.scientific_status
        _LOG.warning("Execution tracking delivery failed for %s: %s", self.run_id, error)
        try:
            self.client.set_tag(self.run_id, TAG_PREFIX + "delivery", "failed")
            self.client.set_tag(self.run_id, TAG_PREFIX + "data_incomplete", str(self.data_incomplete).lower())
            self.client.set_tag(self.run_id, TAG_PREFIX + "scientific_status", scientific_status)
            if scientific_status != "running":
                self.client.set_terminated(self.run_id, status="FAILED")
        except Exception:
            pass


def start_execution(protocol, records, configuration, *, root):
    """Create one owned tracking run after the durable attempt already exists."""
    try:
        client = tracking_client()
        algorithm = (configuration.get("recipe") or {}).get("algorithm", "research")
        name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or f"{algorithm}-cc4"
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(name)
            except MlflowException as error:
                if error.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment_id = client.get_experiment_by_name(name).experiment_id
        else:
            experiment_id = experiment.experiment_id
        attempt = records[0]
        tags = execution_tags(attempt, [protocol, *records]) | {
            TAG_PREFIX + "role": "execution",
            TAG_PREFIX + "state": "unverified",
            TAG_PREFIX + "delivery": "pending",
        }
        if len(records) > 1 and records[1]["kind"] == "evaluation":
            tags[TAG_PREFIX + "evaluation_id"] = records[1]["id"]
        run = client.create_run(experiment_id, tags=tags, run_name=f"{attempt['execution_kind']} {attempt['trial_id']}")
    except Exception as error:
        _LOG.warning("Execution tracking creation failed for %s: %s", records[0]["id"], error)
        return None
    return ExecutionRun(client, run.info.run_id, protocol, configuration, Path(root).resolve())


def sync_execution(root, attempt_id):
    """Refresh one existing live index from terminal evidence, without creating or resealing it."""
    from jaxborg.evaluation.capture_store import load_records

    if not destination_configured():
        raise ValueError("Tracking synchronization requires an explicit destination")
    root = Path(root).resolve()
    records = load_records(root)
    attempt = next((record for record in records if record["id"] == attempt_id and record["kind"] == "attempt"), None)
    if attempt is None or attempt["status"] == "running":
        raise ValueError("Tracking synchronization requires a terminal attempt")
    protocol = next(record for record in records if record["id"] == attempt["protocol_id"])
    client = tracking_client()
    matches = []
    token = None
    while True:
        page = client.search_experiments(view_type=ViewType.ACTIVE_ONLY, page_token=token)
        for experiment in page:
            matches.extend(
                client.search_runs(
                    [experiment.experiment_id],
                    filter_string=(
                        f"tags.`{TAG_PREFIX}role` = 'execution' AND tags.`{TAG_PREFIX}attempt_id` = '{attempt_id}'"
                    ),
                    max_results=2,
                )
            )
            if len(matches) > 1:
                raise ValueError("Multiple execution runs match the exact attempt identity")
        token = page.token
        if not token:
            break
    if len(matches) != 1:
        raise ValueError("No existing execution run matches the exact attempt identity")
    run = matches[0]
    expected = execution_tags(attempt, records)
    for key in ("record_id", "protocol_id", "trial_id"):
        if run.data.tags.get(TAG_PREFIX + key) != expected[TAG_PREFIX + key]:
            raise ValueError("Execution run identity differs from the retained attempt")
    directory = root / "attempts" / attempt_id.rsplit(":", 1)[1]
    snapshot = json.loads((directory / "terminal.json").read_text())
    configuration = attempt["facts"].get("resolved_config", {}).get("value")
    binding = ExecutionRun(
        client,
        run.info.run_id,
        protocol,
        configuration,
        root,
        data_incomplete=run.data.tags.get(TAG_PREFIX + "data_incomplete") == "true",
    )
    if not binding.update(snapshot):
        raise OSError("Tracking delivery failed; terminal scientific evidence is unchanged")
    return {
        "run_id": run.info.run_id,
        "attempt_id": attempt_id,
        "scientific_status": attempt["status"],
        "delivery": "failed" if binding.data_incomplete else "current",
        "data_incomplete": binding.data_incomplete,
    }
