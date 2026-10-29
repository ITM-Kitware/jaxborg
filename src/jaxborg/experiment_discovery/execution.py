"""Best-effort native MLflow navigation over authoritative execution snapshots."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException

from jaxborg.mlflow_setup import tracking_client, training_run_metadata

from .mlflow import TAG_PREFIX, execution_tags

_LOG = logging.getLogger(__name__)


def destination_configured():
    """Evaluation indexing requires an explicitly chosen destination."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("JAXBORG_EXP_DIR"))


@dataclass
class ExecutionRun:
    client: object
    run_id: str
    protocol: dict
    configuration: dict | None
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
            owners = [record for record in records if record["kind"] in {"attempt", "evaluation"}]
            self.client.log_dict(self.run_id, {"protocol": self.protocol, "records": owners}, "execution.json")
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


def start_execution(protocol, records, configuration):
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
    return ExecutionRun(client, run.info.run_id, protocol, configuration)


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
