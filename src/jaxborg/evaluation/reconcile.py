"""Seal interrupted executions only after checking their local writer is absent."""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from jaxborg.evaluation.capture_store import Capture, finish_evaluation, finish_training, load_records, record_facts
from jaxborg.evaluation.provenance import writer_identity


def reconcile_interrupted(root, attempt_id, *, reason):
    root = Path(root).resolve()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Interruption requires an explicit operator reason")
    records = load_records(root)
    attempt = next((record for record in records if record["id"] == attempt_id and record["kind"] == "attempt"), None)
    if attempt is None or attempt["status"] != "running":
        raise ValueError("Reconciliation requires an unresolved running attempt")
    writer = attempt["facts"].get("resolved_config", {}).get("value", {}).get("execution", {}).get("writer")
    local = writer_identity()
    if (
        not writer
        or any(
            not writer.get(key) or writer.get(key) != local.get(key) for key in ("hostname", "boot_id", "pid_namespace")
        )
        or type(writer.get("pid")) is not int
        or writer["pid"] <= 0
    ):
        raise ValueError(
            "Writer identity is unavailable or belongs to another host, boot or PID namespace; "
            "inspect in the original writer environment"
        )
    try:
        os.kill(writer["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Recorded writer is still alive; reconciliation cannot stop an active execution")
    evaluation = next(
        (record for record in records if record["kind"] == "evaluation" and record["attempt_id"] == attempt_id), None
    )
    directory = root / "attempts" / attempt_id.rsplit(":", 1)[1]
    capture = Capture(root, directory, attempt_id, evaluation["id"] if evaluation else None, {})
    record_facts(
        capture,
        {
            "interruption": {
                "value": {
                    "reason": reason,
                    "writer": writer,
                    "checked_by_pid": os.getpid(),
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                    "check": "local writer PID absent",
                },
                "source": "explicit operator reconciliation after local process check",
            }
        },
    )
    finish = finish_evaluation if evaluation else finish_training
    finish(capture, status="cancelled", error=InterruptedError(reason))
    from jaxborg.experiment_discovery.execution import destination_configured, sync_execution

    if destination_configured():
        try:
            sync_execution(root, attempt_id)
        except Exception as error:
            logging.getLogger(__name__).warning(
                "Tracking synchronization failed after scientific cancellation: %s", error
            )
    return load_records(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--attempt", required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--reason")
    operation.add_argument(
        "--sync-tracking", action="store_true", help="Refresh the existing index from terminal evidence"
    )
    args = parser.parse_args()
    try:
        if args.sync_tracking:
            from jaxborg.experiment_discovery.execution import sync_execution

            result = sync_execution(args.root, args.attempt)
        else:
            result = reconcile_interrupted(args.root, args.attempt, reason=args.reason)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Reconciliation rejected: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
