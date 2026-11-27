"""Verify one MLflow evaluation and rebuild its mean Blue return from episode records."""

import argparse
import json
from pathlib import Path
from statistics import mean

from mlflow import MlflowClient

from jaxborg.evaluation.capture_store import load_records
from jaxborg.evaluation.tracking import TAG_PREFIX


def rebuild_blue_return(client, run_id, destination):
    tags = client.get_run(run_id).data.tags
    evaluation_id = tags.get(TAG_PREFIX + "evaluation_id")
    if not evaluation_id or tags.get(TAG_PREFIX + "evidence_path") != "evidence":
        raise ValueError("Select an evaluation run with delivered evidence")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(client.download_artifacts(run_id, "evidence", str(destination)))
    records = load_records(root, verify=True)
    evaluation = next((r for r in records if r["kind"] == "evaluation" and r["id"] == evaluation_id), None)
    if evaluation is None or evaluation["attempt_id"] != tags.get(TAG_PREFIX + "attempt_id"):
        raise ValueError("MLflow and downloaded evaluation identities differ")
    if evaluation["status"] != "succeeded":
        raise ValueError("A completed evaluation is required for a mean return")
    metric = next((m for m in evaluation["metrics"] if m["name"] == "blue_return" and m["unit"] == "score"), None)
    if metric is None:
        raise ValueError("The selected evaluation has no mean Blue return")
    observations = [
        json.loads((root / r["uri"]).read_text())
        for r in records
        if r["kind"] == "artifact"
        and r["producer_id"] == evaluation_id
        and r["role"] == "evaluation episode observation"
    ]
    value = mean(row["reward"] for row in observations)
    if value != metric["value"]:
        raise ValueError("Episode mean differs from the retained evaluation metric")
    return {"run_id": run_id, "evaluation_id": evaluation_id, "episodes": len(observations), "blue_return": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = rebuild_blue_return(MlflowClient(tracking_uri=args.tracking_uri), args.run_id, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
