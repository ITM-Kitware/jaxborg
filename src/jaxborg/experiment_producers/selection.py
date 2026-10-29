"""Verify saved collation snapshots before an existing bundle analysis runs."""

from jaxborg.experiment_records import collate, validate_collation, validate_records

from .local import frozen_bundle


def verify_frozen_selection(selection, root):
    validate_collation(selection)
    records = selection["records"]
    indexed = validate_records(records)
    expected = frozen_bundle(root)
    protocol = next(r for r in expected if r["kind"] == "protocol")
    if selection["protocol_id"] != protocol["id"]:
        raise ValueError("Selection must identify this frozen bundle protocol")
    for item in expected:
        if indexed.get(item["id"]) != item:
            raise ValueError(f"Saved record differs from retained frozen bundle: {item['id']}")
    included = set(selection["included"])
    selected_attempts = {
        item["trial_id"]: item["id"]
        for item in expected
        if item["kind"] == "attempt" and isinstance(item["trial_id"], str)
    }
    selected_evaluations = [item["id"] for item in expected if item["kind"] == "evaluation"]
    if included != set(selected_attempts.values()) | set(selected_evaluations):
        raise ValueError("Frozen analysis requires exactly the recorded scientific attempts and evaluations")
    rebuilt = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts=selected_attempts,
        selected_evaluations=selected_evaluations,
        analysis=selection["analysis"],
    )
    if selection != rebuilt:
        raise ValueError("Saved collation differs from its explicit selection")
