"""A small executable schema and semantic checks for retained evidence."""

import hashlib
import json
import math
import re
from copy import deepcopy


class RecordError(ValueError):
    """A record or selection violates the retained evidence contract."""


# These specifications are the canonical v1 schema, used directly by validation.
SCHEMAS = {
    "protocol": {
        "study_id": str,
        "revision": str,
        "selection_rule": str,
        "required_facts": dict,
        "trials": list,
        "metric_definitions": list,
    },
    "attempt": {
        "protocol_id": (str, dict),
        "trial_id": (str, dict),
        "status": str,
        "execution_kind": str,
        "retry_of": (str, dict, type(None)),
        "facts": dict,
        "budgets": list,
        "inputs": list,
    },
    "artifact": {
        "producer_id": (str, type(None)),
        "uri": str,
        "sha256": str,
        "size_bytes": int,
        "role": str,
    },
    "evaluation": {
        "attempt_id": (str, dict),
        "inputs": list,
        "status": str,
        "facts": dict,
        "budgets": list,
        "metrics": list,
        "raw_artifacts": list,
    },
}
STATUSES = {"planned", "running", "succeeded", "failed", "cancelled"}


def _require(ok, message):
    if not ok:
        raise RecordError(message)


def _fields(value, spec, context, optional=()):
    _require(isinstance(value, dict), f"{context}: expected object")
    _require(set(spec) <= value.keys(), f"{context}: missing fields {sorted(set(spec) - value.keys())}")
    _require(
        value.keys() <= set(spec) | set(optional),
        f"{context}: unknown fields {sorted(value.keys() - set(spec) - set(optional))}",
    )
    for key, expected in spec.items():
        item = value[key]
        _require(
            isinstance(item, expected) and not (isinstance(item, bool) and expected is int),
            f"{context}.{key}: invalid type",
        )
        if isinstance(item, str):
            _require(bool(item.strip()), f"{context}.{key}: empty string")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _facts(facts, context):
    for name, fact in facts.items():
        _require(isinstance(fact, dict), f"{context}.{name}: expected fact object")
        if fact.get("value") is None:
            _fields(fact, {"value": type(None), "reason": str, "source": str}, f"{context}.{name}")
        else:
            _fields(fact, {"value": object, "source": str}, f"{context}.{name}")


def validate_records(records):
    """Validate v1 fields and graph links, returning a detached ID index."""
    index = {}
    trials = {}
    for original in records:
        record = deepcopy(original)
        _require(isinstance(record, dict), "record: expected object")
        kind = record.get("kind")
        _require(isinstance(kind, str) and kind in SCHEMAS, f"unknown record kind: {kind}")
        _fields(record, {"schema_version": int, "kind": str, "id": str, **SCHEMAS[kind]}, kind, ("extensions",))
        _require(record["schema_version"] == 1, "unsupported schema version")
        _require(record["id"] not in index, f"duplicate record ID: {record['id']}")
        if "extensions" in record:
            _require(isinstance(record["extensions"], dict), "extensions: expected object")
        for namespace, extension in record.get("extensions", {}).items():
            _require(isinstance(namespace, str) and bool(namespace.strip()), "invalid extension namespace")
            _fields(extension, {"schema_version": int, "source": str, "values": dict}, "extension")
            _require(extension["schema_version"] > 0, "invalid extension version")
        for key in ("protocol_id", "trial_id", "retry_of", "attempt_id"):
            if isinstance(record.get(key), dict):
                _fields(record[key], {"value": type(None), "reason": str, "source": str}, key)
        index[record["id"]] = record
        if kind == "protocol":
            _require(record["selection_rule"] == "explicit", "unsupported selection rule")
            _fields(record["required_facts"], {"attempt": list, "evaluation": list}, "required_facts")
            for names_required in record["required_facts"].values():
                _require(
                    all(isinstance(name, str) and name.strip() for name in names_required), "invalid required fact name"
                )
            cell_keys = set()
            for trial in record["trials"]:
                _fields(
                    trial,
                    {"id": str, "condition_id": str, "replicate_id": str, "matched_unit_id": (str, type(None))},
                    "trial",
                )
                _require(trial["id"] not in trials, f"duplicate trial ID: {trial['id']}")
                cell_key = (trial["condition_id"], trial["replicate_id"], trial["matched_unit_id"])
                _require(cell_key not in cell_keys, "duplicate scientific trial cell")
                cell_keys.add(cell_key)
                trials[trial["id"]] = record["id"]
            names = set()
            for metric in record["metric_definitions"]:
                _fields(
                    metric,
                    {key: str for key in ("name", "unit", "direction", "aggregation", "sampling_unit", "uncertainty")},
                    "metric definition",
                )
                _require(metric["direction"] in {"higher", "lower", "neutral"}, "invalid metric direction")
                _require(metric["name"] not in names, "duplicate metric definition")
                names.add(metric["name"])
        elif kind == "artifact":
            _require(bool(re.fullmatch(r"[0-9a-f]{64}", record["sha256"])), "invalid sha256")
            _require(record["size_bytes"] >= 0, "negative artifact size")
        else:
            _require(record["status"] in STATUSES, "invalid status")
            _facts(record["facts"], record["id"])
            for budget in record["budgets"]:
                _fields(budget, {"unit": str, "requested": object, "completed": object}, "budget")
                for key in ("requested", "completed"):
                    if isinstance(budget[key], dict):
                        _fields(budget[key], {"value": type(None), "reason": str, "source": str}, f"budget.{key}")
                    else:
                        _require(_number(budget[key]) and budget[key] >= 0, f"invalid {key} budget")
            if kind == "evaluation":
                names = set()
                for metric in record["metrics"]:
                    _fields(metric, {"name": str, "value": object, "unit": str}, "metric")
                    _require(_number(metric["value"]), "invalid metric value")
                    _require(metric["name"] not in names, "duplicate evaluation metric")
                    names.add(metric["name"])

    _require(not (set(trials) & set(index)), "trial and record IDs collide")

    def link(identity, kinds):
        _require(isinstance(identity, str), "reference must be a string")
        _require(identity in index, f"dangling reference: {identity}")
        target = index[identity]
        _require(target["kind"] in kinds, f"wrong-kind reference: {identity}")
        return target

    for record in index.values():
        kind = record["kind"]
        if kind == "attempt":
            if isinstance(record["protocol_id"], str):
                link(record["protocol_id"], {"protocol"})
            if isinstance(record["trial_id"], str):
                _require(isinstance(record["protocol_id"], str), "known trial requires known protocol")
                _require(
                    trials.get(record["trial_id"]) == record["protocol_id"], "attempt trial does not belong to protocol"
                )
            seen = {record["id"]}
            cursor = record
            while isinstance(cursor["retry_of"], str):
                cursor = link(cursor["retry_of"], {"attempt"})
                _require(cursor["id"] not in seen, "retry cycle")
                seen.add(cursor["id"])
                if isinstance(cursor["protocol_id"], str) and isinstance(record["protocol_id"], str):
                    _require(cursor["protocol_id"] == record["protocol_id"], "retry must retain protocol identity")
                if isinstance(cursor["trial_id"], str) and isinstance(record["trial_id"], str):
                    _require(cursor["trial_id"] == record["trial_id"], "retry must retain trial identity")
        elif kind == "artifact" and record["producer_id"] is not None:
            link(record["producer_id"], {"attempt", "evaluation"})
        elif kind == "evaluation" and isinstance(record["attempt_id"], str):
            attempt = link(record["attempt_id"], {"attempt"})
            if isinstance(attempt["protocol_id"], str):
                protocol = link(attempt["protocol_id"], {"protocol"})
                definitions = {m["name"]: m for m in protocol["metric_definitions"]}
                for metric in record["metrics"]:
                    _require(metric["name"] in definitions, "undefined metric")
                    _require(metric["unit"] == definitions[metric["name"]]["unit"], "metric unit mismatch")
        if kind in {"attempt", "evaluation"}:
            for identity in record["inputs"] + record.get("raw_artifacts", []):
                link(identity, {"artifact"})
    return index


def collate(records, *, protocol_id, selected_attempts, selected_evaluations, analysis):
    """Freeze explicit successful selections; never choose by name or recency."""
    index = validate_records(records)
    _require(isinstance(protocol_id, str), "protocol selection must be a string")
    _require(
        isinstance(selected_attempts, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in selected_attempts.items()),
        "invalid attempt selection",
    )
    _require(
        isinstance(selected_evaluations, list) and all(isinstance(v, str) for v in selected_evaluations),
        "invalid evaluation selection",
    )
    _fields(analysis, {"source": str, "configuration": dict}, "analysis")
    _require(protocol_id in index and index[protocol_id]["kind"] == "protocol", "unknown protocol")
    protocol = index[protocol_id]

    def required_known(record, kind):
        for name in protocol["required_facts"][kind]:
            _require(
                name in record["facts"] and record["facts"][name]["value"] is not None,
                f"selected {kind} lacks required known fact: {name}",
            )

    cells = {trial["id"]: trial for trial in protocol["trials"]}
    _require(set(selected_attempts) <= cells.keys(), "selection contains unknown trial")
    _require(len(selected_evaluations) == len(set(selected_evaluations)), "duplicate evaluation selection")
    included, excluded, missing, rows = [], [], [], []
    for trial_id in sorted(cells):
        cell = cells[trial_id]
        identity = selected_attempts.get(trial_id)
        if identity is None:
            reason = (
                "no_attempt"
                if not any(r["kind"] == "attempt" and r["trial_id"] == trial_id for r in index.values())
                else "no_explicit_selection"
            )
            missing.append({"trial_id": trial_id, "reason": reason, **cell})
            continue
        attempt = index.get(identity)
        _require(attempt is not None and attempt["kind"] == "attempt", "selected attempt missing or wrong kind")
        _require(
            attempt["trial_id"] == trial_id and attempt["protocol_id"] == protocol_id,
            "selected attempt belongs to another trial",
        )
        _require(attempt["status"] == "succeeded", "selected attempt is not succeeded")
        required_known(attempt, "attempt")
        included.append(identity)
        rows.append({**cell, "trial_id": trial_id, "attempt_id": identity, "evaluation_id": None})
    for identity in sorted(selected_evaluations):
        evaluation = index.get(identity)
        _require(
            evaluation is not None and evaluation["kind"] == "evaluation", "selected evaluation missing or wrong kind"
        )
        _require(evaluation["status"] == "succeeded", "selected evaluation is not succeeded")
        _require(isinstance(evaluation["attempt_id"], str), "selected evaluation has unknown attempt")
        required_known(evaluation, "evaluation")
        _require(evaluation["attempt_id"] in selected_attempts.values(), "evaluation attempt is not selected")
        included.append(identity)
        attempt = index[evaluation["attempt_id"]]
        rows.append(
            {
                **cells[attempt["trial_id"]],
                "trial_id": attempt["trial_id"],
                "attempt_id": attempt["id"],
                "evaluation_id": identity,
                "metrics": evaluation["metrics"],
            }
        )
    for identity, record in sorted(index.items()):
        if record["kind"] == "attempt" and (
            isinstance(record["protocol_id"], dict) or isinstance(record["trial_id"], dict)
        ):
            excluded.append({"id": identity, "reason": "unresolved_membership", "status": record["status"]})
            continue
        if record["kind"] == "evaluation":
            parent = index.get(record["attempt_id"]) if isinstance(record["attempt_id"], str) else None
            if parent is None or isinstance(parent["protocol_id"], dict) or isinstance(parent["trial_id"], dict):
                excluded.append({"id": identity, "reason": "unresolved_membership", "status": record["status"]})
                continue
        if record["kind"] == "attempt" and record["protocol_id"] == protocol_id and identity not in included:
            excluded.append({"id": identity, "reason": "not_selected", "status": record["status"]})
        if (
            record["kind"] == "evaluation"
            and isinstance(record["attempt_id"], str)
            and index[record["attempt_id"]]["protocol_id"] == protocol_id
            and identity not in included
        ):
            excluded.append({"id": identity, "reason": "not_selected", "status": record["status"]})
    frozen = {
        "kind": "collation",
        "schema_version": 1,
        "protocol_id": protocol_id,
        "included": sorted(included),
        "excluded": excluded,
        "missing": missing,
        "rows": rows,
        "metric_definitions": protocol["metric_definitions"],
        "analysis": deepcopy(analysis),
        "selected_attempts": dict(sorted(selected_attempts.items())),
        "selected_evaluations": sorted(selected_evaluations),
        "records": [index[key] for key in sorted(index)],
    }
    frozen["sha256"] = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    frozen["id"] = "collation:sha256:" + frozen["sha256"]
    return frozen


def validate_collation(value):
    """Recompute a saved selection and reject altered rows or evidence."""
    _require(isinstance(value, dict), "collation must be an object")
    fields = ("records", "protocol_id", "selected_attempts", "selected_evaluations", "analysis")
    _require(all(key in value for key in fields), "collation lacks reproduction inputs")
    rebuilt = collate(value["records"], **{key: value[key] for key in fields if key != "records"})
    _require(value == rebuilt, "collation does not match recomputed selection")
    return rebuilt
