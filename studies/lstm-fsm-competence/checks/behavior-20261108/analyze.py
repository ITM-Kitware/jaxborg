"""Summarize retained behavior traces without running policies or simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import mean

NAMES = ("Sleep", "Monitor", "Analyse", "Remove", "Restore", "Decoy", "BlockTraffic", "AllowTraffic")
# Stock enhanced-v2 Blue: four groups of 48 host slots, two groups of 24 routes.
RANGES = ((0, 1), (1, 2), (2, 50), (50, 98), (98, 146), (146, 194), (194, 218), (218, 242))
COMPONENTS = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    chosen, completed, opportunities, legal_choices = (Counter() for _ in range(4))
    totals = Counter()
    targets = [set() for _ in range(5)]
    target_opportunities = [set() for _ in range(5)]
    subnet_targets = [set() for _ in range(5)]
    previous = [None] * 5
    events = []
    first = None
    valid_rows = values.get("valid", [True] * len(values["blue_reward"]))
    for time, valid in enumerate(valid_rows):
        if not valid:
            continue
        row = {key: value[time] for key, value in values.items()}
        first = row if first is None else first
        totals["steps"] += 1
        totals["return"] += row["blue_reward"]
        for name in (*COMPONENTS, "green_lwf_count", "green_asf_count", "impact_count"):
            totals[name] += row[name]
        if abs(sum(row[name] for name in COMPONENTS) - row["blue_reward"]) > 1e-5:
            raise ValueError("Reward components do not reproduce the observed stock return")
        totals["blocked_directed_pair_ticks"] += sum(sum(bool(x) for x in line) for line in row["post_blocked_zones"])
        for agent, action in enumerate(row["blue_actions"]):
            mask = row["blue_action_masks"][agent]
            if len(mask) != 242:
                raise ValueError("Analyzer requires the declared enhanced-v2 242-action contract")
            if not mask[action]:
                totals["illegal_submissions"] += 1
            busy = row["blue_busy"][agent]
            kind = row["blue_effective_type"][agent]
            name = NAMES[kind]
            host = row["blue_effective_host"][agent]
            if busy:
                totals["busy_agent_ticks"] += 1
                totals["forced_sleep_ticks"] += int(action == 0)
                if action != 0:
                    raise ValueError("Busy Blue received a non-Sleep submission")
            else:
                totals["idle_decisions"] += 1
                chosen[name] += 1
                totals["same_action_as_previous_idle_decision"] += int(previous[agent] == action)
                previous[agent] = action
                for label, (start, end) in zip(NAMES, RANGES, strict=True):
                    count = sum(bool(x) for x in mask[start:end])
                    opportunities[label] += int(count > 0)
                    legal_choices[label] += count
                for start, end in RANGES[2:6]:
                    for slot, available in enumerate(mask[start:end]):
                        if available:
                            subnet = row["blue_observed_subnets"][agent][slot // 16]
                            if subnet < 0:
                                raise ValueError("Legal action targets a padded subnet")
                            target_opportunities[agent].add(row["observation_host_map"][subnet][slot % 16])
                if host >= 0:
                    targets[agent].add(host)
                    subnet_targets[agent].add(row["host_subnet"][host])
            if row["blue_effective_completes"][agent]:
                completed[name] += 1
                if kind in (3, 4, 6, 7):
                    event = {
                        "step": time,
                        "agent": agent,
                        "action": name,
                        "host": host,
                        "src_subnet": row["blue_effective_src_subnet"][agent],
                        "dst_subnet": row["blue_effective_dst_subnet"][agent],
                        "pending_before": busy,
                        "reward_components": {k: row[k] for k in COMPONENTS},
                    }
                    if host >= 0:
                        event.update(
                            {
                                "pre_compromised": row["pre_host_compromised"][host],
                                "post_compromised": row["post_host_compromised"][host],
                            }
                        )
                    events.append(event)
    if first is None:
        raise ValueError("Trace has no active steps")
    if totals["illegal_submissions"]:
        raise ValueError("Trace has illegal Blue submissions")
    return {
        "totals": dict(totals),
        "chosen_counts": dict(chosen),
        "completed_counts": dict(completed),
        "idle_decisions_with_available_type": dict(opportunities),
        "legal_action_slots_at_idle": dict(legal_choices),
        "chosen_share_of_idle_decisions": {k: chosen[k] / totals["idle_decisions"] for k in NAMES},
        "chosen_per_available_type_decision": {
            k: chosen[k] / opportunities[k] if opportunities[k] else None for k in NAMES
        },
        "targeted_hosts_by_agent": [sorted(x) for x in targets],
        "targetable_hosts_at_idle_by_agent": [sorted(x) for x in target_opportunities],
        "targeted_subnets_by_agent": [sorted(x) for x in subnet_targets],
        "host_coverage_of_idle_opportunities": [
            len(targets[a]) / len(target_opportunities[a]) if target_opportunities[a] else None for a in range(5)
        ],
        "owned_host_counts": [sum(bool(x) for x in row) for row in first["blue_owned_hosts"]],
        "events": events,
    }


def analyze(plan_path, output):
    from jaxborg.evaluation.capture_store import load_records

    if output.exists():
        raise FileExistsError("Choose a new analysis output; do not overwrite retained results")
    plan = read(plan_path)
    evidence = Path(plan["study_root"]) / "evidence"
    records = load_records(evidence)
    indexed = {record["id"]: record for record in records}
    episodes, missing = [], []
    selected = []
    for cell in plan["cells"]:
        receipt_path = plan_path.parent / "cells" / cell["id"] / "receipt.json"
        if not receipt_path.exists():
            missing.append(cell["id"])
            continue
        receipt = read(receipt_path)
        if (
            receipt["cell"] != cell["id"]
            or not receipt["canonical_parity"]
            or receipt["plan_sha256"] != digest(plan_path)
        ):
            raise ValueError("Canonical parity failed or receipt belongs to another plan")
        evaluation = indexed[receipt["evaluation_id"]]
        attempt = indexed[evaluation["attempt_id"]]
        if (
            evaluation["kind"] != "evaluation"
            or evaluation["status"] != "succeeded"
            or attempt["status"] != "succeeded"
        ):
            raise ValueError("Trace evaluation or attempt did not succeed")
        if cell["checkpoint"]["model_artifact"] not in evaluation["inputs"]:
            raise ValueError("Trace evaluation consumes another checkpoint")
        config = read(evidence / "attempts" / attempt["id"].rsplit(":", 1)[-1] / "configuration.json")
        scripted = cell["opponent"] == "fsm"
        roots = config["seeds"] if scripted else config["seed_roles"]["episode_roots"]
        if (
            config["execution"]["source_revision"] != plan["source_revision"]
            or roots != plan["protocol"]["episode_roots"]
            or config["topology"]["sha256"] != [p["sha256"] for p in plan["topologies"]]
            or config["base_variant" if scripted else "variant"] != cell["reference"]["case"]["variant"]
            or config["deterministic"] is not False
            or config["episodes_per_seed"] != 1
            or config["trajectories"] is not True
        ):
            raise ValueError("Trace configuration differs from the frozen behavior plan")
        if len(receipt["episodes"]) != len(cell["reference"]["episodes"]):
            raise ValueError("Receipt has another case count")
        for expected, observed in zip(cell["reference"]["episodes"], receipt["episodes"], strict=True):
            if any(observed.get(k) != v for k, v in expected.items()) or not observed["matches_canonical"]:
                raise ValueError("Receipt's canonical episode identity differs from the frozen plan")
        artifacts = [
            r
            for r in records
            if r["kind"] == "artifact"
            and r.get("producer_id") == evaluation["id"]
            and r.get("role") == "JAX policy trajectory"
        ]
        traces = {}
        for artifact in artifacts:
            path = evidence / artifact["uri"]
            if digest(path) != artifact["sha256"]:
                raise ValueError("Retained trajectory bytes changed")
            trace = read(path)
            if trace["episode_index"] in traces:
                raise ValueError("Duplicate trajectory episode index")
            traces[trace["episode_index"]] = (trace, artifact)
        if set(traces) != set(range(len(receipt["episodes"]))):
            raise ValueError("Expected exactly one retained trajectory per declared episode")
        selected.append(
            {"cell": cell["id"], "receipt_sha256": digest(receipt_path), "evaluation_id": receipt["evaluation_id"]}
        )
        for index, case in enumerate(receipt["episodes"]):
            trace, artifact = traces[index]
            summary = summarize(trace["steps"])
            if summary["totals"]["steps"] != plan["protocol"]["horizon"]:
                raise ValueError("Trace does not cover the declared full episode horizon")
            if summary["totals"]["return"] != case["canonical_return"]:
                raise ValueError("Trajectory reward sum differs from canonical return")
            episodes.append(
                {
                    "cell": cell["id"],
                    "checkpoint_steps": cell["checkpoint"]["steps"],
                    "opponent": cell["opponent"],
                    "case": case,
                    "trace_artifact": artifact["id"],
                    "trace_sha256": artifact["sha256"],
                    **summary,
                }
            )
    pairs = []
    for opponent, topology, root in product(
        plan["protocol"]["opponents"], range(len(plan["topologies"])), plan["protocol"]["episode_roots"]
    ):
        rows = [
            e
            for e in episodes
            if e["opponent"] == opponent
            and e["case"]["topology_index"] == topology
            and e["case"]["episode_root"] == root
        ]
        if len(rows) != 2:
            continue
        early, late = sorted(rows, key=lambda e: e["checkpoint_steps"])
        pairs.append(
            {
                "opponent": opponent,
                "topology_index": topology,
                "episode_root": root,
                "totals_delta_final_minus_primary": {
                    k: late["totals"].get(k, 0) - early["totals"].get(k, 0)
                    for k in set(early["totals"]) | set(late["totals"])
                },
                "idle_choice_share_delta": {
                    k: late["chosen_share_of_idle_decisions"][k] - early["chosen_share_of_idle_decisions"][k]
                    for k in NAMES
                },
            }
        )
    result = {
        "schema_version": 1,
        "plan_sha256": digest(plan_path),
        "analyzer_sha256": digest(__file__),
        "selection": selected,
        "missing_cells": missing,
        "episodes": episodes,
        "paired_cases": pairs,
        "limits": [
            "One Blue training seed; four paired cases per opponent; action rows are correlated.",
            "Completed means the pending action reached its execution tick, not successful remediation.",
            "Pre/post target changes span the full multi-agent step and do not establish action causality.",
            "Global compromised state is diagnostic ground truth, not Blue's observation.",
            "Monitor also runs automatically every tick; explicit Monitor choices do not measure sensor coverage.",
            "Green failure counts have no attempted-action denominator; report counts, not rates.",
        ],
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "analyze.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "paired_cases": len(pairs),
                "missing_cells": missing,
                "paired_return_delta": {
                    o: mean(p["totals_delta_final_minus_primary"]["return"] for p in pairs if p["opponent"] == o)
                    for o in plan["protocol"]["opponents"]
                    if any(p["opponent"] == o for p in pairs)
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.plan.resolve(), args.output.resolve())
