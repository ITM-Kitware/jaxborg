"""Read parity-checked traces for per-agent and per-host descriptive diagnostics."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NAMES = ("Sleep", "Monitor", "Analyse", "Remove", "Restore", "Decoy", "Block", "Allow")
SUBNETS = (
    "restricted A",
    "restricted B",
    "operational A",
    "operational B",
    "contractor",
    "admin",
    "office",
    "public",
    "internet",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(values, weights):
    agents = [Counter() for _ in range(5)]
    hosts = defaultdict(Counter)
    remediation = []
    phases = defaultdict(Counter)
    for t, actions in enumerate(values["blue_actions"]):
        phase = int(values["post_mission_phase"][t])
        for component in ("reward_ria", "reward_lwf", "reward_asf", "action_cost"):
            phases[phase][component] += values[component][t]
        active = values["host_active"][t]
        subnet = values["host_subnet"][t]
        for host, enabled in enumerate(active):
            if not enabled:
                continue
            record = hosts[host]
            record["subnet"] = int(subnet[host])
            record["compromised_ticks"] += int(values["post_host_compromised"][t][host] > 0)
            record["privileged_ticks"] += int(values["post_host_compromised"][t][host] == 2)
            for field, label, offset in [
                ("green_lwf_hosts", "lwf", 0),
                ("green_asf_hosts", "asf", 1),
                ("red_impact_hosts", "ria", 2),
            ]:
                marked = bool(values[field][t][host])
                record[label + "_count"] += int(marked)
                record[label + "_penalty"] += float(marked * weights[phase, int(subnet[host]), offset])
        for agent, action in enumerate(actions):
            if values["blue_busy"][t][agent]:
                agents[agent]["forced_wait_ticks"] += 1
                continue
            kind = int(values["blue_submitted_type"][t][agent])
            host = int(values["blue_submitted_host"][t][agent])
            agents[agent][NAMES[kind]] += 1
            agents[agent]["idle_decisions"] += 1
            if host >= 0:
                hosts[host]["chosen_" + NAMES[kind]] += 1
            if kind in (3, 4):
                remediation.append(
                    {
                        "row": t,
                        "agent": agent,
                        "host": host,
                        "action": NAMES[kind],
                        "pre_compromised": values["pre_host_compromised"][t][host],
                        "post_compromised": values["post_host_compromised"][t][host],
                    }
                )
    for label in ("lwf", "asf", "ria"):
        expected = sum(values["reward_" + label])
        actual = sum(h[label + "_penalty"] for h in hosts.values())
        if abs(actual - expected) > 1e-4:
            raise ValueError(f"{label} host accounting mismatch: {actual} != {expected}")
    return {
        "agents": [dict(x) for x in agents],
        "hosts": {str(h): {**dict(x), "subnet_name": SUBNETS[x["subnet"]]} for h, x in hosts.items()},
        "phase_rewards": dict(phases),
        "remediation_starts": remediation,
    }


def main():
    from jaxborg.evaluation.capture_store import load_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    summary = json.loads(args.summary.read_text())
    if summary["plan_sha256"] != sha(args.plan):
        raise ValueError("Summary/plan identity mismatch")
    evidence = Path(plan["study_root"]) / "evidence"
    records = {r["id"]: r for r in load_records(evidence)}
    outputs, first_paths, divergence = [], {}, []
    for episode in summary["episodes"]:
        artifact = records[episode["trace_artifact"]]
        path = evidence / artifact["uri"]
        if sha(path) != episode["trace_sha256"]:
            raise ValueError("Trace hash mismatch")
        values = json.loads(path.read_text())["steps"]
        topology_index = episode["case"]["topology_index"]
        snapshot = plan["topologies"][topology_index]
        if sha(snapshot["path"]) != snapshot["sha256"]:
            raise ValueError("Topology hash mismatch")
        with np.load(snapshot["path"], allow_pickle=False) as topology:
            data = diagnose(values, topology["phase_rewards"])
        outputs.append({"cell": episode["cell"], "case": episode["case"], "trace_sha256": artifact["sha256"], **data})
        key = (episode["opponent"], topology_index, episode["case"]["episode_root"])
        compact = {
            k: values[k]
            for k in (
                "blue_actions",
                "blue_observations",
                "blue_action_masks",
                "blue_submitted_type",
                "blue_submitted_host",
            )
        }
        if key in first_paths:
            prior = first_paths.pop(key)
            first = next(
                (
                    t
                    for t in range(len(values["blue_actions"]))
                    if prior["blue_actions"][t] != values["blue_actions"][t]
                ),
                None,
            )
            divergence.append(
                {
                    "opponent": key[0],
                    "topology_index": key[1],
                    "episode_root": key[2],
                    "first_sampled_action_difference_row": first,
                    "observations_equal_at_first_difference": first is not None
                    and prior["blue_observations"][first] == values["blue_observations"][first],
                    "masks_equal_at_first_difference": first is not None
                    and prior["blue_action_masks"][first] == values["blue_action_masks"][first],
                    "primary_action_types": None if first is None else prior["blue_submitted_type"][first],
                    "final_action_types": None if first is None else values["blue_submitted_type"][first],
                    "primary_hosts": None if first is None else prior["blue_submitted_host"][first],
                    "final_hosts": None if first is None else values["blue_submitted_host"][first],
                }
            )
        else:
            first_paths[key] = compact
    result = {
        "plan_sha256": sha(args.plan),
        "summary_sha256": sha(args.summary),
        "source_sha256": sha(__file__),
        "episodes": outputs,
        "first_divergence": divergence,
        ("meaning"): (
            "Descriptive on-policy state/action diagnostics. Net states and host costs are not "
            "individual-action causal effects."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"episodes": len(outputs), "pairs": len(divergence), "output": str(args.output)}))


if __name__ == "__main__":
    main()
