"""Read retained behavior traces without executing policies or simulator steps.

Usage: python mechanism_probe.py TRACE.json TOPOLOGY.snapshot.npz [--output REPORT.json]
Requires NumPy only to read the topology snapshot. JSON trace analysis uses stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ACTION_NAMES = ("sleep", "monitor", "analyse", "remove", "restore", "decoy", "block", "allow")
CHANNELS = ("file", "process", "network", "ioc")
HOST_STARTS = ((2, 50, 2), (50, 98, 3), (98, 146, 4), (146, 194, 5))


def decode(action, agent, topology):
    if not 0 <= action < 242:
        raise ValueError(f"Action outside the 242-action contract: {action}")
    if action < 2:
        return action, None, None, None
    for start, stop, kind in HOST_STARTS:
        if start <= action < stop:
            slot = action - start
            subnet = topology["blue_obs_subnets"][agent][slot // 16]
            host = None if subnet < 0 else topology["obs_host_map"][subnet][slot % 16]
            if host is not None and not 0 <= host < len(topology["host_active"]):
                host = None
            return kind, host, None, None
    kind, offset = (6, action - 194) if action < 218 else (7, action - 218)
    dst = topology["blue_obs_subnets"][agent][offset % 3]
    src = offset // 3
    src += src >= dst
    return kind, None, src, dst


def visible_hosts(observations, agent, topology):
    if len(observations) != 450:
        raise ValueError("Expected the enhanced-v2 450-element Blue observation")
    result = {}
    for slot in range(48):
        if not observations[210 + slot]:
            continue
        subnet = topology["blue_obs_subnets"][agent][slot // 16]
        if subnet < 0:
            raise ValueError("Valid observation slot has no observed subnet")
        host = topology["obs_host_map"][subnet][slot % 16]
        if not (0 <= host < len(topology["host_active"]) and topology["host_active"][host]):
            raise ValueError("Valid observation slot maps to an inactive or padded host")
        if not topology["blue_agent_hosts"][agent][host]:
            raise ValueError("Valid observation slot maps outside the agent's owned hosts")
        result[host] = tuple(
            name for plane, name in enumerate(CHANNELS, 1) if observations[210 + plane * 48 + slot] > 0
        )
    return result


def session_count(steps, prefix, row, host):
    field = f"{prefix}_red_session_count"
    if field in steps:
        return sum(int(agent[host]) for agent in steps[field][row])
    return sum(bool(agent[host]) for agent in steps[f"{prefix}_red_sessions"][row])


def distribution(values):
    return {"n": len(values), "min": min(values), "median": median(values), "max": max(values)} if values else {"n": 0}


def intervals(rows, times):
    output = []
    for row in rows:
        if output and row == output[-1]["end_row_inclusive"] + 1:
            output[-1]["end_row_inclusive"] = row
            output[-1]["end_tick_inclusive"] = times[row]
        else:
            output.append(
                {"start_row": row, "end_row_inclusive": row, "start_tick": times[row], "end_tick_inclusive": times[row]}
            )
    return output


def analyze(payload, topology, *, window_count=12):
    steps = payload["steps"]
    n = len(steps["blue_actions"])
    if not n:
        raise ValueError("Empty trace")
    if "valid" in steps:
        if not all(steps["valid"]):
            raise ValueError("Trim invalid/padded trace rows before analysis")
    if any(len(row) != 5 for row in steps["blue_actions"]):
        raise ValueError("Expected five Blue agents")
    if any(len(mask) != 242 for row in steps["blue_action_masks"] for mask in row):
        raise ValueError("Expected the 242-action availability mask")
    times = [int(value) for value in steps["pre_step_count"]]
    if any(b != a + 1 for a, b in zip(times, times[1:])):
        raise ValueError("Trace must contain contiguous world ticks")
    visible = [[visible_hosts(steps["blue_observations"][t][a], a, topology) for a in range(5)] for t in range(n)]
    decoded = [[decode(int(steps["blue_actions"][t][a]), a, topology) for a in range(5)] for t in range(n)]
    idle = [[not bool(steps["blue_busy"][t][a]) for a in range(5)] for t in range(n)]
    remediation_starts = defaultdict(list)
    target_categories = {name: Counter() for name in ACTION_NAMES[2:6]}
    starts = []
    invalid_idle = []
    idle_counts = Counter()
    for t in range(n):
        for a in range(5):
            action = int(steps["blue_actions"][t][a])
            if not idle[t][a]:
                continue
            kind, host, _, _ = decoded[t][a]
            idle_counts[ACTION_NAMES[kind]] += 1
            if not steps["blue_action_masks"][t][a][action]:
                invalid_idle.append({"row": t, "tick": times[t], "agent": a, "action": action})
            if kind not in (2, 3, 4, 5) or host is None:
                continue
            channels = visible[t][a].get(host, ())
            compromised = int(steps["pre_host_compromised"][t][host])
            sessions = session_count(steps, "pre", t, host)
            latent = compromised > 0 or sessions > 0
            category = (
                ("visible_and_latent" if latent else "visible_only")
                if channels
                else ("latent_only" if latent else "neither")
            )
            target_categories[ACTION_NAMES[kind]][category] += 1
            if kind in (3, 4):
                remediation_starts[a, host].append(t)
                starts.append(
                    {
                        "row": t,
                        "tick": times[t],
                        "agent": a,
                        "host": host,
                        "action": ACTION_NAMES[kind],
                        "channels": list(channels),
                        "category": category,
                        "pre_compromised_code": compromised,
                        "pre_session_count": sessions,
                    }
                )

    completions = []
    for t in range(n):
        for a in range(5):
            kind = int(steps["blue_effective_type"][t][a])
            if kind not in (3, 4) or not steps["blue_effective_completes"][t][a]:
                continue
            host = int(steps["blue_effective_host"][t][a])
            if not 0 <= host < len(topology["host_active"]):
                continue
            before, after = (int(steps[f"{prefix}_host_compromised"][t][host]) for prefix in ("pre", "post"))
            pre_sessions, post_sessions = (session_count(steps, prefix, t, host) for prefix in ("pre", "post"))
            completions.append(
                {
                    "row": t,
                    "tick": times[t],
                    "agent": a,
                    "host": host,
                    "action": ACTION_NAMES[kind],
                    "pre_compromised_code": before,
                    "post_compromised_code": after,
                    "pre_session_count": pre_sessions,
                    "post_session_count": post_sessions,
                    "whole_step_net_sessions": post_sessions - pre_sessions,
                    "whole_step_compromise_became_zero": before > 0 and after == 0,
                    "visible_channels_before": list(visible[t][a].get(host, ())),
                }
            )

    alert_events = []
    for a in range(5):
        previous = {}
        for t in range(n):
            for host, channels in visible[t][a].items():
                if not channels or previous.get(host):
                    continue
                event = {
                    "row": t,
                    "tick": times[t],
                    "agent": a,
                    "host": host,
                    "channels_at_onset": list(channels),
                    "pre_compromised_code_diagnostic": int(steps["pre_host_compromised"][t][host]),
                    "pre_session_count_diagnostic": session_count(steps, "pre", t, host),
                }
                same_pending = (
                    not idle[t][a]
                    and int(steps["blue_effective_type"][t][a]) in (3, 4)
                    and int(steps["blue_effective_host"][t][a]) == host
                )
                first_idle = next((u for u in range(t, n) if idle[u][a]), None)
                event["compulsory_busy_delay_ticks"] = None if first_idle is None else times[first_idle] - times[t]
                if same_pending:
                    event["status"] = "same_host_remediation_already_pending"
                    end = next((u for u in range(t, n) if steps["blue_effective_completes"][u][a]), None)
                    event["pending_completion_row"] = end
                else:
                    response = next((u for u in remediation_starts[a, host] if u >= t), None)
                    if response is None:
                        event.update(
                            status="episode_end_censored",
                            followup_ticks=times[-1] - times[t] + 1,
                            idle_opportunities_observed=sum(idle[u][a] for u in range(t, n)),
                        )
                    else:
                        event.update(
                            status="subsequent_idle_remediation_start",
                            response_row=response,
                            response_tick=times[response],
                            response_action=ACTION_NAMES[decoded[response][a][0]],
                            latency_world_ticks=times[response] - times[t],
                            earlier_idle_choices=sum(idle[u][a] for u in range(t, response)),
                            visible_evidence_cleared_before_start=any(
                                not visible[u][a].get(host) for u in range(t, response)
                            ),
                        )
                        end = next((u for u in range(response, n) if steps["blue_effective_completes"][u][a]), None)
                        event["response_completion_row"] = end
                        event["start_to_completion_ticks"] = None if end is None else times[end] - times[response]
                alert_events.append(event)
            previous = visible[t][a]

    route_rows = defaultdict(list)
    per_tick_routes = []
    phase_route_ticks = Counter()
    route_blocks_started = []
    allowed = topology["allowed_subnet_pairs"]
    for t in range(n):
        phase = int(steps["post_mission_phase"][t])
        blocks = steps["post_blocked_zones"][t]
        pairs = []
        for source in range(9):
            for destination in range(source + 1, 9):
                permitted = allowed[phase][source][destination] or allowed[phase][destination][source]
                if permitted and (blocks[source][destination] or blocks[destination][source]):
                    route_rows[source, destination].append(t)
                    pairs.append([source, destination])
                    phase_route_ticks[str(phase)] += 1
        per_tick_routes.append(pairs)
        for a in range(5):
            kind, _, src, dst = decoded[t][a]
            if idle[t][a] and kind == 6 and src >= 0 and dst >= 0:
                old = steps["pre_blocked_zones"][t]
                route_blocks_started.append(
                    {
                        "row": t,
                        "tick": times[t],
                        "agent": a,
                        "src": src,
                        "dst": dst,
                        "mission_phase": phase,
                        "pair_permitted_for_green": bool(allowed[phase][src][dst] or allowed[phase][dst][src]),
                        "pair_already_blocked_either_direction": bool(old[src][dst] or old[dst][src]),
                    }
                )

    window_candidates = []
    responded = [event for event in alert_events if event["status"] == "subsequent_idle_remediation_start"]
    for event in sorted(responded, key=lambda e: e["latency_world_ticks"], reverse=True)[:4]:
        window_candidates.append(("slow_alert_response", event["response_row"], event["agent"], event["host"]))
    for event in [e for e in alert_events if e["status"] != "subsequent_idle_remediation_start"][:3]:
        window_candidates.append((event["status"], event["row"], event["agent"], event["host"]))
    for event in [e for e in completions if e["pre_session_count"] > 0 and e["whole_step_net_sessions"] >= 0][:3]:
        window_candidates.append(
            ("remediation_completion_without_net_session_decrease", event["row"], event["agent"], event["host"])
        )
    for t in range(1, n):
        if steps["post_mission_phase"][t] != steps["post_mission_phase"][t - 1]:
            window_candidates.append(("mission_phase_boundary", t, None, None))
    if not window_candidates:
        window_candidates.append(("episode_opening", 0, None, None))
    windows = []
    seen = set()
    for label, anchor, agent, host in window_candidates:
        first = max(0, min(anchor - 5, n - 20))
        stop = min(n, first + 20)
        identity = first, stop, agent, host
        if identity in seen:
            continue
        seen.add(identity)
        rows = []
        for t in range(first, stop):
            record = {
                "row": t,
                "tick": times[t],
                "post_phase": int(steps["post_mission_phase"][t]),
                "blue_actions": [ACTION_NAMES[item[0]] for item in decoded[t]],
                "blue_targets": [item[1] for item in decoded[t]],
                "blue_busy": [not value for value in idle[t]],
                "mission_pairs_blocked": per_tick_routes[t],
                "reward": {
                    name: steps[name][t]
                    for name in ("blue_reward", "reward_ria", "reward_lwf", "reward_asf", "action_cost")
                },
            }
            if agent is not None and host is not None:
                record.update(
                    visible_channels=list(visible[t][agent].get(host, ())),
                    pre_compromised=int(steps["pre_host_compromised"][t][host]),
                    post_compromised=int(steps["post_host_compromised"][t][host]),
                    pre_sessions=session_count(steps, "pre", t, host),
                    post_sessions=session_count(steps, "post", t, host),
                    effective_action=ACTION_NAMES[int(steps["blue_effective_type"][t][agent])],
                    effective_host=int(steps["blue_effective_host"][t][agent]),
                    effective_completes=bool(steps["blue_effective_completes"][t][agent]),
                )
            rows.append(record)
        windows.append(
            {
                "reason": label,
                "anchor_row": anchor,
                "agent": agent,
                "host": host,
                "start_row": first,
                "end_row_exclusive": stop,
                "rows": rows,
            }
        )
        if len(windows) >= window_count:
            break
    response_reuse = Counter((e["agent"], e["host"], e["response_row"]) for e in responded)
    return {
        "schema_version": 1,
        "episode_index": payload.get("episode_index"),
        "world_steps": n,
        "interpretation": [
            "Visible alerts include persistent history, Green false positives and blocked-access network events.",
            (
                "Alert response is temporal association; policy states and attacker opportunities can "
                "diverge across paired cases."
            ),
            (
                "A response may match multiple prior alert onsets; counts below disclose reuse and "
                "clearing before response."
            ),
            "Remediation completion is an attempt. Net state changes span Blue, Green, Red and session reassignment.",
            (
                "Latent category means pre-step compromised code>0 or session count>0; it is unavailable "
                "to Blue as ground truth."
            ),
            (
                "Green route occupancy uses exact allowed_subnet_pairs and post-step phase; failures do "
                "not identify the sampled route."
            ),
        ],
        "idle_action_counts": dict(idle_counts),
        "forced_busy_rows": sum(not value for row in idle for value in row),
        "invalid_idle_action_rows": invalid_idle,
        "target_start_categories": {name: dict(counts) for name, counts in target_categories.items()},
        "remediation_starts": starts,
        "alert_summary": {
            "onsets": len(alert_events),
            "statuses": dict(Counter(e["status"] for e in alert_events)),
            "latency_world_ticks": distribution([e["latency_world_ticks"] for e in responded]),
            "earlier_idle_choices": distribution([e["earlier_idle_choices"] for e in responded]),
            "responses_matched_to_multiple_onsets": sum(count > 1 for count in response_reuse.values()),
            "onsets_matching_reused_response": sum(count for count in response_reuse.values() if count > 1),
        },
        "alert_events": alert_events,
        "completed_remediation_net_diagnostics": completions,
        "green_failures": {
            name: sum(sum(bool(value) for value in row) for row in steps[name])
            for name in ("green_lwf_hosts", "green_asf_hosts")
        },
        "reward_components": {
            name: sum(steps[name]) for name in ("blue_reward", "reward_ria", "reward_lwf", "reward_asf", "action_cost")
        },
        "mission_route_occupancy": {
            "undirected_pair_ticks": sum(len(rows) for rows in route_rows.values()),
            "by_phase": dict(phase_route_ticks),
            "routes": [
                {"subnets": list(pair), "ticks": len(rows), "runs": intervals(rows, times)}
                for pair, rows in sorted(route_rows.items())
            ],
        },
        "block_starts": route_blocks_started,
        "manual_windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("topology", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--windows", type=int, default=12)
    args = parser.parse_args()
    import numpy as np

    trace_bytes = args.trace.read_bytes()
    snapshot_bytes = args.topology.read_bytes()
    with np.load(args.topology, allow_pickle=False) as data:
        topology = {
            name: data[name].tolist()
            for name in ("allowed_subnet_pairs", "blue_obs_subnets", "obs_host_map", "host_active", "blue_agent_hosts")
        }
    report = analyze(json.loads(trace_bytes), topology, window_count=args.windows)
    report["inputs"] = {
        "trace": str(args.trace.resolve()),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "topology": str(args.topology.resolve()),
        "topology_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
