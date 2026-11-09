"""Read retained layout200 traces; write a new compact paired evidence file."""

import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1]).expanduser()
sys.path.insert(0, str(OUT))
import mechanism_probe as probe  # noqa: E402

full = json.loads((OUT / "full-agent-host-diagnostics.json").read_text())
old = json.loads((OUT / "paired-fsm-compact-evidence.json").read_text())
result = {
    k: old[k]
    for k in [
        "schema_version",
        "trace_source_revision",
        "canonical_evaluation_source_revision",
        "training_seed",
        "episode_root",
        "phase_boundaries",
        "visible_plane_columns",
        "probe_sha256",
    ]
}
result["layout"] = 200
result["opponent"] = "red-diverse-100-final"
result["weighted_host_cost_columns"] = ["host", "subnet", "LWF", "RIA", "ASF"]
result["selected_tick_columns"] = [
    "tick",
    "post_phase",
    "submitted_action_id",
    "submitted_action",
    "submitted_host",
    "busy",
    "pending_ticks",
    "pending_action_id",
    "analyse_remove_restore_structural_masks",
    "analyse_remove_restore_policy_masks",
    "effective_action",
    "effective_host",
    "completes",
    "visible_before",
    "visible_next_tick",
    "pre_compromised",
    "post_compromised",
    "pre_sessions",
    "post_sessions",
    "pre_service_reliability",
    "post_service_reliability",
    "pre_OT_stopped",
    "post_OT_stopped",
    "completed_recovery_attempt_flag",
    "Red_idle_choices_targeting_host",
]
result["interpretation"] = old["interpretation"] + [
    (
        "RIA records penalized Impact attempts, including repeated attempts on already stopped "
        "OT; it is not a count of new outages."
    ),
    (
        "Red Withdraw can clear all sessions without Blue remediation. Red decision rows are "
        "gated by active and pre-step pending ticks zero."
    ),
    (
        "Case-paired trajectories have the same topology and root but diverged states; selective "
        "repair examples do not establish common-state preference or causal attribution of every "
        "return difference."
    ),
    "Service reliability vectors include all five service slots; the snapshot determines which services exist.",
    "Mission-permitted route occupancy is from exact snapshot allowed_subnet_pairs and post-step phase.",
]
result["cells"] = []
names = [
    "aggressive_scan",
    "stealth_scan",
    "discover_deception",
    "exploit",
    "privilege_escalate",
    "impact",
    "degrade",
    "withdraw",
]
focus = {73: 0, 86: 2, 31: 1, 38: 1, 44: 3, 46: 3}
ticks = {
    "primary": {
        73: [49, 63, 212, 216],
        86: [119, 382, 386],
        31: [153, 157, 207, 209, 343],
        38: [168, 170, 340, 344],
        44: [34],
        46: [226, 403],
    },
    "final": {
        73: [45, 46, 49, 53, 164],
        86: [119, 123],
        31: [246, 248, 449],
        38: [168, 453, 476],
        44: [34, 461],
        46: [190, 403],
    },
}

for label, checkpoint in [("primary", 49968000), ("final", 124992000)]:
    analysis = json.loads((OUT / f"{label}-learned-layout200-probe.json").read_text())
    trace_path = pathlib.Path(analysis["inputs"]["trace"])
    raw = trace_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == analysis["inputs"]["trace_sha256"]
    trace = json.loads(raw)
    s = trace["steps"]
    topo = dict(np.load(analysis["inputs"]["topology"], allow_pickle=False))
    diag = next(
        e
        for e in full["episodes"]
        if e["cell"] == f"{checkpoint}-red-diverse-100-final"
        and e["case"]["topology_index"] == 0
        and e["case"]["episode_root"] == 21000
    )

    def red_choices(t, h):
        found = []
        for r, x in enumerate(s["red_actions"][t]):
            if not s["red_active_before"][t][r] or s["red_pending_ticks_before"][t][r] != 0:
                continue
            if 10 <= x < 10 + 8 * 137 and (x - 10) % 137 == h:
                found.append([r, names[(x - 10) // 137], x])
        return found

    def slot(a, h):
        return next(
            k
            for k in range(48)
            if int(topo["blue_obs_subnets"][a][k // 16]) >= 0
            and int(topo["obs_host_map"][int(topo["blue_obs_subnets"][a][k // 16])][k % 16]) == h
        )

    def planes(t, a, k):
        if t >= len(s["blue_actions"]):
            return None
        v = s["blue_observations"][t][a]
        base = 1 + (k // 16) * 59
        return [
            v[210 + k],
            v[258 + k],
            v[306 + k],
            v[354 + k],
            v[402 + k],
            v[base + 27 + k % 16],
            v[base + 43 + k % 16],
        ]

    def row(t, a, h, k):
        ids = [2 + k, 50 + k, 98 + k]
        return [
            t,
            s["post_mission_phase"][t],
            s["blue_actions"][t][a],
            probe.ACTION_NAMES[s["blue_submitted_type"][t][a]],
            s["blue_submitted_host"][t][a],
            s["blue_busy"][t][a],
            s["blue_pending_ticks"][t][a],
            s["blue_pending_action"][t][a],
            [s["blue_structural_masks"][t][a][i] for i in ids],
            [s["blue_action_masks"][t][a][i] for i in ids],
            probe.ACTION_NAMES[s["blue_effective_type"][t][a]],
            s["blue_effective_host"][t][a],
            s["blue_effective_completes"][t][a],
            planes(t, a, k),
            planes(t + 1, a, k),
            s["pre_host_compromised"][t][h],
            s["post_host_compromised"][t][h],
            probe.session_count(s, "pre", t, h),
            probe.session_count(s, "post", t, h),
            s["pre_host_service_reliability"][t][h],
            s["post_host_service_reliability"][t][h],
            s["pre_ot_service_stopped"][t][h],
            s["post_ot_service_stopped"][t][h],
            s["blue_recovery_attempt_completed"][t][h],
            red_choices(t, h),
        ]

    cell = {
        "checkpoint": label,
        "training_steps": checkpoint,
        "trace": analysis["inputs"],
        "canonical_episode_receipt": diag["case"],
        "return_components": analysis["reward_components"],
        "agents": diag["agents"],
        "alert_summary": analysis["alert_summary"],
        "mission_route_occupancy": analysis["mission_route_occupancy"],
        "weighted_host_costs": [
            [int(h), d["subnet"], d["lwf_penalty"], d["ria_penalty"], d["asf_penalty"]]
            for h, d in diag["hosts"].items()
            if d["lwf_penalty"] or d["ria_penalty"] or d["asf_penalty"]
        ],
        "focus_hosts": [],
    }
    for h, a in focus.items():
        k = slot(a, h)
        chosen = collections.Counter()
        choices = []
        for t in range(500):
            if not s["blue_busy"][t][a] and s["blue_submitted_host"][t][a] == h:
                name = probe.ACTION_NAMES[s["blue_submitted_type"][t][a]]
                chosen[name] += 1
                if name in ("analyse", "remove", "restore"):
                    choices.append([t, name, planes(t, a, k)])
        impacts = [t for t in range(500) if s["red_impact_hosts"][t][h]]
        withdraws = [t for t in range(500) if any(x[1] == "withdraw" for x in red_choices(t, h))]
        selected = sorted(set(ticks[label][h] + withdraws))
        item = {
            "agent": a,
            "host": h,
            "subnet": int(topo["host_subnet"][h]),
            "relative_slot": k,
            "restore_action": 98 + k,
            "idle_target_action_counts": dict(chosen),
            "analyse_remove_restore_starts": [x[:2] for x in choices] if h in (73, 86, 31) else [],
            "first_visible_indicator_onset": next(
                (
                    {
                        key: e[key]
                        for key in ("row", "tick", "channels_at_onset", "status", "response_tick", "response_action")
                        if key in e
                    }
                    for e in analysis["alert_events"]
                    if e["agent"] == a and e["host"] == h
                ),
                None,
            ),
            "penalized_Impact_attempts": len(impacts),
            "penalized_Impact_attempts_pre_OT_already_stopped": sum(
                bool(s["pre_ot_service_stopped"][t][h]) for t in impacts
            ),
            "first_last_penalized_Impact_attempt_ticks": [impacts[0], impacts[-1]] if impacts else [],
            "Red_Withdraw_choice_ticks": withdraws,
            "selected_ticks": [row(t, a, h, k) for t in selected] if h not in (44, 46) else [],
            "Withdraw_net_session_changes": [
                [
                    t,
                    red_choices(t, h),
                    probe.session_count(s, "pre", t, h),
                    probe.session_count(s, "post", t, h),
                    s["pre_ot_service_stopped"][t][h],
                    s["post_ot_service_stopped"][t][h],
                    s["blue_recovery_attempt_completed"][t][h],
                ]
                for t in withdraws
            ],
        }
        cell["focus_hosts"].append(item)
    result["cells"].append(cell)

result["paired_component_delta"] = {
    k: result["cells"][1]["return_components"][k] - v for k, v in result["cells"][0]["return_components"].items()
}
blob = (json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n").encode()
if len(blob) >= 25000:
    print(
        [
            (c["checkpoint"], {k: len(json.dumps(v, separators=(",", ":"))) for k, v in c.items()})
            for c in result["cells"]
        ]
    )
assert len(blob) < 25000, len(blob)
path = OUT / "paired-learned-layout200-compact-evidence.json"
with path.open("xb") as f:
    f.write(blob)
print(json.dumps({"path": str(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}))
for cell in result["cells"]:
    print(
        cell["checkpoint"],
        [
            (
                h["host"],
                h["penalized_Impact_attempts"],
                h["penalized_Impact_attempts_pre_OT_already_stopped"],
                h["Red_Withdraw_choice_ticks"],
            )
            for h in cell["focus_hosts"]
        ],
    )
