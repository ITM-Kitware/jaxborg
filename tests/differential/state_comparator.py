import numpy as np
from CybORG.Shared.Enums import DecoyType

from jaxborg.constants import (
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
    SERVICE_IDS,
)
from jaxborg.translate import CC4Mappings
from tests.differential.harness import StateDiff, StateSnapshot

_DECOY_SVC_TO_IDX = {"haraka": 0, "apache2": 1, "tomcat": 2, "vsftpd": 3}

_ERROR_FIELDS = {
    "identity_primary_host",
    "identity_primary_pid",
    "identity_primary_abstract",
    "identity_pending_busy",
    "identity_pending_ticks",
    "identity_pending_action",
    "host_compromised",
    "red_privilege",
    "red_sessions",
    "red_discovered_hosts",
    "red_scanned_hosts",
    "host_services",
    "host_service_reliability",
    "host_has_malware",
    "host_decoys",
    "ot_service_stopped",
    "blocked_zones",
    "mission_phase",
    "rewards",
    "observation",
}

_WARNING_FIELDS = {
    "host_activity_detected",
}


def extract_cyborg_snapshot(cyborg_env, mappings: CC4Mappings) -> StateSnapshot:
    state = cyborg_env.environment_controller.state
    sorted_hosts = sorted(state.hosts.keys())

    host_compromised = {}
    host_services = {}
    host_service_reliability = {}
    host_has_malware = {}
    host_decoys = {}
    ot_service_stopped = {}
    for hostname in sorted_hosts:
        idx = mappings.hostname_to_idx[hostname]
        host = state.hosts[hostname]
        max_level = 0
        for agent_name, sessions in state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            for sess in sessions.values():
                if sess.hostname == hostname:
                    if hasattr(sess, "username"):
                        if sess.username in ("root", "SYSTEM"):
                            max_level = max(max_level, 2)
                        else:
                            max_level = max(max_level, 1)
                    else:
                        max_level = max(max_level, 1)
        host_compromised[idx] = max_level

        services = [False] * NUM_SERVICES
        service_reliability = [100] * NUM_SERVICES
        decoys = [False] * NUM_DECOY_TYPES
        ot_stopped = False
        for svc_name, svc in host.services.items():
            svc_str = str(svc_name).split(".")[-1] if "." in str(svc_name) else str(svc_name)
            sid = SERVICE_IDS.get(svc_str)
            if sid is not None:
                services[sid] = bool(svc.active)
                service_reliability[sid] = int(svc.get_service_reliability())
                if svc_str == "OTSERVICE" and not bool(svc.active):
                    ot_stopped = True

            decoy_idx = _DECOY_SVC_TO_IDX.get(svc_str.lower())
            if decoy_idx is None:
                continue
            pid = svc.process if hasattr(svc, "process") else svc.get("process")
            if pid is None:
                continue
            proc = host.get_process(pid)
            if proc is not None and proc.decoy_type & DecoyType.EXPLOIT:
                decoys[decoy_idx] = True

        host_services[idx] = tuple(services)
        host_service_reliability[idx] = tuple(service_reliability)
        host_has_malware[idx] = any(getattr(file_obj, "name", "") == "cmd.sh" for file_obj in host.files)
        if any(decoys):
            host_decoys[idx] = tuple(decoys)
        if ot_stopped:
            ot_service_stopped[idx] = True

    red_privilege = {}
    red_sessions = {}
    for agent_name, sessions in state.sessions.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        session_hosts = set()
        priv = {}
        for sess in sessions.values():
            if sess.hostname in mappings.hostname_to_idx:
                hidx = mappings.hostname_to_idx[sess.hostname]
                session_hosts.add(hidx)
                level = 1
                if hasattr(sess, "username") and sess.username in ("root", "SYSTEM"):
                    level = 2
                priv[hidx] = max(priv.get(hidx, 0), level)
        red_sessions[red_idx] = session_hosts
        red_privilege[red_idx] = priv

    blocked = set()
    for to_subnet, from_list in getattr(state, "blocks", {}).items():
        for from_subnet in from_list:
            if from_subnet in mappings.cidr_to_subnet_idx or from_subnet in {v for v in mappings.subnet_names.values()}:
                pass
            blocked.add((str(from_subnet), str(to_subnet)))

    mission_phase = getattr(state, "mission_phase", 0)

    return StateSnapshot(
        time=getattr(state, "step_count", 0),
        mission_phase=mission_phase,
        host_compromised=host_compromised,
        red_privilege=red_privilege,
        red_sessions=red_sessions,
        host_services=host_services,
        host_service_reliability=host_service_reliability,
        host_has_malware=host_has_malware,
        host_decoys=host_decoys,
        ot_service_stopped=ot_service_stopped,
        blocked_zones=blocked,
    )


def extract_jax_snapshot(state, const, mappings: CC4Mappings) -> StateSnapshot:
    n = mappings.num_hosts
    host_compromised = {}
    host_services = {}
    host_service_reliability = {}
    host_has_malware = {}
    for h in range(n):
        host_compromised[h] = int(state.host_compromised[h])
        host_services[h] = tuple(bool(v) for v in np.asarray(state.host_services[h, :NUM_SERVICES]).tolist())
        host_service_reliability[h] = tuple(
            int(v) for v in np.asarray(state.host_service_reliability[h, :NUM_SERVICES]).tolist()
        )
        host_has_malware[h] = bool(state.host_has_malware[h])

    red_privilege = {}
    red_sessions = {}
    for r in range(NUM_RED_AGENTS):
        sessions_set = set()
        priv = {}
        for h in range(n):
            if bool(state.red_sessions[r, h]):
                sessions_set.add(h)
            p = int(state.red_privilege[r, h])
            if p > 0:
                priv[h] = p
        red_sessions[r] = sessions_set
        red_privilege[r] = priv

    host_decoys = {}
    for h in range(n):
        decoys = np.array(state.host_decoys[h])
        if np.any(decoys):
            host_decoys[h] = tuple(bool(d) for d in decoys)

    ot_service_stopped = {}
    for h in range(n):
        if bool(state.ot_service_stopped[h]):
            ot_service_stopped[h] = True

    blocked = set()
    blocked_arr = np.array(state.blocked_zones)
    for dst in range(NUM_SUBNETS):
        for src in range(NUM_SUBNETS):
            if blocked_arr[dst, src]:
                src_name = mappings.subnet_names.get(src, str(src))
                dst_name = mappings.subnet_names.get(dst, str(dst))
                blocked.add((src_name, dst_name))

    return StateSnapshot(
        time=int(state.time),
        mission_phase=int(state.mission_phase),
        host_compromised=host_compromised,
        red_privilege=red_privilege,
        red_sessions=red_sessions,
        host_services=host_services,
        host_service_reliability=host_service_reliability,
        host_has_malware=host_has_malware,
        host_decoys=host_decoys,
        ot_service_stopped=ot_service_stopped,
        blocked_zones=blocked,
    )


def compare_snapshots(cyborg: StateSnapshot, jax: StateSnapshot) -> list[StateDiff]:
    diffs = []

    if cyborg.mission_phase != jax.mission_phase:
        diffs.append(StateDiff("mission_phase", cyborg.mission_phase, jax.mission_phase))

    for h in set(cyborg.host_compromised) | set(jax.host_compromised):
        cv = cyborg.host_compromised.get(h, 0)
        jv = jax.host_compromised.get(h, 0)
        if cv != jv:
            diffs.append(StateDiff("host_compromised", cv, jv, f"host_{h}"))

    for r in set(cyborg.red_sessions) | set(jax.red_sessions):
        cs = cyborg.red_sessions.get(r, set())
        js = jax.red_sessions.get(r, set())
        if cs != js:
            diffs.append(StateDiff("red_sessions", cs, js, f"red_agent_{r}"))

    for r in set(cyborg.red_privilege) | set(jax.red_privilege):
        cp = cyborg.red_privilege.get(r, {})
        jp = jax.red_privilege.get(r, {})
        for h in set(cp) | set(jp):
            cv = cp.get(h, 0)
            jv = jp.get(h, 0)
            if cv != jv:
                diffs.append(StateDiff("red_privilege", cv, jv, f"red_{r}_host_{h}"))

    for h in set(cyborg.host_services) | set(jax.host_services):
        cs = cyborg.host_services.get(h)
        js = jax.host_services.get(h)
        if cs != js:
            diffs.append(StateDiff("host_services", cs, js, f"host_{h}"))

    for h in set(cyborg.host_service_reliability) | set(jax.host_service_reliability):
        cs = cyborg.host_service_reliability.get(h)
        js = jax.host_service_reliability.get(h)
        if cs != js:
            diffs.append(StateDiff("host_service_reliability", cs, js, f"host_{h}"))

    for h in set(cyborg.host_has_malware) | set(jax.host_has_malware):
        cv = cyborg.host_has_malware.get(h, False)
        jv = jax.host_has_malware.get(h, False)
        if cv != jv:
            diffs.append(StateDiff("host_has_malware", cv, jv, f"host_{h}"))

    for h in set(cyborg.host_decoys) | set(jax.host_decoys):
        cd = cyborg.host_decoys.get(h)
        jd = jax.host_decoys.get(h)
        if cd != jd:
            diffs.append(StateDiff("host_decoys", cd, jd, f"host_{h}"))

    for h in set(cyborg.ot_service_stopped) | set(jax.ot_service_stopped):
        cv = cyborg.ot_service_stopped.get(h, False)
        jv = jax.ot_service_stopped.get(h, False)
        if cv != jv:
            diffs.append(StateDiff("ot_service_stopped", cv, jv, f"host_{h}"))

    if cyborg.blocked_zones != jax.blocked_zones:
        diffs.append(StateDiff("blocked_zones", cyborg.blocked_zones, jax.blocked_zones))

    if cyborg.rewards != jax.rewards:
        diffs.append(StateDiff("rewards", cyborg.rewards, jax.rewards))

    return diffs


def snapshots_match(cyborg: StateSnapshot, jax: StateSnapshot) -> bool:
    return len(compare_snapshots(cyborg, jax)) == 0


def compare_fast(cyborg_env, jax_state, jax_const, mappings) -> list[StateDiff]:
    """Array-based comparison — bulk-transfers JAX arrays to numpy instead of per-element access."""
    n = mappings.num_hosts
    diffs = []

    # JAX arrays → numpy in bulk (one transfer per array)
    jax_compromised = np.asarray(jax_state.host_compromised[:n])
    jax_sessions = np.asarray(jax_state.red_sessions[:NUM_RED_AGENTS, :n])
    jax_priv = np.asarray(jax_state.red_privilege[:NUM_RED_AGENTS, :n])
    jax_discovered = np.asarray(jax_state.red_discovered_hosts[:NUM_RED_AGENTS, :n])
    jax_scanned = np.asarray(jax_state.red_scanned_hosts[:NUM_RED_AGENTS, :n])
    jax_services = np.asarray(jax_state.host_services[:n, :NUM_SERVICES])
    jax_service_reliability = np.asarray(jax_state.host_service_reliability[:n, :NUM_SERVICES])
    jax_decoys = np.asarray(jax_state.host_decoys[:n])
    jax_ot_stopped = np.asarray(jax_state.ot_service_stopped[:n])
    jax_has_malware = np.asarray(jax_state.host_has_malware[:n])
    jax_blocked = np.asarray(jax_state.blocked_zones)
    jax_phase = int(jax_state.mission_phase)

    # CybORG → numpy arrays
    cyborg_state = cyborg_env.environment_controller.state
    cyborg_compromised = np.zeros(n, dtype=np.int32)
    cyborg_sessions = np.zeros((NUM_RED_AGENTS, n), dtype=np.bool_)
    cyborg_priv = np.zeros((NUM_RED_AGENTS, n), dtype=np.int32)
    cyborg_discovered = np.zeros((NUM_RED_AGENTS, n), dtype=np.bool_)
    cyborg_scanned = np.zeros((NUM_RED_AGENTS, n), dtype=np.bool_)
    cyborg_services = np.zeros((n, NUM_SERVICES), dtype=np.bool_)
    cyborg_service_reliability = np.full((n, NUM_SERVICES), 100, dtype=np.int32)
    cyborg_has_malware = np.zeros(n, dtype=np.bool_)
    cyborg_ot_stopped = np.zeros(n, dtype=np.bool_)

    for hostname, host in cyborg_state.hosts.items():
        if hostname not in mappings.hostname_to_idx:
            continue
        hidx = mappings.hostname_to_idx[hostname]

        if any(getattr(f, "name", "") == "cmd.sh" for f in host.files):
            cyborg_has_malware[hidx] = True

        for svc_name, svc in host.services.items():
            svc_str = str(svc_name).split(".")[-1] if "." in str(svc_name) else str(svc_name)
            if svc_str not in SERVICE_IDS:
                continue
            sid = SERVICE_IDS[svc_str]
            cyborg_services[hidx, sid] = bool(svc.active)
            cyborg_service_reliability[hidx, sid] = int(svc.get_service_reliability())
            if svc_str == "OTSERVICE" and not bool(svc.active):
                cyborg_ot_stopped[hidx] = True

    for agent_name, sessions in cyborg_state.sessions.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        for sess in sessions.values():
            if sess.hostname not in mappings.hostname_to_idx:
                continue
            hidx = mappings.hostname_to_idx[sess.hostname]
            cyborg_sessions[red_idx, hidx] = True
            level = 2 if (hasattr(sess, "username") and sess.username in ("root", "SYSTEM")) else 1
            cyborg_priv[red_idx, hidx] = max(cyborg_priv[red_idx, hidx], level)
            cyborg_compromised[hidx] = max(cyborg_compromised[hidx], level)
            for ip in getattr(sess, "ports", {}).keys():
                host_from_ip = cyborg_state.ip_addresses.get(ip)
                if host_from_ip in mappings.hostname_to_idx:
                    cyborg_scanned[red_idx, mappings.hostname_to_idx[host_from_ip]] = True

    cyborg_phase = getattr(cyborg_state, "mission_phase", 0)

    controller = cyborg_env.environment_controller
    for r in range(NUM_RED_AGENTS):
        iface = controller.agent_interfaces.get(f"red_agent_{r}")
        if iface is None:
            continue
        aspace = iface.action_space

        for ip, known in getattr(aspace, "ip_address", {}).items():
            if not known:
                continue
            hostname = cyborg_state.ip_addresses.get(ip)
            if hostname in mappings.hostname_to_idx:
                cyborg_discovered[r, mappings.hostname_to_idx[hostname]] = True

    # Compare with vectorized ops
    if cyborg_phase != jax_phase:
        diffs.append(StateDiff("mission_phase", cyborg_phase, jax_phase))

    comp_mismatch = np.where(cyborg_compromised != jax_compromised)[0]
    for h in comp_mismatch:
        diffs.append(StateDiff("host_compromised", int(cyborg_compromised[h]), int(jax_compromised[h]), f"host_{h}"))

    for r in range(NUM_RED_AGENTS):
        if not np.array_equal(cyborg_sessions[r], jax_sessions[r]):
            cs = set(np.where(cyborg_sessions[r])[0].tolist())
            js = set(np.where(jax_sessions[r])[0].tolist())
            diffs.append(StateDiff("red_sessions", cs, js, f"red_agent_{r}"))

    for r in range(NUM_RED_AGENTS):
        if not np.array_equal(cyborg_discovered[r], jax_discovered[r]):
            cs = set(np.where(cyborg_discovered[r])[0].tolist())
            js = set(np.where(jax_discovered[r])[0].tolist())
            diffs.append(StateDiff("red_discovered_hosts", cs, js, f"red_agent_{r}"))

    for r in range(NUM_RED_AGENTS):
        if not np.array_equal(cyborg_scanned[r], jax_scanned[r]):
            cs = set(np.where(cyborg_scanned[r])[0].tolist())
            js = set(np.where(jax_scanned[r])[0].tolist())
            diffs.append(StateDiff("red_scanned_hosts", cs, js, f"red_agent_{r}"))

    for r in range(NUM_RED_AGENTS):
        priv_mismatch = np.where(cyborg_priv[r] != jax_priv[r])[0]
        for h in priv_mismatch:
            diffs.append(StateDiff("red_privilege", int(cyborg_priv[r, h]), int(jax_priv[r, h]), f"red_{r}_host_{h}"))

    for h in range(n):
        if not np.array_equal(cyborg_services[h], jax_services[h]):
            cs = tuple(bool(x) for x in cyborg_services[h].tolist())
            js = tuple(bool(x) for x in jax_services[h].tolist())
            diffs.append(StateDiff("host_services", cs, js, f"host_{h}"))

    for h in range(n):
        if not np.array_equal(cyborg_service_reliability[h], jax_service_reliability[h]):
            cs = tuple(int(x) for x in cyborg_service_reliability[h].tolist())
            js = tuple(int(x) for x in jax_service_reliability[h].tolist())
            diffs.append(StateDiff("host_service_reliability", cs, js, f"host_{h}"))

    malware_mismatch = np.where(cyborg_has_malware != jax_has_malware)[0]
    for h in malware_mismatch:
        diffs.append(StateDiff("host_has_malware", bool(cyborg_has_malware[h]), bool(jax_has_malware[h]), f"host_{h}"))

    cyborg_decoys = np.zeros((n, NUM_DECOY_TYPES), dtype=np.bool_)
    for hostname, host in cyborg_state.hosts.items():
        if hostname not in mappings.hostname_to_idx:
            continue
        hidx = mappings.hostname_to_idx[hostname]
        for svc_name, svc in host.services.items():
            svc_str = str(svc_name).split(".")[-1].lower()
            if svc_str not in _DECOY_SVC_TO_IDX:
                continue
            pid = svc.process if hasattr(svc, "process") else svc.get("process")
            if pid is None:
                continue
            proc = host.get_process(pid)
            if proc is not None and proc.decoy_type & DecoyType.EXPLOIT:
                cyborg_decoys[hidx, _DECOY_SVC_TO_IDX[svc_str]] = True

    for h in range(n):
        if not np.array_equal(cyborg_decoys[h], jax_decoys[h]):
            cd = tuple(bool(d) for d in cyborg_decoys[h])
            jd = tuple(bool(d) for d in jax_decoys[h])
            diffs.append(StateDiff("host_decoys", cd, jd, f"host_{h}"))

    ot_stopped_mismatch = np.where(cyborg_ot_stopped != jax_ot_stopped)[0]
    for h in ot_stopped_mismatch:
        diffs.append(StateDiff("ot_service_stopped", bool(cyborg_ot_stopped[h]), bool(jax_ot_stopped[h]), f"host_{h}"))

    cyborg_blocked = set()
    for to_subnet, from_list in getattr(cyborg_state, "blocks", {}).items():
        for from_subnet in from_list:
            cyborg_blocked.add((str(from_subnet), str(to_subnet)))
    jax_blocked_set = set()
    for dst in range(NUM_SUBNETS):
        for src in range(NUM_SUBNETS):
            if jax_blocked[dst, src]:
                src_name = mappings.subnet_names.get(src, str(src))
                dst_name = mappings.subnet_names.get(dst, str(dst))
                jax_blocked_set.add((src_name, dst_name))
    if cyborg_blocked != jax_blocked_set:
        diffs.append(StateDiff("blocked_zones", cyborg_blocked, jax_blocked_set))

    return diffs


def format_diffs(diffs: list[StateDiff]) -> str:
    if not diffs:
        return "No differences"
    lines = []
    for d in diffs:
        loc = f" [{d.host_or_agent}]" if d.host_or_agent else ""
        severity = "ERROR" if d.field_name in _ERROR_FIELDS else "WARN"
        lines.append(f"  {severity} {d.field_name}{loc}: cyborg={d.cyborg_value} jax={d.jax_value}")
    return "\n".join(lines)
