"""Explicit differential regressions for cross-subnet red session reassignment parity."""

import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.encoding import encode_red_action
from jaxborg.constants import COMPROMISE_USER, NUM_RED_AGENTS, NUM_SUBNETS
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.scenarios.cc4.red_fsm import FSM_K, FSM_KD, FSM_S, FSM_SD, FSM_U
from jaxborg.scenarios.cc4.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg
from jaxborg.state import create_initial_state


def _make_env(seed: int = 0):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    env = CybORG(scenario_generator=sg, seed=seed)
    env.reset()
    return env


_const_cache: dict[int, tuple] = {}


def _cached_const_and_mappings(seed: int = 0):
    if seed not in _const_cache:
        env = _make_env(seed)
        _const_cache[seed] = (build_const_from_cyborg(env), build_mappings_from_cyborg(env))
    return _const_cache[seed]


def _find_reassignment_case(const, source_agent: int):
    for host_idx in range(int(const.num_hosts)):
        if not bool(const.host_active[host_idx]) or bool(const.host_is_router[host_idx]):
            continue
        subnet_idx = int(const.host_subnet[host_idx])
        owners = np.flatnonzero(np.asarray(const.red_agent_subnets[:, subnet_idx]))
        if owners.size != 1:
            continue
        owner = int(owners[0])
        if owner == source_agent:
            continue
        if bool(const.red_agent_subnets[source_agent, subnet_idx]):
            continue
        return host_idx, owner
    return None, None


def _cy_scanned_hosts(state, mappings, agent_id: int):
    scanned = set()
    for sess in state.sessions[f"red_agent_{agent_id}"].values():
        for ip in getattr(sess, "ports", {}).keys():
            hostname = state.ip_addresses.get(ip)
            if hostname in mappings.hostname_to_idx:
                scanned.add(mappings.hostname_to_idx[hostname])
    return scanned


def test_red_agent_allowed_subnets_match_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    const, _ = _cached_const_and_mappings(seed=0)
    for red_id in range(NUM_RED_AGENTS):
        cy_allowed = set(controller.agent_interfaces[f"red_agent_{red_id}"].allowed_subnets)
        cy_allowed_ids = {CYBORG_SUFFIX_TO_ID[name] for name in cy_allowed if name in CYBORG_SUFFIX_TO_ID}
        jax_allowed_ids = {sid for sid in range(NUM_SUBNETS) if bool(const.red_agent_subnets[red_id, sid])}
        assert jax_allowed_ids == cy_allowed_ids


def test_cross_subnet_reassignment_drops_session_scan_memory_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 5
    target_idx = None
    dest_agent = None
    for host_idx in range(int(const.num_hosts)):
        if not bool(const.host_active[host_idx]) or bool(const.host_is_router[host_idx]):
            continue
        subnet_idx = int(const.host_subnet[host_idx])
        owners = np.flatnonzero(np.asarray(const.red_agent_subnets[:, subnet_idx]))
        if owners.size != 1:
            continue
        owner = int(owners[0])
        if owner == source_agent or bool(const.red_agent_subnets[source_agent, subnet_idx]):
            continue
        hostname = mappings.idx_to_hostname[host_idx]
        owner_has_host_session = any(
            sess.hostname == hostname for sess in cy_state.sessions[f"red_agent_{owner}"].values()
        )
        if owner_has_host_session:
            continue
        target_idx = host_idx
        dest_agent = owner
        break
    assert target_idx is not None and dest_agent is not None

    target_hostname = mappings.idx_to_hostname[target_idx]
    seeded_scan_hosts = [target_idx]
    for h in range(int(const.num_hosts)):
        if h == target_idx:
            continue
        seeded_scan_hosts.append(h)
        if len(seeded_scan_hosts) == 5:
            break

    red_session = RedAbstractSession(
        ident=None,
        hostname=target_hostname,
        username="user",
        agent=f"red_agent_{source_agent}",
        parent=0,
        session_type="shell",
        pid=None,
    )
    cy_state.add_session(red_session)
    cy_src_session = next(
        sess for sess in cy_state.sessions[f"red_agent_{source_agent}"].values() if sess.hostname == target_hostname
    )
    for h in seeded_scan_hosts:
        ip = mappings.hostname_to_ip[mappings.idx_to_hostname[h]]
        cy_src_session.addport(ip, 22)
    assert _cy_scanned_hosts(cy_state, mappings, source_agent) == set(seeded_scan_hosts)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    red_sessions = jax_state.red_sessions.at[source_agent, target_idx].set(True)
    red_privilege = jax_state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER)
    red_discovered = jax_state.red_discovered_hosts.at[source_agent, target_idx].set(True)
    red_session_count = jax_state.red_session_count.at[source_agent, target_idx].set(1)
    red_scanned = jax_state.red_scanned_hosts
    for h in seeded_scan_hosts:
        red_scanned = red_scanned.at[source_agent, h].set(True)
    jax_state = jax_state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned,
        host_compromised=jax_state.host_compromised.at[target_idx].set(COMPROMISE_USER),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_src_has_target_session = any(
        sess.hostname == target_hostname for sess in cy_state.sessions[f"red_agent_{source_agent}"].values()
    )
    cy_dst_has_target_session = any(
        sess.hostname == target_hostname for sess in cy_state.sessions[f"red_agent_{dest_agent}"].values()
    )

    assert not cy_src_has_target_session
    assert cy_dst_has_target_session
    assert bool(jax_after.red_sessions[source_agent, target_idx]) == cy_src_has_target_session
    assert bool(jax_after.red_sessions[dest_agent, target_idx]) == cy_dst_has_target_session

    cy_src_scanned = _cy_scanned_hosts(cy_state, mappings, source_agent)
    cy_dst_scanned = _cy_scanned_hosts(cy_state, mappings, dest_agent)
    assert cy_src_scanned == set()
    assert cy_dst_scanned == set()

    jax_src_scanned = {h for h in range(int(const.num_hosts)) if bool(jax_after.red_scanned_hosts[source_agent, h])}
    jax_dst_scanned = {h for h in range(int(const.num_hosts)) if bool(jax_after.red_scanned_hosts[dest_agent, h])}
    assert jax_src_scanned == cy_src_scanned
    assert jax_dst_scanned == cy_dst_scanned


def test_cross_subnet_reassignment_preserves_nonabstract_session_type_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 5
    target_idx = None
    dest_agent = None
    for host_idx in range(int(const.num_hosts)):
        if not bool(const.host_active[host_idx]) or bool(const.host_is_router[host_idx]):
            continue
        subnet_idx = int(const.host_subnet[host_idx])
        owners = np.flatnonzero(np.asarray(const.red_agent_subnets[:, subnet_idx]))
        if owners.size != 1:
            continue
        owner = int(owners[0])
        if owner == source_agent or bool(const.red_agent_subnets[source_agent, subnet_idx]):
            continue
        hostname = mappings.idx_to_hostname[host_idx]
        owner_has_session = any(sess.hostname == hostname for sess in cy_state.sessions[f"red_agent_{owner}"].values())
        if owner_has_session:
            continue
        target_idx = host_idx
        dest_agent = owner
        break
    assert target_idx is not None and dest_agent is not None

    target_hostname = mappings.idx_to_hostname[target_idx]
    cy_state.add_session(
        Session(
            ident=999,
            hostname=target_hostname,
            username="user",
            agent=f"red_agent_{source_agent}",
            pid=9001,
            parent=None,
        )
    )

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, target_idx].set(True),
        red_session_count=jax_state.red_session_count.at[source_agent, target_idx].set(1),
        red_session_pids=jax_state.red_session_pids.at[source_agent, target_idx, 0].set(9001),
        red_privilege=jax_state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
        red_discovered_hosts=jax_state.red_discovered_hosts.at[source_agent, target_idx].set(True),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[source_agent, target_idx].set(False),
        host_compromised=jax_state.host_compromised.at[target_idx].set(COMPROMISE_USER),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_dst_sessions = [
        sess for sess in cy_state.sessions[f"red_agent_{dest_agent}"].values() if sess.hostname == target_hostname
    ]
    assert cy_dst_sessions, "CybORG should move session to subnet owner"
    cy_has_abstract = any(type(sess).__name__ == "RedAbstractSession" for sess in cy_dst_sessions)

    assert bool(jax_after.red_sessions[dest_agent, target_idx])
    assert bool(jax_after.red_session_is_abstract[dest_agent, target_idx]) == cy_has_abstract


def test_cross_subnet_reassignment_does_not_overclear_existing_scan_memory_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 0
    target_idx, _ = _find_reassignment_case(const, source_agent)
    assert target_idx is not None
    target_hostname = mappings.idx_to_hostname[target_idx]

    keep_host_idx = int(const.red_start_hosts[source_agent])
    keep_host_name = mappings.idx_to_hostname[keep_host_idx]
    keep_ip = mappings.hostname_to_ip[keep_host_name]

    base_session = cy_state.sessions[f"red_agent_{source_agent}"][0]
    base_session.addport(keep_ip, 22)

    reassign_session = RedAbstractSession(
        ident=None,
        hostname=target_hostname,
        username="user",
        agent=f"red_agent_{source_agent}",
        parent=0,
        session_type="shell",
        pid=None,
    )
    cy_state.add_session(reassign_session)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_session_abstract_pids=jax_state.red_session_abstract_pids.at[source_agent, keep_host_idx, 0]
        .set(100)
        .at[source_agent, target_idx, 0]
        .set(101),
        red_session_count=jax_state.red_session_count.at[source_agent, keep_host_idx]
        .set(1)
        .at[source_agent, target_idx]
        .set(1),
        red_privilege=jax_state.red_privilege.at[source_agent, keep_host_idx]
        .set(COMPROMISE_USER)
        .at[source_agent, target_idx]
        .set(COMPROMISE_USER),
        red_discovered_hosts=jax_state.red_discovered_hosts.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_scanned_hosts=jax_state.red_scanned_hosts.at[source_agent, keep_host_idx].set(True),
        red_scanned_source_hosts=jax_state.red_scanned_source_hosts.at[source_agent, keep_host_idx, keep_host_idx].set(
            True
        ),
        host_compromised=jax_state.host_compromised.at[keep_host_idx]
        .set(COMPROMISE_USER)
        .at[target_idx]
        .set(COMPROMISE_USER),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_src_scanned = _cy_scanned_hosts(cy_state, mappings, source_agent)
    jax_src_scanned = {h for h in range(int(const.num_hosts)) if bool(jax_after.red_scanned_hosts[source_agent, h])}
    assert cy_src_scanned == {keep_host_idx}
    assert jax_src_scanned == cy_src_scanned


def test_cross_subnet_reassignment_keeps_remote_scan_memory_when_unrelated_session_moves_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 0
    target_idx, _ = _find_reassignment_case(const, source_agent)
    assert target_idx is not None

    base_session = cy_state.sessions[f"red_agent_{source_agent}"][0]
    keep_host_idx = mappings.hostname_to_idx[base_session.hostname]
    remote_scan_host = next(h for h in range(int(const.num_hosts)) if h not in {keep_host_idx, target_idx})
    base_session.addport(mappings.hostname_to_ip[mappings.idx_to_hostname[remote_scan_host]], 22)

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[target_idx],
            username="user",
            agent=f"red_agent_{source_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    assert _cy_scanned_hosts(cy_state, mappings, source_agent) == {remote_scan_host}

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_session_abstract_pids=jax_state.red_session_abstract_pids.at[source_agent, keep_host_idx, 0]
        .set(100)
        .at[source_agent, target_idx, 0]
        .set(101),
        red_session_count=jax_state.red_session_count.at[source_agent, keep_host_idx]
        .set(1)
        .at[source_agent, target_idx]
        .set(1),
        red_privilege=jax_state.red_privilege.at[source_agent, keep_host_idx]
        .set(COMPROMISE_USER)
        .at[source_agent, target_idx]
        .set(COMPROMISE_USER),
        red_discovered_hosts=jax_state.red_discovered_hosts.at[source_agent, keep_host_idx]
        .set(True)
        .at[source_agent, target_idx]
        .set(True),
        red_scanned_hosts=jax_state.red_scanned_hosts.at[source_agent, remote_scan_host].set(True),
        red_scanned_source_hosts=jax_state.red_scanned_source_hosts.at[
            source_agent, remote_scan_host, keep_host_idx
        ].set(True),
        red_scan_anchor_host=jax_state.red_scan_anchor_host.at[source_agent].set(keep_host_idx),
        host_compromised=jax_state.host_compromised.at[keep_host_idx]
        .set(COMPROMISE_USER)
        .at[target_idx]
        .set(COMPROMISE_USER),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_src_scanned = _cy_scanned_hosts(cy_state, mappings, source_agent)
    jax_src_scanned = {h for h in range(int(const.num_hosts)) if bool(jax_after.red_scanned_hosts[source_agent, h])}
    assert cy_src_scanned == {remote_scan_host}
    assert jax_src_scanned == cy_src_scanned


def test_reassignment_does_not_rebind_busy_scan_when_bound_source_becomes_invalid():
    const, _ = _cached_const_and_mappings(seed=0)

    agent_id = 0
    candidate_hosts = [
        h for h in range(int(const.num_hosts)) if bool(const.host_active[h]) and not bool(const.host_is_router[h])
    ]
    assert len(candidate_hosts) >= 3
    source_host, alt_host, target_host = candidate_hosts[:3]

    state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    scan_idx = encode_red_action("StealthServiceDiscovery", target_host, agent_id)
    state = state.replace(
        red_sessions=state.red_sessions.at[agent_id, alt_host].set(True),
        red_session_count=state.red_session_count.at[agent_id, alt_host].set(1),
        red_session_is_abstract=state.red_session_is_abstract.at[agent_id, alt_host].set(True),
        red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
        red_pending_ticks=state.red_pending_ticks.at[agent_id].set(1),
        red_pending_action=state.red_pending_action.at[agent_id].set(scan_idx),
        red_pending_source_host=state.red_pending_source_host.at[agent_id].set(source_host),
    )

    out = reassign_cross_subnet_sessions(state, const)

    assert int(out.red_pending_source_host[agent_id]) == source_host


def test_cross_subnet_reassignment_clears_remote_scan_memory_when_scan_owner_session_moves_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 0
    target_idx, _ = _find_reassignment_case(const, source_agent)
    assert target_idx is not None
    target_hostname = mappings.idx_to_hostname[target_idx]
    source_sessions = cy_state.sessions[f"red_agent_{source_agent}"]
    base_session = source_sessions[0]
    old_base_host = base_session.hostname
    cy_state.hosts[old_base_host].sessions[f"red_agent_{source_agent}"].remove(0)
    base_session.hostname = target_hostname
    cy_state.hosts[target_hostname].sessions[f"red_agent_{source_agent}"].append(0)
    source_hosts = {mappings.hostname_to_idx[s.hostname] for s in source_sessions.values()}
    remote_scan_host = next(h for h in range(int(const.num_hosts)) if h not in source_hosts)
    base_session.addport(mappings.hostname_to_ip[mappings.idx_to_hostname[remote_scan_host]], 22)

    retained_hosts = []
    for h in range(int(const.num_hosts)):
        if h == target_idx:
            continue
        if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
            continue
        subnet_idx = int(const.host_subnet[h])
        if not bool(const.red_agent_subnets[source_agent, subnet_idx]):
            continue
        if h in source_hosts:
            continue
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[h],
                username="user",
                agent=f"red_agent_{source_agent}",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        retained_hosts.append(h)
        if len(retained_hosts) == 2:
            break
    assert len(retained_hosts) == 2
    assert _cy_scanned_hosts(cy_state, mappings, source_agent) == {remote_scan_host}

    all_source_hosts = sorted(source_hosts | set(retained_hosts))
    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    for h in all_source_hosts:
        jax_state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[source_agent, h].set(True),
            red_session_count=jax_state.red_session_count.at[source_agent, h].set(1),
            red_privilege=jax_state.red_privilege.at[source_agent, h].set(COMPROMISE_USER),
            red_discovered_hosts=jax_state.red_discovered_hosts.at[source_agent, h].set(True),
            host_compromised=jax_state.host_compromised.at[h].set(COMPROMISE_USER),
        )
    jax_state = jax_state.replace(
        red_scanned_hosts=jax_state.red_scanned_hosts.at[source_agent, remote_scan_host].set(True),
        red_scanned_source_hosts=jax_state.red_scanned_source_hosts.at[source_agent, remote_scan_host, target_idx].set(
            True
        ),
        red_scan_anchor_host=jax_state.red_scan_anchor_host.at[source_agent].set(target_idx),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_src_scanned = _cy_scanned_hosts(cy_state, mappings, source_agent)
    jax_src_scanned = {h for h in range(int(const.num_hosts)) if bool(jax_after.red_scanned_hosts[source_agent, h])}
    assert cy_src_scanned == set()
    assert jax_src_scanned == cy_src_scanned


def test_cross_subnet_reassignment_preserves_scan_anchor_host_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 0
    red_name = f"red_agent_{source_agent}"
    cy_anchor_host = cy_state.sessions[red_name][0].hostname
    cy_anchor_idx = mappings.hostname_to_idx[cy_anchor_host]

    keep_idx = None
    for h in range(int(const.num_hosts)):
        if h == cy_anchor_idx:
            continue
        if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
            continue
        subnet_idx = int(const.host_subnet[h])
        if bool(const.red_agent_subnets[source_agent, subnet_idx]):
            keep_idx = h
            break
    assert keep_idx is not None

    transfer_idx, _ = _find_reassignment_case(const, source_agent)
    assert transfer_idx is not None

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[keep_idx],
            username="user",
            agent=red_name,
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[transfer_idx],
            username="user",
            agent=red_name,
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, cy_anchor_idx]
        .set(True)
        .at[source_agent, keep_idx]
        .set(True)
        .at[source_agent, transfer_idx]
        .set(True),
        red_session_count=jax_state.red_session_count.at[source_agent, cy_anchor_idx]
        .set(1)
        .at[source_agent, keep_idx]
        .set(1)
        .at[source_agent, transfer_idx]
        .set(1),
        red_privilege=jax_state.red_privilege.at[source_agent, cy_anchor_idx]
        .set(COMPROMISE_USER)
        .at[source_agent, keep_idx]
        .set(COMPROMISE_USER)
        .at[source_agent, transfer_idx]
        .set(COMPROMISE_USER),
        red_discovered_hosts=jax_state.red_discovered_hosts.at[source_agent, cy_anchor_idx]
        .set(True)
        .at[source_agent, keep_idx]
        .set(True)
        .at[source_agent, transfer_idx]
        .set(True),
        red_scan_anchor_host=jax_state.red_scan_anchor_host.at[source_agent].set(cy_anchor_idx),
        host_compromised=jax_state.host_compromised.at[cy_anchor_idx]
        .set(COMPROMISE_USER)
        .at[keep_idx]
        .set(COMPROMISE_USER)
        .at[transfer_idx]
        .set(COMPROMISE_USER),
    )

    controller.different_subnet_agent_reassignment()
    jax_after = reassign_cross_subnet_sessions(jax_state, const)

    cy_anchor_after = cy_state.sessions[red_name][0].hostname
    cy_anchor_after_idx = mappings.hostname_to_idx[cy_anchor_after]
    assert cy_anchor_after_idx == cy_anchor_idx
    assert int(jax_after.red_scan_anchor_host[source_agent]) == cy_anchor_after_idx


def test_cross_subnet_reassignment_sums_transferred_suspicious_counts_per_session_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const, mappings = _cached_const_and_mappings(seed=0)

    source_agent = 0
    target_idx, dest_agent = _find_reassignment_case(const, source_agent)
    assert target_idx is not None
    assert dest_agent is not None

    target_subnet = int(const.host_subnet[target_idx])
    extra_source = next(
        r
        for r in range(NUM_RED_AGENTS)
        if r not in {source_agent, dest_agent} and not bool(const.red_agent_subnets[r, target_subnet])
    )
    target_hostname = mappings.idx_to_hostname[target_idx]

    injected = [(source_agent, 7100), (extra_source, 7101)]
    for src, pid in injected:
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent=f"red_agent_{src}",
                parent=0,
                session_type="shell",
                pid=pid,
            )
        )

    controller.different_subnet_agent_reassignment()
    cy_dest_pid_sessions = [
        sess
        for sess in cy_state.sessions[f"red_agent_{dest_agent}"].values()
        if sess.hostname == target_hostname and getattr(sess, "pid", None) in {7100, 7101}
    ]
    assert len(cy_dest_pid_sessions) == len(injected)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    for src, pid in injected:
        pid_row = jax_state.red_session_pids[src, target_idx]
        pid_row = pid_row.at[0].set(pid)
        jax_state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[src, target_idx].set(True),
            red_session_count=jax_state.red_session_count.at[src, target_idx].set(1),
            red_privilege=jax_state.red_privilege.at[src, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=jax_state.red_discovered_hosts.at[src, target_idx].set(True),
            red_suspicious_process_count=jax_state.red_suspicious_process_count.at[src, target_idx].set(1),
            red_session_pids=jax_state.red_session_pids.at[src, target_idx].set(pid_row),
            host_compromised=jax_state.host_compromised.at[target_idx].set(COMPROMISE_USER),
        )

    jax_after = reassign_cross_subnet_sessions(jax_state, const)
    expected = len(cy_dest_pid_sessions)
    assert int(jax_after.red_session_count[dest_agent, target_idx]) == expected
    assert int(jax_after.red_suspicious_process_count[dest_agent, target_idx]) == expected


class TestReassignmentPreservesFsmState:
    """CybORG's FSM (FiniteStateRedAgent.host_states) is NOT modified by
    session reassignment.  The FSM state only changes through the agent's own
    actions or session-loss recovery.  JAX must match this behaviour."""

    def test_reassignment_preserves_sd_state(self):
        """Host at FSM_SD must stay SD after reassignment, not become U."""
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        # Dest agent already has host in FSM at SD (scanned + decoy found)
        state = state.replace(
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx].set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            # Dest agent already tracks this host in FSM at SD
            fsm_host_states=state.fsm_host_states.at[dest_agent, target_idx].set(FSM_SD),
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, target_idx].set(True),
        )

        after = reassign_cross_subnet_sessions(state, const)

        # Session should be reassigned to dest
        assert bool(after.red_sessions[dest_agent, target_idx])
        # FSM must stay at SD, NOT become U or UD
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_SD, (
            f"FSM should stay SD after reassignment, got {int(after.fsm_host_states[dest_agent, target_idx])}"
        )

    def test_reassignment_preserves_s_state(self):
        """Host at FSM_S must stay S after reassignment."""
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx].set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            fsm_host_states=state.fsm_host_states.at[dest_agent, target_idx].set(FSM_S),
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, target_idx].set(True),
        )

        after = reassign_cross_subnet_sessions(state, const)
        assert bool(after.red_sessions[dest_agent, target_idx])
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_S

    def test_reassignment_preserves_kd_state(self):
        """Host at FSM_KD must stay KD after reassignment."""
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx].set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            fsm_host_states=state.fsm_host_states.at[dest_agent, target_idx].set(FSM_KD),
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, target_idx].set(True),
        )

        after = reassign_cross_subnet_sessions(state, const)
        assert bool(after.red_sessions[dest_agent, target_idx])
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_KD

    def test_reassignment_sets_u_for_new_host_not_in_fsm(self):
        """Host NOT yet in FSM (entered=False) gets U on user-session reassignment."""
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx].set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            # Host NOT entered in dest agent's FSM
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, target_idx].set(False),
        )

        after = reassign_cross_subnet_sessions(state, const)
        assert bool(after.red_sessions[dest_agent, target_idx])
        # Newly activated agent: hosts enter as U (CybORG step 0 behavior)
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_U

    def test_reassignment_sets_k_for_new_host_on_already_active_agent(self):
        """Host NOT yet in FSM on an ALREADY-ACTIVE agent gets K on reassignment.

        CybORG's _process_new_observations assigns 'K' (not 'U') when step > 0.
        Only newly activated agents (step 0) get 'U'.
        """
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        # dest_agent is ALREADY active (has sessions elsewhere)
        other_host = 0  # some other host
        state = state.replace(
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True).at[dest_agent, other_host].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx]
            .set(1)
            .at[dest_agent, other_host]
            .set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            red_agent_active=state.red_agent_active.at[dest_agent].set(True),
            # Host NOT entered in dest agent's FSM
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, target_idx].set(False),
        )

        after = reassign_cross_subnet_sessions(state, const)
        assert bool(after.red_sessions[dest_agent, target_idx])
        # Already-active agent: hosts enter as K (CybORG step > 0 behavior)
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_K
        # Must be marked as entered in FSM
        assert bool(after.fsm_host_entered[dest_agent, target_idx])


class TestReactivatedAgentFsmState:
    """Regression: when an agent is REACTIVATED (was active, lost all sessions,
    then received new sessions), CybORG's step counter never resets.  So
    _process_new_observations runs with step > 0, assigning 'K' (not 'U') to
    newly observed hosts.  JAX must match this by distinguishing first-time
    activation (U) from reactivation (K)."""

    def test_reactivated_agent_gets_k_not_u(self):
        """Reactivated agent must assign FSM_K to new hosts, not FSM_U."""
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        # Pick a host that was previously in the FSM (from prior activation)
        prior_host = 0

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            # Source agent has a session on target (will be reassigned to dest)
            red_sessions=state.red_sessions.at[source_agent, target_idx].set(True),
            red_session_count=state.red_session_count.at[source_agent, target_idx].set(1),
            red_privilege=state.red_privilege.at[source_agent, target_idx].set(COMPROMISE_USER),
            red_discovered_hosts=state.red_discovered_hosts.at[source_agent, target_idx].set(True),
            host_compromised=state.host_compromised.at[target_idx].set(COMPROMISE_USER),
            # dest_agent was PREVIOUSLY active but is now INACTIVE (lost all sessions)
            red_agent_active=state.red_agent_active.at[dest_agent].set(False),
            # Prior activation left FSM entries — proves agent has been active before
            fsm_host_entered=state.fsm_host_entered.at[dest_agent, prior_host].set(True),
            fsm_host_states=state.fsm_host_states.at[dest_agent, prior_host].set(FSM_KD),
            # target_idx NOT yet in dest's FSM (fsm_host_entered default False)
        )

        after = reassign_cross_subnet_sessions(state, const)
        assert bool(after.red_sessions[dest_agent, target_idx])
        # Reactivated agent: hosts enter as K (CybORG step > 0), NOT U
        assert int(after.fsm_host_states[dest_agent, target_idx]) == FSM_K, (
            f"Reactivated agent should get FSM_K, got {int(after.fsm_host_states[dest_agent, target_idx])}"
        )
        assert bool(after.fsm_host_entered[dest_agent, target_idx])


class TestReassignmentAnchorUsesRank:
    """Regression: when a newly activated agent receives sessions on multiple
    hosts simultaneously, the scan anchor must be set to the host with the
    lowest abstract rank (ident=0) — matching CybORG's add_session order.

    Before the fix, recompute_scan_anchor_hosts used jnp.argmax (lowest host
    index) as fallback, which doesn't match CybORG's ident=0 assignment when
    the lowest-index host isn't the first one reassigned.
    """

    def test_anchor_prefers_lowest_rank_host(self):
        const, mappings = _cached_const_and_mappings(seed=0)
        source_agent = 5
        target_idx, dest_agent = _find_reassignment_case(const, source_agent)
        assert target_idx is not None

        # Test the anchor selection directly:
        # Set up dest_agent as newly activated with sessions on two hosts,
        # host_lo (low index) with rank=1 and host_hi (high index) with rank=0.
        # The anchor should pick host_hi (lowest rank), not host_lo (lowest index).
        # Both hosts must be in dest_agent's allowed subnets so reassignment
        # doesn't move them away.
        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))

        allowed_subnets = np.array(const.red_agent_subnets[dest_agent])
        host_lo = None
        host_hi = None
        for h in range(int(const.num_hosts)):
            sub = int(const.host_subnet[h])
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and allowed_subnets[sub]:
                if host_lo is None:
                    host_lo = h
                elif host_hi is None and h > host_lo:
                    host_hi = h
                    break
        assert host_lo is not None and host_hi is not None, (
            f"Need 2 active non-router hosts in dest_agent={dest_agent}'s allowed subnets"
        )

        state = state.replace(
            red_sessions=state.red_sessions.at[dest_agent, host_lo].set(True).at[dest_agent, host_hi].set(True),
            red_session_count=state.red_session_count.at[dest_agent, host_lo].set(1).at[dest_agent, host_hi].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[dest_agent, host_lo]
            .set(True)
            .at[dest_agent, host_hi]
            .set(True),
            # host_hi has rank 0 (ident=0, CybORG primary), host_lo has rank 1
            red_abstract_host_rank=state.red_abstract_host_rank.at[dest_agent, host_hi]
            .set(0)
            .at[dest_agent, host_lo]
            .set(1),
            red_agent_active=state.red_agent_active.at[dest_agent].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[dest_agent].set(jnp.int32(-1)),
        )

        after = reassign_cross_subnet_sessions(state, const)
        anchor = int(after.red_scan_anchor_host[dest_agent])
        assert anchor == host_hi, (
            f"Anchor should be host_hi={host_hi} (rank=0) but got host_lo={host_lo} (rank=1). "
            f"recompute_scan_anchor_hosts must prefer lowest rank, not lowest index."
        )
