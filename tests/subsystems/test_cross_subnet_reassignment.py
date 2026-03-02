"""Explicit differential regressions for cross-subnet red session reassignment parity."""

import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Shared.Session import RedAbstractSession
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import COMPROMISE_USER, NUM_RED_AGENTS, NUM_SUBNETS
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.state import create_initial_state
from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg
from jaxborg.translate import build_mappings_from_cyborg


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
    const = build_const_from_cyborg(env)
    for red_id in range(NUM_RED_AGENTS):
        cy_allowed = set(controller.agent_interfaces[f"red_agent_{red_id}"].allowed_subnets)
        cy_allowed_ids = {CYBORG_SUFFIX_TO_ID[name] for name in cy_allowed if name in CYBORG_SUFFIX_TO_ID}
        jax_allowed_ids = {sid for sid in range(NUM_SUBNETS) if bool(const.red_agent_subnets[red_id, sid])}
        assert jax_allowed_ids == cy_allowed_ids


def test_cross_subnet_reassignment_drops_session_scan_memory_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const = build_const_from_cyborg(env)
    mappings = build_mappings_from_cyborg(env)

    source_agent = 5
    target_idx, dest_agent = _find_reassignment_case(const, source_agent)
    assert target_idx is not None
    assert dest_agent is not None

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


def test_cross_subnet_reassignment_does_not_overclear_existing_scan_memory_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const = build_const_from_cyborg(env)
    mappings = build_mappings_from_cyborg(env)

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
    const = build_const_from_cyborg(env)
    mappings = build_mappings_from_cyborg(env)

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


def test_cross_subnet_reassignment_clears_remote_scan_memory_when_scan_owner_session_moves_matches_cyborg():
    env = _make_env(seed=0)
    controller = env.environment_controller
    cy_state = controller.state
    const = build_const_from_cyborg(env)
    mappings = build_mappings_from_cyborg(env)

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
        red_scanned_via=jax_state.red_scanned_via.at[source_agent, remote_scan_host].set(target_idx),
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
    const = build_const_from_cyborg(env)
    mappings = build_mappings_from_cyborg(env)

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
