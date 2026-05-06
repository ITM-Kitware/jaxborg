"""Green-agent CybORG parity tests, ported to ``IndexedRNGTape``.

These verify that JAX's ``apply_green_agents`` matches CybORG's behaviour for
edge cases that the live differential harness might not exercise:

* ASF reward keys off the *source* subnet, not the destination.
* Phishing prefers a same-subnet source agent over a remote one.
* Phishing creates an *abstract* session at the target.
* Phishing ignores subnet blocks when picking a source agent.
* Phishing does not reuse a stale blue suspicious PID.
* ``Remove`` clears phishing + follow-on compromise.

The originals injected per-(time, host, field) values via the retired
``green_randoms`` const fields.  These ports use :class:`IndexedRNGTape`'s
``set_green_uniform`` / ``set_green_int_range`` and consume them via
``indexed_rng_impls(**tape.as_overrides())`` — the same mechanism the
differential harness uses for full-episode parity replay.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Actions import Remove
from CybORG.Simulator.Actions.ConcreteActions.PhishingEmail import PhishingEmail
from CybORG.Simulator.Actions.GreenActions.GreenAccessService import GreenAccessService
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.encoding import encode_blue_action
from jaxborg.actions.green import (
    FP_DETECTION_RATE,
    GREEN_ACCESS_SERVICE,
    GREEN_LOCAL_WORK,
    apply_green_agents,
)
from jaxborg.actions.green_vmap import (
    PHISHING_ERROR_RATE,
    apply_green_agents_vmapped,
)
from jaxborg.actions.red_common import apply_exploit_success
from jaxborg.actions.rng import indexed_rng_impls
from jaxborg.constants import (
    CC4_CONFIG,
    COMPROMISE_NONE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.env import _init_red_state
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.rewards import ASF, compute_reward_breakdown
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, build_topology
from jaxborg.state import create_initial_state
from tests.differential.parity_rng_replay import IndexedRNGTape

pytestmark = pytest.mark.slow


def _make_green_tape(*, host_fields):
    """Build an :class:`IndexedRNGTape` populated with the per-host green draws.

    ``host_fields`` maps ``host_idx -> {field_idx: value}``.  Field semantics
    (must match :func:`jaxborg.actions.green.apply_green_agents`):

      * 0 (action selector, int_range=NUM_GREEN_ACTIONS): integer 0..2 →
        DO_NOTHING / LOCAL_WORK / ACCESS_SERVICE.
      * 1 (service token, int_range=num_available): integer index into the
        host's sorted active-tokens list (use 0 for "first").
      * 2 (reliability roll, int_range=100): integer 0..99 (lower → succeed).
      * 3 (LWF false-positive roll, float in [0,1)).
      * 4 (phishing roll, float in [0,1)).
      * 5 (access-service dest host index): float(dest_host_idx).
      * 6 (access-service FP roll, float in [0,1)).
      * 7 (pid delta, int_range=9): integer 0..8; the green action then adds 1.

    Float-typed fields are written into ``set_green_uniform``; int_range
    fields are also written there but additionally pinned via
    ``set_green_int_range`` so the tape returns the integer override
    directly (rather than ``floor(uniform * int_range)``).
    """
    table = np.zeros((GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    int_table = np.full((GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), -1, dtype=np.int32)
    int_fields = {0, 1, 2, 7}
    for host_idx, fields in host_fields.items():
        for field_idx, value in fields.items():
            table[host_idx, field_idx] = float(value)
            if field_idx in int_fields:
                int_table[host_idx, field_idx] = int(value)
    tape = IndexedRNGTape(strict=False)
    tape.set_green_uniform(table)
    tape.set_green_int_range(int_table)
    return tape


def _phishing_fields(target_host: int) -> dict:
    """Common per-host green-tape fields that force a phishing event."""
    return {
        target_host: {
            0: GREEN_LOCAL_WORK,  # action selector
            1: 0,  # service token (first sorted)
            2: 0,  # reliability roll: succeed
            3: 0.5,  # LWF FP roll: no FP
            4: 0.0,  # phishing roll: trigger
        }
    }


def _fresh_cyborg_and_const(seed: int = 0):
    """Build a fresh CybORG env + matching JAX const/mappings."""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=seed)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)
    return cyborg_env, cy_state, const, mappings


def _clear_red_sessions(cy_state):
    for r in range(NUM_RED_AGENTS):
        red_name = f"red_agent_{r}"
        cy_state.sessions[red_name] = {}
        cy_state.sessions_count[red_name] = 0
    for host in cy_state.hosts.values():
        for r in range(NUM_RED_AGENTS):
            host.sessions[f"red_agent_{r}"] = []


def _run_green_with_tape(jax_state, const, tape, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    with indexed_rng_impls(**tape.as_overrides()):
        return apply_green_agents(jax_state, const, key)


def test_green_access_service_reward_uses_source_subnet_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()

    source_host = None
    dest_host = None
    phase = 0
    asf_weights = np.array(const.phase_rewards[phase, :, ASF])
    for candidate_src in range(int(const.num_hosts)):
        if not bool(const.green_agent_active[candidate_src]):
            continue
        src_subnet = int(const.host_subnet[candidate_src])
        for candidate_dst in range(int(const.num_hosts)):
            if candidate_dst == candidate_src:
                continue
            if not bool(const.host_active[candidate_dst]) or not bool(const.host_is_server[candidate_dst]):
                continue
            dst_subnet = int(const.host_subnet[candidate_dst])
            if asf_weights[src_subnet] == asf_weights[dst_subnet]:
                continue
            source_host = candidate_src
            dest_host = candidate_dst
            break
        if source_host is not None:
            break

    assert source_host is not None and dest_host is not None

    source_subnet = int(const.host_subnet[source_host])
    dest_subnet = int(const.host_subnet[dest_host])
    expected_reward = float(const.phase_rewards[phase, source_subnet, ASF])
    wrong_dest_reward = float(const.phase_rewards[phase, dest_subnet, ASF])
    assert expected_reward != wrong_dest_reward

    source_hostname = mappings.idx_to_hostname[source_host]
    dest_hostname = mappings.idx_to_hostname[dest_host]
    source_ip = mappings.hostname_to_ip[source_hostname]
    dest_ip = mappings.hostname_to_ip[dest_hostname]
    green_idx = int(const.green_agent_host[source_host])
    green_name = f"green_agent_{green_idx}"
    allowed_subnets = cy_state.scenario.agents[green_name].allowed_subnets

    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[source_hostname].value, []).append(
        cy_state.hostname_subnet_map[dest_hostname].value
    )
    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[dest_hostname].value, []).append(
        cy_state.hostname_subnet_map[source_hostname].value
    )
    cy_action = GreenAccessService(
        agent=green_name,
        session_id=0,
        src_ip=source_ip,
        allowed_subnets=allowed_subnets,
        fp_detection_rate=FP_DETECTION_RATE,
    )
    cy_action.random_reachable_ip = lambda state: dest_ip
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "FALSE"
    assert cy_state.hosts[dest_hostname].events.network_connections

    class _ObsWrapper:
        def __init__(self, obs):
            self.observations = [obs]

    cy_reward = BlueRewardMachine("").calculate_reward(
        current_state={},
        action_dict={green_name: [cy_action]},
        agent_observations={green_name: _ObsWrapper(cy_obs)},
        done=False,
        state=cy_state,
    )
    assert cy_reward == pytest.approx(expected_reward)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    active_green = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    active_green[source_host] = True
    blocked_zones = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    blocked_zones[source_subnet, dest_subnet] = True
    blocked_zones[dest_subnet, source_subnet] = True
    jax_state = jax_state.replace(
        blocked_zones=jnp.array(blocked_zones),
    )
    const = const.replace(green_agent_active=jnp.array(active_green))

    tape = _make_green_tape(
        host_fields={
            source_host: {
                0: GREEN_ACCESS_SERVICE,
                5: dest_host,
                6: 0.5,
            }
        }
    )
    jax_after = _run_green_with_tape(jax_state, const, tape)
    jax_reward = compute_reward_breakdown(
        jax_after,
        const,
        jax_after.red_impact_attempted,
        jax_after.green_lwf_this_step,
        jax_after.green_asf_this_step,
    )

    assert bool(jax_after.green_asf_this_step[source_host])
    assert not bool(jax_after.green_asf_this_step[dest_host])
    assert bool(jax_after.host_activity_detected[dest_host])
    assert not bool(jax_after.host_activity_detected[source_host])
    assert float(jax_reward.total) == pytest.approx(cy_reward)


def test_phishing_prefers_same_subnet_source_agent_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()
    _clear_red_sessions(cy_state)

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    target_subnet = int(const.host_subnet[target_host])
    same_subnet_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) == target_subnet
    )
    diff_subnet_host = next(
        h
        for h in range(int(const.num_hosts))
        if h not in {target_host, same_subnet_host}
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) != target_subnet
    )

    low_agent = 0
    high_agent = NUM_RED_AGENTS - 1
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[diff_subnet_host],
            username="user",
            agent=f"red_agent_{low_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[same_subnet_host],
            username="user",
            agent=f"red_agent_{high_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    cy_owner = next(
        r
        for r in range(NUM_RED_AGENTS)
        if any(
            sess.hostname == mappings.idx_to_hostname[target_host]
            for sess in cy_state.sessions[f"red_agent_{r}"].values()
        )
    )

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    for r, h in ((low_agent, diff_subnet_host), (high_agent, same_subnet_host)):
        jax_state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[r, h].set(True),
            red_session_count=jax_state.red_session_count.at[r, h].set(1),
            red_session_is_abstract=jax_state.red_session_is_abstract.at[r, h].set(True),
        )

    jax_after = _run_green_with_tape(jax_state, const, _make_green_tape(host_fields=_phishing_fields(target_host)))
    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after.red_sessions[r, target_host]))

    assert cy_owner == high_agent
    assert jax_owner == cy_owner


def test_phishing_creates_abstract_session_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()
    _clear_red_sessions(cy_state)

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_agent = 0
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[source_host],
            username="user",
            agent=f"red_agent_{source_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    cy_created = [
        sess
        for r in range(NUM_RED_AGENTS)
        for sess in cy_state.sessions[f"red_agent_{r}"].values()
        if sess.hostname == mappings.idx_to_hostname[target_host]
    ]
    assert cy_created
    assert all(isinstance(sess, RedAbstractSession) for sess in cy_created)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[source_agent, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[source_agent, source_host].set(True),
    )
    jax_after = _run_green_with_tape(jax_state, const, _make_green_tape(host_fields=_phishing_fields(target_host)))
    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after.red_sessions[r, target_host]))
    assert bool(jax_after.red_session_is_abstract[jax_owner, target_host])


def test_phishing_ignores_subnet_blocks_for_source_selection_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()
    _clear_red_sessions(cy_state)

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    target_subnet = int(const.host_subnet[target_host])
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) != target_subnet
    )
    source_subnet = int(const.host_subnet[source_host])
    target_hostname = mappings.idx_to_hostname[target_host]
    source_hostname = mappings.idx_to_hostname[source_host]

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=source_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[target_hostname].value, []).append(
        cy_state.hostname_subnet_map[source_hostname].value
    )
    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[source_hostname].value, []).append(
        cy_state.hostname_subnet_map[target_hostname].value
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[target_hostname]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"
    assert any(sess.hostname == target_hostname for sess in cy_state.sessions["red_agent_0"].values())

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    blocked_zones = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    blocked_zones[target_subnet, source_subnet] = True
    blocked_zones[source_subnet, target_subnet] = True
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[0, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[0, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[0, source_host].set(True),
        blocked_zones=jnp.array(blocked_zones),
    )

    jax_after = _run_green_with_tape(jax_state, const, _make_green_tape(host_fields=_phishing_fields(target_host)))
    assert bool(jax_after.red_sessions[0, target_host])


def test_phishing_does_not_reuse_stale_blue_suspicious_pid_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()
    _clear_red_sessions(cy_state)

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) == int(const.host_subnet[target_host])
    )
    target_hostname = mappings.idx_to_hostname[target_host]
    source_hostname = mappings.idx_to_hostname[source_host]

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=source_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    blue_idx = next(b for b in range(const.blue_agent_hosts.shape[0]) if bool(const.blue_agent_hosts[b, target_host]))
    stale_pid = 424242
    cy_blue_parent = cy_state.sessions[f"blue_agent_{blue_idx}"][0]
    cy_blue_parent.add_sus_pids(hostname=target_hostname, pid=stale_pid)

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[target_hostname]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"
    cy_target_sessions = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_target_sessions) == 1
    cy_new_pid = int(cy_target_sessions[0].pid)
    assert cy_new_pid != stale_pid

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[0, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[0, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[0, source_host].set(True),
        red_privilege=jax_state.red_privilege.at[0, source_host].set(COMPROMISE_USER),
        host_compromised=jax_state.host_compromised.at[source_host].set(COMPROMISE_USER),
        blue_suspicious_pids=jax_state.blue_suspicious_pids.at[blue_idx, target_host, 0].set(stale_pid),
    )

    jax_after = _run_green_with_tape(jax_state, const, _make_green_tape(host_fields=_phishing_fields(target_host)))

    jax_target_pids = np.array(jax_after.red_session_pids[0, target_host])
    jax_target_pids = jax_target_pids[jax_target_pids >= 0]
    assert len(jax_target_pids) == 1
    assert int(jax_target_pids[0]) != stale_pid


def test_remove_clears_sessions_from_phishing_and_follow_on_compromise_matches_cyborg():
    cyborg_env, cy_state, const, mappings = _fresh_cyborg_and_const()
    _clear_red_sessions(cy_state)

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    target_hostname = mappings.idx_to_hostname[target_host]
    source_hostname = mappings.idx_to_hostname[source_host]

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=source_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[target_hostname]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    cy_state.add_session(
        Session(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_target_sessions = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_target_sessions) == 2

    blue_idx = next(b for b in range(const.blue_agent_hosts.shape[0]) if bool(const.blue_agent_hosts[b, target_host]))
    cy_blue_parent = cy_state.sessions[f"blue_agent_{blue_idx}"][0]
    for sess in cy_target_sessions:
        cy_blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)
    cy_sus_pids = cy_blue_parent.sus_pids.get(target_hostname, [])
    assert set(cy_sus_pids) == {sess.pid for sess in cy_target_sessions}

    cy_remove = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
    cy_remove.duration = 1
    cy_remove_obs = cy_remove.execute(cy_state)
    assert cy_remove_obs.success
    cy_remaining = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_remaining) == 0

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[0, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[0, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[0, source_host].set(True),
        red_scan_anchor_host=jax_state.red_scan_anchor_host.at[0].set(source_host),
        red_privilege=jax_state.red_privilege.at[0, source_host].set(COMPROMISE_USER),
        host_compromised=jax_state.host_compromised.at[source_host].set(COMPROMISE_USER),
    )

    jax_after_green = _run_green_with_tape(
        jax_state, const, _make_green_tape(host_fields=_phishing_fields(target_host))
    )

    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after_green.red_sessions[r, target_host]))
    phish_pid_row = np.array(jax_after_green.red_session_pids[jax_owner, target_host])
    phish_pid = int(phish_pid_row[phish_pid_row >= 0][0])
    jax_after_green = jax_after_green.replace(
        blue_suspicious_pids=jax_after_green.blue_suspicious_pids.at[blue_idx, target_host, 0].set(phish_pid),
    )

    pre_exploit_pid_row = np.array(jax_after_green.red_session_pids[jax_owner, target_host])
    pre_exploit_pids = {int(pid) for pid in pre_exploit_pid_row if int(pid) >= 0}
    jax_before_remove = apply_exploit_success(
        jax_after_green,
        const,
        jax_owner,
        jnp.int32(target_host),
        jnp.array(True),
        key=jax.random.PRNGKey(1),
    )
    post_exploit_pid_row = np.array(jax_before_remove.red_session_pids[jax_owner, target_host])
    post_exploit_pids = {int(pid) for pid in post_exploit_pid_row if int(pid) >= 0}
    new_exploit_pids = post_exploit_pids - pre_exploit_pids
    assert len(new_exploit_pids) == 1
    follow_on_pid = next(iter(new_exploit_pids))
    jax_before_remove = apply_blue_monitor(jax_before_remove, const, blue_idx)

    jax_sus_pids = np.array(jax_before_remove.blue_suspicious_pids[blue_idx, target_host])
    jax_sus_pids = jax_sus_pids[jax_sus_pids >= 0].tolist()
    assert phish_pid in jax_sus_pids
    assert follow_on_pid in jax_sus_pids

    remove_idx = encode_blue_action("Remove", target_host, blue_idx, const=const)
    jax_after_remove = apply_blue_action(jax_before_remove, const, blue_idx, remove_idx)

    assert int(jax_after_remove.red_session_count[jax_owner, target_host]) == len(cy_remaining)
    assert bool(jax_after_remove.red_sessions[jax_owner, target_host]) == (len(cy_remaining) > 0)
    assert int(jax_after_remove.red_privilege[jax_owner, target_host]) == COMPROMISE_NONE
    assert int(jax_after_remove.host_compromised[target_host]) == COMPROMISE_NONE


def test_vmap_matches_sequential_with_two_phishings_same_step():
    """Two greens in the same subnet both phish in the same step.

    The second phishing's source-agent selection must see the first's newly
    created red session.  The vmap path computes intents in parallel but
    applies phishing inside a sequential ``fori_loop``; this test pins
    sequential ≡ vmap for the load-bearing concurrent-phishing case.
    """
    const = build_topology(jax.random.PRNGKey(42), num_steps=500)
    state = create_initial_state(CC4_CONFIG)
    state = state.replace(
        host_services=jnp.array(const.initial_services),
        host_max_pid=const.host_initial_max_pid,
    )
    state = _init_red_state(const, state)

    active_services_np = np.asarray(state.host_services)
    candidates = [
        int(h) for h in range(GLOBAL_MAX_HOSTS) if bool(const.green_agent_active[h]) and active_services_np[h].any()
    ]
    pair = None
    for i, h1 in enumerate(candidates):
        s1 = int(const.host_subnet[h1])
        for h2 in candidates[i + 1 :]:
            if int(const.host_subnet[h2]) == s1:
                pair = (h1, h2)
                break
        if pair:
            break
    assert pair is not None, "fixture: need two greens in same subnet with active services"
    host_a, host_b = pair
    svc_a = int(np.flatnonzero(active_services_np[host_a])[0])
    svc_b = int(np.flatnonzero(active_services_np[host_b])[0])

    fields = {}
    for h, svc in ((host_a, svc_a), (host_b, svc_b)):
        fields[h] = {
            0: GREEN_LOCAL_WORK,
            1: 0,
            2: 0,
            3: 0.5,
            4: PHISHING_ERROR_RATE * 0.1,
            5: 0,
            6: 0.5,
            7: 4,
        }
        # NB: pin service token to the host's first sorted service via
        # ``set_green_int_range``; but green's apply_local_work uses a
        # subset-aware selector keyed off ``host_services``, so any
        # consistently-pinned token suffices.
        del svc  # noqa: F841

    tape = _make_green_tape(host_fields=fields)

    key = jax.random.PRNGKey(99)
    with indexed_rng_impls(**tape.as_overrides()):
        state_seq = apply_green_agents(state, const, key)
        state_vmap = apply_green_agents_vmapped(state, const, key)

    new_seq = int(jnp.sum(state_seq.red_session_count) - jnp.sum(state.red_session_count))
    new_vmap = int(jnp.sum(state_vmap.red_session_count) - jnp.sum(state.red_session_count))
    assert new_seq >= 2, f"fixture failed to create two sequential sessions (got {new_seq})"
    assert new_vmap >= 2, f"fixture failed to create two vmap sessions (got {new_vmap})"

    diffs = []
    for field in state_seq.__dataclass_fields__:
        a = np.asarray(getattr(state_seq, field))
        b = np.asarray(getattr(state_vmap, field))
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(field)
    assert not diffs, f"two-phishing same-step divergence: {diffs}"
