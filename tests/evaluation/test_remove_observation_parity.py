"""Real Remove outcomes and enhanced observation history must agree across backends."""

import jax
import numpy as np
import pytest
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Shared.Session import RedAbstractSession
from CybORG.Simulator.Actions import Monitor, Remove

from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.blue_ioc import initialize_blue_ioc
from jaxborg.evaluation.enhanced_blue_wrapper import EnhancedBlueFlatWrapper
from jaxborg.observations import get_blue_obs, update_blue_observation_memory
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from jaxborg.state import create_initial_state


@pytest.mark.parametrize("username", ["root", "user"])
def test_real_remove_resets_observed_history_even_when_protected_session_survives(cyborg_env, username):
    # Execute the action directly at its completion boundary, as existing Remove
    # parity tests do. This isolates action/observation semantics from other agents.
    wrapper = EnhancedBlueFlatWrapper(cyborg_env, pad_spaces=True)
    wrapper.reset(seed=42)
    native = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env).replace(cage4_enhanced_obs=True, blue_observation_version=2)
    mappings = build_mappings_from_cyborg(cyborg_env)
    agent = 1
    agent_name = f"blue_agent_{agent}"
    target = next(
        h
        for h in range(int(const.num_hosts))
        if const.blue_agent_hosts[agent, h] and const.host_active[h] and not const.host_is_router[h]
    )
    hostname = next(name for name, index in mappings.hostname_to_idx.items() if index == target)
    slot = int(np.flatnonzero(np.asarray(const.obs_host_map[const.blue_obs_subnets[agent], :16]) == target)[0])
    host = native.hosts[hostname]
    assert not any(s.hostname == hostname for s in native.sessions["red_agent_0"].values())
    attacker = RedAbstractSession(
        ident=None, hostname=hostname, username=username, agent="red_agent_0", parent=0, session_type="shell", pid=None
    )
    native.add_session(attacker)
    assert host.get_process(attacker.pid).user == username
    protected = username == "root"
    privilege = 2 if protected else 1

    state = initialize_blue_ioc(create_initial_state())
    state = state.replace(
        host_services=const.initial_services,
        red_sessions=state.red_sessions.at[0, target].set(True),
        red_session_count=state.red_session_count.at[0, target].set(1),
        red_abstract_session_count=state.red_abstract_session_count.at[0, target].set(1),
        red_session_is_abstract=state.red_session_is_abstract.at[0, target].set(True),
        red_session_pids=state.red_session_pids.at[0, target, 0].set(attacker.pid),
        red_session_abstract_pids=state.red_session_abstract_pids.at[0, target, 0].set(attacker.pid),
        red_session_privileged_pids=state.red_session_privileged_pids.at[0, target, 0].set(
            attacker.pid if protected else -1
        ),
        red_privilege=state.red_privilege.at[0, target].set(privilege),
        red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
        host_compromised=state.host_compromised.at[target].set(privilege),
        host_process_creation_pids=state.host_process_creation_pids.at[target, 0].set(attacker.pid),
    )
    # A real Monitor ingests the actual session PID; no success or memory flags
    # are fabricated. The next Monitor ages the event out while history persists.
    host.events.process_creation.append({"pid": attacker.pid})
    monitor = Monitor(session=0, agent=agent_name)
    for current_alert in (True, False):
        result = monitor.execute(native)
        assert result.success == TernaryEnum.TRUE
        state = apply_blue_monitor(state, const, agent)
        state = update_blue_observation_memory(state, const)
        observed = wrapper.observation_change(agent_name, {**result.get_dict(), "action": monitor})
        assert observed.shape == (450,)
        np.testing.assert_array_equal(observed, get_blue_obs(state, const, agent))
        assert bool(state.old_host_exploit_detected[target]) is current_alert
        assert observed[210:].reshape(5, 48)[2, slot] == 1
    assert attacker.pid in native.sessions[agent_name][0].sus_pids[hostname]
    assert attacker.pid in np.asarray(state.blue_suspicious_pids[agent, target])

    remove = Remove(session=0, agent=agent_name, hostname=hostname)
    result = remove.execute(native)
    assert result.success == TernaryEnum.TRUE
    state = jax.jit(apply_blue_remove, static_argnums=(2,))(state, const, agent, target)
    state = update_blue_observation_memory(state, const)
    observed = wrapper.observation_change(agent_name, {**result.get_dict(), "action": remove})

    # The same real action reports TRUE in both cases, but only user is eradicated.
    survivors = [s for s in native.sessions["red_agent_0"].values() if s.hostname == hostname]
    assert len(survivors) == int(protected)
    assert (host.get_process(attacker.pid) is not None) is protected
    assert bool(state.red_sessions[0, target]) is protected
    assert int(state.red_session_count[0, target]) == len(survivors)
    assert int(state.host_compromised[target]) == (privilege if protected else 0)
    assert bool(state.blue_recovered_this_step[target])
    np.testing.assert_array_equal(observed, get_blue_obs(state, const, agent))
    np.testing.assert_array_equal(observed[210:].reshape(5, 48)[1:, slot], np.zeros(4))
