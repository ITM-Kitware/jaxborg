from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.action_defs import (
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_START,
    RED_DISCOVER_START,
    RED_EXPLOIT_HTTP_START,
    RED_EXPLOIT_SSH_START,
    RED_IMPACT_START,
    RED_PRIVESC_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_END,
    RED_WITHDRAW_START,
)
from jaxborg.actions.red_common import sync_scan_memory_fields
from jaxborg.actions.red_policy import (
    RED_POLICY_ACTION_DIM,
    RED_POLICY_AGGRESSIVE_SCAN_START,
    RED_POLICY_DEGRADE_START,
    RED_POLICY_DISCOVER_DECEPTION_START,
    RED_POLICY_DISCOVER_START,
    RED_POLICY_EXPLOIT_START,
    RED_POLICY_IMPACT_START,
    RED_POLICY_PRIVESC_START,
    RED_POLICY_SLEEP,
    RED_POLICY_STEALTH_SCAN_START,
    RED_POLICY_WITHDRAW_END,
    RED_POLICY_WITHDRAW_START,
    RedPolicyActionType,
    decode_red_policy_action,
)
from jaxborg.actions.red_scan_unified import apply_scan_unified
from jaxborg.constants import (
    BLUE_OBS_SIZE,
    COMPROMISE_PRIVILEGED,
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    RED_OBS_SIZE,
    RED_SCANNED_PORT_IDS,
    SERVICE_IDS,
)
from jaxborg.joint_env import JointPolicyCC4Env, force_sleep_for_unavailable_actions
from jaxborg.learned_red import (
    compact_red_action_to_raw,
    compute_red_policy_action_mask,
    get_red_policy_obs,
)

_META_SIZE = 3 + 1 + 1 + 1 + NUM_RED_AGENTS + NUM_SUBNETS
_DISCOVERED = slice(_META_SIZE, _META_SIZE + GLOBAL_MAX_HOSTS)
_SCANNED = slice(_DISCOVERED.stop, _DISCOVERED.stop + GLOBAL_MAX_HOSTS)
_SESSIONS = slice(_SCANNED.stop, _SCANNED.stop + GLOBAL_MAX_HOSTS)
_PRIVILEGED = slice(_SESSIONS.stop, _SESSIONS.stop + GLOBAL_MAX_HOSTS)
_PRIMARY = slice(_PRIVILEGED.stop, _PRIVILEGED.stop + GLOBAL_MAX_HOSTS)


@pytest.fixture(scope="module")
def joint_reset():
    env = JointPolicyCC4Env(num_steps=20)
    obs, state = env.reset(jax.random.PRNGKey(17))
    return env, obs, state


def _active_hosts(const) -> np.ndarray:
    return np.flatnonzero(np.asarray(const.host_active))


def test_policy_dimensions_and_raw_abi_are_distinct(joint_reset):
    env, obs, _ = joint_reset
    assert RED_OBS_SIZE == 706
    assert RED_POLICY_ACTION_DIM == 1106
    assert RED_POLICY_WITHDRAW_END == 1106
    assert RED_WITHDRAW_END == 2202
    assert env.observation_space("blue_0").shape == (BLUE_OBS_SIZE,)
    assert env.observation_space("red_0").shape == (RED_OBS_SIZE,)
    assert env.action_space("red_0").n == RED_POLICY_ACTION_DIM
    assert obs["red_0"].shape == (RED_OBS_SIZE,)


def test_compact_schema_decode():
    assert decode_red_policy_action(RED_POLICY_SLEEP) == (RedPolicyActionType.SLEEP, -1)
    assert decode_red_policy_action(RED_POLICY_DISCOVER_START + 4) == (RedPolicyActionType.DISCOVER, 4)
    assert decode_red_policy_action(RED_POLICY_EXPLOIT_START + 23) == (RedPolicyActionType.EXPLOIT, 23)
    assert decode_red_policy_action(RED_POLICY_WITHDRAW_START + 136) == (RedPolicyActionType.WITHDRAW, 136)
    with pytest.raises(ValueError, match="out of range"):
        decode_red_policy_action(RED_POLICY_ACTION_DIM)


def test_observation_layout_has_local_metadata(joint_reset):
    _, obs, env_state = joint_reset
    red0 = np.asarray(obs["red_0"])
    red1 = np.asarray(obs["red_1"])

    np.testing.assert_array_equal(red0[:3], np.array([1.0, 0.0, 0.0]))
    assert red0[3] == 0.0  # normalized time
    assert red0[4] == 1.0  # red_0 is initially active
    assert red1[4] == 0.0
    assert red0[5] == 0.0  # not busy
    assert red1[_DISCOVERED].sum() == 0.0
    assert red1[_SCANNED].sum() == 0.0
    assert red1[_SESSIONS].sum() == 0.0
    assert red1[_PRIVILEGED].sum() == 0.0
    assert red1[_PRIMARY].sum() == 0.0
    np.testing.assert_array_equal(red0[6 : 6 + NUM_RED_AGENTS], np.eye(NUM_RED_AGENTS)[0])
    np.testing.assert_array_equal(
        red0[6 + NUM_RED_AGENTS : _META_SIZE],
        np.asarray(env_state.const.red_agent_subnets[0], dtype=np.float32),
    )
    assert _PRIMARY.stop == RED_OBS_SIZE


def test_preseeded_future_start_host_requires_actual_observation(joint_reset):
    _, _, env_state = joint_reset
    agent_id = 1
    start_host = int(env_state.const.red_start_hosts[agent_id])
    state = env_state.state.replace(
        red_agent_active=env_state.state.red_agent_active.at[agent_id].set(True),
        red_discovered_hosts=env_state.state.red_discovered_hosts.at[agent_id, start_host].set(True),
        fsm_host_entered=env_state.state.fsm_host_entered.at[agent_id, start_host].set(False),
    )

    obs = get_red_policy_obs(state, env_state.const, agent_id)
    mask = compute_red_policy_action_mask(state, env_state.const, agent_id)
    assert obs[_DISCOVERED.start + start_host] == 0.0
    assert not mask[RED_POLICY_AGGRESSIVE_SCAN_START + start_host]

    observed = state.replace(fsm_host_entered=state.fsm_host_entered.at[agent_id, start_host].set(True))
    observed_obs = get_red_policy_obs(observed, env_state.const, agent_id)
    observed_mask = compute_red_policy_action_mask(observed, env_state.const, agent_id)
    assert observed_obs[_DISCOVERED.start + start_host] == 1.0
    assert observed_mask[RED_POLICY_AGGRESSIVE_SCAN_START + start_host]


def test_observation_does_not_leak_hidden_or_other_agent_state(joint_reset):
    _, _, env_state = joint_reset
    state = env_state.state
    const = env_state.const
    before = get_red_policy_obs(state, const, 0)

    hidden_state = state.replace(
        host_services=~state.host_services,
        host_decoys=~state.host_decoys,
        blocked_zones=~state.blocked_zones,
        host_compromised=jnp.full_like(state.host_compromised, COMPROMISE_PRIVILEGED),
        fsm_host_states=state.fsm_host_states.at[0].add(7),
        red_sessions=state.red_sessions.at[1].set(~state.red_sessions[1]),
        red_privilege=state.red_privilege.at[1].set(COMPROMISE_PRIVILEGED),
    )
    hidden_const = const.replace(
        initial_services=~const.initial_services,
        data_links=~const.data_links,
        comms_policy=~const.comms_policy,
    )
    after = get_red_policy_obs(hidden_state, hidden_const, 0)
    np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

    candidates = _active_hosts(const)[~np.asarray(state.red_discovered_hosts[0])[_active_hosts(const)]]
    if candidates.size:
        target = int(candidates[0])
        local_change = state.replace(
            red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
            fsm_host_entered=state.fsm_host_entered.at[0, target].set(True),
        )
        changed = get_red_policy_obs(local_change, const, 0)
        assert changed[_DISCOVERED.start + target] == 1.0
        assert not jnp.array_equal(changed, before)


def test_primary_scan_plane_and_exploit_mask_are_pid_scoped(joint_reset):
    _, _, env_state = joint_reset
    state = env_state.state
    const = env_state.const
    active_hosts = _active_hosts(const)
    source, target = map(int, active_hosts[:2])
    primary_pid = 9123
    scoped = state.replace(
        red_agent_active=state.red_agent_active.at[0].set(True),
        red_sessions=state.red_sessions.at[0, source].set(True),
        red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(source),
        red_primary_pid=state.red_primary_pid.at[0].set(primary_pid),
        red_scan_source_pid=state.red_scan_source_pid.at[0, source].set(primary_pid),
        red_scanned_source_hosts=state.red_scanned_source_hosts.at[0, target, source].set(True),
    )

    obs = get_red_policy_obs(scoped, const, 0)
    mask = compute_red_policy_action_mask(scoped, const, 0)
    assert obs[_SCANNED.start + target] == 1.0
    assert obs[_PRIMARY.start + source] == 1.0
    assert mask[RED_POLICY_EXPLOIT_START + target]

    stale = scoped.replace(red_scan_source_pid=scoped.red_scan_source_pid.at[0, source].set(primary_pid + 1))
    stale_obs = get_red_policy_obs(stale, const, 0)
    stale_mask = compute_red_policy_action_mask(stale, const, 0)
    assert stale_obs[_SCANNED.start + target] == 0.0
    assert not stale_mask[RED_POLICY_EXPLOIT_START + target]


def test_mask_encodes_knowledge_sessions_privilege_and_availability(joint_reset):
    _, _, env_state = joint_reset
    state = env_state.state
    const = env_state.const
    target, privileged_target = map(int, _active_hosts(const)[:2])
    clean = state.replace(
        red_agent_active=state.red_agent_active.at[0].set(True),
        red_pending_ticks=state.red_pending_ticks.at[0].set(0),
        red_discovered_hosts=state.red_discovered_hosts.at[0].set(False).at[0, target].set(True),
        fsm_host_entered=(
            state.fsm_host_entered.at[0].set(False).at[0, target].set(True).at[0, privileged_target].set(True)
        ),
        red_sessions=state.red_sessions.at[0].set(False).at[0, target].set(True).at[0, privileged_target].set(True),
        red_privilege=(state.red_privilege.at[0].set(0).at[0, privileged_target].set(COMPROMISE_PRIVILEGED)),
        red_scanned_source_hosts=state.red_scanned_source_hosts.at[0].set(False),
        red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(-1),
        red_primary_pid=state.red_primary_pid.at[0].set(-1),
    )
    mask = compute_red_policy_action_mask(clean, const, 0)
    np.testing.assert_array_equal(
        np.asarray(mask[RED_POLICY_DISCOVER_START : RED_POLICY_DISCOVER_START + NUM_SUBNETS]),
        np.asarray(const.red_agent_subnets[0]),
    )
    assert mask[RED_POLICY_AGGRESSIVE_SCAN_START + target]
    assert mask[RED_POLICY_STEALTH_SCAN_START + target]
    assert mask[RED_POLICY_DISCOVER_DECEPTION_START + target]
    assert not mask[RED_POLICY_EXPLOIT_START + target]
    assert mask[RED_POLICY_PRIVESC_START + target]
    assert mask[RED_POLICY_WITHDRAW_START + target]
    assert not mask[RED_POLICY_IMPACT_START + target]
    assert not mask[RED_POLICY_DEGRADE_START + target]
    assert mask[RED_POLICY_IMPACT_START + privileged_target]
    assert mask[RED_POLICY_DEGRADE_START + privileged_target]

    for unavailable in (
        clean.replace(red_agent_active=clean.red_agent_active.at[0].set(False)),
        clean.replace(red_pending_ticks=clean.red_pending_ticks.at[0].set(2)),
    ):
        unavailable_mask = np.asarray(compute_red_policy_action_mask(unavailable, const, 0))
        assert unavailable_mask.sum() == 1
        assert unavailable_mask[RED_POLICY_SLEEP]


def test_compact_to_raw_mapping_and_generic_exploit(joint_reset):
    _, _, env_state = joint_reset
    state = env_state.state
    const = env_state.const
    target = int(_active_hosts(const)[0])
    controlled = state.replace(
        red_agent_active=state.red_agent_active.at[0].set(True),
        red_pending_ticks=state.red_pending_ticks.at[0].set(0),
        host_services=(state.host_services.at[target].set(False).at[target, SERVICE_IDS["SSHD"]].set(True)),
        red_scanned_ports=state.red_scanned_ports.at[0, target, RED_SCANNED_PORT_IDS[22]].set(True),
    )
    key = jax.random.PRNGKey(9)
    cases = (
        (RED_POLICY_DISCOVER_START + 3, RED_DISCOVER_START + 3),
        (RED_POLICY_AGGRESSIVE_SCAN_START + target, RED_AGGRESSIVE_SCAN_START + target),
        (RED_POLICY_STEALTH_SCAN_START + target, RED_STEALTH_SCAN_START + target),
        (RED_POLICY_DISCOVER_DECEPTION_START + target, RED_DISCOVER_DECEPTION_START + target),
        (RED_POLICY_EXPLOIT_START + target, RED_EXPLOIT_SSH_START + target),
        (RED_POLICY_PRIVESC_START + target, RED_PRIVESC_START + target),
        (RED_POLICY_IMPACT_START + target, RED_IMPACT_START + target),
        (RED_POLICY_DEGRADE_START + target, RED_DEGRADE_START + target),
        (RED_POLICY_WITHDRAW_START + target, RED_WITHDRAW_START + target),
    )
    for compact, raw in cases:
        assert int(compact_red_action_to_raw(controlled, const, 0, compact, key)) == raw

    busy = controlled.replace(red_pending_ticks=controlled.red_pending_ticks.at[0].set(1))
    assert int(compact_red_action_to_raw(busy, const, 0, RED_POLICY_EXPLOIT_START + target, key)) == RED_SLEEP
    assert int(compact_red_action_to_raw(controlled, const, 0, RED_POLICY_ACTION_DIM, key)) == RED_SLEEP


def test_learned_exploit_uses_scan_snapshot_not_live_services(joint_reset):
    _, _, env_state = joint_reset
    state = env_state.state
    const = env_state.const
    source = int(state.red_scan_anchor_host[0])
    target = next(int(h) for h in _active_hosts(const) if int(h) != source)

    scan_state = state.replace(
        red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
        blocked_zones=jnp.zeros_like(state.blocked_zones),
        host_services=state.host_services.at[target].set(False).at[target, SERVICE_IDS["SSHD"]].set(True),
        host_decoys=state.host_decoys.at[target].set(False),
    )
    scanned = apply_scan_unified(
        scan_state,
        const,
        0,
        jnp.int32(target),
        jax.random.PRNGKey(21),
        jnp.bool_(False),
        jnp.float32(0.0),
    )
    assert scanned.red_scanned_ports[0, target, RED_SCANNED_PORT_IDS[22]]

    # Changing the real network without another scan must not change the
    # generic exploit chosen for learned Red.
    changed_live_state = scanned.replace(
        host_services=scanned.host_services.at[target].set(False).at[target, SERVICE_IDS["APACHE2"]].set(True),
        host_decoys=scanned.host_decoys.at[target, DECOY_IDS["Tomcat"]].set(True),
    )
    compact = RED_POLICY_EXPLOIT_START + target
    assert int(compact_red_action_to_raw(changed_live_state, const, 0, compact, jax.random.PRNGKey(22))) == (
        RED_EXPLOIT_SSH_START + target
    )

    # A re-scan overwrites the snapshot, including ports exposed by decoys.
    decoy_only = changed_live_state.replace(
        host_services=changed_live_state.host_services.at[target].set(False),
        host_decoys=changed_live_state.host_decoys.at[target].set(False).at[target, DECOY_IDS["Apache"]].set(True),
    )
    rescanned = apply_scan_unified(
        decoy_only,
        const,
        0,
        jnp.int32(target),
        jax.random.PRNGKey(23),
        jnp.bool_(False),
        jnp.float32(0.0),
    )
    assert rescanned.red_scanned_ports[0, target, RED_SCANNED_PORT_IDS[80]]
    assert int(compact_red_action_to_raw(rescanned, const, 0, compact, jax.random.PRNGKey(24))) == (
        RED_EXPLOIT_HTTP_START + target
    )

    # Port knowledge disappears with the abstract session that owns the scan.
    owner = int(rescanned.red_scan_anchor_host[0])
    without_owner = rescanned.replace(
        red_sessions=rescanned.red_sessions.at[0, owner].set(False),
        red_session_is_abstract=rescanned.red_session_is_abstract.at[0, owner].set(False),
    )
    cleared = sync_scan_memory_fields(without_owner, const)
    assert not jnp.any(cleared.red_scanned_ports[0, target])


def test_joint_force_sleep_handles_blue_busy_and_red_inactive(joint_reset):
    env, _, env_state = joint_reset
    sim = env_state.state.replace(
        blue_pending_ticks=env_state.state.blue_pending_ticks.at[0].set(2),
        red_agent_active=env_state.state.red_agent_active.at[0].set(True).at[1].set(False),
        red_pending_ticks=env_state.state.red_pending_ticks.at[0].set(2),
    )
    state = env_state.replace(state=sim)
    actions = {agent: jnp.int32(2) for agent in env.agents}
    forced = force_sleep_for_unavailable_actions(state, actions)
    assert int(forced["blue_0"]) == 0
    assert int(forced["red_0"]) == RED_POLICY_SLEEP
    assert int(forced["red_1"]) == RED_POLICY_SLEEP
    assert int(forced["blue_1"]) == 2


def test_joint_submits_all_eleven_actions_in_one_inner_step(joint_reset, monkeypatch):
    env, _, env_state = joint_reset
    ready = env_state.state.replace(
        blue_pending_ticks=jnp.zeros_like(env_state.state.blue_pending_ticks),
        red_pending_ticks=jnp.zeros_like(env_state.state.red_pending_ticks),
        red_agent_active=jnp.ones_like(env_state.state.red_agent_active),
    )
    env_state = env_state.replace(state=ready)
    actions = {agent: jnp.int32(i) for i, agent in enumerate(env.blue_agents)}
    actions.update(
        {agent: jnp.int32(RED_POLICY_DISCOVER_START + (i % NUM_SUBNETS)) for i, agent in enumerate(env.red_agents)}
    )
    calls = []

    def record_step(key, state, raw_actions, red_creation_visible_sessions_override=None):
        del key
        calls.append((raw_actions, red_creation_visible_sessions_override))
        rewards = {agent: jnp.float32(0.0) for agent in env.agents}
        dones = {agent: jnp.bool_(False) for agent in env.agents}
        dones["__all__"] = jnp.bool_(False)
        return {}, state, rewards, dones, {}

    monkeypatch.setattr(env._env, "step_env", record_step)
    with jax.disable_jit():
        env.step_env(jax.random.PRNGKey(31), env_state, actions)

    assert len(calls) == 1
    raw_actions, visible_sessions = calls[0]
    assert set(raw_actions) == set(env.agents)
    for i, agent in enumerate(env.blue_agents):
        assert int(raw_actions[agent]) == i
    for i, agent in enumerate(env.red_agents):
        assert int(raw_actions[agent]) == RED_DISCOVER_START + (i % NUM_SUBNETS)
    np.testing.assert_array_equal(np.asarray(visible_sessions), np.ones(NUM_RED_AGENTS, dtype=np.int32))


def test_joint_step_has_single_game_tick_and_zero_sum_team_rewards(joint_reset):
    env, _, env_state = joint_reset
    actions = {agent: jnp.int32(0) for agent in env.agents}
    with jax.disable_jit():
        obs, next_state, rewards, dones, _ = env.step_env(jax.random.PRNGKey(81), env_state, actions)

    assert int(next_state.state.time) == int(env_state.state.time) + 1
    assert set(obs) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert set(dones) == set(env.agents) | {"__all__"}
    for blue in env.blue_agents:
        assert float(rewards[blue]) == float(rewards["blue_0"])
    for red in env.red_agents:
        assert float(rewards[red]) == -float(rewards["blue_0"])
