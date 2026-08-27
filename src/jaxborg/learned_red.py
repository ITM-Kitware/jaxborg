"""JAX observation, masking, and action translation for learned Red policies.

``ScenarioEnv`` deliberately retains the raw 2,202-action Red ABI used by the
parity/replay tooling.  This module defines the smaller policy-facing contract
and is consumed by :class:`jaxborg.joint_env.JointPolicyCC4Env` at its edge.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.action_defs import (
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_START,
    RED_DISCOVER_START,
    RED_IMPACT_START,
    RED_PRIVESC_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_START,
)
from jaxborg.actions.red_policy import (
    RED_POLICY_ACTION_DIM,
    RED_POLICY_AGGRESSIVE_SCAN_END,
    RED_POLICY_AGGRESSIVE_SCAN_START,
    RED_POLICY_DEGRADE_END,
    RED_POLICY_DEGRADE_START,
    RED_POLICY_DISCOVER_DECEPTION_END,
    RED_POLICY_DISCOVER_DECEPTION_START,
    RED_POLICY_DISCOVER_END,
    RED_POLICY_DISCOVER_START,
    RED_POLICY_EXPLOIT_END,
    RED_POLICY_EXPLOIT_START,
    RED_POLICY_IMPACT_END,
    RED_POLICY_IMPACT_START,
    RED_POLICY_PRIVESC_END,
    RED_POLICY_PRIVESC_START,
    RED_POLICY_SLEEP,
    RED_POLICY_STEALTH_SCAN_END,
    RED_POLICY_STEALTH_SCAN_START,
    RED_POLICY_WITHDRAW_END,
    RED_POLICY_WITHDRAW_START,
)
from jaxborg.constants import (
    COMPROMISE_PRIVILEGED,
    GLOBAL_MAX_HOSTS,
    MISSION_PHASES,
    NUM_RED_AGENTS,
    RED_OBS_SIZE,
)
from jaxborg.scenarios.cc4.red_fsm import _pick_exploit_action
from jaxborg.state import SimulatorConst, SimulatorState


def _red_policy_discovered_hosts(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
) -> chex.Array:
    """Hosts that have actually entered this agent's local knowledge.

    ``ScenarioEnv`` retains CybORG's pre-seeded start-host action-space bit for
    inactive Red agents so raw translated-action replay stays faithful.  That
    bit is not an observation processed by ``FiniteStateRedAgent``.  The
    ``fsm_host_entered`` field tracks hosts that were genuinely observed or
    acquired through a session for every Red execution mode, so intersecting
    it here prevents the learned policy from gaining the pre-seeded host when
    the agent later activates.
    """

    return state.red_discovered_hosts[agent_id] & state.fsm_host_entered[agent_id] & const.host_active


def red_primary_session_host(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
) -> tuple[chex.Array, chex.Array]:
    """Return ``(safe_host_index, is_live)`` for the agent's session-0 equivalent."""

    primary_host = state.red_scan_anchor_host[agent_id]
    safe_host = jnp.clip(primary_host, 0, GLOBAL_MAX_HOSTS - 1)
    is_live = (
        (primary_host >= 0)
        & (primary_host < GLOBAL_MAX_HOSTS)
        & (state.red_primary_pid[agent_id] >= 0)
        & state.red_sessions[agent_id, safe_host]
        & const.host_active[safe_host]
    )
    return safe_host, is_live


def red_primary_scanned_hosts(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
) -> chex.Array:
    """Hosts whose scan knowledge belongs to the current primary session.

    Scan memory is keyed by both source host and owning PID.  The PID check is
    important when session 0 is removed/promoted while another session remains
    on the same host: knowledge owned by the old session must not leak into the
    learned policy's observation or exploit mask.
    """

    source_host, primary_live = red_primary_session_host(state, const, agent_id)
    owner_pid = state.red_scan_source_pid[agent_id, source_host]
    owned_by_primary = (owner_pid >= 0) & (owner_pid == state.red_primary_pid[agent_id])
    return (
        state.red_scanned_source_hosts[agent_id, :, source_host] & const.host_active & primary_live & owned_by_primary
    )


def get_red_policy_obs(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
) -> chex.Array:
    """Encode one Red agent's fixed-size, strictly local 706-value view.

    Layout: phase one-hot, normalized time, active, busy, agent one-hot,
    allowed-subnet mask, then five 137-host planes (discovered, scanned by the
    current primary session, own session, own privileged session, primary host).
    Global services, decoys, traffic blocks, topology, and other agents' session
    state are intentionally absent.
    """

    phase = jax.nn.one_hot(state.mission_phase, MISSION_PHASES, dtype=jnp.float32)
    time = jnp.asarray(state.time, dtype=jnp.float32)
    max_steps = jnp.maximum(jnp.asarray(const.max_steps, dtype=jnp.float32), 1.0)
    normalized_time = jnp.clip(time / max_steps, 0.0, 1.0)[None]
    agent_active = state.red_agent_active[agent_id]
    active = agent_active.astype(jnp.float32)[None]
    busy = (state.red_pending_ticks[agent_id] > 0).astype(jnp.float32)[None]
    identity = jax.nn.one_hot(agent_id, NUM_RED_AGENTS, dtype=jnp.float32)
    allowed_subnets = const.red_agent_subnets[agent_id].astype(jnp.float32)

    discovered = (_red_policy_discovered_hosts(state, const, agent_id) & agent_active).astype(jnp.float32)
    primary_scanned = (red_primary_scanned_hosts(state, const, agent_id) & agent_active).astype(jnp.float32)
    own_session = (state.red_sessions[agent_id] & const.host_active & agent_active).astype(jnp.float32)
    own_privileged = (
        state.red_sessions[agent_id]
        & (state.red_privilege[agent_id] >= COMPROMISE_PRIVILEGED)
        & const.host_active
        & agent_active
    ).astype(jnp.float32)
    primary_host, primary_live = red_primary_session_host(state, const, agent_id)
    primary_host_plane = (
        jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.float32)
        .at[primary_host]
        .set((primary_live & agent_active).astype(jnp.float32))
    )

    obs = jnp.concatenate(
        [
            phase,
            normalized_time,
            active,
            busy,
            identity,
            allowed_subnets,
            discovered,
            primary_scanned,
            own_session,
            own_privileged,
            primary_host_plane,
        ]
    )
    if obs.shape != (RED_OBS_SIZE,):  # static invariant, checked while tracing
        raise ValueError(f"learned-Red observation has shape {obs.shape}, expected {(RED_OBS_SIZE,)}")
    return obs


def compute_red_policy_action_mask(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
) -> chex.Array:
    """Return the state-aware compact action mask for one Red agent."""

    discovered = _red_policy_discovered_hosts(state, const, agent_id)
    primary_scanned = red_primary_scanned_hosts(state, const, agent_id)
    own_session = state.red_sessions[agent_id] & const.host_active
    own_privileged = own_session & (state.red_privilege[agent_id] >= COMPROMISE_PRIVILEGED)

    mask = jnp.concatenate(
        [
            jnp.ones(1, dtype=jnp.bool_),  # Sleep
            const.red_agent_subnets[agent_id] & jnp.ones_like(const.red_agent_subnets[agent_id], dtype=jnp.bool_),
            discovered,  # aggressive scan
            discovered,  # stealth scan
            discovered,  # discover deception
            primary_scanned,  # generic exploit
            own_session,  # privilege escalation
            own_privileged,  # impact
            own_privileged,  # degrade
            own_session,  # withdraw
        ]
    )
    available = state.red_agent_active[agent_id] & (state.red_pending_ticks[agent_id] <= 0)
    sleep_only = jnp.zeros(RED_POLICY_ACTION_DIM, dtype=jnp.bool_).at[RED_POLICY_SLEEP].set(True)
    return jnp.where(available, mask, sleep_only)


def compact_red_action_to_raw(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    action_idx: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
    """Translate a learned-Red action to the unchanged raw simulator ABI.

    Generic Exploit is resolved inside this adapter from the agent's remembered
    scan-time ports.  The policy only sees whether its primary session scanned
    the host; it never receives the port values themselves.  Inactive/busy
    agents and out-of-range indices are forced to raw Sleep.
    """

    action = jnp.asarray(action_idx, dtype=jnp.int32)
    in_range = (action >= 0) & (action < RED_POLICY_ACTION_DIM)
    available = state.red_agent_active[agent_id] & (state.red_pending_ticks[agent_id] <= 0)

    raw_action = jnp.int32(RED_SLEEP)
    is_discover = (action >= RED_POLICY_DISCOVER_START) & (action < RED_POLICY_DISCOVER_END)
    raw_action = jnp.where(is_discover, RED_DISCOVER_START + action - RED_POLICY_DISCOVER_START, raw_action)

    deterministic_blocks = (
        (RED_POLICY_AGGRESSIVE_SCAN_START, RED_POLICY_AGGRESSIVE_SCAN_END, RED_AGGRESSIVE_SCAN_START),
        (RED_POLICY_STEALTH_SCAN_START, RED_POLICY_STEALTH_SCAN_END, RED_STEALTH_SCAN_START),
        (
            RED_POLICY_DISCOVER_DECEPTION_START,
            RED_POLICY_DISCOVER_DECEPTION_END,
            RED_DISCOVER_DECEPTION_START,
        ),
        (RED_POLICY_PRIVESC_START, RED_POLICY_PRIVESC_END, RED_PRIVESC_START),
        (RED_POLICY_IMPACT_START, RED_POLICY_IMPACT_END, RED_IMPACT_START),
        (RED_POLICY_DEGRADE_START, RED_POLICY_DEGRADE_END, RED_DEGRADE_START),
        (RED_POLICY_WITHDRAW_START, RED_POLICY_WITHDRAW_END, RED_WITHDRAW_START),
    )
    for compact_start, compact_end, raw_start in deterministic_blocks:
        in_block = (action >= compact_start) & (action < compact_end)
        raw_action = jnp.where(in_block, raw_start + action - compact_start, raw_action)

    is_exploit = (action >= RED_POLICY_EXPLOIT_START) & (action < RED_POLICY_EXPLOIT_END)
    exploit_host = jnp.clip(action - RED_POLICY_EXPLOIT_START, 0, GLOBAL_MAX_HOSTS - 1)
    concrete_exploit = _pick_exploit_action(state, agent_id, exploit_host, key)
    raw_action = jnp.where(is_exploit, concrete_exploit, raw_action)

    return jnp.where(in_range & available, raw_action, jnp.int32(RED_SLEEP))


__all__ = [
    "RED_OBS_SIZE",
    "RED_POLICY_ACTION_DIM",
    "compact_red_action_to_raw",
    "compute_red_policy_action_mask",
    "get_red_policy_obs",
    "red_primary_scanned_hosts",
    "red_primary_session_host",
]
