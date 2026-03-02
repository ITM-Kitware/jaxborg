"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_STEALTH_SCAN,
    decode_red_action,
)
from jaxborg.actions.pids import append_pid_to_row, first_valid_pid
from jaxborg.actions.red_common import select_scan_execution_source_host
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import GLOBAL_MAX_HOSTS, MAX_TRACKED_SUSPICIOUS_PIDS, NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def reassign_cross_subnet_sessions(state: CC4State, const: CC4Const) -> CC4State:
    owner_mask = const.red_agent_subnets
    any_owner = jnp.any(owner_mask, axis=0)
    subnet_owner = jnp.where(any_owner, jnp.argmax(owner_mask, axis=0), -1)

    host_owner = subnet_owner[const.host_subnet]  # (GLOBAL_MAX_HOSTS,)

    session_counts = effective_session_counts(state)
    allowed = const.red_agent_subnets[:, const.host_subnet]  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
    needs_reassign = (session_counts > 0) & ~allowed & const.host_active[None, :]

    red_session_count = session_counts
    red_suspicious_process_count = state.red_suspicious_process_count
    red_privilege = state.red_privilege
    red_session_is_abstract = state.red_session_is_abstract
    red_session_pids = state.red_session_pids
    red_discovered = state.red_discovered_hosts

    for src in range(NUM_RED_AGENTS):
        src_mask = needs_reassign[src]
        src_counts = red_session_count[src]
        src_suspicious = red_suspicious_process_count[src]
        src_privilege = red_privilege[src]
        src_pid_rows = red_session_pids[src]
        src_abstract = red_session_is_abstract[src]

        red_session_count = red_session_count.at[src].set(jnp.where(src_mask, 0, src_counts))
        red_suspicious_process_count = red_suspicious_process_count.at[src].set(jnp.where(src_mask, 0, src_suspicious))
        red_privilege = red_privilege.at[src].set(jnp.where(src_mask, 0, src_privilege))
        red_session_is_abstract = red_session_is_abstract.at[src].set(jnp.where(src_mask, False, src_abstract))
        red_session_pids = red_session_pids.at[src].set(jnp.where(src_mask[:, None], -1, src_pid_rows))

        for dst in range(NUM_RED_AGENTS):
            dst_mask = src_mask & (host_owner == dst)
            moved_counts = jnp.where(dst_mask, src_counts, 0)
            moved_suspicious = jnp.where(dst_mask, src_suspicious, 0)
            moved_privilege = jnp.where(dst_mask, src_privilege, 0)

            red_session_count = red_session_count.at[dst].set(red_session_count[dst] + moved_counts)
            red_suspicious_process_count = red_suspicious_process_count.at[dst].set(
                red_suspicious_process_count[dst] + moved_suspicious
            )
            red_privilege = red_privilege.at[dst].set(jnp.maximum(red_privilege[dst], moved_privilege))
            red_discovered = red_discovered.at[dst].set(jnp.where(dst_mask, True, red_discovered[dst]))
            red_session_is_abstract = red_session_is_abstract.at[dst].set(
                jnp.where(dst_mask, True, red_session_is_abstract[dst])
            )

            dst_rows = red_session_pids[dst]
            for slot in range(MAX_TRACKED_SUSPICIOUS_PIDS):
                incoming_pid = src_pid_rows[:, slot]
                dst_rows = jax.vmap(
                    lambda row, pid, do_assign: jnp.where(
                        do_assign & (pid >= 0),
                        append_pid_to_row(row, pid),
                        row,
                    )
                )(dst_rows, incoming_pid, dst_mask)
            red_session_pids = red_session_pids.at[dst].set(dst_rows)

    red_sessions = red_session_count > 0
    red_session_multiple = red_session_count > 1
    red_session_many = red_session_count > 2
    red_session_pid = jax.vmap(jax.vmap(first_valid_pid))(red_session_pids)
    red_session_pid = jnp.where(red_sessions, red_session_pid, -1)

    # Any host with an active red session must be discoverable by that red agent.
    red_discovered = red_discovered | red_sessions

    host_compromised = state.host_compromised

    # Keep anchor bound only while that specific source host session survives.
    anchor = state.red_scan_anchor_host
    has_any_sessions_now = jnp.any(red_sessions, axis=1)
    anchor_idx = jnp.clip(anchor, 0, red_sessions.shape[1] - 1)
    anchor_had_session = (anchor >= 0) & (session_counts[jnp.arange(NUM_RED_AGENTS), anchor_idx] > 0)
    anchor_has_session = anchor_had_session & red_sessions[jnp.arange(NUM_RED_AGENTS), anchor_idx]
    red_scan_anchor_host = jnp.where(
        has_any_sessions_now,
        jnp.where(anchor_has_session, anchor, -1),
        -1,
    )

    removed_session_hosts = (session_counts > 0) & (red_session_count == 0)
    via_hosts = state.red_scanned_via
    valid_via = via_hosts >= 0
    clipped_via = jnp.clip(via_hosts, 0, GLOBAL_MAX_HOSTS - 1)
    via_removed = removed_session_hosts[jnp.arange(NUM_RED_AGENTS)[:, None], clipped_via] & valid_via
    full_clear = (~has_any_sessions_now)[:, None]
    red_scanned_hosts = state.red_scanned_hosts & ~(full_clear | via_removed)
    red_scanned_via = jnp.where(full_clear | via_removed, -1, state.red_scanned_via)
    host_suspicious_process = jnp.any(red_suspicious_process_count > 0, axis=0)
    red_pending_source_host = state.red_pending_source_host

    source_state = state.replace(
        red_sessions=red_sessions,
        red_session_is_abstract=red_session_is_abstract,
        red_scan_anchor_host=red_scan_anchor_host,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
    )
    for r in range(NUM_RED_AGENTS):
        is_busy = state.red_pending_ticks[r] > 0
        action_type, _, target_host = decode_red_action(state.red_pending_action[r], r, const)
        is_scan_action = (
            (action_type == ACTION_TYPE_SCAN)
            | (action_type == ACTION_TYPE_AGGRESSIVE_SCAN)
            | (action_type == ACTION_TYPE_STEALTH_SCAN)
        )
        source_host = red_pending_source_host[r]
        source_idx = jnp.clip(source_host, 0, GLOBAL_MAX_HOSTS - 1)
        source_valid = (
            (source_host >= 0)
            & red_sessions[r, source_idx]
            & red_session_is_abstract[r, source_idx]
            & const.host_active[source_idx]
        )
        rebound_source = select_scan_execution_source_host(source_state, const, r, target_host)
        can_rebind = is_busy & is_scan_action & ~source_valid
        red_pending_source_host = red_pending_source_host.at[r].set(
            jnp.where(can_rebind, rebound_source, source_host)
        )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_session_pid=red_session_pid,
        red_session_pids=red_session_pids,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
        red_session_is_abstract=red_session_is_abstract,
        red_pending_source_host=red_pending_source_host,
    )
