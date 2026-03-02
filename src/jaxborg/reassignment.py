"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row, first_valid_pid
from jaxborg.actions.red_common import recompute_scan_anchor_hosts, scan_sources_with_fallback, sync_scan_memory_fields
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import MAX_TRACKED_SESSION_PIDS, NUM_RED_AGENTS
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
    red_abstract_host_rank = state.red_abstract_host_rank
    red_session_pids = state.red_session_pids
    red_discovered = state.red_discovered_hosts

    for src in range(NUM_RED_AGENTS):
        src_mask = needs_reassign[src]
        src_counts = red_session_count[src]
        src_suspicious = red_suspicious_process_count[src]
        src_privilege = red_privilege[src]
        src_pid_rows = red_session_pids[src]
        src_abstract = red_session_is_abstract[src]
        src_abstract_rank = red_abstract_host_rank[src]

        red_session_count = red_session_count.at[src].set(jnp.where(src_mask, 0, src_counts))
        red_suspicious_process_count = red_suspicious_process_count.at[src].set(jnp.where(src_mask, 0, src_suspicious))
        red_privilege = red_privilege.at[src].set(jnp.where(src_mask, 0, src_privilege))
        red_session_is_abstract = red_session_is_abstract.at[src].set(jnp.where(src_mask, False, src_abstract))
        red_abstract_host_rank = red_abstract_host_rank.at[src].set(
            jnp.where(src_mask, jnp.int32(1_000_000), src_abstract_rank)
        )
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
            moved_ranks = jnp.where(dst_mask, src_abstract_rank, jnp.int32(1_000_000))
            merged_ranks = jnp.minimum(red_abstract_host_rank[dst], moved_ranks)
            red_abstract_host_rank = red_abstract_host_rank.at[dst].set(merged_ranks)

            dst_rows = red_session_pids[dst]
            slot_indices = jnp.arange(MAX_TRACKED_SESSION_PIDS, dtype=jnp.int32)
            max_slot_by_host = jnp.max(jnp.where(src_pid_rows >= 0, slot_indices[None, :], -1), axis=1) + 1
            slot_limit = jnp.clip(jnp.max(jnp.where(dst_mask, max_slot_by_host, 0)), 0, MAX_TRACKED_SESSION_PIDS)

            def _move_slot(slot, rows):
                incoming_pid = src_pid_rows[:, slot]
                return jax.vmap(
                    lambda row, pid, do_assign: jnp.where(
                        do_assign & (pid >= 0),
                        append_pid_to_row(row, pid),
                        row,
                    )
                )(rows, incoming_pid, dst_mask)

            dst_rows = jax.lax.fori_loop(0, slot_limit, _move_slot, dst_rows)
            red_session_pids = red_session_pids.at[dst].set(dst_rows)

    red_sessions = red_session_count > 0
    red_session_multiple = red_session_count > 1
    red_session_many = red_session_count > 2
    red_session_pid = jax.vmap(jax.vmap(first_valid_pid))(red_session_pids)
    red_session_pid = jnp.where(red_sessions, red_session_pid, -1)

    # Any host with an active red session must be discoverable by that red agent.
    red_discovered = red_discovered | red_sessions

    host_compromised = state.host_compromised

    has_any_sessions_now = jnp.any(red_sessions, axis=1)
    red_scan_anchor_host = recompute_scan_anchor_hosts(
        state.red_scan_anchor_host,
        red_sessions,
        red_session_is_abstract,
        const.host_active,
    )
    full_clear = (~has_any_sessions_now)[:, None]
    scan_sources = scan_sources_with_fallback(state)
    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=red_sessions,
            red_session_is_abstract=red_session_is_abstract,
        ),
        const,
        scan_sources=scan_sources,
    )
    red_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    red_scanned_via = jnp.where(full_clear, -1, scan_synced.red_scanned_via)
    red_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)
    host_suspicious_process = jnp.any(red_suspicious_process_count > 0, axis=0)

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
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_pending_source_host=state.red_pending_source_host,
    )
