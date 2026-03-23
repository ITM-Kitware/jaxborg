"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row, first_valid_pid
from jaxborg.actions.red_common import recompute_scan_anchor_hosts, scan_sources, sync_scan_memory_fields
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.agents.fsm_red import FSM_R, FSM_RD, FSM_U, FSM_UD
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    MAX_TRACKED_SESSION_PIDS,
    NUM_RED_AGENTS,
)
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
    red_session_abstract_pids = state.red_session_abstract_pids
    red_session_privileged_pids = state.red_session_privileged_pids
    red_discovered = state.red_discovered_hosts
    reassigned_hosts = jnp.zeros_like(state.red_sessions, dtype=jnp.bool_)

    for src in range(NUM_RED_AGENTS):
        src_mask = needs_reassign[src]
        src_counts = red_session_count[src]
        src_suspicious = red_suspicious_process_count[src]
        src_privilege = red_privilege[src]
        src_pid_rows = red_session_pids[src]
        src_priv_pid_rows = red_session_privileged_pids[src]
        src_abstract = red_session_is_abstract[src]
        src_abstract_rank = red_abstract_host_rank[src]

        red_session_count = red_session_count.at[src].set(jnp.where(src_mask, 0, src_counts))
        red_suspicious_process_count = red_suspicious_process_count.at[src].set(jnp.where(src_mask, 0, src_suspicious))
        red_privilege = red_privilege.at[src].set(jnp.where(src_mask, 0, src_privilege))
        red_session_is_abstract = red_session_is_abstract.at[src].set(jnp.where(src_mask, False, src_abstract))
        red_abstract_host_rank = red_abstract_host_rank.at[src].set(
            jnp.where(src_mask, jnp.int32(ABSTRACT_RANK_NONE), src_abstract_rank)
        )
        red_session_pids = red_session_pids.at[src].set(jnp.where(src_mask[:, None], -1, src_pid_rows))
        red_session_abstract_pids = red_session_abstract_pids.at[src].set(
            jnp.where(src_mask[:, None], -1, red_session_abstract_pids[src])
        )
        red_session_privileged_pids = red_session_privileged_pids.at[src].set(
            jnp.where(src_mask[:, None], -1, src_priv_pid_rows)
        )

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
            reassigned_hosts = reassigned_hosts.at[dst].set(reassigned_hosts[dst] | dst_mask)
            red_session_is_abstract = red_session_is_abstract.at[dst].set(red_session_is_abstract[dst] | dst_mask)
            moved_ranks = jnp.where(dst_mask, src_abstract_rank, jnp.int32(ABSTRACT_RANK_NONE))
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

            dst_abstract_rows = red_session_abstract_pids[dst]

            def _move_abstract_slot(slot, rows):
                # CybORG converts ALL reassigned sessions to RedAbstractSession,
                # so copy from src_pid_rows (all PIDs), not just abstract ones.
                incoming_pid = src_pid_rows[:, slot]
                return jax.vmap(
                    lambda row, pid, do_assign: jnp.where(
                        do_assign & (pid >= 0),
                        append_pid_to_row(row, pid),
                        row,
                    )
                )(rows, incoming_pid, dst_mask)

            dst_abstract_rows = jax.lax.fori_loop(0, slot_limit, _move_abstract_slot, dst_abstract_rows)
            red_session_abstract_pids = red_session_abstract_pids.at[dst].set(dst_abstract_rows)

            dst_priv_rows = red_session_privileged_pids[dst]
            priv_max_slot_by_host = jnp.max(jnp.where(src_priv_pid_rows >= 0, slot_indices[None, :], -1), axis=1) + 1
            priv_slot_limit = jnp.clip(
                jnp.max(jnp.where(dst_mask, priv_max_slot_by_host, 0)),
                0,
                MAX_TRACKED_SESSION_PIDS,
            )

            def _move_priv_slot(slot, rows):
                incoming_pid = src_priv_pid_rows[:, slot]
                return jax.vmap(
                    lambda row, pid, do_assign: jnp.where(
                        do_assign & (pid >= 0),
                        append_pid_to_row(row, pid),
                        row,
                    )
                )(rows, incoming_pid, dst_mask)

            dst_priv_rows = jax.lax.fori_loop(0, priv_slot_limit, _move_priv_slot, dst_priv_rows)
            red_session_privileged_pids = red_session_privileged_pids.at[dst].set(dst_priv_rows)

    red_sessions = red_session_count > 0
    newly_active = ~state.red_agent_active & jnp.any(red_sessions, axis=1)
    red_agent_active = state.red_agent_active | newly_active
    # Any host with an active red session must be discoverable by that red agent.
    red_discovered = red_discovered | red_sessions
    # Newly activated agents also discover their start host — CybORG pre-seeds
    # aspace.ip_address with the start host at reset, and it persists through
    # activation.
    for r in range(NUM_RED_AGENTS):
        start_h = const.red_start_hosts[r]
        red_discovered = jnp.where(
            newly_active[r],
            red_discovered.at[r, start_h].set(True),
            red_discovered,
        )
    red_session_is_abstract = jnp.any(red_session_abstract_pids >= 0, axis=2) & red_sessions
    red_privilege = jnp.where(
        jnp.any(red_session_privileged_pids >= 0, axis=2),
        jnp.maximum(red_privilege, COMPROMISE_PRIVILEGED),
        jnp.where(red_sessions, jnp.maximum(red_privilege, COMPROMISE_USER), red_privilege),
    )

    host_compromised = state.host_compromised

    has_any_sessions_now = jnp.any(red_sessions, axis=1)
    current_fsm = state.fsm_host_states
    # CybORG's FSM (FiniteStateRedAgent.host_states) is NOT modified by
    # session reassignment — the agent's own actions drive FSM transitions
    # via _host_state_transition, and new hosts enter as K through
    # _process_new_observations.  Only set FSM for hosts not yet tracked
    # (fsm_host_entered=False); already-tracked hosts keep their state.
    not_yet_entered = ~state.fsm_host_entered
    discovered_decoy = (current_fsm == FSM_UD) | (current_fsm == FSM_RD)
    privileged_session = reassigned_hosts & (red_privilege >= COMPROMISE_PRIVILEGED)
    user_session = reassigned_hosts & ~privileged_session
    fsm_with_sessions = jnp.where(
        privileged_session & not_yet_entered,
        jnp.where(discovered_decoy, FSM_RD, FSM_R),
        current_fsm,
    )
    uncompromised_state = (
        (current_fsm != FSM_U) & (current_fsm != FSM_UD) & (current_fsm != FSM_R) & (current_fsm != FSM_RD)
    )
    fsm_with_sessions = jnp.where(
        user_session & uncompromised_state & not_yet_entered,
        jnp.where(discovered_decoy, FSM_UD, FSM_U),
        fsm_with_sessions,
    )
    red_scan_anchor_host = recompute_scan_anchor_hosts(
        state.red_scan_anchor_host,
        red_sessions,
        red_session_is_abstract,
        const.host_active,
    )

    # Set red_primary_pid for newly activated agents so the post-step
    # apply_red_session_check sees a valid primary and doesn't re-sample.
    # CybORG's add_session assigns ident=0 to the first session, and
    # RedSessionCheck.execute sees session 0 exists → no RNG consumed.
    red_primary_pid = state.red_primary_pid
    red_primary_is_abstract = state.red_primary_is_abstract
    for r in range(NUM_RED_AGENTS):
        anchor_h = red_scan_anchor_host[r]
        anchor_h_idx = jnp.clip(anchor_h, 0, red_session_pids.shape[1] - 1)
        pid_at_anchor = first_valid_pid(red_session_pids[r, anchor_h_idx])
        is_abstract_at_anchor = jnp.any(red_session_abstract_pids[r, anchor_h_idx] >= 0)
        red_primary_pid = jnp.where(
            newly_active[r] & (anchor_h >= 0) & (pid_at_anchor >= 0),
            red_primary_pid.at[r].set(pid_at_anchor),
            red_primary_pid,
        )
        red_primary_is_abstract = jnp.where(
            newly_active[r] & (anchor_h >= 0),
            red_primary_is_abstract.at[r].set(is_abstract_at_anchor),
            red_primary_is_abstract,
        )

    full_clear = (~has_any_sessions_now)[:, None]
    source_matrix = scan_sources(state)
    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=red_sessions,
            red_session_is_abstract=red_session_is_abstract,
        ),
        const,
        source_matrix=source_matrix,
    )
    red_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    red_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)
    red_scan_source_pid = jnp.where(full_clear, jnp.int32(-1), scan_synced.red_scan_source_pid)
    host_suspicious_process = jnp.any(red_suspicious_process_count > 0, axis=0)

    # Deactivate agents that lost all sessions (CybORG line 901-902).
    # Agent 0 stays active (initial agent, never deactivated in CC4).
    has_sessions_after = jnp.any(red_sessions, axis=1)
    lost_all = red_agent_active & ~has_sessions_after
    red_agent_active = red_agent_active & ~lost_all
    red_agent_active = red_agent_active.at[0].set(True)
    # NOTE: Do NOT clear red_discovered for deactivated agents.
    # CybORG's aspace.ip_address retains known IPs across deactivation/
    # re-activation cycles, so discovery must persist too.

    # Mark reassigned hosts as entered in the FSM — CybORG's FSM agent
    # will observe these hosts in its next observation and add them to
    # host_states.  Also preserve existing entries.
    fsm_host_entered = state.fsm_host_entered | (fsm_with_sessions > 0)

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_session_privileged_pids=red_session_privileged_pids,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_source_pid=red_scan_source_pid,
        red_scan_anchor_host=red_scan_anchor_host,
        red_primary_pid=red_primary_pid,
        red_primary_is_abstract=red_primary_is_abstract,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
        fsm_host_states=fsm_with_sessions,
        fsm_host_entered=fsm_host_entered,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_pending_source_host=state.red_pending_source_host,
        red_agent_active=red_agent_active,
    )
