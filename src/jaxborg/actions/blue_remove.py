import jax
import jax.numpy as jnp

from jaxborg.actions.pids import (
    pid_row_contains,
    recompute_host_max_pid,
    remove_pid_from_row,
)
from jaxborg.actions.red_common import sync_scan_memory_fields
from jaxborg.actions.rng import sample_blue_decoy_pid_delta
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    MAX_TRACKED_SUSPICIOUS_PIDS,
)
from jaxborg.state import SimulatorConst, SimulatorState


def apply_blue_remove(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    target_host: int,
    key=None,
) -> SimulatorState:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    suspicious_pid_row = state.blue_suspicious_pids[agent_id, target_host]

    session_count_before = effective_session_counts(state)
    row_indices = jnp.arange(MAX_TRACKED_SUSPICIOUS_PIDS, dtype=jnp.int32)
    row_max_slot = jnp.max(jnp.where(suspicious_pid_row >= 0, row_indices, -1)) + 1
    slot_limit = jnp.clip(row_max_slot, 0, MAX_TRACKED_SUSPICIOUS_PIDS)

    def _remove_slot(slot, carry):
        (
            new_session_count,
            new_abstract_session_count,
            new_suspicious_count,
            new_privilege,
            new_session_pids,
            new_session_abstract_pids,
            new_session_privileged_pids,
            new_session_is_abstract,
            new_abstract_host_rank,
            any_removed,
        ) = carry
        sus_pid = suspicious_pid_row[slot]
        has_pid = sus_pid >= 0

        has_sessions = new_session_count[:, target_host] > 0
        has_live_pid = jax.vmap(pid_row_contains, in_axes=(0, None))(new_session_pids[:, target_host, :], sus_pid)
        # CybORG StopProcess checks proc.user in ('root', 'SYSTEM') per-PID.
        # Only PIDs explicitly tracked as privileged (via privesc) are protected.
        has_privileged_pid = jax.vmap(pid_row_contains, in_axes=(0, None))(
            new_session_privileged_pids[:, target_host, :], sus_pid
        )
        pid_is_privileged = has_privileged_pid
        match_by_red = covers_host & has_pid & has_sessions & has_live_pid & ~pid_is_privileged

        any_match = jnp.any(match_by_red)
        matched_red = jnp.argmax(match_by_red)

        matched_count = new_session_count[matched_red, target_host]
        matched_suspicious = new_suspicious_count[matched_red, target_host]
        count_after = jnp.maximum(matched_count - 1, 0)
        removed_one = any_match & (count_after < matched_count)
        any_removed = any_removed | removed_one
        suspicious_after = jnp.where(removed_one, jnp.maximum(matched_suspicious - 1, 0), matched_suspicious)

        matched_pid_row = new_session_pids[matched_red, target_host]
        matched_abstract_pid_row = new_session_abstract_pids[matched_red, target_host]
        matched_privileged_pid_row = new_session_privileged_pids[matched_red, target_host]
        updated_pid_row = jnp.where(removed_one, remove_pid_from_row(matched_pid_row, sus_pid), matched_pid_row)
        updated_abstract_pid_row = jnp.where(
            removed_one,
            remove_pid_from_row(matched_abstract_pid_row, sus_pid),
            matched_abstract_pid_row,
        )
        updated_privileged_pid_row = jnp.where(
            removed_one,
            remove_pid_from_row(matched_privileged_pid_row, sus_pid),
            matched_privileged_pid_row,
        )

        cleared_pid_row = jnp.full_like(updated_pid_row, -1)
        cleared_abstract_pid_row = jnp.full_like(updated_abstract_pid_row, -1)
        cleared_privileged_pid_row = jnp.full_like(updated_privileged_pid_row, -1)
        updated_pid_row = jnp.where(removed_one & (count_after == 0), cleared_pid_row, updated_pid_row)
        updated_abstract_pid_row = jnp.where(
            removed_one & (count_after == 0),
            cleared_abstract_pid_row,
            updated_abstract_pid_row,
        )
        updated_privileged_pid_row = jnp.where(
            removed_one & (count_after == 0),
            cleared_privileged_pid_row,
            updated_privileged_pid_row,
        )

        has_abstract_after = (count_after > 0) & jnp.any(updated_abstract_pid_row >= 0)
        has_privileged_after = (count_after > 0) & jnp.any(updated_privileged_pid_row >= 0)
        priv_after = jnp.where(
            count_after == 0,
            COMPROMISE_NONE,
            jnp.where(has_privileged_after, COMPROMISE_PRIVILEGED, COMPROMISE_USER),
        )
        abstract_rank_after = jnp.where(
            has_abstract_after,
            new_abstract_host_rank[matched_red, target_host],
            jnp.int32(ABSTRACT_RANK_NONE),
        )

        # Check if the removed PID was abstract
        removed_was_abstract = removed_one & jnp.any(matched_abstract_pid_row == sus_pid)
        abstract_count_before = new_abstract_session_count[matched_red, target_host]
        abstract_count_after = jnp.where(
            removed_was_abstract, jnp.maximum(abstract_count_before - 1, 0), abstract_count_before
        )
        new_abstract_session_count = jnp.where(
            removed_was_abstract,
            new_abstract_session_count.at[matched_red, target_host].set(abstract_count_after),
            new_abstract_session_count,
        )
        new_session_count = jnp.where(
            removed_one,
            new_session_count.at[matched_red, target_host].set(count_after),
            new_session_count,
        )
        new_suspicious_count = jnp.where(
            removed_one,
            new_suspicious_count.at[matched_red, target_host].set(suspicious_after),
            new_suspicious_count,
        )
        new_privilege = jnp.where(
            removed_one,
            new_privilege.at[matched_red, target_host].set(priv_after),
            new_privilege,
        )
        new_session_pids = jnp.where(
            removed_one,
            new_session_pids.at[matched_red, target_host].set(updated_pid_row),
            new_session_pids,
        )
        new_session_abstract_pids = jnp.where(
            removed_one,
            new_session_abstract_pids.at[matched_red, target_host].set(updated_abstract_pid_row),
            new_session_abstract_pids,
        )
        new_session_privileged_pids = jnp.where(
            removed_one,
            new_session_privileged_pids.at[matched_red, target_host].set(updated_privileged_pid_row),
            new_session_privileged_pids,
        )
        new_session_is_abstract = jnp.where(
            removed_one,
            new_session_is_abstract.at[matched_red, target_host].set(has_abstract_after),
            new_session_is_abstract,
        )
        new_abstract_host_rank = jnp.where(
            removed_one,
            new_abstract_host_rank.at[matched_red, target_host].set(abstract_rank_after),
            new_abstract_host_rank,
        )
        return (
            new_session_count,
            new_abstract_session_count,
            new_suspicious_count,
            new_privilege,
            new_session_pids,
            new_session_abstract_pids,
            new_session_privileged_pids,
            new_session_is_abstract,
            new_abstract_host_rank,
            any_removed,
        )

    init = (
        session_count_before,
        state.red_abstract_session_count,
        state.red_suspicious_process_count,
        state.red_privilege,
        state.red_session_pids,
        state.red_session_abstract_pids,
        state.red_session_privileged_pids,
        state.red_session_is_abstract,
        state.red_abstract_host_rank,
        jnp.array(False),
    )
    (
        new_session_count,
        new_abstract_session_count,
        new_suspicious_count,
        new_privilege,
        new_session_pids,
        new_session_abstract_pids,
        new_session_privileged_pids,
        red_session_is_abstract,
        red_abstract_host_rank,
        any_removed,
    ) = jax.lax.fori_loop(0, slot_limit, _remove_slot, init)

    new_sessions = new_session_count > 0
    remaining_max_priv = jnp.max(new_privilege[:, target_host])
    new_host_compromised = jnp.where(
        covers_host & any_removed,
        state.host_compromised.at[target_host].set(remaining_max_priv),
        state.host_compromised,
    )

    # CybORG's StopProcess.kill_process respawns service processes: when a
    # killed PID belongs to a service (e.g. a decoy), the process is
    # removed and re-created with a new PID via host.create_pid(). This
    # lowers host_max_pid (because the old high PID is gone) and the
    # respawned process gets a new, potentially lower PID.  Mirror this by
    # detecting suspicious PIDs that match decoy process PIDs and
    # respawning them.
    any_decoy_respawned = jnp.array(False)
    host_decoy_process_pids = state.host_decoy_process_pids

    def _check_decoy_respawn(slot, carry):
        decoy_pids_carry, any_respawned = carry
        sus_pid = suspicious_pid_row[slot]
        has_pid = sus_pid >= 0
        is_decoy = jnp.any(decoy_pids_carry[target_host] == sus_pid) & has_pid & covers_host
        # Only respawn if the PID is NOT a red session (CybORG's StopProcess
        # kills the process regardless, but only respawns services).
        is_red_session = jnp.any(
            jax.vmap(pid_row_contains, in_axes=(0, None))(new_session_pids[:, target_host, :], sus_pid)
        )
        should_respawn = is_decoy & ~is_red_session
        # Remove old PID from decoy tracking
        matching_slot = jnp.argmax(decoy_pids_carry[target_host] == sus_pid)
        cleared = decoy_pids_carry.at[target_host, matching_slot].set(
            jnp.where(should_respawn, jnp.int32(-1), decoy_pids_carry[target_host, matching_slot])
        )
        decoy_pids_carry = jnp.where(should_respawn, cleared, decoy_pids_carry)
        any_respawned = any_respawned | should_respawn
        return decoy_pids_carry, any_respawned

    host_decoy_process_pids, any_decoy_respawned = jax.lax.fori_loop(
        0, slot_limit, _check_decoy_respawn, (host_decoy_process_pids, any_decoy_respawned)
    )

    # Recompute host_max_pid from remaining processes. CybORG's
    # Host.create_pid() uses max(current processes) which decreases when
    # processes are removed.
    #
    # For the respawn PID base, use PRE-removal session PIDs.  CybORG kills
    # suspicious PIDs one at a time (Remove.execute loops over sus_pids);
    # when a decoy is killed and respawned, the other session PIDs that
    # haven't been killed yet are still in the process list, so create_pid()
    # sees them in max(processes).  Using pre-removal PIDs matches this.
    max_red_pre_removal = jnp.max(
        jnp.where(state.red_session_pids[:, target_host, :] >= 0, state.red_session_pids[:, target_host, :], 0)
    )
    base_max_no_decoy = jnp.maximum(const.host_initial_max_pid[target_host], max_red_pre_removal)
    max_orphan = state.host_orphaned_decoy_max_pid[target_host]
    base_max_w_orphan = jnp.maximum(base_max_no_decoy, max_orphan)
    # Standard recompute (includes original decoy PIDs) for non-decoy-respawn case.
    recomputed_max_standard = recompute_host_max_pid(state, const, target_host, new_session_pids)

    # Respawn ALL killed decoys with new PIDs (matching CybORG's service
    # respawn in StopProcess.kill_process).  CybORG kills and respawns one
    # decoy at a time: after killing decoy X, it computes
    # max(remaining_processes) which includes not-yet-killed decoys but
    # excludes X.  Each respawn PID > the base, so subsequent respawns use
    # the latest respawn PID as their base.
    if key is None:
        key = jax.random.PRNGKey(0)

    def _respawn_one_decoy(respawn_idx, carry):
        pids, prev_respawn_max = carry
        cleared = pids[target_host] < 0
        originals = state.host_decoy_process_pids[target_host] >= 0
        needs = cleared & originals
        has_slot = jnp.any(needs)
        slot = jnp.argmax(needs)
        # Max of original decoy PIDs EXCLUDING the current slot.  This
        # represents the not-yet-killed decoys still in CybORG's process
        # list at the time this decoy is killed.
        orig_excl = state.host_decoy_process_pids[target_host].at[slot].set(jnp.int32(-1))
        max_orig_excl = jnp.max(jnp.where(orig_excl >= 0, orig_excl, jnp.int32(0)))
        # CybORG's base: max(non-decoy procs, other decoys, previous respawns)
        effective_max = jnp.maximum(jnp.maximum(base_max_w_orphan, max_orig_excl), prev_respawn_max)
        delta = sample_blue_decoy_pid_delta(const, state.time, agent_id, key, respawn_index=respawn_idx)
        new_pid = effective_max + delta
        pids = jnp.where(
            has_slot,
            pids.at[target_host, slot].set(new_pid),
            pids,
        )
        prev_respawn_max = jnp.where(has_slot, new_pid, prev_respawn_max)
        return pids, prev_respawn_max

    from jaxborg.constants import NUM_DECOY_TYPES

    host_decoy_process_pids, _ = jax.lax.cond(
        any_decoy_respawned,
        lambda args: jax.lax.fori_loop(0, NUM_DECOY_TYPES, _respawn_one_decoy, args),
        lambda args: args,
        (host_decoy_process_pids, jnp.int32(0)),
    )
    # Final host_max_pid: max of all remaining PIDs (surviving + respawned decoys + base)
    max_decoy_final = jnp.max(
        jnp.where(host_decoy_process_pids[target_host] >= 0, host_decoy_process_pids[target_host], jnp.int32(0))
    )
    recomputed_max_after_respawn = jnp.maximum(base_max_w_orphan, max_decoy_final)
    recomputed_max = jnp.where(any_decoy_respawned, recomputed_max_after_respawn, recomputed_max_standard)

    host_max_pid = jnp.where(
        covers_host & (any_removed | any_decoy_respawned),
        state.host_max_pid.at[target_host].set(recomputed_max),
        state.host_max_pid,
    )
    had_any_sessions = jnp.any(session_count_before > 0, axis=1)
    has_any_sessions_now = jnp.any(new_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    full_clear = cleared_all_sessions[:, None]
    any_suspicious_after = jnp.any(new_suspicious_count[:, target_host] > 0)
    new_suspicious_process = jnp.where(
        covers_host,
        state.host_suspicious_process.at[target_host].set(any_suspicious_after),
        state.host_suspicious_process,
    )
    # When Remove destroys session 0, invalidate the anchor (set to -1)
    # rather than promoting a fallback. In CybORG, session 0 is simply gone
    # until RedSessionCheck promotes a new primary. Host-level session
    # presence is not enough here because other sessions may survive on the
    # same host after session 0 is killed.
    anchor_on_target = state.red_scan_anchor_host == target_host
    lost_all_on_target = covers_host & (session_count_before[:, target_host] > 0) & ~new_sessions[:, target_host]
    primary_pid_survives = jax.vmap(pid_row_contains, in_axes=(0, 0))(
        new_session_pids[:, target_host, :],
        state.red_primary_pid,
    )
    primary_pid_killed = covers_host & anchor_on_target & (state.red_primary_pid >= 0) & ~primary_pid_survives
    primary_invalidated = anchor_on_target & (lost_all_on_target | primary_pid_killed)
    red_scan_anchor_host = jnp.where(
        primary_invalidated,
        jnp.int32(-1),
        state.red_scan_anchor_host,
    )
    red_primary_is_abstract = jnp.where(
        primary_invalidated,
        False,
        state.red_primary_is_abstract,
    )
    red_primary_pid = jnp.where(
        primary_invalidated,
        jnp.int32(-1),
        state.red_primary_pid,
    )

    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=new_sessions,
            red_session_is_abstract=red_session_is_abstract,
            red_abstract_host_rank=red_abstract_host_rank,
        ),
        const,
    )
    new_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    new_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)
    # Per-session scan memory clearing: CybORG stores port knowledge on the
    # specific session object that performed the scan.  When blue Remove kills
    # that session (but other sessions survive on the same host), the scan
    # knowledge is lost.  Check if the scan-owning PID on target_host was
    # killed; if so, clear scan records sourced from target_host.
    scan_owner_pids = state.red_scan_source_pid[:, target_host]  # (NUM_RED_AGENTS,)
    owner_survived = jax.vmap(pid_row_contains, in_axes=(0, 0))(new_session_pids[:, target_host, :], scan_owner_pids)
    owner_killed = covers_host & any_removed & (scan_owner_pids >= 0) & ~owner_survived
    # Clear scan source records from target_host for agents whose owner died.
    cleared_src = new_scanned_source_hosts.at[:, :, target_host].set(
        jnp.where(owner_killed[:, None], False, new_scanned_source_hosts[:, :, target_host])
    )
    new_scanned_source_hosts = jnp.where(
        jnp.any(owner_killed),
        cleared_src,
        new_scanned_source_hosts,
    )
    new_scanned_hosts = jnp.where(
        owner_killed[:, None],
        jnp.any(new_scanned_source_hosts, axis=2),
        new_scanned_hosts,
    )
    new_scan_source_pid = state.red_scan_source_pid.at[:, target_host].set(
        jnp.where(owner_killed, jnp.int32(-1), state.red_scan_source_pid[:, target_host])
    )
    return state.replace(
        red_sessions=new_sessions,
        red_session_count=new_session_count,
        red_abstract_session_count=new_abstract_session_count,
        red_session_pids=new_session_pids,
        red_session_abstract_pids=new_session_abstract_pids,
        red_session_privileged_pids=new_session_privileged_pids,
        red_suspicious_process_count=new_suspicious_count,
        red_privilege=new_privilege,
        red_scan_anchor_host=red_scan_anchor_host,
        red_scanned_hosts=new_scanned_hosts,
        red_scanned_source_hosts=new_scanned_source_hosts,
        red_scan_source_pid=new_scan_source_pid,
        host_compromised=new_host_compromised,
        host_max_pid=host_max_pid,
        host_decoy_process_pids=host_decoy_process_pids,
        host_suspicious_process=new_suspicious_process,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_primary_is_abstract=red_primary_is_abstract,
        red_primary_pid=red_primary_pid,
    )
