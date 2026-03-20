import jax.numpy as jnp


def host_current_max_pid(state, const, host_idx):
    """Mirror CybORG Host.create_pid base: max(current host processes).

    Uses the running per-host max PID tracked in state.host_max_pid, which is
    updated whenever any process is created (red exploit, blue decoy, green
    phishing, etc.).  This matches CybORG's Host.create_pid() which computes
    max(all host processes).
    """
    return state.host_max_pid[host_idx]


def allocate_host_pid_from_delta(state, const, host_idx, delta):
    """Allocate PID as max(current) + delta where delta is in [1, 9]."""
    return host_current_max_pid(state, const, host_idx) + delta


def append_pid_to_row(pid_row, pid):
    already_present = jnp.any(pid_row == pid)
    empty_mask = pid_row < 0
    has_empty = jnp.any(empty_mask)
    insert_idx = jnp.argmax(empty_mask)
    updated = pid_row.at[insert_idx].set(pid)
    return jnp.where(already_present | ~has_empty, pid_row, updated)


def append_pid_to_row_allow_duplicates(pid_row, pid):
    empty_mask = pid_row < 0
    has_empty = jnp.any(empty_mask)
    insert_idx = jnp.argmax(empty_mask)
    updated = pid_row.at[insert_idx].set(pid)
    return jnp.where((pid >= 0) & has_empty, updated, pid_row)


# Sentinel for process_creation events without a PID (CybORG green FP events).
# Must be != -1 (so it's not treated as an empty slot) and < 0 (so Monitor
# doesn't ingest it as a suspicious PID).
PROCESS_EVENT_NO_PID = jnp.int32(-2)


def append_process_event(event_row, pid):
    """Append a process creation event PID to the event row.

    Uses == -1 for empty-slot detection so that PROCESS_EVENT_NO_PID sentinels
    (-2) are treated as occupied slots, matching CybORG's list append semantics
    where green FP events (no PID) occupy a slot.
    """
    empty_mask = event_row == -1
    has_empty = jnp.any(empty_mask)
    insert_idx = jnp.argmax(empty_mask)
    updated = event_row.at[insert_idx].set(pid)
    return jnp.where(has_empty, updated, event_row)


def recompute_host_max_pid(state, const, target_host, session_pids):
    """Recompute host_max_pid from remaining processes after removal.

    CybORG's Host.create_pid() uses max(all current process PIDs). When a
    process is removed (by blue Remove, red Withdraw, etc.), the max can
    decrease. This recomputes from: initial service PIDs, remaining red
    session PIDs, and blue decoy process PIDs.

    Args:
        session_pids: The *updated* red_session_pids array (after removal).
    """
    initial_max = const.host_initial_max_pid[target_host]
    # Max across all red agents' remaining session PIDs on this host
    all_red_pids = session_pids[:, target_host, :]  # (NUM_RED_AGENTS, MAX_TRACKED_SESSION_PIDS)
    max_red = jnp.max(jnp.where(all_red_pids >= 0, all_red_pids, jnp.int32(0)))
    # Max across blue decoy process PIDs on this host
    decoy_pids = state.host_decoy_process_pids[target_host]
    max_decoy = jnp.max(jnp.where(decoy_pids >= 0, decoy_pids, jnp.int32(0)))
    return jnp.maximum(jnp.maximum(initial_max, max_red), max_decoy)


def pid_row_contains(pid_row, pid):
    return (pid >= 0) & jnp.any(pid_row == pid)


def remove_pid_from_row(pid_row, pid):
    """Remove pid and compact: remaining entries slide left, -1s fill the tail.

    CybORG's session dict preserves insertion order when entries are deleted.
    Compacting ensures slot position continues to correspond to session creation
    order, which is critical for privesc session selection (nth_valid_pid).
    Without compaction, append_pid_to_row reuses gaps left by removal, breaking
    the slot-order ↔ creation-order correspondence.
    """
    match = pid_row == pid
    has_match = jnp.any(match)
    # Valid PIDs that aren't being removed sort first (preserving original order)
    is_valid_kept = (pid_row >= 0) & ~match
    priority = (~is_valid_kept).astype(jnp.int32)
    order = jnp.argsort(priority)  # JAX argsort is always stable
    compacted = pid_row[order]
    result = jnp.where(is_valid_kept[order], compacted, jnp.int32(-1))
    return jnp.where(has_match & (pid >= 0), result, pid_row)


def move_pid_to_row_end(pid_row, pid):
    """Mirror CybORG session-dict reinsertion order for promoted session 0."""
    idx = jnp.arange(pid_row.shape[0], dtype=jnp.int32)
    scores = jnp.where(
        pid_row < 0,
        pid_row.shape[0] * 2 + idx,
        jnp.where(pid_row == pid, pid_row.shape[0] + idx, idx),
    )
    order = jnp.argsort(scores)
    moved = pid_row[order]
    return jnp.where((pid >= 0) & jnp.any(pid_row == pid), moved, pid_row)


def count_pid_matches(pid_row, candidate_pids):
    return jnp.sum((candidate_pids[:, None] >= 0) & (pid_row[None, :] == candidate_pids[:, None]))


def count_valid_pids(pid_row):
    return jnp.sum(pid_row >= 0).astype(jnp.int32)


def first_valid_pid(pid_row):
    valid = pid_row >= 0
    idx = jnp.argmax(valid)
    return jnp.where(jnp.any(valid), pid_row[idx], -1)


def nth_valid_pid(pid_row, n):
    valid = pid_row >= 0
    ordinals = jnp.cumsum(valid.astype(jnp.int32)) - 1
    target = valid & (ordinals == n)
    idx = jnp.argmax(target)
    return jnp.where(jnp.any(target), pid_row[idx], first_valid_pid(pid_row))


def nth_valid_pid_sorted(pid_row, n):
    """Return the nth valid PID in ascending order.

    CybORG iterates sessions in dict insertion order (by ident), which
    correlates with ascending PID order on the same host because create_pid()
    is monotonically increasing.  Sorting valid PIDs ascending matches this
    ordering even when JAX slot order diverges due to slot reuse after removal.
    """
    max_pid = jnp.int32(2**30)
    sorted_pids = jnp.sort(jnp.where(pid_row >= 0, pid_row, max_pid))
    valid_sorted = sorted_pids < max_pid
    ordinals = jnp.cumsum(valid_sorted.astype(jnp.int32)) - 1
    target = valid_sorted & (ordinals == n)
    idx = jnp.argmax(target)
    fallback = jnp.where(jnp.any(valid_sorted), sorted_pids[0], jnp.int32(-1))
    return jnp.where(jnp.any(target), sorted_pids[idx], fallback)
