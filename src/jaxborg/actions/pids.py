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


def pid_row_contains(pid_row, pid):
    return (pid >= 0) & jnp.any(pid_row == pid)


def remove_pid_from_row(pid_row, pid):
    return jnp.where(pid_row == pid, -1, pid_row)


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
