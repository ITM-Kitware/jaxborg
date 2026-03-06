import jax.numpy as jnp


def host_current_max_pid(state, const, host_idx):
    """Mirror CybORG Host.create_pid base: max(current host processes)."""
    host_session_pids = state.red_session_pids[:, host_idx, :]
    session_max = jnp.max(jnp.where(host_session_pids >= 0, host_session_pids, -1))
    host_decoy_pids = state.host_decoy_process_pids[host_idx]
    decoy_max = jnp.max(jnp.where(host_decoy_pids >= 0, host_decoy_pids, -1))
    return jnp.maximum(const.host_initial_max_pid[host_idx], jnp.maximum(session_max, decoy_max))


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


def pid_row_contains(pid_row, pid):
    return (pid >= 0) & jnp.any(pid_row == pid)


def remove_pid_from_row(pid_row, pid):
    return jnp.where(pid_row == pid, -1, pid_row)


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
