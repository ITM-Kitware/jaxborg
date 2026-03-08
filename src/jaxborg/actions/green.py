import jax
import jax.numpy as jnp

from jaxborg.actions.pids import allocate_host_pid_from_delta, append_pid_to_row
from jaxborg.actions.rng import sample_green_random
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
)
from jaxborg.state import CC4Const, CC4State

FP_DETECTION_RATE = 0.01
PHISHING_ERROR_RATE = 0.01

GREEN_SLEEP = 0
GREEN_LOCAL_WORK = 1
GREEN_ACCESS_SERVICE = 2
NUM_GREEN_ACTIONS = 3


def _find_phishing_red_agent(
    state: CC4State,
    const: CC4Const,
    host_idx: jnp.int32,
    key: jax.Array,
) -> jnp.int32:
    target_subnet = const.host_subnet[host_idx]
    session_pairs = state.red_sessions & const.host_active[None, :]
    source_subnets = jnp.broadcast_to(const.host_subnet[None, :], session_pairs.shape)
    routable_pairs = session_pairs & ~state.blocked_zones[source_subnets, target_subnet]
    same_subnet_pairs = routable_pairs & (source_subnets == target_subnet)

    same_subnet_by_host = jnp.any(same_subnet_pairs, axis=0)
    first_same_agent_by_host = jnp.argmax(same_subnet_pairs, axis=0).astype(jnp.int32)
    host_indices = jnp.arange(same_subnet_by_host.shape[0], dtype=jnp.int32)
    last_same_host = jnp.max(jnp.where(same_subnet_by_host, host_indices, jnp.int32(-1)))
    has_same_subnet = last_same_host >= 0
    same_subnet_agent = first_same_agent_by_host[jnp.clip(last_same_host, 0, same_subnet_by_host.shape[0] - 1)]

    routable_flat = jnp.transpose(routable_pairs, (1, 0)).reshape(-1)
    num_routable = jnp.sum(routable_flat.astype(jnp.int32))
    candidate_indices = jnp.where(routable_flat, jnp.arange(routable_flat.shape[0]), routable_flat.shape[0])
    sorted_candidates = jnp.sort(candidate_indices)
    fallback_pick = jax.random.randint(key, (), 0, jnp.maximum(num_routable, 1))
    fallback_flat_idx = sorted_candidates[fallback_pick]
    fallback_agent = (fallback_flat_idx % NUM_RED_AGENTS).astype(jnp.int32)
    fallback_agent = jnp.where(num_routable > 0, fallback_agent, jnp.int32(-1))

    return jnp.where(has_same_subnet, same_subnet_agent, fallback_agent)


def _apply_single_green(
    state: CC4State,
    const: CC4Const,
    host_idx: jnp.int32,
    key: jax.Array,
) -> CC4State:
    k1, k2, k3, k4, k5, k_svc, k_rel, k_phish_src, k_pid = jax.random.split(key, 9)
    t = state.time

    action = sample_green_random(state, t, host_idx, 0, k1, int_range=NUM_GREEN_ACTIONS)

    active_services = state.host_services[host_idx]
    active_decoys = state.host_decoys[host_idx]
    service_tokens = jnp.where(active_services, jnp.arange(NUM_SERVICES), NUM_SERVICES + NUM_DECOY_TYPES)
    decoy_tokens = jnp.where(
        active_decoys,
        NUM_SERVICES + jnp.arange(NUM_DECOY_TYPES),
        NUM_SERVICES + NUM_DECOY_TYPES,
    )
    available_tokens = jnp.concatenate([service_tokens, decoy_tokens], axis=0)
    num_available = jnp.sum(active_services.astype(jnp.int32)) + jnp.sum(active_decoys.astype(jnp.int32))
    sorted_tokens = jnp.sort(available_tokens)
    chosen_token_precomputed = jnp.int32(state.green_randoms[t, host_idx, 1])
    chosen_token_rng = sorted_tokens[sample_green_random(state, t, host_idx, 1, k_svc, int_range=num_available)]
    chosen_token = jax.lax.cond(
        state.use_green_randoms,
        lambda _: chosen_token_precomputed,
        lambda _: chosen_token_rng,
        None,
    )
    has_service = num_available > 0
    chosen_token = jnp.where(has_service, chosen_token, jnp.int32(0))
    chosen_is_decoy = chosen_token >= NUM_SERVICES
    chosen_service = jnp.minimum(chosen_token, jnp.int32(NUM_SERVICES - 1))
    chosen_decoy = jnp.clip(chosen_token - NUM_SERVICES, 0, NUM_DECOY_TYPES - 1)
    chosen_valid = jnp.where(
        chosen_is_decoy,
        active_decoys[chosen_decoy],
        active_services[chosen_service],
    )
    svc_reliability = jnp.where(
        chosen_is_decoy,
        state.host_decoy_reliability[host_idx, chosen_decoy],
        state.host_service_reliability[host_idx, chosen_service],
    )
    rel_roll = sample_green_random(state, t, host_idx, 2, k_rel, int_range=100)
    work_succeeds = has_service & chosen_valid & (rel_roll < svc_reliability)

    # -- GreenLocalWork --
    local_work_failed = (action == GREEN_LOCAL_WORK) & ~work_succeeds
    green_lwf_this_step = jnp.where(
        local_work_failed,
        state.green_lwf_this_step.at[host_idx].set(True),
        state.green_lwf_this_step,
    )

    fp_roll = sample_green_random(state, t, host_idx, 3, k2)
    fp_triggered = fp_roll < FP_DETECTION_RATE
    local_fp = (action == GREEN_LOCAL_WORK) & work_succeeds & fp_triggered

    phish_roll = sample_green_random(state, t, host_idx, 4, k3)
    phish_triggered = phish_roll < PHISHING_ERROR_RATE
    do_phish = (action == GREEN_LOCAL_WORK) & work_succeeds & phish_triggered

    red_agent = _find_phishing_red_agent(state, const, host_idx, k_phish_src)
    any_red_on_host = jnp.any(state.red_sessions[:, host_idx])
    phish_creates_session = do_phish & (red_agent >= 0) & ~any_red_on_host
    red_agent_idx = jnp.maximum(red_agent, 0)
    session_counts = effective_session_counts(state)
    had_count_for_agent = jnp.where(red_agent >= 0, session_counts[red_agent_idx, host_idx], 0)
    new_count_for_agent = had_count_for_agent + phish_creates_session.astype(jnp.int32)

    red_sessions = jnp.where(
        phish_creates_session,
        state.red_sessions.at[red_agent_idx, host_idx].set(new_count_for_agent > 0),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        phish_creates_session,
        session_counts.at[red_agent_idx, host_idx].set(new_count_for_agent),
        session_counts,
    )
    red_session_is_abstract = jnp.where(
        phish_creates_session,
        state.red_session_is_abstract.at[red_agent_idx, host_idx].set(True),
        state.red_session_is_abstract,
    )
    abstract_rank_before = state.red_abstract_host_rank[red_agent_idx, host_idx]
    next_abstract_rank = state.red_next_abstract_rank[red_agent_idx]
    assigned_rank = jnp.where(
        abstract_rank_before < jnp.int32(ABSTRACT_RANK_NONE),
        abstract_rank_before,
        next_abstract_rank,
    )
    red_abstract_host_rank = jnp.where(
        phish_creates_session,
        state.red_abstract_host_rank.at[red_agent_idx, host_idx].set(assigned_rank),
        state.red_abstract_host_rank,
    )
    red_next_abstract_rank = jnp.where(
        phish_creates_session,
        state.red_next_abstract_rank.at[red_agent_idx].set(next_abstract_rank + 1),
        state.red_next_abstract_rank,
    )
    pid_delta = sample_green_random(state, t, host_idx, 7, k_pid, int_range=9) + 1
    new_pid = allocate_host_pid_from_delta(state, const, host_idx, pid_delta)
    red_next_pid = jnp.where(
        phish_creates_session,
        jnp.maximum(state.red_next_pid, new_pid + 1),
        state.red_next_pid,
    )
    pid_row = state.red_session_pids[red_agent_idx, host_idx]
    updated_pid_row = append_pid_to_row(pid_row, new_pid)
    red_session_pids = jnp.where(
        phish_creates_session,
        state.red_session_pids.at[red_agent_idx, host_idx].set(updated_pid_row),
        state.red_session_pids,
    )
    abstract_pid_row = state.red_session_abstract_pids[red_agent_idx, host_idx]
    updated_abstract_pid_row = append_pid_to_row(abstract_pid_row, new_pid)
    red_session_abstract_pids = jnp.where(
        phish_creates_session,
        state.red_session_abstract_pids.at[red_agent_idx, host_idx].set(updated_abstract_pid_row),
        state.red_session_abstract_pids,
    )
    red_privilege = jnp.where(
        phish_creates_session,
        state.red_privilege.at[red_agent_idx, host_idx].set(
            jnp.maximum(state.red_privilege[red_agent_idx, host_idx], COMPROMISE_USER)
        ),
        state.red_privilege,
    )
    host_compromised = jnp.where(
        phish_creates_session,
        state.host_compromised.at[host_idx].set(jnp.maximum(state.host_compromised[host_idx], COMPROMISE_USER)),
        state.host_compromised,
    )
    red_scan_anchor_host = jnp.where(
        phish_creates_session & (red_agent >= 0) & (state.red_scan_anchor_host[red_agent_idx] < 0),
        state.red_scan_anchor_host.at[red_agent_idx].set(host_idx),
        state.red_scan_anchor_host,
    )
    # -- GreenAccessService --
    src_subnet = const.host_subnet[host_idx]
    phase = state.mission_phase
    allowed = const.allowed_subnet_pairs[phase]
    own_subnet = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_).at[src_subnet].set(True)
    src_in_allowed = jnp.any(allowed[src_subnet])
    reachable_subnets = jnp.where(src_in_allowed, allowed[src_subnet] | own_subnet, own_subnet)

    is_reachable_server = (
        const.host_active
        & const.host_is_server
        & reachable_subnets[const.host_subnet]
        & (jnp.arange(GLOBAL_MAX_HOSTS) != host_idx)
    )
    num_reachable = jnp.sum(is_reachable_server)
    has_reachable = num_reachable > 0

    server_indices = jnp.where(is_reachable_server, jnp.arange(GLOBAL_MAX_HOSTS), GLOBAL_MAX_HOSTS)
    sorted_servers = jnp.sort(server_indices)

    # In precomputed mode, field 5 stores the actual JAX host index directly
    # (to match CybORG's server ordering). In RNG mode, pick randomly from sorted list.
    dest_host_precomputed = jnp.int32(state.green_randoms[t, host_idx, 5])
    rand_idx_rng = jax.random.randint(k4, (), 0, jnp.maximum(num_reachable, 1))
    dest_host_rng = sorted_servers[rand_idx_rng]
    dest_host = jax.lax.cond(
        state.use_green_randoms,
        lambda _: dest_host_precomputed,
        lambda _: dest_host_rng,
        None,
    )
    dest_host = jnp.where(has_reachable, dest_host, jnp.int32(0))

    dest_subnet = const.host_subnet[dest_host]
    blocked_src_to_dst = state.blocked_zones[src_subnet, dest_subnet]
    blocked_dst_to_src = state.blocked_zones[dest_subnet, src_subnet]
    is_blocked = blocked_src_to_dst | blocked_dst_to_src

    do_access = (action == GREEN_ACCESS_SERVICE) & has_reachable

    # CybORG only fails GreenAccessService on blocked traffic — the dest service
    # availability check is dead code (method ref not called). See CYBORG_DIFFERENCES.md.
    access_blocked = do_access & is_blocked
    access_fp_roll = sample_green_random(state, t, host_idx, 6, k5)
    access_fp = do_access & ~is_blocked & (access_fp_roll < FP_DETECTION_RATE)

    # CybORG rewards GreenAccessService failures against the source green host's
    # subnet, even though the network event is recorded on the destination host.
    green_asf_this_step = jnp.where(
        access_blocked,
        state.green_asf_this_step.at[host_idx].set(True),
        state.green_asf_this_step,
    )

    # CybORG: GreenAccessService blocked/FP creates network_connections events on dest host
    host_activity_detected = state.host_activity_detected
    host_activity_detected = jnp.where(
        access_blocked | access_fp,
        host_activity_detected.at[dest_host].set(True),
        host_activity_detected,
    )

    # CybORG: GreenLocalWork FP creates process_creation events on source host
    host_exploit_detected = state.host_exploit_detected
    host_exploit_detected = jnp.where(
        local_fp,
        host_exploit_detected.at[host_idx].set(True),
        host_exploit_detected,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_next_abstract_rank=red_next_abstract_rank,
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_next_pid=red_next_pid,
        red_privilege=red_privilege,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        host_activity_detected=host_activity_detected,
        host_exploit_detected=host_exploit_detected,
        green_lwf_this_step=green_lwf_this_step,
        green_asf_this_step=green_asf_this_step,
    )


def apply_green_agents(state: CC4State, const: CC4Const, key: jax.Array) -> CC4State:
    keys = jax.random.split(key, GLOBAL_MAX_HOSTS)

    def step_fn(carry_state, idx):
        is_active = const.green_agent_active[idx]
        new_state = _apply_single_green(carry_state, const, idx, keys[idx])
        out_state = jax.tree.map(
            lambda new, old: jnp.where(is_active, new, old),
            new_state,
            carry_state,
        )
        return out_state, None

    final_state, _ = jax.lax.scan(
        step_fn,
        state,
        jnp.arange(GLOBAL_MAX_HOSTS),
    )
    return final_state
