"""Vectorized green agent processing for training mode.

Splits green agent logic into:
1. Pure per-host decisions (vmappable) — computes what each green agent does
2. Scatter phase — applies per-host results to state in one pass
3. Sequential phishing — applies rare session creation events (~1% of hosts)

This replaces the sequential fori_loop over ~80 active hosts with one
vmapped pass + a small sequential loop for the rare phishing events.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxborg.actions.pids import (
    PROCESS_EVENT_NO_PID,
    allocate_host_pid_from_delta,
    append_pid_to_row,
    append_process_event,
)
from jaxborg.actions.rng import sample_green_random
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    MAX_SERVER_HOSTS,
    MAX_USER_HOSTS,
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

# Max phishing events to process sequentially per step.
# With ~80 active hosts and 1% phishing rate, expect ~0.8 per step.
# 8 gives >10x headroom for worst-case clustering.
MAX_PHISHING_PER_STEP = 8


class GreenDecision(NamedTuple):
    """Per-host green agent decision outputs (vmappable)."""
    host_idx: jnp.int32
    local_work_failed: jnp.bool_
    local_fp: jnp.bool_
    access_blocked: jnp.bool_
    access_fp: jnp.bool_
    dest_host: jnp.int32
    phish_creates_session: jnp.bool_
    red_agent_idx: jnp.int32
    pid_delta: jnp.int32


def _find_phishing_red_agent(state, const, host_idx, key):
    """Find which red agent gets a phishing session on this host."""
    host_subnet = const.host_subnet[host_idx]
    same_subnet_agents = const.red_agent_subnets[:, host_subnet]
    has_same_subnet = jnp.any(same_subnet_agents)
    same_candidates = jnp.where(same_subnet_agents, jnp.arange(NUM_RED_AGENTS), -1)
    same_valid = same_candidates >= 0
    same_count = jnp.sum(same_valid)
    same_idx = jax.random.randint(key, (), 0, jnp.maximum(same_count, 1))
    same_sorted = jnp.sort(jnp.where(same_valid, same_candidates, NUM_RED_AGENTS))
    same_subnet_agent = same_sorted[same_idx]

    any_active = jnp.any(state.red_agent_active)
    active_agents = jnp.where(state.red_agent_active, jnp.arange(NUM_RED_AGENTS), NUM_RED_AGENTS)
    active_sorted = jnp.sort(active_agents)
    num_active = jnp.sum(state.red_agent_active)
    fallback_flat_idx = jax.random.randint(key, (), 0, jnp.maximum(num_active, 1))
    fallback_agent = active_sorted[fallback_flat_idx]
    num_candidates = jnp.where(has_same_subnet, same_count, num_active)
    fallback_agent = jnp.where(num_candidates > 0, fallback_agent, jnp.int32(-1))

    return jnp.where(has_same_subnet, same_subnet_agent, fallback_agent)


def _compute_green_decision(
    state: CC4State,
    const: CC4Const,
    host_idx: jnp.int32,
    key: jax.Array,
) -> GreenDecision:
    """Pure function: compute what a green agent does (no state modification)."""
    k1, k2, k3, k4, k5, k_svc, k_rel, k_phish_src, k_pid = jax.random.split(key, 9)
    t = state.time

    action = sample_green_random(const, t, host_idx, 0, k1, int_range=NUM_GREEN_ACTIONS)

    # Service selection
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
    chosen_token_precomputed = jnp.int32(const.green_randoms[t, host_idx, 1])
    chosen_token_rng = sorted_tokens[sample_green_random(const, t, host_idx, 1, k_svc, int_range=num_available)]
    chosen_token = jax.lax.cond(
        const.use_green_randoms,
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
    rel_roll = sample_green_random(const, t, host_idx, 2, k_rel, int_range=100)
    work_succeeds = has_service & chosen_valid & (rel_roll < svc_reliability)

    # Local work
    local_work_failed = (action == GREEN_LOCAL_WORK) & ~work_succeeds
    fp_roll = sample_green_random(const, t, host_idx, 3, k2)
    fp_triggered = fp_roll < FP_DETECTION_RATE
    local_fp = (action == GREEN_LOCAL_WORK) & work_succeeds & fp_triggered

    # Phishing
    phish_roll = sample_green_random(const, t, host_idx, 4, k3)
    phish_triggered = phish_roll < PHISHING_ERROR_RATE
    do_phish = (action == GREEN_LOCAL_WORK) & work_succeeds & phish_triggered
    red_agent = _find_phishing_red_agent(state, const, host_idx, k_phish_src)
    any_red_on_host = jnp.any(state.red_sessions[:, host_idx])
    phish_creates_session = do_phish & (red_agent >= 0) & ~any_red_on_host
    red_agent_idx = jnp.maximum(red_agent, 0)
    pid_delta = sample_green_random(const, t, host_idx, 7, k_pid, int_range=9) + 1

    # Access service
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
    dest_host_precomputed = jnp.int32(const.green_randoms[t, host_idx, 5])
    rand_idx_rng = jax.random.randint(k4, (), 0, jnp.maximum(num_reachable, 1))
    dest_host_rng = sorted_servers[rand_idx_rng]
    dest_host = jax.lax.cond(
        const.use_green_randoms,
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
    access_blocked = do_access & is_blocked
    access_fp_roll = sample_green_random(const, t, host_idx, 6, k5)
    access_fp = do_access & ~is_blocked & (access_fp_roll < FP_DETECTION_RATE)

    return GreenDecision(
        host_idx=host_idx,
        local_work_failed=local_work_failed,
        local_fp=local_fp,
        access_blocked=access_blocked,
        access_fp=access_fp,
        dest_host=dest_host,
        phish_creates_session=phish_creates_session,
        red_agent_idx=red_agent_idx,
        pid_delta=pid_delta,
    )


def apply_green_agents_vmapped(
    state: CC4State,
    const: CC4Const,
    key: jax.Array,
) -> CC4State:
    """Apply all green agents using vmap + scatter instead of sequential fori_loop.

    1. Vmap: compute per-host decisions in parallel
    2. Scatter: apply per-host state changes (LWF, ASF, activity detection)
    3. Sequential: apply rare phishing events (~1% of hosts)
    """
    from jaxborg.actions.green import _ordered_green_hosts

    green_keys = jax.random.split(key, GLOBAL_MAX_HOSTS)
    green_host_order = _ordered_green_hosts(const)
    num_active = const.num_green_agents

    # Pad host_order to GLOBAL_MAX_HOSTS for static vmap shape, mask later.
    active_hosts = green_host_order[:GLOBAL_MAX_HOSTS]
    active_keys = green_keys  # indexed by host_idx inside

    # --- Phase 1: Vmap decisions over ALL host slots (masked) ---
    def decide_for_slot(slot_idx):
        host_idx = active_hosts[slot_idx]
        return _compute_green_decision(state, const, host_idx, green_keys[host_idx])

    decisions = jax.vmap(decide_for_slot)(jnp.arange(GLOBAL_MAX_HOSTS))
    # Mask: only first num_active slots are valid
    active_mask = jnp.arange(GLOBAL_MAX_HOSTS) < num_active

    # --- Phase 2: Scatter per-host results ---
    host_indices = decisions.host_idx  # (GLOBAL_MAX_HOSTS,) — actual host idx per slot

    # LWF: green_lwf_this_step[host_idx] |= local_work_failed
    lwf_mask = active_mask & decisions.local_work_failed
    green_lwf = state.green_lwf_this_step.at[host_indices].max(lwf_mask)

    # ASF: green_asf_this_step[host_idx] |= access_blocked
    asf_mask = active_mask & decisions.access_blocked
    green_asf = state.green_asf_this_step.at[host_indices].max(asf_mask)

    # host_activity_detected[dest_host] |= (access_blocked | access_fp)
    activity_mask = active_mask & (decisions.access_blocked | decisions.access_fp)
    host_activity = state.host_activity_detected.at[decisions.dest_host].max(activity_mask)

    # host_exploit_detected[host_idx] |= local_fp
    exploit_mask = active_mask & decisions.local_fp
    host_exploit = state.host_exploit_detected.at[host_indices].max(exploit_mask)

    # host_process_creation_pids: append FP events
    # This is trickier — each FP host needs an append_process_event.
    # For training mode, we can use a fori_loop over FP hosts (rare: ~0.8 per step).
    fp_mask = active_mask & decisions.local_fp
    fp_host_indices = jnp.where(fp_mask, host_indices, GLOBAL_MAX_HOSTS)
    fp_sorted = jnp.sort(fp_host_indices)
    num_fp = jnp.sum(fp_mask)
    proc_pids = state.host_process_creation_pids

    def _apply_fp_event(i, pids):
        h = fp_sorted[i]
        valid = h < GLOBAL_MAX_HOSTS
        row = pids[h]
        updated = append_process_event(row, PROCESS_EVENT_NO_PID)
        return jnp.where(valid, pids.at[h].set(updated), pids)

    proc_pids = jax.lax.fori_loop(0, jnp.minimum(num_fp, 8), _apply_fp_event, proc_pids)

    state = state.replace(
        green_lwf_this_step=green_lwf,
        green_asf_this_step=green_asf,
        host_activity_detected=host_activity,
        host_exploit_detected=host_exploit,
        host_process_creation_pids=proc_pids,
    )

    # --- Phase 3: Sequential phishing (rare events) ---
    phish_mask = active_mask & decisions.phish_creates_session
    phish_host_indices = jnp.where(phish_mask, host_indices, GLOBAL_MAX_HOSTS)
    phish_sorted_slots = jnp.argsort(~phish_mask)  # phishing slots first
    num_phish = jnp.sum(phish_mask)

    def _apply_phishing(i, carry_state):
        slot = phish_sorted_slots[i]
        valid = i < num_phish
        h = decisions.host_idx[slot]
        red_idx = decisions.red_agent_idx[slot]
        delta = decisions.pid_delta[slot]

        # Re-validate against current carry_state (vmapped decisions used stale
        # state — another green agent may have already created a session on this
        # host or for this red agent since the decisions were computed).
        any_red_on_host_now = jnp.any(carry_state.red_sessions[:, h])
        valid = valid & ~any_red_on_host_now

        # Allocate PID
        new_pid = carry_state.host_max_pid[h] + delta

        # Session creation
        session_counts = effective_session_counts(carry_state)
        had_count = session_counts[red_idx, h]
        new_count = had_count + 1

        # Abstract rank
        abstract_rank_before = carry_state.red_abstract_host_rank[red_idx, h]
        next_rank = carry_state.red_next_abstract_rank[red_idx]
        assigned_rank = jnp.where(
            abstract_rank_before < jnp.int32(ABSTRACT_RANK_NONE),
            abstract_rank_before, next_rank,
        )

        # PID rows
        pid_row = carry_state.red_session_pids[red_idx, h]
        abstract_row = carry_state.red_session_abstract_pids[red_idx, h]

        # Anchor
        agent_has_no_sessions = ~jnp.any(carry_state.red_sessions[red_idx])
        should_set_anchor = (
            valid
            & (carry_state.red_scan_anchor_host[red_idx] < 0)
            & agent_has_no_sessions
        )

        new_state = carry_state.replace(
            red_sessions=carry_state.red_sessions.at[red_idx, h].set(True),
            red_session_count=session_counts.at[red_idx, h].set(new_count),
            red_session_is_abstract=carry_state.red_session_is_abstract.at[red_idx, h].set(True),
            red_abstract_host_rank=carry_state.red_abstract_host_rank.at[red_idx, h].set(assigned_rank),
            red_next_abstract_rank=carry_state.red_next_abstract_rank.at[red_idx].set(next_rank + 1),
            red_session_pids=carry_state.red_session_pids.at[red_idx, h].set(
                append_pid_to_row(pid_row, new_pid)
            ),
            red_session_abstract_pids=carry_state.red_session_abstract_pids.at[red_idx, h].set(
                append_pid_to_row(abstract_row, new_pid)
            ),
            red_next_pid=jnp.maximum(carry_state.red_next_pid, new_pid + 1),
            host_max_pid=carry_state.host_max_pid.at[h].set(
                jnp.maximum(carry_state.host_max_pid[h], new_pid)
            ),
            red_privilege=carry_state.red_privilege.at[red_idx, h].set(
                jnp.maximum(carry_state.red_privilege[red_idx, h], COMPROMISE_USER)
            ),
            host_compromised=carry_state.host_compromised.at[h].set(
                jnp.maximum(carry_state.host_compromised[h], COMPROMISE_USER)
            ),
            red_scan_anchor_host=jnp.where(
                should_set_anchor,
                carry_state.red_scan_anchor_host.at[red_idx].set(h),
                carry_state.red_scan_anchor_host,
            ),
        )

        return jax.tree.map(
            lambda new, old: jnp.where(valid, new, old),
            new_state, carry_state,
        )

    state = jax.lax.fori_loop(0, MAX_PHISHING_PER_STEP, _apply_phishing, state)

    return state
