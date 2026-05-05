"""Vectorized green agent processing.

Splits green agent logic into:
1. Pure per-host decisions (vmappable) — computes what each green agent does
2. Scatter phase — applies per-host results to state in one pass
3. Sequential phishing — applies rare session creation events (~1% of hosts)

This replaces the sequential fori_loop over ~80 active hosts with one
vmapped pass + a small sequential loop for the rare phishing events.

CybORG-faithfulness contract
----------------------------
The vmap intent reads the following pre-green state. For each, the
green phase MUST NOT mutate it (otherwise sequential and vmap diverge):

  field                          | mutated within green phase?
  -------------------------------|---------------------------
  host_services / host_decoys    | no — only blue Restore (phase 1)
  *_reliability                  | no
  blocked_zones                  | no — only blue traffic (phase 1)
  host_active / host_is_server   | no — const
  mission_phase / allowed pairs  | no
  red_sessions                   | YES — by phishing. NEVER read in vmap intent

Because phishing mutates `red_sessions`, the phishing-source derivation
(the only state-sensitive intent) is NOT in the vmap pass. It is
re-derived against `carry_state` inside the sequential phishing fori_loop,
matching `_apply_single_green` in `green.py`.

If a future change adds another green-phase mutation to a field listed
"no" above, this audit becomes invalid and the vmap path will silently
diverge from CybORG. The forced-two-phishing test in
`tests/subsystems/test_green_vmap_pure_parity.py` is the regression gate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxborg.actions.pids import (
    PROCESS_EVENT_NO_PID,
    append_pid_to_row,
    append_process_event,
)
from jaxborg.actions.rng import sample_green_dest_host, sample_green_random
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_DECOY_TYPES,
    NUM_SERVICES,
    NUM_SUBNETS,
)
from jaxborg.state import SimulatorConst, SimulatorState

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
    """Per-host green agent decision outputs (vmappable).

    State-independent intent. The phishing source agent is NOT in this
    struct — it is derived live inside the phishing fori_loop, see
    module docstring.
    """

    host_idx: jnp.int32
    local_work_failed: jnp.bool_
    local_fp: jnp.bool_
    access_blocked: jnp.bool_
    access_fp: jnp.bool_
    dest_host: jnp.int32
    phish_intent: jnp.bool_  # do_phish & work_succeeds & phish_triggered & ~any_red_on_host_pre
    pid_delta: jnp.int32


def _compute_green_decision(
    state: SimulatorState,
    const: SimulatorConst,
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
    chosen_token = sorted_tokens[sample_green_random(const, t, host_idx, 1, k_svc, int_range=num_available)]
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

    # Phishing intent (state-independent rolls only).
    # Source-agent selection is state-dependent (reads red_sessions which is
    # mutated by prior phishing in this same step), so it MUST be derived
    # live inside the phishing fori_loop, not here. The pre-state
    # `~any_red_on_host` is a useful intent gate: red_sessions only grows
    # during the green phase, so a host with a red session pre-green will
    # still have one when this slot would run sequentially.
    del k_phish_src  # source derivation moved to phishing fori_loop
    phish_roll = sample_green_random(const, t, host_idx, 4, k3)
    phish_triggered = phish_roll < PHISHING_ERROR_RATE
    do_phish = (action == GREEN_LOCAL_WORK) & work_succeeds & phish_triggered
    any_red_on_host_pre = jnp.any(state.red_sessions[:, host_idx])
    phish_intent = do_phish & ~any_red_on_host_pre
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
    dest_host = sample_green_dest_host(const, t, host_idx, k4, sorted_servers, num_reachable)
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
        phish_intent=phish_intent,
        pid_delta=pid_delta,
    )


def apply_green_agents_vmapped(
    state: SimulatorState,
    const: SimulatorConst,
    key: jax.Array,
) -> SimulatorState:
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

    # --- Phase 1: Vmap decisions over ALL host slots (masked) ---
    def decide_for_slot(slot_idx):
        host_idx = active_hosts[slot_idx]
        return _compute_green_decision(state, const, host_idx, green_keys[host_idx])

    decisions = jax.vmap(decide_for_slot)(jnp.arange(GLOBAL_MAX_HOSTS))
    # Mask: only first num_active slots are valid. AND in the global
    # `green_agents_active` flag so that the entire green phase becomes a
    # no-op when CybORG uses SleepAgent for green (the harness sets this
    # flag to False to mirror that case).
    active_mask = (jnp.arange(GLOBAL_MAX_HOSTS) < num_active) & const.green_agents_active

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

    # --- Phase 3: Sequential phishing (rare events, ~0.8/step expected) ---
    # Order matters: each green's phishing must see prior greens' state
    # mutations (matching CybORG's sequential dispatch). Source agent and
    # `any_red_on_host` are re-derived from carry_state inside the loop.
    from jaxborg.actions.green import _find_phishing_red_agent

    phish_mask = active_mask & decisions.phish_intent
    phish_sorted_slots = jnp.argsort(~phish_mask)  # phishing slots first, in green-iteration order
    num_phish = jnp.sum(phish_mask)

    def _apply_phishing(i, carry_state):
        slot = phish_sorted_slots[i]
        valid = i < num_phish
        h = decisions.host_idx[slot]
        delta = decisions.pid_delta[slot]

        # Live source-agent derivation against current carry_state (mirrors
        # _apply_single_green's branch in green.py).
        sub_keys = jax.random.split(green_keys[h], 9)
        k_phish_src = sub_keys[7]
        red_agent_live = _find_phishing_red_agent(carry_state, const, h, k_phish_src)

        # Re-validate against current carry_state.
        any_red_on_host_now = jnp.any(carry_state.red_sessions[:, h])
        valid = valid & (red_agent_live >= 0) & ~any_red_on_host_now
        red_idx = jnp.maximum(red_agent_live, 0)

        # Allocate PID
        prev_host_max_pid = carry_state.host_max_pid[h]
        new_pid = prev_host_max_pid + delta

        # Session creation
        session_counts = effective_session_counts(carry_state)
        had_count = session_counts[red_idx, h]
        new_count = had_count + 1

        # Abstract rank
        abstract_rank_before = carry_state.red_abstract_host_rank[red_idx, h]
        next_rank = carry_state.red_next_abstract_rank[red_idx]
        assigned_rank = jnp.where(
            abstract_rank_before < jnp.int32(ABSTRACT_RANK_NONE),
            abstract_rank_before,
            next_rank,
        )

        # PID rows
        pid_row = carry_state.red_session_pids[red_idx, h]
        abstract_row = carry_state.red_session_abstract_pids[red_idx, h]

        # Anchor
        agent_has_no_sessions = ~jnp.any(carry_state.red_sessions[red_idx])
        should_set_anchor = valid & (carry_state.red_scan_anchor_host[red_idx] < 0) & agent_has_no_sessions

        # Push the `valid` gate from a full-state jnp.where down into each
        # individual row/scalar scatter so XLA does not need to materialize a
        # full-state select for every field on every iteration.  When valid is
        # False each scatter writes the original value back at its index
        # (no-op).
        prev_red_sessions_row = carry_state.red_sessions[red_idx, h]
        prev_abs_count_row = carry_state.red_abstract_session_count[red_idx, h]
        prev_server_count = carry_state.red_server_session_count[red_idx]
        prev_is_abstract_row = carry_state.red_session_is_abstract[red_idx, h]
        prev_next_abstract = carry_state.red_next_abstract_rank[red_idx]
        prev_privilege_row = carry_state.red_privilege[red_idx, h]
        prev_host_compromised_row = carry_state.host_compromised[h]
        prev_anchor = carry_state.red_scan_anchor_host[red_idx]

        new_pids_row = jnp.where(valid, append_pid_to_row(pid_row, new_pid), pid_row)
        new_abstract_pids_row = jnp.where(valid, append_pid_to_row(abstract_row, new_pid), abstract_row)

        return carry_state.replace(
            red_sessions=carry_state.red_sessions.at[red_idx, h].set(jnp.where(valid, True, prev_red_sessions_row)),
            red_session_count=carry_state.red_session_count.at[red_idx, h].set(
                jnp.where(valid, new_count, carry_state.red_session_count[red_idx, h])
            ),
            red_abstract_session_count=carry_state.red_abstract_session_count.at[red_idx, h].set(
                jnp.where(valid, prev_abs_count_row + 1, prev_abs_count_row)
            ),
            # CybORG's server_session dict grows by one entry for each new
            # RedAbstractSession (phishing).  Increment the cumulative counter.
            red_server_session_count=carry_state.red_server_session_count.at[red_idx].set(
                jnp.where(valid, prev_server_count + 1, prev_server_count)
            ),
            red_session_is_abstract=carry_state.red_session_is_abstract.at[red_idx, h].set(
                jnp.where(valid, True, prev_is_abstract_row)
            ),
            red_abstract_host_rank=carry_state.red_abstract_host_rank.at[red_idx, h].set(
                jnp.where(valid, assigned_rank, abstract_rank_before)
            ),
            red_next_abstract_rank=carry_state.red_next_abstract_rank.at[red_idx].set(
                jnp.where(valid, prev_next_abstract + 1, prev_next_abstract)
            ),
            red_session_pids=carry_state.red_session_pids.at[red_idx, h].set(new_pids_row),
            red_session_abstract_pids=carry_state.red_session_abstract_pids.at[red_idx, h].set(new_abstract_pids_row),
            red_next_pid=jnp.where(valid, jnp.maximum(carry_state.red_next_pid, new_pid + 1), carry_state.red_next_pid),
            host_max_pid=carry_state.host_max_pid.at[h].set(
                jnp.where(valid, jnp.maximum(prev_host_max_pid, new_pid), prev_host_max_pid)
            ),
            red_privilege=carry_state.red_privilege.at[red_idx, h].set(
                jnp.where(valid, jnp.maximum(prev_privilege_row, COMPROMISE_USER), prev_privilege_row)
            ),
            host_compromised=carry_state.host_compromised.at[h].set(
                jnp.where(
                    valid,
                    jnp.maximum(prev_host_compromised_row, COMPROMISE_USER),
                    prev_host_compromised_row,
                )
            ),
            red_scan_anchor_host=carry_state.red_scan_anchor_host.at[red_idx].set(
                jnp.where(should_set_anchor, h, prev_anchor)
            ),
        )

    state = jax.lax.fori_loop(0, MAX_PHISHING_PER_STEP, _apply_phishing, state)

    return state
