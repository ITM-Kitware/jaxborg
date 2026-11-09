"""Read-only per-step evidence for policy behavior checks.

Pre/post state differences span the whole simulator step. Recovery flags mark
completed remediation attempts, not successful removal of an attacker.
"""

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import BLUE_ACTION_DURATIONS, decode_blue_action
from jaxborg.actions.masking import compute_blue_action_mask


def behavior_fields(before, after, blue_actions, blue_masks, info, *, blue_block_policy="cc4"):
    """Extract arrays without drawing random keys or changing simulator state."""
    pre, post, const = before.state, after.state, before.const
    agents = jnp.arange(blue_actions.shape[0])
    busy = pre.blue_pending_ticks > 0
    effective = jnp.where(busy, pre.blue_pending_action, blue_actions)
    kinds, hosts, _, sources, destinations = jax.vmap(lambda action, agent: decode_blue_action(action, agent, const))(
        effective, agents
    )
    submitted_types, submitted_hosts, _, submitted_sources, submitted_destinations = jax.vmap(
        lambda action, agent: decode_blue_action(action, agent, const)
    )(blue_actions, agents)
    ticks = jnp.where(busy, pre.blue_pending_ticks, BLUE_ACTION_DURATIONS[kinds])
    structural = jnp.stack(
        [
            compute_blue_action_mask(const, agent, pre, blue_block_policy=blue_block_policy)
            for agent in range(blue_actions.shape[0])
        ]
    )
    result = {
        "blue_busy": busy,
        "pre_step_count": jnp.asarray(pre.time),
        "post_step_count": jnp.asarray(post.time),
        "pre_mission_phase": pre.mission_phase,
        "post_mission_phase": post.mission_phase,
        "pre_mission_comms_policy": const.comms_policy[pre.mission_phase],
        "blue_submitted_type": submitted_types,
        "blue_submitted_host": submitted_hosts,
        "blue_submitted_src_subnet": submitted_sources,
        "blue_submitted_dst_subnet": submitted_destinations,
        "blue_pending_ticks": pre.blue_pending_ticks,
        "blue_pending_action": pre.blue_pending_action,
        "blue_action_masks": blue_masks,
        "blue_structural_masks": structural,
        "blue_effective_action": effective,
        "blue_effective_type": kinds,
        "blue_effective_host": hosts,
        "blue_effective_src_subnet": sources,
        "blue_effective_dst_subnet": destinations,
        "blue_effective_completes": ticks <= 1,
        "blue_observed_subnets": const.blue_obs_subnets,
        "observation_host_map": const.obs_host_map,
        "host_subnet": const.host_subnet,
        "host_active": const.host_active,
        "blue_owned_hosts": const.blue_agent_hosts,
        "blue_recovery_attempt_completed": post.blue_recovered_this_step,
        "red_effective_internal_action": post.red_pending_action,
        "red_pending_ticks_before": pre.red_pending_ticks,
        "red_active_before": pre.red_agent_active,
        "green_lwf_hosts": post.green_lwf_this_step,
        "green_asf_hosts": post.green_asf_this_step,
        "red_impact_hosts": post.red_impact_attempted,
    }
    for prefix, state in (("pre", pre), ("post", post)):
        for field in (
            "host_compromised",
            "red_sessions",
            "red_session_count",
            "host_services",
            "host_service_reliability",
            "host_decoys",
            "host_decoy_reliability",
            "ot_service_stopped",
            "blocked_zones",
            "host_exploit_detected",
            "old_host_exploit_detected",
            "host_activity_detected",
            "old_host_activity_detected",
            "blue_file_evidence",
            "blue_ioc_codes",
        ):
            value = getattr(state, field)
            if value is not None:
                result[f"{prefix}_{field}"] = value
        if state.blue_ioc_memory is not None:
            for name in ("process", "network", "files", "decoys", "delivered"):
                result[f"{prefix}_observed_{name}"] = state.blue_ioc_memory[name]
    for name in (
        "reward_ria",
        "reward_lwf",
        "reward_asf",
        "action_cost",
        "impact_count",
        "green_lwf_count",
        "green_asf_count",
    ):
        result[name] = jnp.asarray(info.get(name, 0), dtype=jnp.float32)
    return result
