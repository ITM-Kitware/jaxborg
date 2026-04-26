import jax
import jax.numpy as jnp

from jaxborg.actions.blue_analyse import apply_blue_analyse
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.blue_traffic import apply_allow_traffic, apply_block_traffic
from jaxborg.actions.encoding import (
    ACTION_TYPE_EXPLOIT_SSH,
    ACTION_TYPE_STEALTH_SCAN,
    decode_blue_action,
    decode_red_action,
)
from jaxborg.actions.red_degrade import apply_degrade
from jaxborg.actions.red_discover import apply_discover
from jaxborg.actions.red_discover_deception import apply_discover_deception
from jaxborg.actions.red_exploit_unified import apply_exploit_unified
from jaxborg.actions.red_impact import apply_impact
from jaxborg.actions.red_privesc import apply_privesc
from jaxborg.actions.red_scan_unified import apply_scan_unified
from jaxborg.actions.red_withdraw import apply_withdraw
from jaxborg.state import SimulatorConst, SimulatorState

_RED_BRANCH_MAP = jnp.array(
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 2, 2, 6, 7, 8],
    dtype=jnp.int32,
)
_SCAN_DETECTION_RATE = jnp.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0],
    dtype=jnp.float32,
)
_SCAN_HAS_ROLL = jnp.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
)


def apply_red_action(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    action_idx: int,
    key: jax.Array,
) -> SimulatorState:
    action_type, target_subnet, target_host = decode_red_action(action_idx, agent_id, const)
    k_agg, k_stealth, k_deception = jax.random.split(key, 3)
    k_exploit = jax.random.fold_in(key, 0xCC4)
    k_privesc = jax.random.fold_in(key, 0xC4E5)
    k_scan = jnp.where(action_type == ACTION_TYPE_STEALTH_SCAN, k_stealth, k_agg)

    exploit_subtype = jnp.clip(action_type - ACTION_TYPE_EXPLOIT_SSH, 0, 7)
    branch_idx = _RED_BRANCH_MAP[action_type]
    scan_detection_rate = _SCAN_DETECTION_RATE[action_type]
    scan_has_roll = _SCAN_HAS_ROLL[action_type]

    branches = [
        lambda s: s,  # 0: Sleep
        lambda s: apply_discover(s, const, agent_id, target_subnet),
        lambda s: apply_scan_unified(s, const, agent_id, target_host, k_scan, scan_has_roll, scan_detection_rate),
        lambda s: apply_exploit_unified(s, const, agent_id, target_host, k_exploit, exploit_subtype),
        lambda s: apply_privesc(s, const, agent_id, target_host, k_privesc),
        lambda s: apply_impact(s, const, agent_id, target_host),
        lambda s: apply_discover_deception(s, const, agent_id, target_host, k_deception),
        lambda s: apply_degrade(s, const, agent_id, target_host),
        lambda s: apply_withdraw(s, const, agent_id, target_host),
    ]

    return jax.lax.switch(branch_idx, branches, state)


def apply_blue_action(state: SimulatorState, const: SimulatorConst, agent_id: int, action_idx: int, key=None) -> SimulatorState:
    action_type, target_host, decoy_type, src_subnet, dst_subnet = decode_blue_action(action_idx, agent_id, const)

    branches = [
        lambda s: s,  # 0: Sleep
        lambda s: apply_blue_monitor(s, const, agent_id),
        lambda s: apply_blue_analyse(s, const, agent_id, target_host),
        lambda s: apply_blue_remove(s, const, agent_id, target_host, key),
        lambda s: apply_blue_restore(s, const, agent_id, target_host),
        lambda s: apply_blue_decoy(s, const, agent_id, target_host, decoy_type, key),
        lambda s: apply_block_traffic(s, const, agent_id, src_subnet, dst_subnet),
        lambda s: apply_allow_traffic(s, const, agent_id, src_subnet, dst_subnet),
    ]

    return jax.lax.switch(action_type, branches, state)
