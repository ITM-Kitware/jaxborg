import jax

from jaxborg.actions.blue_analyse import apply_blue_analyse
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.blue_traffic import apply_allow_traffic, apply_block_traffic
from jaxborg.actions.encoding import (
    decode_blue_action,
    decode_red_action,
)
from jaxborg.actions.red_aggressive_scan import apply_aggressive_scan
from jaxborg.actions.red_degrade import apply_degrade
from jaxborg.actions.red_discover import apply_discover
from jaxborg.actions.red_discover_deception import apply_discover_deception
from jaxborg.actions.red_exploit import (
    apply_exploit_bluekeep,
    apply_exploit_eternalblue,
    apply_exploit_ftp,
    apply_exploit_haraka,
    apply_exploit_http,
    apply_exploit_https,
    apply_exploit_sql,
    apply_exploit_ssh,
)
from jaxborg.actions.red_impact import apply_impact
from jaxborg.actions.red_privesc import apply_privesc
from jaxborg.actions.red_scan import apply_scan
from jaxborg.actions.red_stealth_scan import apply_stealth_scan
from jaxborg.actions.red_withdraw import apply_withdraw
from jaxborg.state import CC4Const, CC4State


def apply_red_action(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
    key: jax.Array,
) -> CC4State:
    action_type, target_subnet, target_host = decode_red_action(action_idx, agent_id, const)
    k_agg, k_stealth, k_deception = jax.random.split(key, 3)
    k_exploit = jax.random.fold_in(key, 0xCC4)
    k_privesc = jax.random.fold_in(key, 0xC4E5)

    branches = [
        lambda s: s,  # 0: Sleep
        lambda s: apply_discover(s, const, agent_id, target_subnet),
        lambda s: apply_scan(s, const, agent_id, target_host),
        lambda s: apply_exploit_ssh(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_ftp(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_http(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_https(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_haraka(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_sql(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_eternalblue(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_exploit_bluekeep(s, const, agent_id, target_host, k_exploit),
        lambda s: apply_privesc(s, const, agent_id, target_host, k_privesc),
        lambda s: apply_impact(s, const, agent_id, target_host),
        lambda s: apply_aggressive_scan(s, const, agent_id, target_host, k_agg),
        lambda s: apply_stealth_scan(s, const, agent_id, target_host, k_stealth),
        lambda s: apply_discover_deception(s, const, agent_id, target_host, k_deception),
        lambda s: apply_degrade(s, const, agent_id, target_host),
        lambda s: apply_withdraw(s, const, agent_id, target_host),
    ]

    return jax.lax.switch(action_type, branches, state)


def apply_blue_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    action_type, target_host, decoy_type, src_subnet, dst_subnet = decode_blue_action(action_idx, agent_id, const)

    branches = [
        lambda s: s,  # 0: Sleep
        lambda s: apply_blue_monitor(s, const, agent_id),
        lambda s: apply_blue_analyse(s, const, agent_id, target_host),
        lambda s: apply_blue_remove(s, const, agent_id, target_host),
        lambda s: apply_blue_restore(s, const, agent_id, target_host),
        lambda s: apply_blue_decoy(s, const, agent_id, target_host, decoy_type),
        lambda s: apply_block_traffic(s, const, agent_id, src_subnet, dst_subnet),
        lambda s: apply_allow_traffic(s, const, agent_id, src_subnet, dst_subnet),
    ]

    return jax.lax.switch(action_type, branches, state)
