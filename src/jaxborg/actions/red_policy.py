"""Backend-neutral learned-Red action-space definition.

The simulator's raw Red action encoding has one host block for every concrete
exploit and is intentionally kept unchanged.  A learned policy instead sees a
compact action space with one generic exploit block.  Environment adapters are
responsible for resolving that generic action to a concrete simulator action.

This module has no JAX dependency so the same layout can be used by CybORG
workers without importing the accelerator runtime.
"""

from __future__ import annotations

from enum import IntEnum

from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_SUBNETS, RED_POLICY_ACTION_DIM


class RedPolicyActionType(IntEnum):
    SLEEP = 0
    DISCOVER = 1
    AGGRESSIVE_SCAN = 2
    STEALTH_SCAN = 3
    DISCOVER_DECEPTION = 4
    EXPLOIT = 5
    PRIVESC = 6
    IMPACT = 7
    DEGRADE = 8
    WITHDRAW = 9


RED_POLICY_SLEEP = 0
RED_POLICY_DISCOVER_START = 1
RED_POLICY_DISCOVER_END = RED_POLICY_DISCOVER_START + NUM_SUBNETS

RED_POLICY_AGGRESSIVE_SCAN_START = RED_POLICY_DISCOVER_END
RED_POLICY_AGGRESSIVE_SCAN_END = RED_POLICY_AGGRESSIVE_SCAN_START + GLOBAL_MAX_HOSTS
RED_POLICY_STEALTH_SCAN_START = RED_POLICY_AGGRESSIVE_SCAN_END
RED_POLICY_STEALTH_SCAN_END = RED_POLICY_STEALTH_SCAN_START + GLOBAL_MAX_HOSTS
RED_POLICY_DISCOVER_DECEPTION_START = RED_POLICY_STEALTH_SCAN_END
RED_POLICY_DISCOVER_DECEPTION_END = RED_POLICY_DISCOVER_DECEPTION_START + GLOBAL_MAX_HOSTS
RED_POLICY_EXPLOIT_START = RED_POLICY_DISCOVER_DECEPTION_END
RED_POLICY_EXPLOIT_END = RED_POLICY_EXPLOIT_START + GLOBAL_MAX_HOSTS
RED_POLICY_PRIVESC_START = RED_POLICY_EXPLOIT_END
RED_POLICY_PRIVESC_END = RED_POLICY_PRIVESC_START + GLOBAL_MAX_HOSTS
RED_POLICY_IMPACT_START = RED_POLICY_PRIVESC_END
RED_POLICY_IMPACT_END = RED_POLICY_IMPACT_START + GLOBAL_MAX_HOSTS
RED_POLICY_DEGRADE_START = RED_POLICY_IMPACT_END
RED_POLICY_DEGRADE_END = RED_POLICY_DEGRADE_START + GLOBAL_MAX_HOSTS
RED_POLICY_WITHDRAW_START = RED_POLICY_DEGRADE_END
RED_POLICY_WITHDRAW_END = RED_POLICY_WITHDRAW_START + GLOBAL_MAX_HOSTS

if RED_POLICY_WITHDRAW_END != RED_POLICY_ACTION_DIM:  # pragma: no cover - import-time invariant
    raise RuntimeError("learned-Red action ranges do not match RED_POLICY_ACTION_DIM")


RED_POLICY_HOST_BLOCKS: tuple[tuple[RedPolicyActionType, int, int], ...] = (
    (
        RedPolicyActionType.AGGRESSIVE_SCAN,
        RED_POLICY_AGGRESSIVE_SCAN_START,
        RED_POLICY_AGGRESSIVE_SCAN_END,
    ),
    (RedPolicyActionType.STEALTH_SCAN, RED_POLICY_STEALTH_SCAN_START, RED_POLICY_STEALTH_SCAN_END),
    (
        RedPolicyActionType.DISCOVER_DECEPTION,
        RED_POLICY_DISCOVER_DECEPTION_START,
        RED_POLICY_DISCOVER_DECEPTION_END,
    ),
    (RedPolicyActionType.EXPLOIT, RED_POLICY_EXPLOIT_START, RED_POLICY_EXPLOIT_END),
    (RedPolicyActionType.PRIVESC, RED_POLICY_PRIVESC_START, RED_POLICY_PRIVESC_END),
    (RedPolicyActionType.IMPACT, RED_POLICY_IMPACT_START, RED_POLICY_IMPACT_END),
    (RedPolicyActionType.DEGRADE, RED_POLICY_DEGRADE_START, RED_POLICY_DEGRADE_END),
    (RedPolicyActionType.WITHDRAW, RED_POLICY_WITHDRAW_START, RED_POLICY_WITHDRAW_END),
)


def decode_red_policy_action(action_idx: int) -> tuple[RedPolicyActionType, int]:
    """Decode a compact action into ``(type, target)``.

    ``target`` is a subnet for :attr:`RedPolicyActionType.DISCOVER`, a global
    host slot for host actions, and ``-1`` for Sleep.  Invalid indices raise a
    ``ValueError`` at non-JAX adapter boundaries instead of silently aliasing a
    valid simulator action.
    """

    action = int(action_idx)
    if action == RED_POLICY_SLEEP:
        return RedPolicyActionType.SLEEP, -1
    if RED_POLICY_DISCOVER_START <= action < RED_POLICY_DISCOVER_END:
        return RedPolicyActionType.DISCOVER, action - RED_POLICY_DISCOVER_START
    for action_type, start, end in RED_POLICY_HOST_BLOCKS:
        if start <= action < end:
            return action_type, action - start
    raise ValueError(f"learned-Red action index out of range: {action}")


__all__ = [
    "RED_POLICY_ACTION_DIM",
    "RED_POLICY_AGGRESSIVE_SCAN_END",
    "RED_POLICY_AGGRESSIVE_SCAN_START",
    "RED_POLICY_DEGRADE_END",
    "RED_POLICY_DEGRADE_START",
    "RED_POLICY_DISCOVER_DECEPTION_END",
    "RED_POLICY_DISCOVER_DECEPTION_START",
    "RED_POLICY_DISCOVER_END",
    "RED_POLICY_DISCOVER_START",
    "RED_POLICY_EXPLOIT_END",
    "RED_POLICY_EXPLOIT_START",
    "RED_POLICY_HOST_BLOCKS",
    "RED_POLICY_IMPACT_END",
    "RED_POLICY_IMPACT_START",
    "RED_POLICY_PRIVESC_END",
    "RED_POLICY_PRIVESC_START",
    "RED_POLICY_SLEEP",
    "RED_POLICY_STEALTH_SCAN_END",
    "RED_POLICY_STEALTH_SCAN_START",
    "RED_POLICY_WITHDRAW_END",
    "RED_POLICY_WITHDRAW_START",
    "RedPolicyActionType",
    "decode_red_policy_action",
]
