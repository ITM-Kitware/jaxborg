from __future__ import annotations

from dataclasses import replace

from jaxborg.scenarios.cc4.game_variant import GameVariant

CC4_STOCK = GameVariant(name="cc4_stock")

CIA_RESILIENCE = GameVariant(
    name="cia_resilience",
    red_agent="resilience",
    op_zone_servers=3,
    resilience_roles=True,
)
CIA_C = replace(CIA_RESILIENCE, name="cia_c", red_agent="c", target_weight=10.0)
CIA_I = replace(CIA_RESILIENCE, name="cia_i", red_agent="i", target_weight=10.0)
CIA_A = replace(CIA_RESILIENCE, name="cia_a", red_agent="a", target_weight=10.0)

VARIANTS: dict[str, GameVariant] = {v.name: v for v in (CC4_STOCK, CIA_RESILIENCE, CIA_C, CIA_I, CIA_A)}


def variant_for_red(
    red_agent: str,
    *,
    resilience_roles: bool = False,
    red_reward: str = "zero_sum",
    blue_block_policy: str = "cc4",
) -> GameVariant:
    """Build the variant for one scripted/learned Red selector.

    ``red_reward`` and ``blue_block_policy`` are carried from the base variant
    by the caller.  ``blue_block_policy`` in particular must survive this hop:
    a Blue policy trained under a restricted BlockTraffic mask has untrained
    logits on the actions that mask removed, so evaluating it under the stock
    mask would score a different policy.
    """
    name = (red_agent or "finite_state").strip().lower()
    overrides = {"red_reward": red_reward, "blue_block_policy": blue_block_policy}
    if name in {"fsm", "finite_state"}:
        return replace(
            CC4_STOCK,
            resilience_roles=resilience_roles,
            op_zone_servers=3 if resilience_roles else None,
            **overrides,
        )
    if name == "resilience":
        return replace(CIA_RESILIENCE, **overrides)
    if name in {"c", "cia_c"}:
        return replace(CIA_C, **overrides)
    if name in {"i", "cia_i"}:
        return replace(CIA_I, **overrides)
    if name in {"a", "cia_a"}:
        return replace(CIA_A, **overrides)
    if name == "sleep":
        return replace(CC4_STOCK, red_agent="sleep", **overrides)
    raise ValueError(f"unknown red_agent: {red_agent}")
