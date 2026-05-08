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
CIA_C = replace(CIA_RESILIENCE, name="cia_c", red_agent="c")
CIA_I = replace(CIA_RESILIENCE, name="cia_i", red_agent="i")
CIA_A = replace(CIA_RESILIENCE, name="cia_a", red_agent="a")

VARIANTS: dict[str, GameVariant] = {v.name: v for v in (CC4_STOCK, CIA_RESILIENCE, CIA_C, CIA_I, CIA_A)}


def variant_for_red(red_agent: str, *, resilience_roles: bool = False) -> GameVariant:
    name = (red_agent or "finite_state").strip().lower()
    if name in {"fsm", "finite_state"}:
        return replace(
            CC4_STOCK,
            resilience_roles=resilience_roles,
            op_zone_servers=3 if resilience_roles else None,
        )
    if name == "resilience":
        return CIA_RESILIENCE
    if name in {"c", "cia_c"}:
        return CIA_C
    if name in {"i", "cia_i"}:
        return CIA_I
    if name in {"a", "cia_a"}:
        return CIA_A
    if name == "sleep":
        return replace(CC4_STOCK, red_agent="sleep")
    raise ValueError(f"unknown red_agent: {red_agent}")
