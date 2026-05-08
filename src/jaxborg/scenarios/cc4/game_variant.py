from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameVariant:
    """Bundle of CC4 game rules. One source of truth for both JAX and CybORG envs."""

    name: str
    red_agent: str = "finite_state"
    target_weight: float = 5.0
    op_zone_servers: int | None = None
    resilience_roles: bool = False
    num_steps: int = 500
