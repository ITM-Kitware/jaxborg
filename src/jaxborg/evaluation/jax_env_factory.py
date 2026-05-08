"""Variant-driven JAX env factory."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jaxborg.parity.fsm_red_env import FsmRedCC4Env, _empty_extras_factory
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.red_selectors import make_red_selector
from jaxborg.scenarios.cc4.topology_roles import assign_resilience_roles_from_const


def _resilience_extras_factory(key, const):
    return {"host_resilience_role": assign_resilience_roles_from_const(const, key)}


def make_jax_env(
    variant: GameVariant,
    *,
    topology_mode: str = "generative",
    training_mode: bool = False,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    name: str | None = None,
) -> FsmRedCC4Env:
    extras = _resilience_extras_factory if variant.resilience_roles else _empty_extras_factory
    selector = make_red_selector(variant.red_agent, target_weight=variant.target_weight)
    return FsmRedCC4Env(
        num_steps=variant.num_steps,
        topology_mode=topology_mode,
        training_mode=training_mode,
        topology_path=topology_path,
        red_selector=selector,
        extras_factory=extras,
        op_zone_min_servers=variant.op_zone_servers,
        name=name,
    )
