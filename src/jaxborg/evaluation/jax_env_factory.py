"""Variant-driven JAX env factory."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jaxborg.parity.fsm_red_env import FsmRedCC4Env, _empty_extras_factory
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.red_selectors import make_red_selector
from jaxborg.scenarios.cc4.topology import load_topology
from jaxborg.scenarios.cc4.topology_roles import (
    assign_resilience_roles_from_const,
    count_resilience_candidates,
)

# build_topology server count per op-zone-B subnet is uniform in [1, 7) → min 1.
# Both alpha-zone subnets are forced to op_zone_min_servers when set; otherwise
# they share the same [1, 6] random range. The candidate pool spans both zones,
# so worst-case = 2 * min(alpha-server-floor) + 2.
_OP_ZONE_B_MIN_PER_SUBNET = 1
_GENERATIVE_ALPHA_RANDOM_FLOOR = 1
_MIN_RESILIENCE_CANDIDATES = 3


def _resilience_extras_factory(key, const):
    return {"host_resilience_role": assign_resilience_roles_from_const(const, key)}


def _generative_min_candidates(op_zone_servers: int | None) -> int:
    alpha_floor = _GENERATIVE_ALPHA_RANDOM_FLOOR if op_zone_servers is None else op_zone_servers
    return 2 * alpha_floor + 2 * _OP_ZONE_B_MIN_PER_SUBNET


def _validate_resilience_topology(
    variant: GameVariant,
    topology_path: str | Path | Sequence[str | Path] | None,
) -> None:
    if topology_path is None:
        worst_case = _generative_min_candidates(variant.op_zone_servers)
        if worst_case < _MIN_RESILIENCE_CANDIDATES:
            raise ValueError(
                f"variant {variant.name!r} has resilience_roles=True but "
                f"op_zone_servers={variant.op_zone_servers} can yield only "
                f"{worst_case} op-zone server candidates in the worst case "
                f"(need ≥{_MIN_RESILIENCE_CANDIDATES} for AUTH/DB/WEB)."
            )
        return

    paths = [topology_path] if isinstance(topology_path, (str, Path)) else list(topology_path)
    for p in paths:
        const = load_topology(p)
        n = count_resilience_candidates(const)
        if n < _MIN_RESILIENCE_CANDIDATES:
            raise ValueError(
                f"topology snapshot {p} has only {n} op-zone server candidates "
                f"(need ≥{_MIN_RESILIENCE_CANDIDATES} for AUTH/DB/WEB roles); "
                f"the resilience CIA metric is undefined on this topology."
            )


def make_jax_env(
    variant: GameVariant,
    *,
    topology_mode: str = "generative",
    training_mode: bool = False,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    mission_bank: Sequence[Sequence[float]] | None = None,
    mission_bank_amplify: float = 1.0,
    name: str | None = None,
) -> FsmRedCC4Env:
    if variant.resilience_roles:
        _validate_resilience_topology(variant, topology_path)
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
        mission_bank=mission_bank,
        mission_bank_amplify=mission_bank_amplify,
        name=name,
    )
