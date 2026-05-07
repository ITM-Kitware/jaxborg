"""Cyborg red-agent class dispatch (shared by runners and trainers).

Maps recipe `red_agent` strings to CybORG agent classes:
  finite_state / fsm  -> FiniteStateRedAgent  (CybORG default)
  sleep               -> SleepAgent           (no-op adversary)
  resilience          -> ResilienceRedAgent   (op-zone biased)
  c / cia_c           -> CRedAgent            (CIA-C biased)
  i / cia_i           -> IRedAgent            (CIA-I biased)
  a / cia_a           -> ARedAgent            (CIA-A biased)

Unknown names raise. The CIA agents and ResilienceRedAgent both subclass
FiniteStateRedAgent and accept a target_weight.
"""

from __future__ import annotations


def cyborg_red_class(red_agent: str, target_weight: float = 5.0):
    from CybORG.Agents import FiniteStateRedAgent, SleepAgent

    from jaxborg.scenarios.cc4.cyborg_resilience_agents import (
        ARedAgent,
        CRedAgent,
        IRedAgent,
        ResilienceRedAgent,
    )

    name = (red_agent or "finite_state").strip().lower()
    if name in {"finite_state", "fsm"}:
        return FiniteStateRedAgent
    if name == "sleep":
        return SleepAgent
    if name == "resilience":
        return ResilienceRedAgent.with_weight(target_weight)
    if name in {"c", "cia_c"}:
        return CRedAgent.with_weight(target_weight)
    if name in {"i", "cia_i"}:
        return IRedAgent.with_weight(target_weight)
    if name in {"a", "cia_a"}:
        return ARedAgent.with_weight(target_weight)
    raise ValueError(
        f"Unknown red_agent name: {red_agent!r} "
        "(expected one of finite_state/fsm/sleep/resilience/c/i/a/cia_c/cia_i/cia_a)"
    )
