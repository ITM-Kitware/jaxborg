"""Variant-driven env construction + explicit reset helper.

Construction is pure (no reset, no inject). Reset + role-map injection
live in `reset_cyborg_env`, which every rollout calls. Inject-after-every-reset
is a hard invariant; this helper makes it explicit at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

from jaxborg.evaluation.cyborg_red_dispatch import cyborg_red_class
from jaxborg.scenarios.cc4.cyborg_resilience_agents import inject_role_map
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.jaxborg_scenario_generator import JaxborgScenarioGenerator


@dataclass(frozen=True)
class CyborgReset:
    obs: dict
    info: dict
    role_map: dict[str, int] | None


def make_cyborg_env(
    variant: GameVariant,
    seed: int,
    *,
    wrapper_class: type,
    wrapper_kwargs: dict | None = None,
):
    sg_kwargs = dict(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=cyborg_red_class(variant.red_agent, variant.target_weight),
        steps=variant.num_steps,
    )
    if variant.op_zone_servers is not None:
        sg_class = JaxborgScenarioGenerator
        sg_kwargs["op_zone_servers"] = variant.op_zone_servers
    else:
        sg_class = EnterpriseScenarioGenerator

    sg = sg_class(**sg_kwargs)
    cyborg = CybORG(sg, "sim", seed=seed)
    return wrapper_class(env=cyborg, **(wrapper_kwargs or {}))


def reset_cyborg_env(env, variant: GameVariant, ep_seed: int) -> CyborgReset:
    obs, info = env.reset()
    role_map = inject_role_map(env, ep_seed=ep_seed) if variant.resilience_roles else None
    return CyborgReset(obs=obs, info=info, role_map=role_map)
