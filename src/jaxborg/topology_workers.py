"""Multiprocessing worker targets for topology bank building.

This module is intentionally free of JAX imports so that spawned worker
processes do not attempt CUDA initialization.
"""

import numpy as np


def _build_one_topology(seed: int, num_steps: int) -> dict:
    """Build one topology from a CybORG seed; returns dict of numpy arrays."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.topology_numpy import build_const_arrays_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    cyborg.reset()
    return build_const_arrays_from_cyborg(cyborg)


def _build_one_green(seed: int, num_steps: int) -> np.ndarray:
    """Record green random tape for one CybORG seed; returns numpy array."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.cyborg_green_recorder import GreenRecorder
    from jaxborg.translate import build_mappings_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    wrapper.reset()

    mappings = build_mappings_from_cyborg(cyborg)
    recorder = GreenRecorder()
    recorder.install(cyborg, mappings)

    sleep_actions = {agent: Sleep() for agent in wrapper.agents}
    for step_idx in range(num_steps):
        wrapper.step(actions=sleep_actions)
        recorder.extract_step(step_idx)

    return recorder.to_numpy_array()


def _build_one_red_policy(seed: int, num_steps: int) -> np.ndarray:
    """Record red policy random tape for one CybORG seed; returns numpy array."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
    from jaxborg.translate import build_mappings_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    wrapper.reset()

    recorder = RedPolicyRecorder()
    recorder.install(cyborg, build_mappings_from_cyborg(cyborg))

    sleep_actions = {agent: Sleep() for agent in wrapper.agents}
    for _ in range(num_steps):
        wrapper.step(actions=sleep_actions)

    return recorder.to_numpy_array()
