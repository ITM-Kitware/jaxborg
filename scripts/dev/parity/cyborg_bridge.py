"""CybORG environment creation and JAX action-space bridge helpers."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import configure_runtime

configure_runtime()

import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.parity.translate import cyborg_blue_to_jax


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    """Translate a CybORG blue action into the JAX canonical action space."""
    cls_name = type(action).__name__
    agent_id = int(agent_name.split("_")[-1])

    if label.startswith("[Padding]"):
        return []

    if cls_name == "Sleep" and not label.startswith("[Invalid]"):
        return [BLUE_SLEEP]

    if cls_name == "Sleep" and label.startswith("[Invalid]"):
        return []

    if cls_name == "DeployDecoy":
        if action.hostname not in mappings.hostname_to_idx:
            return []
        host_idx = mappings.hostname_to_idx[action.hostname]
        jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
        if jax_idx == BLUE_SLEEP:
            return []
        return [jax_idx]

    try:
        jax_idx = cyborg_blue_to_jax(action, agent_name, mappings, const=const)
        if jax_idx == BLUE_SLEEP:
            return []  # host not in agent's observed subnets
        return [jax_idx]
    except (KeyError, ValueError):
        return []


def _build_cyborg_mask_cache(wrapper, mappings, const):
    """Precompute CybORG-to-JAX action translation tables for all agents.

    Returns a dict keyed by agent_name. Each value is a list (one per CybORG
    action slot) of either:
      - list[int]: static JAX indices (for non-Decoy actions)
      - ("decoy", hostname, host_idx, agent_id, factory_jax_indices): precomputed decoy info
      - None: padding/invalid slot (always skipped)
    """
    cache = {}
    controller = wrapper.env.environment_controller
    cyborg_state = controller.state

    for agent_name in wrapper.possible_agents:
        agent_id = int(agent_name.split("_")[-1])
        cyborg_actions = wrapper.actions(agent_name)
        cyborg_labels = wrapper.action_labels(agent_name)
        agent_cache = []
        for action, label in zip(cyborg_actions, cyborg_labels):
            cls_name = type(action).__name__
            if label.startswith("[Padding]") or (cls_name == "Sleep" and label.startswith("[Invalid]")):
                agent_cache.append(None)
            elif cls_name == "DeployDecoy":
                if action.hostname not in mappings.hostname_to_idx:
                    agent_cache.append(None)
                else:
                    host_idx = mappings.hostname_to_idx[action.hostname]
                    jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
                    if jax_idx == BLUE_SLEEP:
                        agent_cache.append(None)
                    else:
                        agent_cache.append([jax_idx])
            else:
                # Static translation — compute once
                jax_indices = _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state)
                agent_cache.append(jax_indices if jax_indices else None)
        cache[agent_name] = agent_cache
    return cache


def _live_blue_wrapper_mask_in_jax_space_cached(wrapper, agent_name, mappings, const, mask_cache):
    """Fast version of mask projection using precomputed translation cache.

    Returns a numpy bool array (caller should stack and convert to jnp once).
    Static mask — does not change for busy agents (matches CybORG).
    """
    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_mask = action_space["mask"]
    agent_cache = mask_cache[agent_name]

    for slot_idx, valid in enumerate(cyborg_mask):
        if not valid:
            continue
        entry = agent_cache[slot_idx]
        if entry is None:
            continue
        for jax_idx in entry:
            jax_mask[jax_idx] = True

    return jax_mask


def _live_blue_wrapper_mask_in_jax_space(wrapper, agent_name, mappings, const):
    """Project BlueFlatWrapper's live action mask into JAX canonical indices.

    Static mask — does not change for busy agents (matches CybORG).
    """
    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_mask = action_space["mask"]
    cyborg_actions = wrapper.actions(agent_name)
    cyborg_labels = wrapper.action_labels(agent_name)
    cyborg_state = wrapper.env.environment_controller.state

    for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
        if not valid:
            continue
        for jax_idx in _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
            jax_mask[jax_idx] = True

    return jnp.array(jax_mask)


def _raw_cyborg_step_with_flat_obs(wrapper, actions, messages=None):
    """Step underlying CybORG with raw actions, then flatten blue observations via the wrapper."""
    obs, rews, dones, info = wrapper.env.parallel_step(
        actions,
        messages=messages,
        skip_valid_action_check=True,
    )

    observations = {
        agent: wrapper.observation_change(agent, obs[agent]) for agent in wrapper.possible_agents if agent in obs
    }
    rewards = {agent: sum(rews[agent].values()) for agent in wrapper.possible_agents if agent in rews}
    terminated = {agent: bool(dones[agent]) for agent in wrapper.possible_agents if agent in dones}
    truncated = terminated.copy()
    info = {agent: {"action_mask": wrapper.get_action_space(agent)["mask"]} for agent in wrapper.possible_agents}
    wrapper.agents = [agent for agent in wrapper.possible_agents if not terminated.get(agent, False)]
    return observations, rewards, terminated, truncated, info


def make_cyborg_env(seed=42):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)
