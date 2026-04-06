import numpy as np

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.translate import cyborg_blue_to_jax, describe_blue_action


def refresh_blue_wrapper_action_space(wrapper) -> None:
    """Refresh BlueFlatWrapper's cached action spaces against the live controller state."""
    if wrapper is None:
        return
    wrapper.agents = list(wrapper.possible_agents)
    for agent_name in wrapper.possible_agents:
        wrapper._populate_action_space(agent_name)


def cyborg_blue_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    """Translate a CybORG blue action slot into one or more JAX canonical indices."""
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
        return [cyborg_blue_to_jax(action, agent_name, mappings, const=const)]
    except (KeyError, ValueError):
        return []


def live_blue_wrapper_mask_in_jax_space(wrapper, agent_name, mappings, const):
    """Project BlueFlatWrapper's live action mask into JAX canonical indices.

    CybORG's BlueFixedActionWrapper mask is based purely on host/session
    validity — pending actions do NOT affect the mask.  CybORG silently
    continues any in-progress action regardless of the agent's choice.
    """
    controller = wrapper.env.environment_controller

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_actions = wrapper.actions(agent_name)
    cyborg_labels = wrapper.action_labels(agent_name)
    for action, valid, label in zip(cyborg_actions, action_space["mask"], cyborg_labels):
        if not valid:
            continue
        for jax_idx in cyborg_blue_action_to_jax_indices(action, label, agent_name, mappings, const, controller.state):
            jax_mask[jax_idx] = True
    return jax_mask


def comparison_blue_mask_in_jax_space(controller, agent_name, agent_idx, state, mappings, const):
    """Return the JAX mask for comparison with CybORG.

    Now that the action space uses a single DeployDecoy per host slot,
    compute_blue_action_mask already produces the correct mask for all cases
    including pending decoy actions.
    """
    return np.asarray(compute_blue_action_mask(const, agent_idx, state), dtype=np.bool_)


def format_action_index_set(indices, mappings, const, *, max_items: int = 5) -> str:
    if not indices:
        return "[]"
    shown = [describe_blue_action(idx, mappings, const=const) for idx in indices[:max_items]]
    suffix = "" if len(indices) <= max_items else f", +{len(indices) - max_items} more"
    return "[" + "; ".join(shown) + suffix + "]"
