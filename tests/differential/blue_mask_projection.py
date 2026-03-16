import numpy as np

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
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
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory

    decoy_factory_actions = (
        (HarakaDecoyFactory(), "DeployDecoy_HarakaSMPT"),
        (ApacheDecoyFactory(), "DeployDecoy_Apache"),
        (TomcatDecoyFactory(), "DeployDecoy_Tomcat"),
        (VsftpdDecoyFactory(), "DeployDecoy_Vsftpd"),
    )

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
        host = cyborg_state.hosts[action.hostname]
        host_idx = mappings.hostname_to_idx[action.hostname]
        return [
            encode_blue_action(action_name, host_idx, agent_id, const=const)
            for factory, action_name in decoy_factory_actions
            if factory.is_host_compatible(host)
        ]

    try:
        return [cyborg_blue_to_jax(action, agent_name, mappings, const=const)]
    except (KeyError, ValueError):
        return []


def live_blue_wrapper_mask_in_jax_space(wrapper, agent_name, mappings, const):
    """Project BlueFlatWrapper's live action mask into JAX canonical indices."""
    controller = wrapper.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in cyborg_blue_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
        return jax_mask

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


def format_action_index_set(indices, mappings, const, *, max_items: int = 5) -> str:
    if not indices:
        return "[]"
    shown = [describe_blue_action(idx, mappings, const=const) for idx in indices[:max_items]]
    suffix = "" if len(indices) <= max_items else f", +{len(indices) - max_items} more"
    return "[" + "; ".join(shown) + suffix + "]"
