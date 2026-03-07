"""Differential test: compare JAX action masks against CybORG BlueFlatWrapper masks."""

import jax.numpy as jnp
import numpy as np
import pytest
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import DecoyApache
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import DecoyHarakaSMPT

from jaxborg.actions.encoding import (
    BLUE_SLEEP,
    encode_blue_action,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import SERVICE_IDS
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg


def _cyborg_action_to_jax_index(action, label, agent_name, mappings, const=None):
    """Translate a CybORG action to JAX index, or None if untranslatable."""
    from jaxborg.translate import cyborg_blue_to_jax

    cls_name = type(action).__name__

    if label.startswith("[Padding]"):
        return None

    if cls_name == "Sleep" and not label.startswith("[Invalid]"):
        return BLUE_SLEEP

    if cls_name == "Sleep" and label.startswith("[Invalid]"):
        return None

    if cls_name == "DeployDecoy":
        return None

    try:
        return cyborg_blue_to_jax(action, agent_name, mappings, const=const)
    except (KeyError, ValueError):
        return None


class TestActionMaskDifferential:
    def test_masks_match_cyborg(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        from jaxborg.translate import build_mappings_from_cyborg

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)

        for agent_idx in range(5):
            agent_name = f"blue_agent_{agent_idx}"
            cyborg_actions = wrapped.actions(agent_name)
            cyborg_mask = wrapped.action_mask(agent_name)
            cyborg_labels = wrapped.action_labels(agent_name)

            jax_mask = np.array(compute_blue_action_mask(const, agent_idx))

            mismatches = []
            for i, (action, valid, label) in enumerate(zip(cyborg_actions, cyborg_mask, cyborg_labels)):
                jax_idx = _cyborg_action_to_jax_index(action, label, agent_name, mappings, const=const)
                if jax_idx is None:
                    continue

                jax_val = bool(jax_mask[jax_idx])
                if valid != jax_val:
                    mismatches.append((i, label, valid, jax_val))

            if mismatches:
                details = "\n".join(f"  [{i}] {label}: cyborg={cv}, jax={jv}" for i, label, cv, jv in mismatches[:20])
                pytest.fail(f"Agent {agent_idx}: {len(mismatches)} mask mismatches:\n{details}")

    def test_valid_action_counts_match(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        from jaxborg.translate import build_mappings_from_cyborg

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)

        for agent_idx in range(5):
            agent_name = f"blue_agent_{agent_idx}"
            cyborg_actions = wrapped.actions(agent_name)
            cyborg_mask = wrapped.action_mask(agent_name)
            cyborg_labels = wrapped.action_labels(agent_name)

            jax_mask = np.array(compute_blue_action_mask(const, agent_idx))

            mapped_agree = 0
            mapped_disagree = 0

            for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
                jax_idx = _cyborg_action_to_jax_index(action, label, agent_name, mappings, const=const)
                if jax_idx is None:
                    continue
                if bool(valid) == bool(jax_mask[jax_idx]):
                    mapped_agree += 1
                else:
                    mapped_disagree += 1

            assert mapped_agree > 0, f"Agent {agent_idx}: no mapped actions found"
            assert mapped_disagree == 0, (
                f"Agent {agent_idx}: {mapped_disagree} disagreements out of {mapped_agree + mapped_disagree}"
            )

    def test_haraka_mask_matches_cyborg_failure_on_smtp_host(self, cyborg_env):
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state
        const = build_const_from_cyborg(cyborg_env)
        jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))

        smtp_hosts = np.where(np.array(const.initial_services[:, SERVICE_IDS["SMTP"]], dtype=bool))[0]
        target = int(next(h for h in smtp_hosts if not bool(const.host_is_router[h])))
        blue_idx = int(next(b for b in range(5) if bool(const.blue_agent_hosts[b, target])))
        hostname = sorted(cy_state.hosts.keys())[target]

        cy_action = DecoyHarakaSMPT(session=0, agent=f"blue_agent_{blue_idx}", hostname=hostname)
        cy_obs = cy_action.execute(cy_state)
        jax_mask = np.array(compute_blue_action_mask(const, blue_idx, jax_state))
        jax_idx = encode_blue_action("DeployDecoy_HarakaSMPT", target, blue_idx, const=const)

        assert str(cy_obs.success).upper() == "FALSE"
        assert not bool(jax_mask[jax_idx])

    def test_apache_mask_matches_cyborg_failure_on_apache_host(self, cyborg_env):
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state
        const = build_const_from_cyborg(cyborg_env)
        jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))

        apache_hosts = np.where(np.array(const.initial_services[:, SERVICE_IDS["APACHE2"]], dtype=bool))[0]
        target = int(next(h for h in apache_hosts if not bool(const.host_is_router[h])))
        blue_idx = int(next(b for b in range(5) if bool(const.blue_agent_hosts[b, target])))
        hostname = sorted(cy_state.hosts.keys())[target]

        cy_action = DecoyApache(session=0, agent=f"blue_agent_{blue_idx}", hostname=hostname)
        cy_obs = cy_action.execute(cy_state)
        jax_mask = np.array(compute_blue_action_mask(const, blue_idx, jax_state))
        jax_idx = encode_blue_action("DeployDecoy_Apache", target, blue_idx, const=const)

        assert str(cy_obs.success).upper() == "FALSE"
        assert not bool(jax_mask[jax_idx])
