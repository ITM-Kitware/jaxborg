"""Differential test: compare JAX action masks against CybORG BlueFlatWrapper masks."""

import jax.numpy as jnp
import numpy as np
import pytest
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import DecoyApache
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import DecoyHarakaSMPT

from jaxborg.actions.encoding import encode_blue_action
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import SERVICE_IDS
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg
from tests.differential.blue_mask_projection import (
    format_action_index_set,
    live_blue_wrapper_mask_in_jax_space,
    refresh_blue_wrapper_action_space,
)


class TestActionMaskDifferential:
    def test_masks_match_cyborg(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        from jaxborg.translate import build_mappings_from_cyborg

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)
        jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        refresh_blue_wrapper_action_space(wrapped)

        for agent_idx in range(5):
            agent_name = f"blue_agent_{agent_idx}"
            cyborg_mask = live_blue_wrapper_mask_in_jax_space(wrapped, agent_name, mappings, const)
            jax_mask = np.asarray(compute_blue_action_mask(const, agent_idx, jax_state), dtype=np.bool_)

            if not np.array_equal(cyborg_mask, jax_mask):
                cyborg_only = np.flatnonzero(cyborg_mask & ~jax_mask).tolist()
                jax_only = np.flatnonzero(jax_mask & ~cyborg_mask).tolist()
                pytest.fail(
                    f"{agent_name}: projected live mask mismatch\n"
                    f"  cyborg_only={format_action_index_set(cyborg_only, mappings, const)}\n"
                    f"  jax_only={format_action_index_set(jax_only, mappings, const)}"
                )

    def test_valid_action_counts_match(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        from jaxborg.translate import build_mappings_from_cyborg

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)
        jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        refresh_blue_wrapper_action_space(wrapped)

        for agent_idx in range(5):
            agent_name = f"blue_agent_{agent_idx}"
            cyborg_mask = live_blue_wrapper_mask_in_jax_space(wrapped, agent_name, mappings, const)
            jax_mask = np.asarray(compute_blue_action_mask(const, agent_idx, jax_state), dtype=np.bool_)
            assert int(cyborg_mask.sum()) == int(jax_mask.sum()), (
                f"{agent_name}: projected valid count {int(cyborg_mask.sum())} != jax {int(jax_mask.sum())}"
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
