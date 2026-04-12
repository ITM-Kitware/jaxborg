"""Differential tests: pure build_topology() vs CybORG-extracted topology.

Verifies that hardcoded constants in the pure path match the live CybORG
values they were copied from.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import MISSION_PHASES, NUM_SUBNETS, SUBNET_NAMES
from jaxborg.topology import build_const_from_cyborg, build_topology
from jaxborg.topology_numpy import (
    _build_allowed_subnet_pairs_pure,
    _build_phase_rewards,
    _build_phase_rewards_from_cyborg,
    _compute_allowed_subnet_pairs,
)


@pytest.fixture(scope="module")
def cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    env = CybORG(scenario_generator=sg, seed=42)
    env.reset()
    return env


@pytest.fixture(scope="module")
def cyborg_const(cyborg_env):
    return build_const_from_cyborg(cyborg_env)


@pytest.fixture(scope="module")
def pure_const():
    return build_topology(jax.random.PRNGKey(42), num_steps=500)


class TestHardcodedConstants:
    def test_phase_rewards_match_cyborg(self, cyborg_env):
        pure = _build_phase_rewards()
        cyborg = _build_phase_rewards_from_cyborg(cyborg_env)
        np.testing.assert_array_equal(pure, cyborg)

    def test_phase_rewards_shape(self, pure_const):
        assert np.array(pure_const.phase_rewards).shape == (MISSION_PHASES, NUM_SUBNETS, 3)

    def test_allowed_subnet_pairs_match_cyborg(self, cyborg_env):
        pure = _build_allowed_subnet_pairs_pure()
        scenario = cyborg_env.environment_controller.state.scenario
        cyborg = _compute_allowed_subnet_pairs(scenario.allowed_subnets_per_mphase)
        np.testing.assert_array_equal(pure, cyborg)

    def test_subnet_adjacency_identical(self, pure_const, cyborg_const):
        np.testing.assert_array_equal(
            np.array(pure_const.subnet_adjacency),
            np.array(cyborg_const.subnet_adjacency),
        )

    def test_comms_policy_identical(self, pure_const, cyborg_const):
        np.testing.assert_array_equal(
            np.array(pure_const.comms_policy),
            np.array(cyborg_const.comms_policy),
        )

    def test_blue_agent_subnets_match(self, pure_const, cyborg_const):
        np.testing.assert_array_equal(
            np.array(pure_const.blue_agent_subnets),
            np.array(cyborg_const.blue_agent_subnets),
        )

    def test_red_agent_subnets_match(self, pure_const, cyborg_const):
        np.testing.assert_array_equal(
            np.array(pure_const.red_agent_subnets),
            np.array(cyborg_const.red_agent_subnets),
        )


class TestHostProperties:
    def test_bruteforceable_matches_servers_and_users(self, cyborg_const):
        """Pure assumes all servers/users are bruteforceable — verify CybORG agrees."""
        c = cyborg_const
        active_srv_usr = np.array(c.host_active) & (np.array(c.host_is_server) | np.array(c.host_is_user))
        bf = np.array(c.host_has_bruteforceable_user)
        assert np.all(bf[active_srv_usr])

    def test_rfi_always_false_in_cyborg(self, cyborg_const):
        """Pure hardcodes host_has_rfi=False — verify CybORG CC4 never sets it."""
        assert not np.any(np.array(cyborg_const.host_has_rfi))

    def test_respond_to_ping_matches_servers_and_users(self, cyborg_const):
        """Pure sets respond_to_ping for servers+users only — verify CybORG agrees."""
        c = cyborg_const
        active = np.array(c.host_active)
        srv_usr = np.array(c.host_is_server) | np.array(c.host_is_user)
        ping = np.array(c.host_respond_to_ping)
        np.testing.assert_array_equal(ping[active], srv_usr[active])

    def test_services_same_set(self, pure_const, cyborg_const):
        """Both paths assign from the same set of service IDs."""
        pure_svcs = set(np.where(np.any(np.array(pure_const.initial_services), axis=0))[0])
        cyborg_svcs = set(np.where(np.any(np.array(cyborg_const.initial_services), axis=0))[0])
        assert pure_svcs == cyborg_svcs

    def test_data_links_symmetric_and_connected(self, cyborg_const):
        """CybORG data_links are symmetric, no self-loops, hosts link to router."""
        dl = np.array(cyborg_const.data_links)
        np.testing.assert_array_equal(dl, dl.T)
        assert not np.any(np.diag(dl))

        for h in range(int(cyborg_const.num_hosts)):
            if not cyborg_const.host_active[h]:
                continue
            sid = int(cyborg_const.host_subnet[h])
            if SUBNET_NAMES[sid] == "INTERNET" or bool(cyborg_const.host_is_router[h]):
                continue
            router_mask = cyborg_const.host_active & cyborg_const.host_is_router & (cyborg_const.host_subnet == sid)
            router_idx = int(jnp.argmax(router_mask))
            assert dl[h, router_idx], f"host {h} not linked to router {router_idx}"


class TestInitialMaxPid:
    def test_pure_pids_in_service_range(self, pure_const):
        """Pure PIDs: [1000, 10000) for servers/users (matching _generate_pid), 0 for routers."""
        c = pure_const
        active = np.array(c.host_active)
        srv_usr = np.array(c.host_is_server) | np.array(c.host_is_user)
        pids = np.array(c.host_initial_max_pid)
        assert np.all(pids[active & srv_usr] >= 1000)
        assert np.all(pids[active & srv_usr] < 10000)
        assert np.all(pids[active & ~srv_usr] == 0)

    def test_cyborg_pids_in_service_range(self, cyborg_const):
        """CybORG PIDs: [1000, 10009] for servers/users (service + session delta)."""
        c = cyborg_const
        active = np.array(c.host_active)
        srv_usr = np.array(c.host_is_server) | np.array(c.host_is_user)
        pids = np.array(c.host_initial_max_pid)
        assert np.all(pids[active & srv_usr] >= 1000)
        assert np.all(pids[active & srv_usr] <= 10009)

    def test_pure_pids_vary(self, pure_const):
        """PIDs should not all be the same value (old hardcoded-5000 bug)."""
        srv_usr = np.array(pure_const.host_is_server) | np.array(pure_const.host_is_user)
        pids = np.array(pure_const.host_initial_max_pid)[srv_usr]
        assert len(set(pids.tolist())) > 5
