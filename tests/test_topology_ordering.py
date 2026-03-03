"""Tests that build_topology() assigns host indices in alphabetical hostname order."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import SUBNET_NAMES
from jaxborg.topology import _generate_host_specs, build_topology


class TestHostOrdering:
    def test_generate_host_specs_sorted(self):
        rng = np.random.default_rng(42)
        specs = _generate_host_specs(rng)
        names = [s["name"] for s in specs]
        assert names == sorted(names)

    def test_sorted_across_seeds(self):
        for seed in [0, 1, 7, 99]:
            rng = np.random.default_rng(seed)
            specs = _generate_host_specs(rng)
            names = [s["name"] for s in specs]
            assert names == sorted(names), f"seed={seed}"

    def test_build_topology_host_order_matches_specs(self):
        """The const arrays match the order from _generate_host_specs."""
        key = jnp.array([42])
        const = build_topology(key, num_steps=100)

        rng = np.random.default_rng(42)
        specs = _generate_host_specs(rng)

        assert const.num_hosts == len(specs)
        for idx, spec in enumerate(specs):
            assert bool(const.host_active[idx])
            assert int(const.host_subnet[idx]) == spec["sid"]
            if spec["is_router"]:
                assert bool(const.host_is_router[idx])
            elif spec["is_server"]:
                assert bool(const.host_is_server[idx])
            elif spec["is_user"]:
                assert bool(const.host_is_user[idx])

    def test_every_subnet_has_hosts(self):
        const = build_topology(jnp.array([42]), num_steps=100)
        for sid, sname in enumerate(SUBNET_NAMES):
            mask = np.array(const.host_subnet[: const.num_hosts]) == sid
            assert mask.any(), f"No hosts in subnet {sname}"


class TestCybORGParity:
    @pytest.fixture
    def cyborg_env(self):
        try:
            from CybORG import CybORG
            from CybORG.Agents import SleepAgent
            from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
        except ImportError:
            pytest.skip("CybORG not installed")
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_cyborg_const_uses_alphabetical_ordering(self, cyborg_env):
        """build_const_from_cyborg assigns indices in alphabetical hostname order."""
        from jaxborg.topology import build_const_from_cyborg

        state = cyborg_env.environment_controller.state
        sorted_hostnames = sorted(state.hosts.keys())
        const = build_const_from_cyborg(cyborg_env)

        assert const.num_hosts == len(sorted_hostnames)
        for idx, hostname in enumerate(sorted_hostnames):
            subnet_name_cyborg = state.hostname_subnet_map[hostname]
            from jaxborg.topology import CYBORG_SUFFIX_TO_ID

            expected_sid = CYBORG_SUFFIX_TO_ID[subnet_name_cyborg]
            assert int(const.host_subnet[idx]) == expected_sid, (
                f"host {hostname} at idx {idx}: expected subnet {expected_sid}, got {int(const.host_subnet[idx])}"
            )

    def test_both_builders_use_same_convention(self, cyborg_env):
        """Both builders order host types the same way within each subnet."""
        from jaxborg.topology import build_const_from_cyborg

        cyborg_const = build_const_from_cyborg(cyborg_env)
        pure_const = build_topology(jnp.array([0]), num_steps=100)

        for sid, sname in enumerate(SUBNET_NAMES):
            cyborg_pattern = _dedup_type_sequence(cyborg_const, sid)
            pure_pattern = _dedup_type_sequence(pure_const, sid)
            assert cyborg_pattern == pure_pattern, (
                f"Subnet {sname}: type ordering pattern differs. "
                f"cyborg={cyborg_pattern}, pure={pure_pattern}"
            )


def _dedup_type_sequence(const, sid: int) -> list[str]:
    """Extract deduplicated host type ordering for a subnet (e.g. ['router', 'server', 'user'])."""
    result = []
    n = const.num_hosts
    for h in range(n):
        if int(const.host_subnet[h]) != sid:
            continue
        if bool(const.host_is_router[h]):
            t = "router"
        elif bool(const.host_is_server[h]):
            t = "server"
        elif bool(const.host_is_user[h]):
            t = "user"
        else:
            t = "internet"
        if not result or result[-1] != t:
            result.append(t)
    return result
