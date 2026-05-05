"""Regression: JAX-derived execution order matches CybORG's dict-insertion order.

The action-ordering rewrite collapsed three apply_all_actions variants into one
typed-phase implementation that derives execution order itself, rather than
consuming a CybORG-recorded order. This test pins down the contract that
the derivation must satisfy: for any topology, the green-agent slot order
JAX iterates must equal CybORG's `agent_interfaces` dict-insertion order.

If this test fails, the differential suite will also fail — but this test
isolates the root cause to a single property and runs in seconds, not
minutes.

Bandwidth-shuffle invariant: CybORG's `sort_action_order` shuffles
`action_index` for bandwidth-overflow accounting only; the returned (executed)
list is the priority-sorted, un-shuffled one. CC4 has every action's
`bandwidth_usage = 0`, so no actions are dropped and the shuffle is a no-op
for execution order. If a future scenario introduces non-zero bandwidth,
this test stays true (priority sort still wins) but the underlying invariant
weakens — see the docstring of `apply_all_actions` for the load-bearing
assumption.
"""

import pytest

pytest.importorskip("CybORG")

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.green import _ordered_green_hosts
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg


def _cyborg_green_dict_order(cyborg, mappings):
    """Walk CybORG's agent_interfaces in dict-insertion order, return host indices."""
    hosts = []
    for agent_name in cyborg.environment_controller.agent_interfaces:
        if not agent_name.startswith("green_agent_"):
            continue
        sess_dict = cyborg.environment_controller.state.sessions.get(agent_name, {})
        for _sid, sess in sess_dict.items():
            hosts.append(mappings.hostname_to_idx[sess.hostname])
            break
    return hosts


def _jax_green_order(const):
    """JAX's _ordered_green_hosts argsort over green_agent_host."""
    n = int(const.num_green_agents)
    order = _ordered_green_hosts(const)
    return [int(order[i]) for i in range(n)]


@pytest.mark.parametrize("seed", list(range(8)))
def test_jax_green_order_matches_cyborg_dict_insertion(seed):
    """For each seed, JAX's _ordered_green_hosts == CybORG's green agent registration order.

    This is the load-bearing assumption that lets `apply_all_actions` derive
    execution order from topology alone, with no replay from CybORG.
    """
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=10,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    cyborg.reset()

    mappings = build_mappings_from_cyborg(cyborg)
    const = build_const_from_cyborg(cyborg)

    cyborg_order = _cyborg_green_dict_order(cyborg, mappings)
    jax_order = _jax_green_order(const)

    assert jax_order == cyborg_order, (
        f"seed={seed}: green order divergence.\n"
        f"  cyborg first 15: {cyborg_order[:15]}\n"
        f"  jax first 15:    {jax_order[:15]}"
    )
