import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

from jaxborg.scenarios.cc4.jaxborg_scenario_generator import JaxborgScenarioGenerator


def _count_op_zone_servers(env):
    ec = env.environment_controller
    hosts = list(ec.state.hosts.keys())
    a = sum(1 for h in hosts if h.startswith("operational_zone_a_subnet_server_host_"))
    b = sum(1 for h in hosts if h.startswith("operational_zone_b_subnet_server_host_"))
    return a, b


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_op_zone_servers_fixed_to_three(seed):
    sg = JaxborgScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
        op_zone_servers=3,
    )
    env = CybORG(sg, "sim", seed=seed)
    a, b = _count_op_zone_servers(env)
    assert a == 3, f"seed={seed} op-zone A had {a} servers, want 3"
    assert b == 3, f"seed={seed} op-zone B had {b} servers, want 3"


def test_stock_generator_unchanged():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=100,
    )
    env = CybORG(sg, "sim", seed=42)
    a, b = _count_op_zone_servers(env)
    assert 1 <= a <= 6 and 1 <= b <= 6
