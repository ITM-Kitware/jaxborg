import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_JAX_CACHE_DIR = Path.home() / ".cache" / "jaxborg" / "xla"
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_JAX_CACHE_DIR))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
_JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import jax  # noqa: E402
import pytest  # noqa: E402

from jaxborg.actions import apply_blue_action, apply_red_action  # noqa: E402
from jaxborg.constants import NUM_BLUE_AGENTS, SUBNET_IDS  # noqa: E402
from jaxborg.state import CC4State  # noqa: E402
from jaxborg.scenarios.cc4.topology import build_topology  # noqa: E402

jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


def setup_red_agent_session(state: CC4State, agent_id: int, host: int) -> CC4State:
    """Set up initial red agent session on a host (session + abstract flag + anchor)."""
    return state.replace(
        red_sessions=state.red_sessions.at[agent_id, host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[agent_id, host].set(True),
        red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(host),
    )


def find_host_in_subnet(const, subnet_name, exclude_router=True):
    """Find an active non-router host in the specified subnet."""
    sid = SUBNET_IDS[subnet_name]
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if int(const.host_subnet[h]) != sid:
            continue
        if exclude_router and bool(const.host_is_router[h]):
            continue
        return h
    return None


def find_blue_for_host(const, host_idx):
    """Find which blue agent (if any) covers a given host."""
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host_idx]):
            return b
    return None


@pytest.fixture(scope="session")
def jax_const():
    return build_topology(jax.random.PRNGKey(42), num_steps=500)


@pytest.fixture
def cyborg_env():
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


# ---------------------------------------------------------------------------
# Session-scoped shared CybORG environments and their extracted CC4Const.
#
# These exist purely for const extraction and state inspection.  Tests that
# need to step/reset CybORG must create their own function-scoped fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cyborg_env_sleep42():
    """CybORG with all SleepAgent agents, seed=42 (most common config for differential tests)."""
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture(scope="session")
def cyborg_const_sleep42(cyborg_env_sleep42):
    """CC4Const extracted from the all-SleepAgent CybORG env (seed=42)."""
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    return build_const_from_cyborg(cyborg_env_sleep42)


@pytest.fixture(scope="session")
def cyborg_env_default42():
    """CybORG with default agents (SleepAgent blue, EnterpriseGreenAgent green, FiniteStateRedAgent red), seed=42."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture(scope="session")
def cyborg_const_default42(cyborg_env_default42):
    """CC4Const extracted from the default-agent CybORG env (seed=42)."""
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    return build_const_from_cyborg(cyborg_env_default42)
