import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

OBS_SIZE = 210

FUZZ_SEEDS = list(range(50))


def _make_cyborg_env(seed, blue_cls=SleepAgent, green_cls=EnterpriseGreenAgent, red_cls=FiniteStateRedAgent):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=blue_cls,
        green_agent_class=green_cls,
        red_agent_class=red_cls,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=seed)


def _check_initial_obs(seed):
    cyborg_env = _make_cyborg_env(seed, blue_cls=SleepAgent, green_cls=SleepAgent, red_cls=SleepAgent)
    wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
    const = build_const_from_cyborg(cyborg_env)

    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))

    observations, _ = wrapped.reset()
    mismatches = []

    for agent_id in range(NUM_BLUE_AGENTS):
        agent_name = f"blue_agent_{agent_id}"
        cyborg_obs = observations[agent_name]
        jax_obs = np.array(get_blue_obs(state, const, agent_id))
        if not np.array_equal(cyborg_obs, jax_obs):
            diff_indices = np.where(cyborg_obs != jax_obs)[0]
            mismatches.append(f"seed={seed} {agent_name}: {len(diff_indices)} diffs at {diff_indices[:10].tolist()}")

    return mismatches


def _check_topology_consistency(seed):
    cyborg_env = _make_cyborg_env(seed, blue_cls=SleepAgent, green_cls=SleepAgent, red_cls=SleepAgent)
    const = build_const_from_cyborg(cyborg_env)
    cyborg_state = cyborg_env.environment_controller.state
    sorted_hosts = sorted(cyborg_state.hosts.keys())
    mismatches = []

    num_hosts = len(sorted_hosts)
    if num_hosts != int(const.num_hosts):
        mismatches.append(f"seed={seed}: host count CybORG={num_hosts} JAX={int(const.num_hosts)}")

    for h in range(num_hosts):
        if not bool(const.host_active[h]):
            mismatches.append(f"seed={seed}: host {h} ({sorted_hosts[h]}) not active in JAX")

    for h in range(num_hosts, GLOBAL_MAX_HOSTS):
        if bool(const.host_active[h]):
            mismatches.append(f"seed={seed}: host {h} active but beyond num_hosts")

    for blue_idx in range(NUM_BLUE_AGENTS):
        agent_name = f"blue_agent_{blue_idx}"
        session = cyborg_state.sessions[agent_name][0]
        children = list(session.children.values()) + [session]
        cyborg_hosts = {sorted_hosts.index(c.hostname) for c in children}
        jax_hosts = {int(h) for h in range(int(const.num_hosts)) if bool(const.blue_agent_hosts[blue_idx, h])}
        if cyborg_hosts != jax_hosts:
            mismatches.append(
                f"seed={seed} {agent_name}: host coverage mismatch CybORG={len(cyborg_hosts)} JAX={len(jax_hosts)}"
            )

    return mismatches


def _check_reward_tables(seed):
    cyborg_env = _make_cyborg_env(seed, blue_cls=SleepAgent, green_cls=SleepAgent, red_cls=SleepAgent)
    const = build_const_from_cyborg(cyborg_env)
    mismatches = []

    from jaxborg.topology import CYBORG_SUFFIX_TO_ID

    brm = BlueRewardMachine("")
    jax_pr = np.array(const.phase_rewards)

    for phase in range(MISSION_PHASES):
        table = brm.get_phase_rewards(phase)
        for cyborg_name, rewards in table.items():
            sid = CYBORG_SUFFIX_TO_ID[cyborg_name]
            for col_idx, key in enumerate(["LWF", "ASF", "RIA"]):
                if jax_pr[phase, sid, col_idx] != rewards[key]:
                    mismatches.append(
                        f"seed={seed} phase={phase} subnet={cyborg_name} {key}: "
                        f"CybORG={rewards[key]} JAX={jax_pr[phase, sid, col_idx]}"
                    )

    return mismatches


def _check_obs_after_sleep_steps(seed, num_sleep_steps=5):
    cyborg_env = _make_cyborg_env(seed, blue_cls=SleepAgent, green_cls=SleepAgent, red_cls=SleepAgent)
    wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
    const = build_const_from_cyborg(cyborg_env)

    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))

    wrapped.reset()
    sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
    for _ in range(num_sleep_steps):
        observations, *_ = wrapped.step(actions=sleep_actions)

    mismatches = []
    for agent_id in range(NUM_BLUE_AGENTS):
        agent_name = f"blue_agent_{agent_id}"
        cyborg_obs = observations[agent_name]
        jax_obs = np.array(get_blue_obs(state, const, agent_id))
        if not np.array_equal(cyborg_obs, jax_obs):
            diff_indices = np.where(cyborg_obs != jax_obs)[0]
            mismatches.append(
                f"seed={seed} step={num_sleep_steps} {agent_name}: "
                f"{len(diff_indices)} diffs at {diff_indices[:10].tolist()}"
            )

    return mismatches


class TestInitialObsFuzzing:
    @pytest.mark.parametrize("seed", FUZZ_SEEDS)
    def test_initial_obs_parity(self, seed):
        mismatches = _check_initial_obs(seed)
        assert len(mismatches) == 0, "\n".join(mismatches)


class TestTopologyFuzzing:
    @pytest.mark.parametrize("seed", FUZZ_SEEDS)
    def test_topology_consistency(self, seed):
        mismatches = _check_topology_consistency(seed)
        assert len(mismatches) == 0, "\n".join(mismatches)


class TestRewardTableFuzzing:
    @pytest.mark.parametrize("seed", FUZZ_SEEDS)
    def test_reward_tables_match(self, seed):
        mismatches = _check_reward_tables(seed)
        assert len(mismatches) == 0, "\n".join(mismatches)


class TestSleepStepFuzzing:
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_obs_stable_after_sleep(self, seed):
        mismatches = _check_obs_after_sleep_steps(seed, num_sleep_steps=5)
        assert len(mismatches) == 0, "\n".join(mismatches)
