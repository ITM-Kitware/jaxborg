"""Reward comparison between JAX FsmRedCC4Env and CybORG baselines."""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.slow


def _run_cyborg_sleep_episode(env, steps=50):
    from statistics import mean

    env.reset()
    actions = {agent: 0 for agent in env.agents}
    total = 0.0
    for _ in range(steps):
        _, rewards, _, _, _ = env.step(actions)
        total += mean(rewards.values())
    return total


def _run_jax_sleep_episode(env, env_state, key, steps=50):
    from jaxborg.constants import NUM_BLUE_AGENTS

    actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
    state = env_state
    total = 0.0
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        _, state, rewards, _, _ = env.step(subkey, state, actions)
        total += float(rewards["blue_0"])
    return total


@pytest.fixture
def cyborg_flat_env():
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg, seed=42)
    return BlueFlatWrapper(env=cyborg)


@pytest.fixture
def jax_fsm_from_cyborg(cyborg_flat_env):
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    const = build_const_from_cyborg(cyborg_flat_env.env)
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)

    env = FsmRedCC4Env(num_steps=500)
    return env, env_state


class TestRewardComparison:
    def test_sleep_baseline_both_nonpositive(self, cyborg_flat_env, jax_fsm_from_cyborg):
        jax_env, jax_state = jax_fsm_from_cyborg

        cyborg_total = _run_cyborg_sleep_episode(cyborg_flat_env)
        jax_total = _run_jax_sleep_episode(jax_env, jax_state, jax.random.PRNGKey(0))

        assert cyborg_total <= 0, f"CybORG sleep baseline should be <= 0, got {cyborg_total}"
        if cyborg_total < -1.0:
            assert jax_total <= 0, f"JAX sleep baseline should be <= 0 when CybORG is {cyborg_total}"

    def test_returns_are_finite(self, jax_fsm_from_cyborg):
        from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
        from jaxborg.constants import NUM_BLUE_AGENTS

        jax_env, jax_state = jax_fsm_from_cyborg
        key = jax.random.PRNGKey(42)
        state = jax_state

        for _ in range(50):
            key, act_key, step_key = jax.random.split(key, 3)
            actions = {
                f"blue_{b}": jax.random.randint(jax.random.fold_in(act_key, b), (), 0, BLUE_ALLOW_TRAFFIC_END)
                for b in range(NUM_BLUE_AGENTS)
            }
            _, state, rewards, _, _ = jax_env.step(step_key, state, actions)
            assert jnp.isfinite(rewards["blue_0"]), "Reward not finite at step"
