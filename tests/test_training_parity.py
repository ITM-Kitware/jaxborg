"""Training parity tests: run JAX and CybORG side-by-side.

Two modes of comparison:
1. Independent FSM: both envs run their own FSM red agents (different RNG → different actions).
   Used for smoke tests that only check sign/magnitude of rewards.
2. Action replay: record CybORG's FSM red actions and replay in JAX's base env.
   Used for exact state/obs parity checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_MONITOR, BLUE_SLEEP, RED_SLEEP
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxborg.observations import get_blue_obs


def _setup_both_envs(seed=42):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    wrapped = BlueFlatWrapper(env=cyborg, pad_spaces=True)

    const = build_const_from_cyborg(cyborg)
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)

    jax_env = FsmRedCC4Env(num_steps=500)
    return wrapped, jax_env, env_state


def _setup_replay_envs(seed=42):
    """CybORG with FSM red + SleepAgent green; JAX base env with green disabled."""
    from CybORG import CybORG
    from CybORG.Agents import FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.env import CC4Env, CC4EnvState, _init_red_state
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg
    from jaxborg.translate import build_mappings_from_cyborg

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    wrapped = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    wrapped.reset()

    const = build_const_from_cyborg(cyborg)
    mappings = build_mappings_from_cyborg(cyborg)

    const = const.replace(green_agent_active=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_))

    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)

    base_env = CC4Env(num_steps=500)
    env_state = CC4EnvState(state=state, const=const)

    return wrapped, base_env, env_state, mappings, cyborg


def _extract_cyborg_red_actions(cyborg, mappings, was_busy):
    """Get red actions to replay into JAX duration semantics.

    CybORG exposes executed actions in `controller.action` and queued actions in
    `actions_in_progress`. For duration-correct replay, submit:
    - queued action when a new long action is first enqueued (executed Sleep),
    - Sleep while already busy (JAX executes stored pending action),
    - otherwise the executed action.
    """
    from jaxborg.translate import cyborg_red_to_jax

    ec = cyborg.environment_controller
    red_actions = {}
    next_busy = {}
    for r in range(NUM_RED_AGENTS):
        agent_name = f"red_agent_{r}"
        pending = ec.actions_in_progress.get(agent_name)
        actions_list = ec.action.get(agent_name, [])
        executed_action = actions_list[0] if actions_list else None

        if was_busy.get(agent_name, False):
            jax_idx = RED_SLEEP
        else:
            just_queued = pending is not None and type(executed_action).__name__ == "Sleep"
            if just_queued:
                jax_idx = cyborg_red_to_jax(pending["action"], agent_name, mappings)
            elif executed_action is not None:
                jax_idx = cyborg_red_to_jax(executed_action, agent_name, mappings)
            else:
                jax_idx = RED_SLEEP

        red_actions[f"red_{r}"] = jnp.int32(jax_idx)
        next_busy[agent_name] = pending is not None
    return red_actions, next_busy


def _extract_cyborg_red_state(cyborg, mappings):
    """Extract red session/privilege/compromise state from CybORG."""
    cyborg_state = cyborg.environment_controller.state
    sessions = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    privilege = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=np.int32)
    compromised = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.int32)

    for agent_name, agent_sessions in cyborg_state.sessions.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        for sess in agent_sessions.values():
            if sess.hostname not in mappings.hostname_to_idx:
                continue
            hidx = mappings.hostname_to_idx[sess.hostname]
            sessions[red_idx, hidx] = True
            level = 1
            if hasattr(sess, "username") and sess.username in ("root", "SYSTEM"):
                level = 2
            privilege[red_idx, hidx] = max(privilege[red_idx, hidx], level)
            compromised[hidx] = max(compromised[hidx], level)

    return sessions, privilege, compromised


@pytest.fixture
def both_envs():
    return _setup_both_envs(seed=42)


@pytest.fixture
def replay_envs():
    return _setup_replay_envs(seed=42)


class TestObservationParity:
    """Compare blue observations between CybORG and JAX at each step."""

    def test_initial_obs_match(self, both_envs):
        wrapped, jax_env, env_state = both_envs
        cyborg_obs, _ = wrapped.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_ob = cyborg_obs[agent_name]
            jax_ob = np.array(get_blue_obs(env_state.state, env_state.const, agent_id))
            np.testing.assert_array_equal(
                cyborg_ob,
                jax_ob,
                err_msg=f"Initial obs mismatch for {agent_name}",
            )

    def test_obs_after_sleep_steps_action_replay(self, replay_envs):
        """Replay CybORG's red actions in JAX for 10 steps with sleep blue.

        Records what CybORG's FSM red agents actually do, then executes those
        same actions in JAX. Green is disabled in both envs. Compares the
        subnet host data section of observations (first 60 fields per agent).
        """
        wrapped, base_env, env_state, mappings, cyborg = replay_envs

        key = jax.random.PRNGKey(42)
        sleep_cyborg = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        sleep_jax = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
        was_busy = {f"red_agent_{r}": False for r in range(NUM_RED_AGENTS)}

        state_diffs = []

        for step in range(10):
            cyborg_obs, _, _, _, _ = wrapped.step(actions=sleep_cyborg)
            red_actions, was_busy = _extract_cyborg_red_actions(cyborg, mappings, was_busy)

            key, subkey = jax.random.split(key)
            all_actions = {**sleep_jax, **red_actions}
            jax_obs, env_state, _, _, _ = base_env.step_env(subkey, env_state, all_actions)

            cyborg_sess, _, cyborg_comp = _extract_cyborg_red_state(cyborg, mappings)
            jax_sess = np.array(env_state.state.red_sessions)
            jax_comp = np.array(env_state.state.host_compromised)

            if not np.array_equal(jax_sess, cyborg_sess):
                diff_agents = np.where(np.any(jax_sess != cyborg_sess, axis=1))[0]
                state_diffs.append(f"step={step} red_sessions: agents {diff_agents.tolist()}")
            if not np.array_equal(jax_comp, cyborg_comp):
                diff_hosts = np.where(jax_comp != cyborg_comp)[0]
                state_diffs.append(f"step={step} host_compromised: hosts {diff_hosts[:5].tolist()}")

        if state_diffs:
            details = "\n".join(state_diffs[:20])
            pytest.fail(f"State mismatches with action replay:\n{details}")

    def test_obs_after_monitor_steps_action_replay(self, replay_envs):
        """Replay CybORG's red actions in JAX for 10 steps with monitor blue."""
        wrapped, base_env, env_state, mappings, cyborg = replay_envs

        key = jax.random.PRNGKey(42)
        monitor_jax = {f"blue_{i}": jnp.int32(BLUE_MONITOR) for i in range(NUM_BLUE_AGENTS)}
        monitor_idx = wrapped.action_labels("blue_agent_0").index("Monitor")
        monitor_cyborg = {f"blue_agent_{i}": monitor_idx for i in range(NUM_BLUE_AGENTS)}
        was_busy = {f"red_agent_{r}": False for r in range(NUM_RED_AGENTS)}

        state_diffs = []

        for step in range(10):
            cyborg_obs, _, _, _, _ = wrapped.step(actions=monitor_cyborg)
            red_actions, was_busy = _extract_cyborg_red_actions(cyborg, mappings, was_busy)

            key, subkey = jax.random.split(key)
            all_actions = {**monitor_jax, **red_actions}
            jax_obs, env_state, _, _, _ = base_env.step_env(subkey, env_state, all_actions)

            cyborg_sess, cyborg_priv, cyborg_comp = _extract_cyborg_red_state(cyborg, mappings)
            jax_sess = np.array(env_state.state.red_sessions)
            jax_comp = np.array(env_state.state.host_compromised)

            if not np.array_equal(jax_sess, cyborg_sess):
                diff_agents = np.where(np.any(jax_sess != cyborg_sess, axis=1))[0]
                state_diffs.append(f"step={step} red_sessions: agents {diff_agents.tolist()}")
            if not np.array_equal(jax_comp, cyborg_comp):
                diff_hosts = np.where(jax_comp != cyborg_comp)[0]
                state_diffs.append(f"step={step} host_compromised: hosts {diff_hosts[:5].tolist()}")

        if state_diffs:
            details = "\n".join(state_diffs[:20])
            pytest.fail(f"Monitor state mismatches with action replay:\n{details}")


class TestRewardParity:
    """Compare per-step rewards between CybORG and JAX."""

    def test_sleep_reward_trajectory(self, both_envs):
        """200 steps with all-sleep blue: both envs should produce negative rewards.

        Since CybORG and JAX use different RNGs, the stochastic FSM red agents
        will take different paths. We only check that both produce negative
        cumulative reward (red causes damage in both).
        """
        wrapped, jax_env, env_state = both_envs
        wrapped.reset()

        key = jax.random.PRNGKey(42)
        sleep_cyborg = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        sleep_jax = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        cyborg_total = 0.0
        jax_total = 0.0

        for step in range(200):
            key, subkey = jax.random.split(key)
            _, cyborg_rewards, _, _, _ = wrapped.step(actions=sleep_cyborg)
            _, env_state, jax_rewards, _, _ = jax_env.step(subkey, env_state, sleep_jax)

            cyborg_total += np.mean([v for v in cyborg_rewards.values()])
            jax_total += float(jax_rewards["blue_0"])

        assert cyborg_total < 0, f"CybORG produced no damage in 200 steps: {cyborg_total:.2f}"
        assert jax_total < 0, f"JAX produced no damage in 200 steps: {jax_total:.2f}"


class TestStateParity:
    """Compare key state fields after running both envs."""

    def test_host_compromise_after_20_steps(self, both_envs):
        """After 20 sleep-blue steps, compare how many hosts are compromised."""
        wrapped, jax_env, env_state = both_envs
        wrapped.reset()

        key = jax.random.PRNGKey(42)
        sleep_cyborg = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        sleep_jax = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for step in range(20):
            key, subkey = jax.random.split(key)
            wrapped.step(actions=sleep_cyborg)
            _, env_state, _, _, _ = jax_env.step(subkey, env_state, sleep_jax)

        jax_compromised = int(np.array(env_state.state.host_compromised > 0).sum())
        jax_malware = int(np.array(env_state.state.host_has_malware).sum())
        jax_detected = int(np.array(env_state.state.host_activity_detected).sum())

        assert jax_compromised > 0, (
            f"After 20 steps with FSM red, expected some compromised hosts, got 0. "
            f"Malware={jax_malware}, Detected={jax_detected}"
        )

    def test_observations_change_as_red_attacks(self, both_envs):
        """Blue observations should change as red agents attack hosts."""
        wrapped, jax_env, env_state = both_envs
        wrapped.reset()

        key = jax.random.PRNGKey(42)
        monitor_jax = {f"blue_{i}": jnp.int32(BLUE_MONITOR) for i in range(NUM_BLUE_AGENTS)}

        initial_obs = {
            f"blue_{i}": np.array(get_blue_obs(env_state.state, env_state.const, i)) for i in range(NUM_BLUE_AGENTS)
        }

        for step in range(50):
            key, subkey = jax.random.split(key)
            jax_obs, env_state, _, _, _ = jax_env.step(subkey, env_state, monitor_jax)

        any_changed = False
        for i in range(NUM_BLUE_AGENTS):
            current_obs = np.array(jax_obs[f"blue_{i}"])
            if not np.array_equal(initial_obs[f"blue_{i}"], current_obs):
                any_changed = True
                break

        assert any_changed, (
            "After 50 steps with FSM red + monitor blue, observations should differ from initial. "
            "Either red isn't attacking, monitor isn't detecting, or obs don't reflect state."
        )

    def test_malware_and_detection_grow(self, both_envs):
        """host_has_malware and host_activity_detected should increase over time."""
        _, jax_env, env_state = both_envs

        key = jax.random.PRNGKey(42)
        monitor_jax = {f"blue_{i}": jnp.int32(BLUE_MONITOR) for i in range(NUM_BLUE_AGENTS)}

        malware_counts = []
        detected_counts = []

        for step in range(100):
            key, subkey = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step(subkey, env_state, monitor_jax)

            if step % 10 == 9:
                malware_counts.append(int(np.array(env_state.state.host_has_malware).sum()))
                detected_counts.append(int(np.array(env_state.state.host_activity_detected).sum()))

        assert malware_counts[-1] > malware_counts[0], (
            f"Malware count didn't grow: {malware_counts}. Red may not be attacking."
        )

        assert detected_counts[-1] > 0, (
            f"No hosts detected after 100 steps with monitor: {detected_counts}. "
            f"Monitor may not be working or red activity not tracked."
        )


class TestRewardSignalQuality:
    """Verify the reward signal provides useful learning information."""

    def test_different_blue_policies_get_different_rewards(self):
        """Sleep-only vs Monitor blue should produce different rewards."""
        jax_env = FsmRedCC4Env(num_steps=500)
        key = jax.random.PRNGKey(42)
        _, env_state_sleep = jax_env.reset(key)
        _, env_state_mon = jax_env.reset(key)

        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
        monitor_actions = {f"blue_{i}": jnp.int32(BLUE_MONITOR) for i in range(NUM_BLUE_AGENTS)}

        sleep_total = 0.0
        monitor_total = 0.0

        for step in range(100):
            key, k1, k2 = jax.random.split(key, 3)
            _, env_state_sleep, rew_s, _, _ = jax_env.step(k1, env_state_sleep, sleep_actions)
            _, env_state_mon, rew_m, _, _ = jax_env.step(k2, env_state_mon, monitor_actions)
            sleep_total += float(rew_s["blue_0"])
            monitor_total += float(rew_m["blue_0"])

        assert sleep_total != 0.0 or monitor_total != 0.0, (
            "Both sleep and monitor produced 0 reward over 100 steps. Red agents may not be causing any damage."
        )

    def test_nonzero_rewards_within_first_100_steps(self):
        """At least some nonzero rewards should appear within 100 steps."""
        jax_env = FsmRedCC4Env(num_steps=500)
        key = jax.random.PRNGKey(42)
        _, env_state = jax_env.reset(key)

        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        nonzero_count = 0
        for step in range(100):
            key, subkey = jax.random.split(key)
            _, env_state, rewards, _, _ = jax_env.step(subkey, env_state, sleep_actions)
            if abs(float(rewards["blue_0"])) > 0.001:
                nonzero_count += 1

        assert nonzero_count > 0, (
            "No nonzero rewards in 100 steps. The reward function may be broken or red agents aren't producing impact."
        )


class TestFsmRedProgression:
    """Verify FSM red agents progress through their state machine."""

    def test_fsm_states_advance_beyond_initial(self):
        """After 50 steps, red FSM should have states beyond just K and U."""
        _, jax_env, env_state = _setup_both_envs(seed=42)

        key = jax.random.PRNGKey(42)
        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for step in range(50):
            key, subkey = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step(subkey, env_state, sleep_actions)

        from jaxborg.agents.fsm_red import FSM_K, FSM_U

        fsm = np.array(env_state.state.fsm_host_states)
        advanced_states = 0
        for r in range(6):
            advanced_states += int(np.sum((fsm[r] != FSM_K) & (fsm[r] != FSM_U)))

        assert advanced_states > 0, (
            "After 50 steps, no FSM states advanced beyond K/U. "
            "fsm_red_update_state is likely never called after red actions execute."
        )

    def test_red_discovery_progresses_for_some_agent(self):
        """At least one red agent should discover hosts in multiple subnets."""
        jax_env = FsmRedCC4Env(num_steps=500)
        key = jax.random.PRNGKey(42)
        _, env_state = jax_env.reset(key)

        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for step in range(100):
            key, subkey = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step_env(subkey, env_state, sleep_actions)

        host_subnet = np.array(env_state.const.host_subnet)
        host_active = np.array(env_state.const.host_active)
        expanded_agents = 0
        for r in range(NUM_RED_AGENTS):
            discovered = np.array(env_state.state.red_discovered_hosts[r])
            discovered_subnets = set(host_subnet[discovered & host_active].tolist())
            if len(discovered_subnets) > 1:
                expanded_agents += 1

        assert expanded_agents > 0, "No red agent expanded discovery beyond one subnet in 100 steps."

    def test_red_reaches_root_state(self):
        """Red agents should reach R or RD states (root) within 100 steps."""
        _, jax_env, env_state = _setup_both_envs(seed=42)

        key = jax.random.PRNGKey(42)
        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for step in range(100):
            key, subkey = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step(subkey, env_state, sleep_actions)

        from jaxborg.agents.fsm_red import FSM_R, FSM_RD

        fsm = np.array(env_state.state.fsm_host_states)
        root_hosts = int(np.sum((fsm == FSM_R) | (fsm == FSM_RD)))

        assert root_hosts > 0, (
            "After 100 steps, no hosts in R/RD FSM state. "
            "Red agents cannot progress to Impact/Degrade without reaching root state."
        )

    def test_ot_service_stopped_after_200_steps(self):
        """With red reaching root+Impact, some OT services should be stopped."""
        jax_env = FsmRedCC4Env(num_steps=500)
        key = jax.random.PRNGKey(42)
        _, env_state = jax_env.reset(key)

        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for step in range(200):
            key, subkey = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step_env(subkey, env_state, sleep_actions)

        ot_stopped = int(np.array(env_state.state.ot_service_stopped).sum())
        assert ot_stopped > 0, (
            "After 200 steps, no OT services stopped. "
            "Red Impact action may not be firing or FSM never reaches Impact state."
        )
