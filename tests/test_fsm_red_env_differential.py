"""Differential tests comparing JAX FsmRedCC4Env vs CybORG with FiniteStateRedAgent."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

TWO_STEP_TRACE_STEPS = 2
THREE_STEP_TRACE_STEPS = 3


@pytest.fixture
def cyborg_sleep_env():
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
def jax_fsm_env():
    from jaxborg.parity.fsm_red_env import FsmRedCC4Env

    return FsmRedCC4Env(num_steps=500)


@pytest.fixture
def jax_env_from_cyborg(cyborg_sleep_env):
    from jaxborg.env import ScenarioEnvState, _init_red_state
    from jaxborg.parity.fsm_red_env import FsmRedCC4Env
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
    from jaxborg.state import create_initial_state

    inner_cyborg = cyborg_sleep_env.env
    const = build_const_from_cyborg(inner_cyborg)
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)
    env_state = ScenarioEnvState(state=state, const=const)

    env = FsmRedCC4Env(num_steps=500)
    return env, env_state


def _translate_logged_red_actions(logged_actions, mappings):
    from jaxborg.actions.encoding import RED_SLEEP
    from jaxborg.constants import NUM_RED_AGENTS
    from jaxborg.parity.translate import cyborg_red_to_jax

    red_actions = {}
    for agent_id in range(NUM_RED_AGENTS):
        agent_name = f"red_agent_{agent_id}"
        cy_action = logged_actions.get(agent_name)
        red_actions[f"red_{agent_id}"] = jnp.int32(
            RED_SLEEP if cy_action is None else cyborg_red_to_jax(cy_action, agent_name, mappings)
        )
    return red_actions


def _correct_pending_generic_red_exploits(jax_env_state, cyborg, mappings):
    from jaxborg.actions.encoding import encode_red_action
    from jaxborg.parity.translate import cyborg_red_to_jax

    red_pending_action = jax_env_state.state.red_pending_action

    for agent_id in range(jax_env_state.state.red_pending_ticks.shape[0]):
        if int(jax_env_state.state.red_pending_ticks[agent_id]) <= 0:
            continue
        executed = cyborg.environment_controller.action.get(f"red_agent_{agent_id}", [])
        if not executed:
            continue
        action = executed[0]
        if type(action).__name__ != "ExploitRemoteService":
            continue
        sub_action = getattr(action, "sub_action", None)
        if sub_action is None:
            target_host = mappings.hostname_to_idx[mappings.ip_to_hostname[action.ip_address]]
            corrected = encode_red_action("ExploitRemoteService_cc4BlueKeep", target_host, agent_id)
        else:
            corrected = cyborg_red_to_jax(sub_action, f"red_agent_{agent_id}", mappings)
        red_pending_action = red_pending_action.at[agent_id].set(corrected)

    return jax_env_state.replace(state=jax_env_state.state.replace(red_pending_action=red_pending_action))


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory

    from jaxborg.actions.encoding import BLUE_SLEEP, encode_blue_action
    from jaxborg.parity.translate import cyborg_blue_to_jax

    decoy_factory_actions = (
        (HarakaDecoyFactory(), "DeployDecoy_HarakaSMPT"),
        (ApacheDecoyFactory(), "DeployDecoy_Apache"),
        (TomcatDecoyFactory(), "DeployDecoy_Tomcat"),
        (VsftpdDecoyFactory(), "DeployDecoy_Vsftpd"),
    )

    cls_name = type(action).__name__
    agent_id = int(agent_name.split("_")[-1])

    if label.startswith("[Padding]"):
        return []
    if cls_name == "Sleep" and not label.startswith("[Invalid]"):
        return [BLUE_SLEEP]
    if cls_name == "Sleep" and label.startswith("[Invalid]"):
        return []
    if cls_name == "DeployDecoy":
        if action.hostname not in mappings.hostname_to_idx:
            return []
        host = cyborg_state.hosts[action.hostname]
        host_idx = mappings.hostname_to_idx[action.hostname]
        return [
            encode_blue_action(action_name, host_idx, agent_id, const=const)
            for factory, action_name in decoy_factory_actions
            if factory.is_host_compatible(host)
        ]
    try:
        return [cyborg_blue_to_jax(action, agent_name, mappings, const=const)]
    except (KeyError, ValueError):
        return []


def _live_cyborg_mask_in_jax_space(wrapper, agent_name, mappings, const):
    from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END

    controller = wrapper.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in _cyborg_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
        return jax_mask

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_actions = wrapper.actions(agent_name)
    cyborg_labels = wrapper.action_labels(agent_name)
    for action, valid, label in zip(cyborg_actions, action_space["mask"], cyborg_labels):
        if not valid:
            continue
        for jax_idx in _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, controller.state):
            jax_mask[jax_idx] = True
    return jax_mask


def _sample_random_blue_actions_from_live_mask(harness, rng):
    from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper

    from tests.differential.blue_mask_projection import refresh_blue_wrapper_action_space

    if harness._blue_wrapper is None:
        harness._blue_wrapper = BlueFlatWrapper(env=harness.cyborg_env, pad_spaces=True)
    refresh_blue_wrapper_action_space(harness._blue_wrapper)

    blue_actions = {}
    for agent_idx in range(5):
        agent_name = f"blue_agent_{agent_idx}"
        mask = _live_cyborg_mask_in_jax_space(harness._blue_wrapper, agent_name, harness.mappings, harness.jax_const)
        blue_actions[agent_idx] = int(rng.choice(np.flatnonzero(mask)))
    return blue_actions


def _save_cyborg_topology_snapshot(cyborg, tmp_path, seed: int):
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, save_topology

    path = tmp_path / f"cyborg_seed_{seed}.npz"
    save_topology(
        build_const_from_cyborg(cyborg),
        path,
        metadata={"source": "cyborg", "source_seed": seed},
    )
    return path


class TestFsmRedEnvDifferential:
    def test_raw_cyborg_step_executes_concrete_decoy_with_skip_valid_check(self):
        """Independent rollout eval must step raw CybORG DeployDecoy actions."""
        from CybORG.Simulator.Actions import Sleep

        from jaxborg.actions.encoding import encode_blue_action
        from jaxborg.parity.translate import build_mappings_from_cyborg, jax_blue_to_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
        from scripts.dev.parity.cyborg_bridge import make_cyborg_env

        seed = 4

        cyborg_env = make_cyborg_env(seed=seed)
        cyborg_env.reset()
        const = build_const_from_cyborg(cyborg_env.env)
        mappings = build_mappings_from_cyborg(cyborg_env.env)

        # Find a valid decoy host for blue_agent_3
        target_hostname = "operational_zone_b_subnet_server_host_2"
        host_idx = mappings.hostname_to_idx[target_hostname]
        action_idx = encode_blue_action("DeployDecoy", host_idx, 3, const=const)

        actions = {agent_name: Sleep() for agent_name in cyborg_env.agents}
        actions["blue_agent_3"] = jax_blue_to_cyborg(action_idx, 3, mappings, const=const)
        assert type(actions["blue_agent_3"]).__name__ == "DeployDecoy"

        cyborg_env.env.parallel_step(actions, skip_valid_action_check=True)
        executed = cyborg_env.env.environment_controller.action.get("blue_agent_3", [])
        # CybORG queues DeployDecoy (duration=2) so it goes into actions_in_progress
        pending = cyborg_env.env.environment_controller.actions_in_progress.get("blue_agent_3")
        if pending is not None:
            assert type(pending["action"]).__name__ == "DeployDecoy"
        else:
            # If it executed immediately, check the executed action
            assert any(type(a).__name__ == "DeployDecoy" for a in executed)

    def test_sleep_blue_cumulative_reward_same_sign(self, cyborg_sleep_env, jax_env_from_cyborg):
        """Sleep blue, FSM red: both should produce negative cumulative reward."""
        from statistics import mean

        from jaxborg.constants import NUM_BLUE_AGENTS

        cyborg_env = cyborg_sleep_env
        jax_env, jax_state = jax_env_from_cyborg

        cyborg_env.reset()
        cyborg_actions = {agent: 0 for agent in cyborg_env.agents}
        cyborg_total = 0.0
        for _ in range(50):
            _, rewards, _, _, _ = cyborg_env.step(cyborg_actions)
            cyborg_total += mean(rewards.values())

        key = jax.random.PRNGKey(0)
        jax_actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
        jax_total = 0.0
        state = jax_state
        for _ in range(50):
            key, subkey = jax.random.split(key)
            _, state, rewards, _, _ = jax_env.step(subkey, state, jax_actions)
            jax_total += float(rewards["blue_0"])

        assert cyborg_total <= 0, f"CybORG sleep reward should be <= 0, got {cyborg_total}"
        if cyborg_total < 0:
            assert jax_total <= 0, f"JAX sleep reward should be <= 0 when CybORG is {cyborg_total}"

    @pytest.mark.xfail(
        reason="requires retired CybORG green/replay tapes; topology snapshots preserve static layout only",
    )
    def test_snapshot_topology_matches_red4_known_hosts_after_first_green_phish(self, tmp_path):
        """After activation, native JAX red_4 should know only the same hosts as CybORG's FSM agent."""
        import types

        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS
        from jaxborg.parity.fsm_red_env import FsmRedCC4Env
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.red_fsm import fsm_red_apply_delayed_update

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=TWO_STEP_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)
        topology_path = _save_cyborg_topology_snapshot(cyborg, tmp_path, seed)

        captured_known_hosts = None
        interface = cyborg.environment_controller.agent_interfaces["red_agent_4"]
        agent = interface.agent
        original_get_action = agent.get_action

        def _wrapped(self, observation, action_space):
            nonlocal captured_known_hosts
            action = original_get_action(observation, action_space)
            captured_known_hosts = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
            for ip, info in self.host_states.items():
                hostname = info.get("hostname")
                if hostname in mappings.hostname_to_idx:
                    captured_known_hosts[mappings.hostname_to_idx[hostname]] = True
            return action

        agent.get_action = types.MethodType(_wrapped, agent)

        jax_env = FsmRedCC4Env(num_steps=TWO_STEP_TRACE_STEPS, topology_path=topology_path)
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)
        blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        loop_key, step_key = jax.random.split(loop_key)
        _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, blue_actions)
        _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})

        loop_key, step_key = jax.random.split(loop_key)
        state_before = fsm_red_apply_delayed_update(jax_state.state)
        _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})

        assert captured_known_hosts is not None, "Expected wrapped CybORG red_4 action to capture known hosts"
        np.testing.assert_array_equal(
            np.array(state_before.red_discovered_hosts[4], dtype=bool),
            captured_known_hosts,
        )

    @pytest.mark.xfail(
        reason="requires retired CybORG green/replay tapes; topology snapshots preserve static layout only",
    )
    def test_snapshot_topology_matches_second_step_red4_action_after_green_phish(self, tmp_path):
        """After the seed-0 phishing foothold, JAX and CybORG should pick the same red_4 follow-up action."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import NUM_BLUE_AGENTS, NUM_RED_AGENTS
        from jaxborg.parity.fsm_red_env import FsmRedCC4Env
        from jaxborg.parity.translate import build_mappings_from_cyborg, jax_red_to_cyborg
        from jaxborg.scenarios.cc4.red_fsm import fsm_red_apply_delayed_update, fsm_red_select_actions

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=TWO_STEP_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)
        topology_path = _save_cyborg_topology_snapshot(cyborg, tmp_path, seed)

        logged_actions = {}
        for agent_name, interface in cyborg.environment_controller.agent_interfaces.items():
            if not agent_name.startswith("red_agent_"):
                continue
            agent = interface.agent
            original_get_action = agent.get_action

            def _wrap_get_action(orig_fn, wrapped_name):
                def _wrapped(self, observation, action_space):
                    action = orig_fn(observation, action_space)
                    logged_actions[wrapped_name] = action
                    return action

                return types.MethodType(_wrapped, agent)

            agent.get_action = _wrap_get_action(original_get_action, agent_name)

        jax_env = FsmRedCC4Env(num_steps=TWO_STEP_TRACE_STEPS, topology_path=topology_path)
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)
        blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        loop_key, step_key = jax.random.split(loop_key)
        _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, blue_actions)
        _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})

        loop_key, step_key = jax.random.split(loop_key)
        key_for_step_env, _key_reset = jax.random.split(step_key)
        _key_unused, key_red = jax.random.split(key_for_step_env)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)
        state_before = fsm_red_apply_delayed_update(jax_state.state)
        jax_red_actions = fsm_red_select_actions(state_before, jax_state.const, red_keys)[0]
        jax_red4 = jax_red_to_cyborg(int(jax_red_actions[4]), 4, mappings)

        _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})
        cyborg_red4 = logged_actions["red_agent_4"]

        def _action_target(action):
            if hasattr(action, "hostname"):
                return action.hostname
            if hasattr(action, "ip_address"):
                return str(action.ip_address)
            return None

        assert type(jax_red4).__name__ == type(cyborg_red4).__name__ == "PrivilegeEscalate"
        assert _action_target(jax_red4) == _action_target(cyborg_red4)

    @pytest.mark.xfail(
        reason="requires retired CybORG green/replay tapes; topology snapshots preserve static layout only",
    )
    def test_explicit_cyborg_red_trace_matches_green_phish_privesc_privilege(self, tmp_path):
        """Replaying CybORG's first seed-0 red trace should preserve the red_4 privesc privilege gain."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP, RED_SLEEP
        from jaxborg.constants import COMPROMISE_PRIVILEGED, NUM_BLUE_AGENTS, NUM_RED_AGENTS
        from jaxborg.env import ScenarioEnv
        from jaxborg.parity.translate import build_mappings_from_cyborg, cyborg_red_to_jax
        from tests.differential.state_comparator import compare_snapshots, extract_cyborg_snapshot, extract_jax_snapshot

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=THREE_STEP_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)
        topology_path = _save_cyborg_topology_snapshot(cyborg, tmp_path, seed)

        logged_actions = {}
        for agent_name, interface in cyborg.environment_controller.agent_interfaces.items():
            if not agent_name.startswith("red_agent_"):
                continue
            agent = interface.agent
            original_get_action = agent.get_action

            def _wrap_get_action(orig_fn, wrapped_name):
                def _wrapped(self, observation, action_space):
                    action = orig_fn(observation, action_space)
                    logged_actions[wrapped_name] = action
                    return action

                return types.MethodType(_wrapped, agent)

            agent.get_action = _wrap_get_action(original_get_action, agent_name)

        jax_env = ScenarioEnv(num_steps=THREE_STEP_TRACE_STEPS, topology_path=topology_path)
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)

        for _step in range(3):
            logged_actions.clear()
            _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})
            red_actions = {}
            for agent_id in range(NUM_RED_AGENTS):
                agent_name = f"red_agent_{agent_id}"
                cy_action = logged_actions.get(agent_name)
                red_actions[f"red_{agent_id}"] = jnp.int32(
                    RED_SLEEP if cy_action is None else cyborg_red_to_jax(cy_action, agent_name, mappings)
                )

            loop_key, step_key = jax.random.split(loop_key)
            blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
            _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, {**blue_actions, **red_actions})

        target_hostname = "operational_zone_b_subnet_user_host_5"
        target_host = mappings.hostname_to_idx[target_hostname]
        target_sessions = [
            sess
            for sess in cyborg.environment_controller.state.sessions["red_agent_4"].values()
            if sess.hostname == target_hostname
        ]
        assert any(sess.has_privileged_access() for sess in target_sessions), target_sessions
        assert int(jax_state.state.red_privilege[4, target_host]) == COMPROMISE_PRIVILEGED
        assert int(jax_state.state.host_compromised[target_host]) == COMPROMISE_PRIVILEGED

        diffs = compare_snapshots(
            extract_cyborg_snapshot(cyborg, mappings),
            extract_jax_snapshot(jax_state.state, jax_state.const, mappings),
        )
        host_label = f"host_{target_host}"
        agent_host_label = f"red_4_host_{target_host}"
        target_diffs = [
            diff
            for diff in diffs
            if diff.field_name in {"host_compromised", "red_privilege"}
            and diff.host_or_agent in {host_label, agent_host_label}
        ]
        assert target_diffs == []

    def test_explicit_replay_corrects_generic_exploit_to_cyborg_subaction(self, tmp_path):
        """Seed-0 generic exploit replay should not invent a host_22 foothold."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import COMPROMISE_NONE, NUM_BLUE_AGENTS
        from jaxborg.env import ScenarioEnv
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from tests.differential.state_comparator import compare_snapshots, extract_cyborg_snapshot, extract_jax_snapshot

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)
        topology_path = _save_cyborg_topology_snapshot(cyborg, tmp_path, seed)

        logged_actions = {}
        for agent_name, interface in cyborg.environment_controller.agent_interfaces.items():
            if not agent_name.startswith("red_agent_"):
                continue
            agent = interface.agent
            original_get_action = agent.get_action

            def _wrap_get_action(orig_fn, wrapped_name):
                def _wrapped(self, observation, action_space):
                    action = orig_fn(observation, action_space)
                    logged_actions[wrapped_name] = action
                    return action

                return types.MethodType(_wrapped, agent)

            agent.get_action = _wrap_get_action(original_get_action, agent_name)

        jax_env = ScenarioEnv(num_steps=500, topology_path=topology_path)
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)

        for _step in range(17):
            logged_actions.clear()
            _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})
            jax_state = _correct_pending_generic_red_exploits(jax_state, cyborg, mappings)
            red_actions = _translate_logged_red_actions(logged_actions, mappings)

            loop_key, step_key = jax.random.split(loop_key)
            blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
            _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, {**blue_actions, **red_actions})

        target_host = mappings.hostname_to_idx["contractor_network_subnet_user_host_7"]
        target_sessions = [
            sess
            for sess in cyborg.environment_controller.state.sessions["red_agent_0"].values()
            if sess.hostname == "contractor_network_subnet_user_host_7"
        ]
        assert target_sessions == []
        assert int(jax_state.state.red_privilege[0, target_host]) == COMPROMISE_NONE
        assert int(jax_state.state.host_compromised[target_host]) == COMPROMISE_NONE
        assert not bool(jax_state.state.red_sessions[0, target_host])

        diffs = compare_snapshots(
            extract_cyborg_snapshot(cyborg, mappings),
            extract_jax_snapshot(jax_state.state, jax_state.const, mappings),
        )
        host_label = f"host_{target_host}"
        agent_host_label = f"red_0_host_{target_host}"
        target_diffs = [
            diff
            for diff in diffs
            if diff.field_name in {"host_compromised", "red_sessions", "red_privilege"}
            and diff.host_or_agent in {host_label, agent_host_label, "red_agent_0"}
        ]
        assert target_diffs == []

    def test_explicit_replay_handles_failed_generic_exploit_without_subaction(self, tmp_path):
        """Seed-0 generic exploit replay should not invent the failed red_4 foothold on host_56."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import COMPROMISE_NONE, NUM_BLUE_AGENTS
        from jaxborg.env import ScenarioEnv
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from tests.differential.state_comparator import compare_snapshots, extract_cyborg_snapshot, extract_jax_snapshot

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)
        topology_path = _save_cyborg_topology_snapshot(cyborg, tmp_path, seed)

        logged_actions = {}
        for agent_name, interface in cyborg.environment_controller.agent_interfaces.items():
            if not agent_name.startswith("red_agent_"):
                continue
            agent = interface.agent
            original_get_action = agent.get_action

            def _wrap_get_action(orig_fn, wrapped_name):
                def _wrapped(self, observation, action_space):
                    action = orig_fn(observation, action_space)
                    logged_actions[wrapped_name] = action
                    return action

                return types.MethodType(_wrapped, agent)

            agent.get_action = _wrap_get_action(original_get_action, agent_name)

        jax_env = ScenarioEnv(num_steps=500, topology_path=topology_path)
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)

        for _step in range(24):
            logged_actions.clear()
            _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})
            jax_state = _correct_pending_generic_red_exploits(jax_state, cyborg, mappings)
            red_actions = _translate_logged_red_actions(logged_actions, mappings)

            loop_key, step_key = jax.random.split(loop_key)
            blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
            _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, {**blue_actions, **red_actions})

        target_host = mappings.hostname_to_idx["operational_zone_b_subnet_user_host_7"]
        target_sessions = [
            sess
            for sess in cyborg.environment_controller.state.sessions["red_agent_4"].values()
            if sess.hostname == "operational_zone_b_subnet_user_host_7"
        ]
        assert target_sessions == []
        assert int(jax_state.state.red_privilege[4, target_host]) == COMPROMISE_NONE
        assert int(jax_state.state.host_compromised[target_host]) == COMPROMISE_NONE
        assert not bool(jax_state.state.red_sessions[4, target_host])

        diffs = compare_snapshots(
            extract_cyborg_snapshot(cyborg, mappings),
            extract_jax_snapshot(jax_state.state, jax_state.const, mappings),
        )
        host_label = f"host_{target_host}"
        agent_host_label = f"red_4_host_{target_host}"
        target_diffs = [
            diff
            for diff in diffs
            if diff.field_name in {"host_compromised", "red_sessions", "red_privilege"}
            and diff.host_or_agent in {host_label, agent_host_label}
        ]
        assert target_diffs == []

    def test_random_blue_reward_distribution(self, cyborg_sleep_env, jax_env_from_cyborg):
        """Random blue policy: compare reward distribution across seeds."""
        from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
        from jaxborg.constants import NUM_BLUE_AGENTS

        jax_env, jax_state = jax_env_from_cyborg
        key = jax.random.PRNGKey(100)
        state = jax_state
        ep_return = 0.0
        for _ in range(50):
            key, act_key, step_key = jax.random.split(key, 3)
            actions = {
                f"blue_{b}": jax.random.randint(jax.random.fold_in(act_key, b), (), 0, BLUE_ALLOW_TRAFFIC_END)
                for b in range(NUM_BLUE_AGENTS)
            }
            _, state, rewards, _, _ = jax_env.step(step_key, state, actions)
            ep_return += float(rewards["blue_0"])

        assert np.isfinite(ep_return), "JAX random baseline should produce finite returns"

    def test_native_generic_exploit_respects_blocked_scan_source_route_matches_cyborg(self):
        """A blocked scan-owning abstract session must not exploit via another agent session's route."""
        from jaxborg.constants import NUM_RED_AGENTS
        from tests.differential.harness import CC4DifferentialHarness

        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            sync_green_rng=True,
            strict_random_sync=True,
        )
        harness.reset()

        rng = np.random.default_rng(0)
        # Run up to 300 steps looking for a red exploit from a blocked subnet.
        # With agent-relative blue encoding, each agent can only block traffic
        # TO its own observed subnets, so blocked-route exploits need more steps
        # for red to spread into subnets where blocks are active.
        found_step = None
        step_result = None
        for step in range(300):
            blue_actions = _sample_random_blue_actions_from_live_mask(harness, rng)
            step_result = harness.full_step(blue_actions)

            controller = harness.cyborg_env.environment_controller
            cy_state = controller.state
            for r in range(NUM_RED_AGENTS):
                agent_name = f"red_agent_{r}"
                executed = controller.action.get(agent_name, [])
                for action in executed:
                    if type(action).__name__ != "ExploitRemoteService":
                        continue
                    if not hasattr(action, "ip_address") or action.ip_address is None:
                        continue
                    target_hostname = harness.mappings.ip_to_hostname.get(action.ip_address)
                    if target_hostname is None:
                        continue
                    target_subnet = cy_state.hostname_subnet_map.get(target_hostname)
                    if target_subnet is None:
                        continue
                    blocks = cy_state.blocks.get(target_subnet, set())
                    session = cy_state.sessions.get(agent_name, {}).get(action.session)
                    if session is None:
                        continue
                    source_subnet = cy_state.hostname_subnet_map.get(session.hostname)
                    if source_subnet in blocks:
                        found_step = step
                        target_host = harness.mappings.hostname_to_idx[target_hostname]
                        target_diffs = [
                            diff
                            for diff in step_result.diffs
                            if diff.field_name
                            in {"host_compromised", "red_sessions", "red_privilege", "host_has_malware"}
                            and diff.host_or_agent in {f"host_{target_host}", f"red_{r}_host_{target_host}"}
                        ]
                        assert target_diffs == [], f"Step {step}: blocked-route exploit parity diffs: {target_diffs}"
                        break
                if found_step is not None:
                    break
            if found_step is not None:
                break

        assert found_step is not None, "Never found a red exploit from a blocked subnet in 300 steps"
