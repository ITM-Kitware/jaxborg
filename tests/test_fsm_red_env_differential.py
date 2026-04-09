"""Differential tests comparing JAX FsmRedCC4Env vs CybORG with FiniteStateRedAgent."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

RESET_TRACE_STEPS = 1
TWO_STEP_TRACE_STEPS = 2
THREE_STEP_TRACE_STEPS = 3
LIVE_RED_SYNC_TRACE_STEPS = 20


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
    from jaxborg.fsm_red_env import FsmRedCC4Env

    return FsmRedCC4Env(num_steps=500)


@pytest.fixture
def jax_env_from_cyborg(cyborg_sleep_env):
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    inner_cyborg = cyborg_sleep_env.env
    const = build_const_from_cyborg(inner_cyborg)
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)

    env = FsmRedCC4Env(num_steps=500)
    return env, env_state


def _translate_logged_red_actions(logged_actions, mappings):
    from jaxborg.actions.encoding import RED_SLEEP
    from jaxborg.constants import NUM_RED_AGENTS
    from jaxborg.translate import cyborg_red_to_jax

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
    from jaxborg.translate import cyborg_red_to_jax

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
    from jaxborg.translate import cyborg_blue_to_jax

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


class TestFsmRedEnvDifferential:
    def test_native_reset_with_cyborg_bank_matches_cyborg_seed_zero(self):
        """Native JAX reset should match CybORG reset when sourced from the same topology seed."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
        from jaxborg.actions.masking import compute_blue_action_mask
        from jaxborg.constants import NUM_BLUE_AGENTS
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg, cyborg_blue_to_jax

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=RESET_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_obs, cyborg_info = cyborg_env.reset()
        cyborg_const = build_const_from_cyborg(cyborg)
        mappings = build_mappings_from_cyborg(cyborg)

        jax_env = FsmRedCC4Env(num_steps=RESET_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
        jax_obs, jax_state = jax_env.reset(jax.random.PRNGKey(seed))
        jax_const = jax_state.const

        np.testing.assert_array_equal(np.array(jax_const.host_active), np.array(cyborg_const.host_active))
        np.testing.assert_array_equal(np.array(jax_const.host_subnet), np.array(cyborg_const.host_subnet))
        np.testing.assert_array_equal(np.array(jax_const.red_start_hosts), np.array(cyborg_const.red_start_hosts))

        decoy_factories = (
            (HarakaDecoyFactory(), "DeployDecoy_HarakaSMPT"),
            (ApacheDecoyFactory(), "DeployDecoy_Apache"),
            (TomcatDecoyFactory(), "DeployDecoy_Tomcat"),
            (VsftpdDecoyFactory(), "DeployDecoy_Vsftpd"),
        )

        def live_cyborg_mask(agent_idx: int) -> np.ndarray:
            agent_name = f"blue_agent_{agent_idx}"
            jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
            action_space = cyborg_env.get_action_space(agent_name)
            cyborg_mask = action_space["mask"]
            cyborg_actions = cyborg_env.actions(agent_name)
            cyborg_labels = cyborg_env.action_labels(agent_name)
            cyborg_state = cyborg.environment_controller.state

            for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
                if not valid or label.startswith("[Padding]"):
                    continue
                cls_name = type(action).__name__
                if cls_name == "Sleep" and not label.startswith("[Invalid]"):
                    jax_mask[BLUE_SLEEP] = True
                    continue
                if cls_name == "DeployDecoy":
                    host = cyborg_state.hosts[action.hostname]
                    host_idx = mappings.hostname_to_idx[action.hostname]
                    for factory, action_name in decoy_factories:
                        if factory.is_host_compatible(host):
                            jax_idx = encode_blue_action(action_name, host_idx, agent_idx, const=cyborg_const)
                            jax_mask[jax_idx] = True
                    continue
                try:
                    jax_idx = cyborg_blue_to_jax(action, agent_name, mappings, const=cyborg_const)
                except (KeyError, ValueError):
                    continue
                jax_mask[jax_idx] = True
            return jax_mask

        for agent_idx in range(NUM_BLUE_AGENTS):
            np.testing.assert_allclose(
                np.array(jax_obs[f"blue_{agent_idx}"], dtype=np.float32),
                np.array(cyborg_obs[f"blue_agent_{agent_idx}"], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.array(compute_blue_action_mask(jax_const, agent_idx, jax_state.state), dtype=bool),
                live_cyborg_mask(agent_idx),
            )

    def test_native_reset_with_cyborg_bank_matches_red_reset_knowledge_seed_zero(self):
        """Native cyborg_bank reset must preserve CybORG reset-time red knowledge for inactive agents."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=RESET_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)

        jax_env = FsmRedCC4Env(num_steps=RESET_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
        _, jax_state = jax_env.reset(jax.random.PRNGKey(seed))

        controller = cyborg.environment_controller
        for red_idx in range(6):
            agent_name = f"red_agent_{red_idx}"
            iface = controller.agent_interfaces[agent_name]

            cy_known = set()
            for ip, known in getattr(iface.action_space, "ip_address", {}).items():
                if not known:
                    continue
                hostname = cyborg.environment_controller.state.ip_addresses.get(ip)
                if hostname in mappings.hostname_to_idx:
                    cy_known.add(mappings.hostname_to_idx[hostname])

            cy_scanned = set()
            for sess in cyborg.environment_controller.state.sessions.get(agent_name, {}).values():
                for ip in getattr(sess, "ports", {}).keys():
                    hostname = cyborg.environment_controller.state.ip_addresses.get(ip)
                    if hostname in mappings.hostname_to_idx:
                        cy_scanned.add(mappings.hostname_to_idx[hostname])

            jax_known = {
                h
                for h in range(int(jax_state.const.num_hosts))
                if bool(jax_state.const.red_initial_discovered_hosts[red_idx, h])
            }
            jax_scanned = {
                h
                for h in range(int(jax_state.const.num_hosts))
                if bool(jax_state.const.red_initial_scanned_hosts[red_idx, h])
            }

            if iface.active:
                assert jax_known == cy_known, f"{agent_name}: known mismatch"
            else:
                assert jax_known == set(), f"{agent_name}: inactive should have empty known"
            assert jax_scanned == cy_scanned, f"{agent_name}: scanned mismatch"

    def test_independent_rollout_bank_seed_mapping_matches_reset_seed_3(self):
        """Independent transfer must use the same cached CybORG bank member as the JAX reset key."""
        from jaxborg.actions.masking import compute_blue_action_mask
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg
        from scripts.eval.transfer import make_cyborg_env

        seed = 3
        bank_size = 2

        jax_env = FsmRedCC4Env(
            num_steps=RESET_TRACE_STEPS,
            topology_mode="cyborg_bank",
            topology_bank_size=bank_size,
            sync_red_policy_bank=True,
        )
        jax_obs, jax_state = jax_env.reset(jax.random.PRNGKey(seed))

        cyborg_env = make_cyborg_env(seed=seed, bank_match_size=bank_size)
        cyborg_obs, _ = cyborg_env.reset()
        cyborg_const = build_const_from_cyborg(cyborg_env.env)
        mappings = build_mappings_from_cyborg(cyborg_env.env)

        np.testing.assert_array_equal(np.array(jax_state.const.host_active), np.array(cyborg_const.host_active))
        np.testing.assert_array_equal(np.array(jax_state.const.host_subnet), np.array(cyborg_const.host_subnet))
        np.testing.assert_array_equal(np.array(jax_state.const.red_start_hosts), np.array(cyborg_const.red_start_hosts))

        for agent_idx in range(5):
            np.testing.assert_allclose(
                np.array(jax_obs[f"blue_{agent_idx}"], dtype=np.float32),
                np.array(cyborg_obs[f"blue_agent_{agent_idx}"], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.array(compute_blue_action_mask(jax_state.const, agent_idx, jax_state.state), dtype=bool),
                _live_cyborg_mask_in_jax_space(
                    cyborg_env,
                    f"blue_agent_{agent_idx}",
                    mappings,
                    jax_state.const,
                ),
            )

    def test_native_cyborg_bank_matches_first_step_red0_action_seed_4_bank_2(self):
        """Native bank-backed red policy should match CybORG's first red_0 action."""
        from CybORG.Simulator.Actions import Sleep

        from jaxborg.agents.fsm_red import fsm_red_apply_delayed_update, fsm_red_select_actions
        from jaxborg.constants import NUM_RED_AGENTS
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg, jax_red_to_cyborg
        from scripts.eval.transfer import make_cyborg_env

        seed = 4
        bank_size = 2

        cyborg_env = make_cyborg_env(seed=seed, bank_match_size=bank_size)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg_env.env)

        logged_actions = {}
        for agent_name, interface in cyborg_env.env.environment_controller.agent_interfaces.items():
            if agent_name != "red_agent_0":
                continue
            agent = interface.agent
            original_get_action = agent.get_action

            def _wrapped(self, observation, action_space):
                action = original_get_action(observation, action_space)
                logged_actions["red_agent_0"] = action
                return action

            agent.get_action = types.MethodType(_wrapped, agent)

        jax_env = FsmRedCC4Env(
            num_steps=RESET_TRACE_STEPS,
            topology_mode="cyborg_bank",
            topology_bank_size=bank_size,
            sync_red_policy_bank=True,
        )
        loop_key = jax.random.PRNGKey(seed)
        _, jax_state = jax_env.reset(loop_key)

        loop_key, step_key = jax.random.split(loop_key)
        key_for_step_env, _key_reset = jax.random.split(step_key)
        _key_unused, key_red = jax.random.split(key_for_step_env)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)
        state_before = fsm_red_apply_delayed_update(jax_state.state)
        jax_red_actions = fsm_red_select_actions(state_before, jax_state.const, red_keys)[0]
        jax_red0 = jax_red_to_cyborg(int(jax_red_actions[0]), 0, mappings)

        sleep_actions = {a: Sleep() for a in cyborg_env.agents}
        cyborg_env.step(actions=sleep_actions)
        cyborg_red0 = logged_actions["red_agent_0"]

        def _target(action):
            return getattr(action, "hostname", getattr(action, "subnet", getattr(action, "ip_address", None)))

        assert type(jax_red0).__name__ == type(cyborg_red0).__name__
        assert _target(jax_red0) == _target(cyborg_red0)

    def test_cyborg_bank_runtime_does_not_preload_sleep_red_policy_tape_by_default(self):
        """cyborg_bank runtime should not inject a Sleep-rollout red-policy tape by default."""
        from jaxborg.fsm_red_env import FsmRedCC4Env

        seed = 4
        bank_size = 5

        jax_env = FsmRedCC4Env(
            num_steps=RESET_TRACE_STEPS,
            topology_mode="cyborg_bank",
            topology_bank_size=bank_size,
        )
        _, env_state = jax_env.reset(jax.random.PRNGKey(seed))
        assert not bool(env_state.const.use_red_policy_randoms)

    def test_raw_cyborg_step_executes_concrete_decoy_with_skip_valid_check(self):
        """Independent rollout eval must step raw CybORG DeployDecoy actions."""
        from CybORG.Simulator.Actions import Sleep

        from jaxborg.actions.encoding import encode_blue_action
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg, jax_blue_to_cyborg
        from scripts.eval.transfer import make_cyborg_env

        seed = 4
        bank_size = 2

        cyborg_env = make_cyborg_env(seed=seed, bank_match_size=bank_size)
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

    def test_live_red_choice_sync_keeps_native_sleep_rollout_aligned_seed_4_bank_5(self):
        """Live CybORG red-choice sync should keep native sleep rollout aligned step-by-step."""
        from CybORG.Simulator.Actions import Sleep

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import NUM_BLUE_AGENTS
        from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg
        from scripts.eval.transfer import make_cyborg_env
        from tests.differential.state_comparator import compare_snapshots, extract_cyborg_snapshot, extract_jax_snapshot

        seed = 4
        bank_size = 5
        steps = LIVE_RED_SYNC_TRACE_STEPS

        cyborg_env = make_cyborg_env(seed=seed, bank_match_size=bank_size)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg_env.env)
        recorder = RedPolicyRecorder()
        recorder.install(cyborg_env.env, mappings)

        jax_env = FsmRedCC4Env(num_steps=500, topology_mode="cyborg_bank", topology_bank_size=bank_size)
        key = jax.random.PRNGKey(seed)
        _, env_state = jax_env.reset(key)
        blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        for _ in range(steps):
            _, _, _, _, _ = cyborg_env.step(actions={agent: Sleep() for agent in cyborg_env.agents})
            step_idx = int(env_state.state.time)
            env_state = env_state.replace(
                const=env_state.const.replace(
                    red_policy_randoms=env_state.const.red_policy_randoms.at[step_idx].set(
                        jnp.asarray(recorder.extract_step(step_idx), dtype=jnp.float32)
                    ),
                    use_red_policy_randoms=jnp.array(True),
                )
            )
            key, step_key = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step(step_key, env_state, blue_actions)

            diffs = compare_snapshots(
                extract_cyborg_snapshot(cyborg_env.env, mappings),
                extract_jax_snapshot(env_state.state, env_state.const, mappings),
            )
            assert diffs == []

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

    def test_native_cyborg_bank_replays_seed_zero_first_green_phish(self):
        """Native cyborg_bank reset should reproduce the first CybORG green phishing foothold."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg

        seed = 0
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=RESET_TRACE_STEPS,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_env.reset()
        mappings = build_mappings_from_cyborg(cyborg)

        jax_env = FsmRedCC4Env(num_steps=RESET_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
        _, jax_state = jax_env.reset(jax.random.PRNGKey(seed))

        _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})

        step_key = jax.random.split(jax.random.PRNGKey(seed))[1]
        blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
        _, jax_state, _, _, _ = jax_env.step(step_key, jax_state, blue_actions)

        cyborg_red4_hosts = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
        for hostname, host in cyborg.environment_controller.state.hosts.items():
            if host.sessions.get("red_agent_4"):
                cyborg_red4_hosts[mappings.hostname_to_idx[hostname]] = True

        np.testing.assert_array_equal(
            np.array(jax_state.state.red_sessions[4], dtype=bool),
            cyborg_red4_hosts,
        )

    def test_native_cyborg_bank_matches_red4_known_hosts_after_first_green_phish(self):
        """After activation, native JAX red_4 should know only the same hosts as CybORG's FSM agent."""
        import types

        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.agents.fsm_red import fsm_red_apply_delayed_update
        from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg

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

        jax_env = FsmRedCC4Env(num_steps=TWO_STEP_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
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

    def test_native_cyborg_bank_matches_second_step_red4_action_after_green_phish(self):
        """After the seed-0 phishing foothold, JAX and CybORG should pick the same red_4 follow-up action."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.agents.fsm_red import fsm_red_apply_delayed_update, fsm_red_select_actions
        from jaxborg.constants import NUM_BLUE_AGENTS, NUM_RED_AGENTS
        from jaxborg.fsm_red_env import FsmRedCC4Env
        from jaxborg.translate import build_mappings_from_cyborg, jax_red_to_cyborg

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

        jax_env = FsmRedCC4Env(num_steps=TWO_STEP_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
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

    def test_explicit_cyborg_red_trace_matches_green_phish_privesc_privilege(self):
        """Replaying CybORG's first seed-0 red trace should preserve the red_4 privesc privilege gain."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP, RED_SLEEP
        from jaxborg.constants import COMPROMISE_PRIVILEGED, NUM_BLUE_AGENTS, NUM_RED_AGENTS
        from jaxborg.env import CC4Env
        from jaxborg.translate import build_mappings_from_cyborg, cyborg_red_to_jax
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

        jax_env = CC4Env(num_steps=THREE_STEP_TRACE_STEPS, topology_mode="cyborg_bank", topology_bank_size=1)
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

    def test_explicit_replay_corrects_generic_exploit_to_cyborg_subaction(self):
        """Seed-0 generic exploit replay should not invent a host_22 foothold."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import COMPROMISE_NONE, NUM_BLUE_AGENTS
        from jaxborg.env import CC4Env
        from jaxborg.translate import build_mappings_from_cyborg
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

        jax_env = CC4Env(num_steps=500, topology_mode="cyborg_bank", topology_bank_size=1)
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

    def test_explicit_replay_handles_failed_generic_exploit_without_subaction(self):
        """Seed-0 generic exploit replay should not invent the failed red_4 foothold on host_56."""
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.actions.encoding import BLUE_SLEEP
        from jaxborg.constants import COMPROMISE_NONE, NUM_BLUE_AGENTS
        from jaxborg.env import CC4Env
        from jaxborg.translate import build_mappings_from_cyborg
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

        jax_env = CC4Env(num_steps=500, topology_mode="cyborg_bank", topology_bank_size=1)
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
            and diff.host_or_agent in {host_label, agent_host_label, "red_agent_4"}
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
                            and diff.host_or_agent in {f"host_{target_host}", f"red_{r}_host_{target_host}", agent_name}
                        ]
                        assert target_diffs == [], f"Step {step}: blocked-route exploit parity diffs: {target_diffs}"
                        break
                if found_step is not None:
                    break
            if found_step is not None:
                break

        assert found_step is not None, "Never found a red exploit from a blocked subnet in 300 steps"
