from ipaddress import ip_network
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from CybORG import CybORG
from CybORG.Agents import FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Actions import DiscoverRemoteSystems, ExploitRemoteService
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.encoding import (
    BLUE_SLEEP,
    RED_EXPLOIT_HARAKA_START,
    RED_EXPLOIT_HTTP_START,
    RED_SLEEP,
)
from jaxborg.agents.fsm_red import (
    FSM_ACT_AGGRESSIVE_SCAN,
    FSM_ACT_DISCOVER,
    FSM_ACT_EXPLOIT,
    FSM_ACT_IMPACT,
    FSM_ACT_PRIVESC,
    FSM_ACT_STEALTH_SCAN,
    FSM_ACT_WITHDRAW,
    FSM_F,
    FSM_K,
    FSM_KD,
    FSM_R,
    FSM_RD,
    FSM_S,
    FSM_U,
    FSM_UD,
    PROBABILITY_MATRIX,
    TRANSITION_FAILURE,
    TRANSITION_SUCCESS,
    _compute_post_step_fsm_states,
    _pick_discover_subnet,
    _pick_exploit_action,
    determine_fsm_success,
    fsm_red_get_action,
    fsm_red_init_states,
    fsm_red_process_session_removal,
    fsm_red_select_actions,
    fsm_red_update_state,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS, SERVICE_IDS
from jaxborg.state import create_initial_state


class TestTransitionMatrices:
    def test_success_matrix_shape(self):
        assert TRANSITION_SUCCESS.shape == (9, 9)

    def test_failure_matrix_shape(self):
        assert TRANSITION_FAILURE.shape == (9, 9)

    def test_probability_matrix_shape(self):
        assert PROBABILITY_MATRIX.shape == (8, 9)

    def test_K_success_discover_goes_KD(self):
        assert int(TRANSITION_SUCCESS[FSM_K, FSM_ACT_DISCOVER]) == FSM_KD

    def test_K_success_aggressive_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_K, FSM_ACT_AGGRESSIVE_SCAN]) == FSM_S

    def test_S_success_exploit_goes_U(self):
        assert int(TRANSITION_SUCCESS[FSM_S, FSM_ACT_EXPLOIT]) == FSM_U

    def test_U_success_privesc_goes_R(self):
        assert int(TRANSITION_SUCCESS[FSM_U, FSM_ACT_PRIVESC]) == FSM_R

    def test_R_success_impact_stays_R(self):
        assert int(TRANSITION_SUCCESS[FSM_R, FSM_ACT_IMPACT]) == FSM_R

    def test_U_success_withdraw_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_U, FSM_ACT_WITHDRAW]) == FSM_S

    def test_R_success_withdraw_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_R, FSM_ACT_WITHDRAW]) == FSM_S

    def test_K_failure_discover_stays_K(self):
        assert int(TRANSITION_FAILURE[FSM_K, FSM_ACT_DISCOVER]) == FSM_K

    def test_S_failure_exploit_stays_S(self):
        assert int(TRANSITION_FAILURE[FSM_S, FSM_ACT_EXPLOIT]) == FSM_S

    def test_F_success_discover_stays_F(self):
        assert int(TRANSITION_SUCCESS[FSM_F, FSM_ACT_DISCOVER]) == FSM_F

    def test_probability_sums_to_one(self):
        for state_idx in range(8):
            valid = PROBABILITY_MATRIX[state_idx] >= 0
            probs = PROBABILITY_MATRIX[state_idx][valid]
            total = float(jnp.sum(probs))
            assert abs(total - 1.0) < 1e-5, f"State {state_idx} probs sum to {total}"


class TestAutonomousActionParity:
    def test_initial_active_red_agent_is_not_forced_to_sleep(self):
        """Initial autonomous red step should not be hard-forced to Sleep."""
        from tests.differential.harness import CC4DifferentialHarness

        harness = CC4DifferentialHarness(seed=0, max_steps=500)
        harness.reset()

        controller = harness.cyborg_env.environment_controller
        cyborg_agent = controller.agent_interfaces["red_agent_0"].agent
        cyborg_obs = harness.cyborg_env.get_observation("red_agent_0")
        cyborg_action_space = controller.get_action_space("red_agent_0")
        cyborg_agent.set_initial_values(cyborg_action_space, cyborg_obs)
        cyborg_action = cyborg_agent.get_action(cyborg_obs, cyborg_action_space)

        red_keys = jax.random.split(jax.random.PRNGKey(0), NUM_RED_AGENTS)
        red_actions, _, _, _, _, _ = fsm_red_select_actions(harness.jax_state, harness.jax_const, red_keys)

        assert type(cyborg_action).__name__ != "Sleep"
        assert int(red_actions[0]) != RED_SLEEP

    def test_fsm_hidden_state_applies_after_completion_step(self):
        """FSM hidden state should update on the next decision step, not immediately on completion."""
        from CybORG.Agents import EnterpriseGreenAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.fsm_red_env import FsmRedCC4Env

        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg_env = BlueFlatWrapper(env=CybORG(scenario, "sim", seed=0), pad_spaces=True)
        cyborg_env.reset()
        cyborg_agent = cyborg_env.env.environment_controller.agent_interfaces["red_agent_0"].agent

        jax_env = FsmRedCC4Env(
            num_steps=500,
            topology_mode="cyborg_bank",
            topology_bank_size=1,
            sync_red_policy_bank=True,
        )
        key = jax.random.PRNGKey(0)
        _, env_state = jax_env.reset(key)
        start_host = int(env_state.const.red_start_hosts[0])

        for expected in (FSM_U, FSM_U, FSM_R):
            _, _, _, _, _ = cyborg_env.step(actions={a: Sleep() for a in cyborg_env.agents})
            key, step_key = jax.random.split(key)
            _, env_state, _, _, _ = jax_env.step(
                step_key,
                env_state,
                {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)},
            )

            cyborg_states = [info["state"] for info in cyborg_agent.host_states.values() if info.get("hostname")]
            expected_cyborg = {FSM_U: "U", FSM_R: "R"}[expected]
            assert cyborg_states == [expected_cyborg]
            assert int(env_state.state.fsm_host_states[0, start_host]) == expected


class TestFsmUpdateState:
    def test_K_success_discover_transitions_to_KD(self, jax_const):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        discovered_hosts = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[5].set(True)
        target_subnet = jax_const.host_subnet[5]
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(5),
            discovered_hosts,
            target_subnet,
            FSM_ACT_DISCOVER,
            jnp.bool_(True),
        )
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_S_failure_exploit_stays_S(self, jax_const):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_S, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(10),
            jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            jnp.int32(0),
            FSM_ACT_EXPLOIT,
            jnp.bool_(False),
        )
        assert int(new_fsm[0, 10]) == FSM_S

    def test_invalid_action_preserves_state(self, jax_const):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(5),
            jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            jnp.int32(0),
            FSM_ACT_EXPLOIT,
            jnp.bool_(True),
        )
        assert int(new_fsm[0, 5]) == FSM_K

    def test_update_does_not_affect_other_hosts(self, jax_const):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        discovered_hosts = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[5].set(True)
        target_subnet = jax_const.host_subnet[5]
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(5),
            discovered_hosts,
            target_subnet,
            FSM_ACT_DISCOVER,
            jnp.bool_(True),
        )
        assert int(new_fsm[0, 5]) == FSM_KD
        assert int(new_fsm[0, 6]) == FSM_K

    def test_update_does_not_affect_other_agents(self, jax_const):
        h = int(jax_const.red_start_hosts[0])
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_S, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(h),
            jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            jnp.int32(0),
            FSM_ACT_EXPLOIT,
            jnp.bool_(True),
        )
        assert int(new_fsm[0, h]) == FSM_U
        assert int(new_fsm[1, h]) == FSM_S

    def test_U_to_F_on_foreign_subnet(self, jax_const):
        """Exploit success on a host outside agent's subnets should transition to F, not U."""
        agent_subnets = jax_const.red_agent_subnets[0]
        foreign_host = None
        for h in range(GLOBAL_MAX_HOSTS):
            if jax_const.host_active[h] and not agent_subnets[jax_const.host_subnet[h]]:
                foreign_host = h
                break
        if foreign_host is None:
            pytest.skip("No foreign host found")
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_S, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(
            fsm,
            jax_const,
            0,
            jnp.int32(foreign_host),
            jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            jnp.int32(0),
            FSM_ACT_EXPLOIT,
            jnp.bool_(True),
        )
        assert int(new_fsm[0, foreign_host]) == FSM_F

    def test_discover_updates_all_known_hosts_in_target_subnet_like_cyborg(self):
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            )
        )
        controller = cyborg_env.environment_controller
        agent = controller.agent_interfaces["red_agent_0"].agent
        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)

        known_ips = list(agent.host_states.keys())
        subnet_to_ips = {}
        for ip in known_ips:
            subnet_to_ips.setdefault(str(ip).rsplit(".", 1)[0], []).append(ip)
        selected_ips = None
        for ips in subnet_to_ips.values():
            if len(ips) >= 2:
                selected_ips = ips[:2]
                break
        if selected_ips is None:
            pytest.skip("Need two known hosts in one subnet")

        for ip, state_name in zip(selected_ips, ["U", "K"], strict=True):
            agent.host_states[ip]["state"] = state_name

        subnet = ip_network(f"{str(selected_ips[0]).rsplit('.', 1)[0]}.0/24")
        action = DiscoverRemoteSystems(subnet=subnet, session=0, agent="red_agent_0")
        agent._host_state_transition(action, SimpleNamespace(name="TRUE", value=1))

        cy_states = [agent.host_states[ip]["state"] for ip in selected_ips]
        assert cy_states == ["UD", "KD"]

        host_indices = [mappings.hostname_to_idx[agent.host_states[ip]["hostname"]] for ip in selected_ips]
        discovered_hosts = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        discovered_hosts = discovered_hosts.at[jnp.array(host_indices, dtype=jnp.int32)].set(True)
        fsm = jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.int32)
        fsm = fsm.at[0, host_indices[0]].set(FSM_U)
        fsm = fsm.at[0, host_indices[1]].set(FSM_K)

        new_fsm = fsm_red_update_state(
            fsm,
            const,
            0,
            jnp.int32(host_indices[0]),
            discovered_hosts,
            const.host_subnet[host_indices[0]],
            FSM_ACT_DISCOVER,
            jnp.bool_(True),
        )

        assert int(new_fsm[0, host_indices[0]]) == FSM_UD
        assert int(new_fsm[0, host_indices[1]]) == FSM_KD


class TestFsmSuccessDetection:
    def test_exploit_session_count_growth_counts_as_success_like_cyborg(self):
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            ),
            seed=42,
        )
        controller = cyborg_env.environment_controller
        agent = controller.agent_interfaces["red_agent_0"].agent
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        start_session = controller.state.sessions["red_agent_0"][0]
        target_hostname = start_session.hostname
        target_ip = next(ip for ip, hostname in controller.state.ip_addresses.items() if hostname == target_hostname)
        target_ip_str = str(target_ip)
        agent.host_states[target_ip_str] = {"hostname": target_hostname, "state": "S"}
        action = ExploitRemoteService(ip_address=target_ip, session=0, agent="red_agent_0")
        agent._host_state_transition(action, SimpleNamespace(name="TRUE", value=1))
        assert agent.host_states[target_ip_str]["state"] == "U"

        target_host = mappings.hostname_to_idx[target_hostname]
        base = create_initial_state()
        before = base.replace(
            red_sessions=base.red_sessions.at[0, target_host].set(True),
            red_session_count=base.red_session_count.at[0, target_host].set(1),
        )
        after = before.replace(
            red_session_count=before.red_session_count.at[0, target_host].set(2),
        )

        assert bool(
            determine_fsm_success(
                before,
                after,
                const,
                0,
                jnp.int32(target_host),
                const.host_subnet[target_host],
                FSM_ACT_EXPLOIT,
            )
        )

    def test_discover_reaffirming_known_hosts_counts_as_success_like_cyborg(self):
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            ),
            seed=42,
        )
        controller = cyborg_env.environment_controller
        agent = controller.agent_interfaces["red_agent_0"].agent
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        session = controller.state.sessions["red_agent_0"][0]
        start_hostname = session.hostname
        start_ip = next(ip for ip, hostname in controller.state.ip_addresses.items() if hostname == start_hostname)
        subnet_prefix = str(start_ip).rsplit(".", 1)[0]

        target_entries = []
        for ip, hostname in controller.state.ip_addresses.items():
            if str(ip).rsplit(".", 1)[0] != subnet_prefix:
                continue
            host_idx = mappings.hostname_to_idx.get(hostname)
            if host_idx is None or not const.host_active[host_idx]:
                continue
            target_entries.append((str(ip), hostname, host_idx))
        if len(target_entries) < 2:
            pytest.skip("Need two active hosts in one subnet")

        for ip_str, hostname, _ in target_entries:
            agent.host_states[ip_str] = {"hostname": hostname, "state": "K"}

        subnet = ip_network(f"{subnet_prefix}.0/24")
        action = DiscoverRemoteSystems(subnet=subnet, session=0, agent="red_agent_0")
        agent._host_state_transition(action, SimpleNamespace(name="TRUE", value=1))
        assert [agent.host_states[ip]["state"] for ip, _, _ in target_entries[:2]] == ["KD", "KD"]

        host_indices = jnp.array([entry[2] for entry in target_entries[:2]], dtype=jnp.int32)
        base = create_initial_state()
        discovered = base.red_discovered_hosts.at[0, host_indices].set(True)
        before = base.replace(red_discovered_hosts=discovered)
        after = before

        assert bool(
            determine_fsm_success(
                before,
                after,
                const,
                0,
                host_indices[0],
                const.host_subnet[host_indices[0]],
                FSM_ACT_DISCOVER,
            )
        )


class TestFsmScanRescanSuccess:
    """Regression: re-scanning an already-scanned host must count as success.

    CybORG always reports scan success when the action executes (agent has
    session, target reachable), regardless of whether the host was previously
    scanned.  JAX's determine_fsm_success used a delta check on
    red_scanned_hosts that returned False for re-scans because the field was
    already True.  This caused KD → KD (failure) instead of KD → SD (success),
    producing systematic FSM divergence across many L3 seeds.
    """

    def test_rescan_aggressive_counts_as_success(self):
        """Aggressive re-scan on already-scanned host → success via red_scan_success flag."""
        base = create_initial_state()
        const = _make_trivial_const()

        target = 5
        before = base.replace(
            red_scanned_hosts=base.red_scanned_hosts.at[0, target].set(True),
        )
        # Re-scan: scanned stays True, but scan action sets red_scan_success
        after = before.replace(
            red_scan_success=before.red_scan_success.at[0].set(True),
        )

        success = determine_fsm_success(
            before,
            after,
            const,
            0,
            jnp.int32(target),
            jnp.int32(0),
            FSM_ACT_AGGRESSIVE_SCAN,
        )
        assert bool(success), "Re-scan should count as success when scan action succeeded"

    def test_rescan_stealth_counts_as_success(self):
        """Stealth re-scan on already-scanned host → success via red_scan_success flag."""
        base = create_initial_state()
        const = _make_trivial_const()

        target = 5
        before = base.replace(
            red_scanned_hosts=base.red_scanned_hosts.at[0, target].set(True),
        )
        after = before.replace(
            red_scan_success=before.red_scan_success.at[0].set(True),
        )

        success = determine_fsm_success(
            before,
            after,
            const,
            0,
            jnp.int32(target),
            jnp.int32(0),
            FSM_ACT_STEALTH_SCAN,
        )
        assert bool(success), "Re-scan should count as success when scan action succeeded"

    def test_fresh_scan_still_detected(self):
        """A fresh scan (not previously scanned) still works via red_scan_success."""
        base = create_initial_state()
        const = _make_trivial_const()

        target = 5
        before = base
        after = base.replace(
            red_scanned_hosts=base.red_scanned_hosts.at[0, target].set(True),
            red_scan_success=base.red_scan_success.at[0].set(True),
        )

        success = determine_fsm_success(
            before,
            after,
            const,
            0,
            jnp.int32(target),
            jnp.int32(0),
            FSM_ACT_AGGRESSIVE_SCAN,
        )
        assert bool(success), "Fresh scan should count as success"

    def test_failed_scan_still_failure(self):
        """A failed scan (red_scan_success not set) → failure."""
        base = create_initial_state()
        const = _make_trivial_const()

        target = 5
        before = base
        after = base  # red_scan_success stays False

        success = determine_fsm_success(
            before,
            after,
            const,
            0,
            jnp.int32(target),
            jnp.int32(0),
            FSM_ACT_AGGRESSIVE_SCAN,
        )
        assert not bool(success), "Failed scan should not count as success"

    def test_failed_scan_with_prior_memory_still_failure(self):
        """Scan fails but old scan memory valid → still failure (no false positive)."""
        base = create_initial_state()
        const = _make_trivial_const()

        target = 5
        before = base.replace(
            red_scanned_hosts=base.red_scanned_hosts.at[0, target].set(True),
        )
        # Scan failed: red_scan_success NOT set, but scanned_hosts still True
        after = before

        success = determine_fsm_success(
            before,
            after,
            const,
            0,
            jnp.int32(target),
            jnp.int32(0),
            FSM_ACT_AGGRESSIVE_SCAN,
        )
        assert not bool(success), "Failed scan with prior memory should not be false positive"


def _make_trivial_const():
    """Build a minimal CC4Const for unit tests that don't need full topology."""
    from jaxborg.topology import build_topology

    return build_topology(jnp.array([42, 0], dtype=jnp.uint32), num_steps=10)


class TestFsmGetAction:
    def test_discover_subnet_matches_cyborg_action_space_allowed_subnet(self):
        from jaxborg.env import _init_red_state
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            ),
            seed=0,
        )
        controller = cyborg_env.environment_controller
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        state = _init_red_state(const, create_initial_state())

        valid_subnets = [
            subnet for subnet, allowed in controller.get_action_space("red_agent_0")["subnet"].items() if allowed
        ]
        assert len(valid_subnets) == 1

        chosen_subnet = int(_pick_discover_subnet(state, const, 0, jax.random.PRNGKey(0)))
        expected_subnet = next(i for i, allowed in enumerate(const.red_agent_subnets[0]) if bool(allowed))
        assert str(valid_subnets[0]) == str(mappings.subnet_cidrs[expected_subnet])
        assert chosen_subnet == expected_subnet

    def test_generic_exploit_selector_matches_cyborg_http_host(self):
        from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import DefaultExploitActionSelector

        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            ),
            seed=0,
        )
        controller = cyborg_env.environment_controller
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        target_host = int(const.red_start_hosts[0])
        hostname = mappings.idx_to_hostname[target_host]
        ip_address = mappings.hostname_to_ip[hostname]
        controller.state.sessions["red_agent_0"][0].ports[ip_address] = [80]
        host_services = jnp.zeros_like(const.initial_services)
        host_services = host_services.at[target_host, SERVICE_IDS["APACHE2"]].set(True)
        state = create_initial_state().replace(host_services=host_services)

        cy_action = DefaultExploitActionSelector().get_exploit_action(
            state=controller.state,
            session=0,
            agent="red_agent_0",
            ip_address=ip_address,
        )
        assert type(cy_action).__name__ == "HTTPRFI"
        assert int(_pick_exploit_action(state, jnp.int32(target_host), jax.random.PRNGKey(0))) == (
            RED_EXPLOIT_HTTP_START + target_host
        )

    def test_generic_exploit_selector_matches_cyborg_smtp_host(self):
        from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import DefaultExploitActionSelector

        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        cyborg_env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            ),
            seed=0,
        )
        controller = cyborg_env.environment_controller
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        target_host = int(const.red_start_hosts[0])
        hostname = mappings.idx_to_hostname[target_host]
        ip_address = mappings.hostname_to_ip[hostname]
        controller.state.sessions["red_agent_0"][0].ports[ip_address] = [25]
        host_services = jnp.zeros_like(const.initial_services)
        host_services = host_services.at[target_host, SERVICE_IDS["SMTP"]].set(True)
        state = create_initial_state().replace(host_services=host_services)

        cy_action = DefaultExploitActionSelector().get_exploit_action(
            state=controller.state,
            session=0,
            agent="red_agent_0",
            ip_address=ip_address,
        )
        assert type(cy_action).__name__ == "HarakaRCE"
        assert int(_pick_exploit_action(state, jnp.int32(target_host), jax.random.PRNGKey(0))) == (
            RED_EXPLOIT_HARAKA_START + target_host
        )

    def test_returns_sleep_when_no_eligible_hosts(self, jax_const):
        state = create_initial_state()
        key = jax.random.PRNGKey(0)
        action = fsm_red_get_action(state, jax_const, 0, key)
        assert int(action) == RED_SLEEP

    def test_returns_valid_action_with_eligible_hosts(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_KD)
        entered = state.fsm_host_entered.at[0, start_host].set(True)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
            fsm_host_entered=entered,
        )
        key = jax.random.PRNGKey(42)
        action = int(fsm_red_get_action(state, jax_const, 0, key))
        assert action != RED_SLEEP

    def test_F_hosts_excluded(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_F)
        entered = state.fsm_host_entered.at[0, start_host].set(True)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
            fsm_host_entered=entered,
        )
        key = jax.random.PRNGKey(0)
        action = int(fsm_red_get_action(state, jax_const, 0, key))
        assert action == RED_SLEEP

    def test_jit_compatible(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_KD)
        entered = state.fsm_host_entered.at[0, start_host].set(True)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
            fsm_host_entered=entered,
        )
        key = jax.random.PRNGKey(42)
        jitted = jax.jit(fsm_red_get_action, static_argnums=(2,))
        action = int(jitted(state, jax_const, 0, key))
        assert action != RED_SLEEP

    def test_multiple_calls_produce_different_actions(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_KD)
        entered = state.fsm_host_entered.at[0, start_host].set(True)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
            fsm_host_entered=entered,
        )

        actions = set()
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            action = int(fsm_red_get_action(state, jax_const, 0, key))
            actions.add(action)

        assert len(actions) > 1


class TestFsmInitStates:
    def test_start_host_gets_U(self, jax_const):
        fsm = fsm_red_init_states(jax_const, 0)
        start_host = int(jax_const.red_start_hosts[0])
        assert int(fsm[start_host]) == FSM_U

    def test_other_hosts_get_K(self, jax_const):
        fsm = fsm_red_init_states(jax_const, 0)
        start_host = int(jax_const.red_start_hosts[0])
        for h in range(GLOBAL_MAX_HOSTS):
            if h != start_host:
                assert int(fsm[h]) == FSM_K


class TestFsmSessionRemoval:
    def test_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_U)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_kept_session_no_change(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_U)
        sessions = state.red_sessions.at[0, 5].set(True)
        state = state.replace(fsm_host_states=fsm, red_sessions=sessions)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_U

    def test_R_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_R)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_RD_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_RD)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_K_state_unaffected(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_K)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_K

    def test_S_state_unaffected(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_S)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_S


class TestFsmRedDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_translate_roundtrip_discover(self, cyborg_env):
        """Verify DiscoverRemoteSystems roundtrips through the translator."""
        from CybORG.Simulator.Actions import DiscoverRemoteSystems

        from jaxborg.actions.encoding import RED_DISCOVER_START
        from jaxborg.translate import (
            build_mappings_from_cyborg,
            cyborg_red_to_jax,
            jax_red_to_cyborg,
        )

        mappings = build_mappings_from_cyborg(cyborg_env)
        subnet_idx = list(mappings.subnet_cidrs.keys())[0]
        cidr = mappings.subnet_cidrs[subnet_idx]

        action = DiscoverRemoteSystems(subnet=cidr, session=0, agent="red_agent_0")
        jax_idx = cyborg_red_to_jax(action, "red_agent_0", mappings)
        assert jax_idx == RED_DISCOVER_START + subnet_idx

        roundtrip = jax_red_to_cyborg(jax_idx, 0, mappings)
        assert type(roundtrip).__name__ == "DiscoverRemoteSystems"
        assert roundtrip.subnet == cidr

    def test_translate_roundtrip_exploit(self, cyborg_env):
        """Verify exploit action roundtrips through the translator."""
        from CybORG.Simulator.Actions import SSHBruteForce

        from jaxborg.actions.encoding import RED_EXPLOIT_SSH_START
        from jaxborg.translate import build_mappings_from_cyborg, cyborg_red_to_jax, jax_red_to_cyborg

        mappings = build_mappings_from_cyborg(cyborg_env)
        host_idx = 0
        hostname = mappings.idx_to_hostname[host_idx]
        ip = mappings.hostname_to_ip[hostname]

        action = SSHBruteForce(session=0, agent="red_agent_0", ip_address=ip)
        jax_idx = cyborg_red_to_jax(action, "red_agent_0", mappings)
        assert jax_idx == RED_EXPLOIT_SSH_START + host_idx

        roundtrip = jax_red_to_cyborg(jax_idx, 0, mappings)
        assert type(roundtrip).__name__ == "ExploitRemoteService"
        assert roundtrip.ip_address == ip


class TestRedAgentActivation:
    """Red agents should only be active when CybORG would activate them.

    CybORG only activates red_agent_0 at episode start. Others activate
    through different_subnet_agent_reassignment when red_0 compromises
    a host in their subnet.
    """

    def test_only_red_0_active_at_reset(self):
        """JAXborg should match CybORG: only red_agent_0 active at reset."""
        from tests.differential.harness import CC4DifferentialHarness

        h = CC4DifferentialHarness(seed=42, max_steps=1)
        h.reset()

        controller = h.cyborg_env.environment_controller
        cyborg_active = {
            r
            for r in range(NUM_RED_AGENTS)
            if controller.agent_interfaces.get(f"red_agent_{r}")
            and controller.agent_interfaces[f"red_agent_{r}"].active
        }

        jax_active = {r for r in range(NUM_RED_AGENTS) if h.jax_state.red_agent_active[r]}

        assert cyborg_active == {0}, f"CybORG should have only red_agent_0 active, got {cyborg_active}"
        assert jax_active == cyborg_active, (
            f"JAXborg activates {jax_active} but CybORG only activates "
            f"{cyborg_active}. Extra agents: {jax_active - cyborg_active}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_only_red_0_active_at_reset_multi_seed(self, seed):
        """CybORG activates only red_0 at reset across seeds."""
        from tests.differential.harness import CC4DifferentialHarness

        h = CC4DifferentialHarness(seed=seed, max_steps=1)
        h.reset()

        controller = h.cyborg_env.environment_controller
        cyborg_active = {
            r
            for r in range(NUM_RED_AGENTS)
            if controller.agent_interfaces.get(f"red_agent_{r}")
            and controller.agent_interfaces[f"red_agent_{r}"].active
        }

        jax_active = {r for r in range(NUM_RED_AGENTS) if h.jax_state.red_agent_active[r]}

        assert jax_active == cyborg_active, (
            f"seed={seed}: JAXborg activates {jax_active} but CybORG "
            f"activates {cyborg_active}. Extra: {jax_active - cyborg_active}"
        )

    def test_fsm_init_state_matches_cyborg(self):
        """JAX FSM initial host state should match CybORG's FSM host_states at step 0."""
        from tests.differential.harness import CC4DifferentialHarness

        h = CC4DifferentialHarness(seed=42, max_steps=1)
        h.reset()

        # CybORG: get_action processes the initial observation and populates host_states
        controller = h.cyborg_env.environment_controller
        cyborg_agent = controller.agent_interfaces["red_agent_0"].agent
        obs = h.cyborg_env.get_observation("red_agent_0")
        aspace = controller.get_action_space("red_agent_0")
        cyborg_agent.set_initial_values(aspace, obs)
        cyborg_agent.get_action(obs, aspace)

        # CybORG FSM state map: IP -> {'state': 'U'/'K'/etc, 'hostname': ...}
        cyborg_fsm_states = {}
        for ip, info in cyborg_agent.host_states.items():
            hostname = info.get("hostname")
            if hostname and hostname in h.mappings.hostname_to_idx:
                hidx = h.mappings.hostname_to_idx[hostname]
                cyborg_fsm_states[hidx] = info["state"]

        # JAX FSM state
        jax_fsm = h.jax_state.fsm_host_states[0]
        # Start host should be in state U in both
        start_host = int(h.jax_const.red_start_hosts[0])
        assert start_host in cyborg_fsm_states, "CybORG should know about start host"
        assert cyborg_fsm_states[start_host] == "U", (
            f"CybORG start host state should be 'U', got '{cyborg_fsm_states[start_host]}'"
        )
        assert int(jax_fsm[start_host]) == FSM_U, (
            f"JAX start host FSM state should be FSM_U={FSM_U}, got {int(jax_fsm[start_host])}"
        )

        # All other hosts should be in state K (unknown) in JAX
        for hidx in range(int(h.jax_const.num_hosts)):
            if hidx == start_host:
                continue
            assert int(jax_fsm[hidx]) == FSM_K, (
                f"JAX host {hidx} should be FSM_K={FSM_K} at reset, got {int(jax_fsm[hidx])}"
            )


class TestInactiveAgentFsmFreeze:
    """CybORG freezes FSM state for inactive agents — JAX must match.

    When an agent loses all sessions and becomes inactive, CybORG's
    get_action() is skipped, so _host_state_transition and
    _session_removal_state_change never run.  The FSM is frozen until
    the agent reactivates.
    """

    def _make_state_pair(self, agent_id, host, fsm_state, *, active, has_session):
        """Build (state_before, state_after) for a single-agent scenario."""
        base = create_initial_state()
        const = _make_trivial_const()
        fsm = base.fsm_host_states.at[agent_id, host].set(fsm_state)
        sessions = base.red_sessions.at[agent_id, host].set(has_session)
        active_arr = base.red_agent_active.at[agent_id].set(active)
        state = base.replace(
            fsm_host_states=fsm,
            red_sessions=sessions,
            red_agent_active=active_arr,
        )
        return state, state, const

    def test_inactive_agent_session_loss_no_downgrade(self):
        """Inactive agent with lost sessions: FSM should NOT downgrade to KD."""
        agent, host = 1, 5
        state_before, state_after, const = self._make_state_pair(
            agent,
            host,
            FSM_R,
            active=False,
            has_session=False,
        )
        target_hosts = [jnp.int32(0)] * NUM_RED_AGENTS
        target_subnets = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions = [jnp.int32(0)] * NUM_RED_AGENTS
        eligible = [jnp.bool_(False)] * NUM_RED_AGENTS

        result = _compute_post_step_fsm_states(
            state_before,
            state_after,
            const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible,
        )
        assert int(result[agent, host]) == FSM_R, (
            f"Inactive agent FSM should stay frozen at R, got {int(result[agent, host])}"
        )

    def test_inactive_agent_RD_session_loss_no_downgrade(self):
        """Inactive agent at RD with lost sessions: FSM should stay at RD."""
        agent, host = 2, 10
        state_before, state_after, const = self._make_state_pair(
            agent,
            host,
            FSM_RD,
            active=False,
            has_session=False,
        )
        target_hosts = [jnp.int32(0)] * NUM_RED_AGENTS
        target_subnets = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions = [jnp.int32(0)] * NUM_RED_AGENTS
        eligible = [jnp.bool_(False)] * NUM_RED_AGENTS

        result = _compute_post_step_fsm_states(
            state_before,
            state_after,
            const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible,
        )
        assert int(result[agent, host]) == FSM_RD, (
            f"Inactive agent FSM should stay frozen at RD, got {int(result[agent, host])}"
        )

    def test_active_agent_session_loss_downgrades(self):
        """Active agent with lost sessions: FSM SHOULD downgrade to KD."""
        agent, host = 1, 5
        state_before, state_after, const = self._make_state_pair(
            agent,
            host,
            FSM_R,
            active=True,
            has_session=False,
        )
        target_hosts = [jnp.int32(0)] * NUM_RED_AGENTS
        target_subnets = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions = [jnp.int32(0)] * NUM_RED_AGENTS
        eligible = [jnp.bool_(False)] * NUM_RED_AGENTS

        result = _compute_post_step_fsm_states(
            state_before,
            state_after,
            const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible,
        )
        assert int(result[agent, host]) == FSM_KD, (
            f"Active agent with lost session should downgrade to KD, got {int(result[agent, host])}"
        )

    def test_active_agent_with_session_keeps_state(self):
        """Active agent that still has a session: FSM should NOT change."""
        agent, host = 1, 5
        state_before, state_after, const = self._make_state_pair(
            agent,
            host,
            FSM_R,
            active=True,
            has_session=True,
        )
        target_hosts = [jnp.int32(0)] * NUM_RED_AGENTS
        target_subnets = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions = [jnp.int32(0)] * NUM_RED_AGENTS
        eligible = [jnp.bool_(False)] * NUM_RED_AGENTS

        result = _compute_post_step_fsm_states(
            state_before,
            state_after,
            const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible,
        )
        assert int(result[agent, host]) == FSM_R, (
            f"Active agent with session should keep R, got {int(result[agent, host])}"
        )

    def test_inactive_agent_action_transition_skipped(self):
        """Inactive agent's action transition should NOT be applied.

        Even if the agent was eligible and its action executed, if the agent
        is inactive in state_after, the FSM transition should be skipped
        (CybORG's get_action() is not called for inactive agents).
        """
        agent, host = 0, 5
        const = _make_trivial_const()
        base = create_initial_state()
        # Agent has scanned host (S) and exploit succeeds (session gained)
        fsm = base.fsm_host_states.at[agent, host].set(FSM_S)
        sessions_before = base.red_session_count
        sessions_after = sessions_before.at[agent, host].set(1)
        state_before = base.replace(
            fsm_host_states=fsm,
            red_agent_active=base.red_agent_active.at[agent].set(True),
        )
        state_after = base.replace(
            fsm_host_states=fsm,
            red_session_count=sessions_after,
            red_sessions=base.red_sessions.at[agent, host].set(True),
            # Agent becomes inactive by step end (e.g., sessions reassigned away)
            red_agent_active=base.red_agent_active.at[agent].set(False),
        )
        target_hosts = [jnp.int32(0)] * NUM_RED_AGENTS
        target_hosts[agent] = jnp.int32(host)
        target_subnets = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions = [jnp.int32(0)] * NUM_RED_AGENTS
        fsm_actions[agent] = jnp.int32(FSM_ACT_EXPLOIT)
        eligible = [jnp.bool_(False)] * NUM_RED_AGENTS
        eligible[agent] = jnp.bool_(True)
        executed = jnp.ones(NUM_RED_AGENTS, dtype=jnp.bool_)

        result = _compute_post_step_fsm_states(
            state_before,
            state_after,
            const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible,
            executed_flags=executed,
        )
        # Exploit succeeded (S→U) but agent is inactive → transition frozen
        assert int(result[agent, host]) == FSM_S, (
            f"Inactive agent should not transition S→U, got {int(result[agent, host])}"
        )
