import numpy as np
from CybORG.Agents import SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Simulator.Actions import Restore, Sleep
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import DecoyTomcat
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import DecoyVsftpd
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import DeployDecoy

from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.constants import DECOY_IDS
from tests.differential.blue_mask_projection import (
    comparison_blue_mask_in_jax_space,
    live_blue_wrapper_mask_in_jax_space,
)
from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness

_DECOY_SERVICE_TO_FLAGS = {
    "haraka": (True, False, False, False),
    "apache2": (False, True, False, False),
    "tomcat": (False, False, True, False),
    "vsftpd": (False, False, False, True),
}


def _make_scripted_blue_agent(target_agent_name, action_factory):
    class ScriptedBlueAgent(BaseAgent):
        def __init__(self, name=None, np_random=None):
            super().__init__(name, np_random)
            self._step_idx = 0

        def train(self, results):
            del results

        def get_action(self, observation, action_space):
            del observation, action_space
            if self.name == target_agent_name and self._step_idx == 0:
                action = action_factory(self.name)
            else:
                action = Sleep()
            self._step_idx += 1
            return action

        def end_episode(self):
            pass

        def set_initial_values(self, action_space, observation):
            del action_space, observation

    return ScriptedBlueAgent


def test_generic_deploy_decoy_pending_ticks_match_jax():
    target_hostname = "restricted_zone_b_subnet_server_host_1"
    blue_cls = _make_scripted_blue_agent(
        "blue_agent_2",
        lambda agent_name: DeployDecoy(session=0, agent=agent_name, hostname=target_hostname),
    )
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=blue_cls,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    controller = harness.cyborg_env.environment_controller
    before_services = {str(name).split(".")[-1].lower() for name in controller.state.hosts[target_hostname].services}
    step0 = harness.full_step()
    pending = controller.actions_in_progress["blue_agent_2"]

    assert step0.diffs == []
    assert type(pending["action"]).__name__ == "DeployDecoy"
    assert pending["action"].hostname == target_hostname
    assert int(pending["remaining_ticks"]) == 1
    assert int(harness.jax_state.blue_pending_ticks[2]) == 1

    step1 = harness.full_step()
    target_host = controller.state.hosts[target_hostname]
    target_host_idx = harness.mappings.hostname_to_idx[target_hostname]
    service_names = {str(name).split(".")[-1].lower() for name in target_host.services}
    added_services = service_names - before_services
    assert len(added_services) == 1
    added_service = next(iter(added_services))
    jax_decoys = tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx])

    assert step1.diffs == []
    assert jax_decoys == _DECOY_SERVICE_TO_FLAGS[added_service]


def test_generic_deploy_decoy_pending_mask_matches_jax_projection():
    target_hostname = "restricted_zone_b_subnet_server_host_1"
    blue_cls = _make_scripted_blue_agent(
        "blue_agent_2",
        lambda agent_name: DeployDecoy(session=0, agent=agent_name, hostname=target_hostname),
    )
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=blue_cls,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    harness.full_step()
    agent_name = "blue_agent_2"
    controller = harness.cyborg_env.environment_controller
    cyborg_mask = live_blue_wrapper_mask_in_jax_space(
        harness._blue_wrapper,
        agent_name,
        harness.mappings,
        harness.jax_const,
    )
    jax_mask = comparison_blue_mask_in_jax_space(
        controller,
        agent_name,
        2,
        harness.jax_state,
        harness.mappings,
        harness.jax_const,
    )

    # With collapsed action space, pending decoy maps to a single index per host slot
    pending_indices = np.flatnonzero(cyborg_mask).tolist()
    assert len(pending_indices) == 1, f"Expected 1 pending decoy index, got {pending_indices}"
    np.testing.assert_array_equal(jax_mask, cyborg_mask)


def test_router_restore_pending_ticks_match_jax():
    target_hostname = "restricted_zone_a_subnet_router"
    blue_cls = _make_scripted_blue_agent(
        "blue_agent_0",
        lambda agent_name: Restore(session=0, agent=agent_name, hostname=target_hostname),
    )
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=blue_cls,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    step0 = harness.full_step()
    controller = harness.cyborg_env.environment_controller
    pending = controller.actions_in_progress["blue_agent_0"]
    pending_name, pending_host, pending_ticks = harness._blue_unsupported_pending[0]

    assert step0.diffs == []
    assert type(pending["action"]).__name__ == "Restore"
    assert pending["action"].hostname == target_hostname
    assert int(pending["remaining_ticks"]) == 4
    assert pending_name == "Restore"
    assert pending_host == harness.mappings.hostname_to_idx[target_hostname]
    assert pending_ticks == 4


def test_router_generic_deploy_decoy_matches_jax():
    target_hostname = "operational_zone_b_subnet_router"
    blue_cls = _make_scripted_blue_agent(
        "blue_agent_3",
        lambda agent_name: DeployDecoy(session=0, agent=agent_name, hostname=target_hostname),
    )
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=blue_cls,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    controller = harness.cyborg_env.environment_controller
    before_services = {str(name).split(".")[-1].lower() for name in controller.state.hosts[target_hostname].services}
    step0 = harness.full_step()
    pending = controller.actions_in_progress["blue_agent_3"]
    pending_name, pending_host, pending_ticks = harness._blue_unsupported_pending[3]

    assert step0.diffs == []
    assert type(pending["action"]).__name__ == "DeployDecoy"
    assert pending["action"].hostname == target_hostname
    assert int(pending["remaining_ticks"]) == 1
    assert pending_name == "DeployDecoy"
    assert pending_host == harness.mappings.hostname_to_idx[target_hostname]
    assert pending_ticks == 1

    step1 = harness.full_step()
    target_host = controller.state.hosts[target_hostname]
    target_host_idx = harness.mappings.hostname_to_idx[target_hostname]
    service_names = {str(name).split(".")[-1].lower() for name in target_host.services}
    added_services = service_names - before_services
    assert len(added_services) == 1
    added_service = next(iter(added_services))
    jax_decoys = tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx])

    assert step1.diffs == []
    assert jax_decoys == _DECOY_SERVICE_TO_FLAGS[added_service]


def test_fuzzer_runs_with_cyborg_random_blue_policy():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=20,
        mismatch_mode="all",
        blue_agent="random",
        blue_action_source="cyborg_policy",
        verbose=False,
    )
    assert report is None, str(report)


def test_reward_parity_when_green_local_work_selects_decoy_service():
    report = run_differential_fuzz(
        seeds=[1],
        max_steps_per_seed=74,
        blue_agent="random",
        blue_action_source="cyborg_policy",
        strict_random_sync=True,
        verbose=False,
    )
    assert report is None, str(report)


def test_reward_parity_handles_sessionless_impact_trace():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=355,
        blue_agent="random",
        blue_action_source="cyborg_policy",
        strict_random_sync=True,
        verbose=False,
    )
    assert report is None, str(report)


def test_generic_deploy_decoy_reusing_service_name_matches_jax():
    target_hostname = "operational_zone_a_subnet_server_host_0"
    blue_cls = _make_scripted_blue_agent(
        "blue_agent_1",
        lambda agent_name: DeployDecoy(session=0, agent=agent_name, hostname=target_hostname),
    )
    harness = CC4DifferentialHarness(
        seed=42,
        max_steps=20,
        blue_cls=blue_cls,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()
    target_host_idx = harness.mappings.hostname_to_idx[target_hostname]
    controller = harness.cyborg_env.environment_controller
    cy_state = controller.state

    decoy_vsftpd = DecoyVsftpd(session=0, agent="blue_agent_1", hostname=target_hostname)
    decoy_tomcat = DecoyTomcat(session=0, agent="blue_agent_1", hostname=target_hostname)
    assert str(decoy_vsftpd.execute(cy_state).success).upper() == "TRUE"
    assert str(decoy_tomcat.execute(cy_state).success).upper() == "TRUE"
    harness.jax_state = apply_blue_decoy(
        harness.jax_state,
        harness.jax_const,
        1,
        target_host_idx,
        DECOY_IDS["Vsftpd"],
    )
    harness.jax_state = apply_blue_decoy(
        harness.jax_state,
        harness.jax_const,
        1,
        target_host_idx,
        DECOY_IDS["Tomcat"],
    )

    cy_state.np_random = np.random.default_rng(0)
    for host in cy_state.hosts.values():
        host.np_random = cy_state.np_random

    step0 = harness.full_step()
    pending = controller.actions_in_progress["blue_agent_1"]
    assert not any(d.field_name == "host_decoys" and d.host_or_agent == f"host_{target_host_idx}" for d in step0.diffs)
    assert type(pending["action"]).__name__ == "DeployDecoy"
    assert int(pending["remaining_ticks"]) == 1
    assert tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx]) == (False, False, True, True)

    step1 = harness.full_step()
    service_names = {str(name).split(".")[-1].lower() for name in controller.state.hosts[target_hostname].services}

    assert not any(d.field_name == "host_decoys" and d.host_or_agent == f"host_{target_host_idx}" for d in step1.diffs)
    assert service_names == {"sshd", "otservice", "tomcat", "vsftpd"}
    assert tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx]) == (False, False, True, True)
