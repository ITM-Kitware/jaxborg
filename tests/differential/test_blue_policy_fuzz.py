from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent, cc4BlueRandomAgent

from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness


def test_generic_deploy_decoy_pending_ticks_match_jax():
    target_hostname = "restricted_zone_b_subnet_server_host_1"
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=cc4BlueRandomAgent,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    step0 = harness.full_step()
    controller = harness.cyborg_env.environment_controller
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
    jax_decoys = tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx])

    assert step1.diffs == []
    assert "tomcat" in service_names
    assert jax_decoys == (False, False, True, False)


def test_router_restore_pending_ticks_match_jax():
    target_hostname = "restricted_zone_a_subnet_router"
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=cc4BlueRandomAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    harness.full_step()
    step1 = harness.full_step()
    controller = harness.cyborg_env.environment_controller
    pending = controller.actions_in_progress["blue_agent_0"]
    pending_name, pending_host, pending_ticks = harness._blue_unsupported_pending[0]

    assert step1.diffs == []
    assert type(pending["action"]).__name__ == "Restore"
    assert pending["action"].hostname == target_hostname
    assert int(pending["remaining_ticks"]) == 4
    assert pending_name == "Restore"
    assert pending_host == harness.mappings.hostname_to_idx[target_hostname]
    assert pending_ticks == 4


def test_router_generic_deploy_decoy_matches_jax():
    target_hostname = "operational_zone_b_subnet_router"
    harness = CC4DifferentialHarness(
        seed=2,
        max_steps=20,
        blue_cls=cc4BlueRandomAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
        use_cyborg_blue_policy=True,
    )
    harness.reset()

    for _ in range(7):
        harness.full_step()

    controller = harness.cyborg_env.environment_controller
    pending = controller.actions_in_progress["blue_agent_3"]
    pending_name, pending_host, pending_ticks = harness._blue_unsupported_pending[3]

    assert type(pending["action"]).__name__ == "DeployDecoy"
    assert pending["action"].hostname == target_hostname
    assert int(pending["remaining_ticks"]) == 1
    assert pending_name == "DeployDecoy"
    assert pending_host == harness.mappings.hostname_to_idx[target_hostname]
    assert pending_ticks == 1

    step7 = harness.full_step()
    target_host = controller.state.hosts[target_hostname]
    target_host_idx = harness.mappings.hostname_to_idx[target_hostname]
    service_names = {str(name).split(".")[-1].lower() for name in target_host.services}
    jax_decoys = tuple(bool(v) for v in harness.jax_state.host_decoys[target_host_idx])

    assert step7.diffs == []
    assert "apache2" in service_names
    assert jax_decoys == (False, True, False, False)


def test_fuzzer_runs_with_cyborg_random_blue_policy():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=20,
        mismatch_mode="all",
        blue_agent="random",
        blue_action_source="cyborg_policy",
        verbose=False,
    )
    assert report is None
