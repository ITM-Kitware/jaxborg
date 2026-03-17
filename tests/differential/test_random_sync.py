import jax.numpy as jnp
import numpy as np

from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import compare_fast


def test_detection_random_sync_advances_jax_index_for_cyborg_scan_trace():
    harness = CC4DifferentialHarness(seed=0, max_steps=500, sync_green_rng=True, strict_random_sync=True)
    harness.reset()

    report = None
    result = None
    for _ in range(20):
        result = harness.full_step()
        report = harness.last_random_sync_report
        if report is not None and report.detection_randoms:
            break

    assert result is not None
    assert result.diffs == []
    assert report is not None
    assert report.detection_randoms
    assert report.detection_sync_supported
    assert not report.has_issues
    assert int(harness.jax_state.detection_random_index) == 1
    np.testing.assert_allclose(
        np.asarray(harness.jax_const.detection_randoms[:1]),
        np.asarray(report.detection_randoms, dtype=np.float32),
        atol=1e-7,
    )


def test_strict_random_sync_handles_pending_ssh_exploit_trace():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=9,
        verbose=False,
        strict_random_sync=True,
    )

    assert report is None


def test_strict_random_sync_handles_discover_deception_trace():
    report = run_differential_fuzz(
        seeds=[1],
        max_steps_per_seed=11,
        verbose=False,
        strict_random_sync=True,
    )

    assert report is None


def test_strict_random_sync_handles_generic_blue_deploy_decoy_trace():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=2,
        verbose=False,
        strict_random_sync=True,
        blue_agent="random",
        blue_action_source="cyborg_policy",
    )

    assert report is None


def test_strict_random_sync_handles_red_session_check_trace():
    report = run_differential_fuzz(
        seeds=[2],
        max_steps_per_seed=74,
        verbose=False,
        strict_random_sync=True,
        blue_agent="random",
        blue_action_source="cyborg_policy",
    )

    assert report is None


def test_strict_random_sync_handles_privesc_session_choice_trace():
    report = run_differential_fuzz(
        seeds=[3],
        max_steps_per_seed=184,
        verbose=False,
        strict_random_sync=True,
        blue_agent="random",
        blue_action_source="cyborg_policy",
    )

    assert report is None


def test_detection_random_sync_handles_failed_scan_trace_with_random_blue_actions():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=43,
        verbose=False,
        strict_random_sync=True,
        blue_agent="random",
        blue_action_source="cyborg_policy",
    )

    assert report is None


def test_detection_random_sync_handles_long_discover_deception_trace():
    report = run_differential_fuzz(
        seeds=[5],
        max_steps_per_seed=274,
        verbose=False,
        strict_random_sync=True,
        blue_agent="random",
        blue_action_source="cyborg_policy",
    )

    assert report is None


class _FakeProcessEvent:
    def __init__(self, pid: int):
        self.pid = pid


def test_compare_fast_reports_detection_event_drift():
    harness = CC4DifferentialHarness(seed=0, max_steps=1)
    harness.reset()

    host_idx = min(harness.mappings.idx_to_hostname)
    hostname = harness.mappings.idx_to_hostname[host_idx]
    events = harness.cyborg_env.environment_controller.state.hosts[hostname].events
    events.network_connections = [object()]
    events.old_network_connections = [object()]
    events.process_creation = [_FakeProcessEvent(4321)]
    events.old_process_creation = [_FakeProcessEvent(1234)]

    diffs = compare_fast(harness.cyborg_env, harness.jax_state, harness.jax_const, harness.mappings)
    fields = {(d.field_name, d.host_or_agent) for d in diffs}

    assert ("host_activity_detected", f"host_{host_idx}") in fields
    assert ("old_host_activity_detected", f"host_{host_idx}") in fields
    assert ("host_exploit_detected", f"host_{host_idx}") in fields
    assert ("old_host_exploit_detected", f"host_{host_idx}") in fields
    assert ("host_process_creation_pids", f"host_{host_idx}") in fields


def test_ordering_resync_rebuilds_detection_event_state():
    harness = CC4DifferentialHarness(seed=0, max_steps=1)
    harness.reset()

    host_idx = min(harness.mappings.idx_to_hostname)
    hostname = harness.mappings.idx_to_hostname[host_idx]
    events = harness.cyborg_env.environment_controller.state.hosts[hostname].events
    events.network_connections = [object()]
    events.old_network_connections = [object()]
    events.process_creation = [_FakeProcessEvent(4321)]
    events.old_process_creation = [_FakeProcessEvent(1234)]

    harness.jax_state = harness.jax_state.replace(
        host_activity_detected=harness.jax_state.host_activity_detected.at[host_idx].set(False),
        old_host_activity_detected=harness.jax_state.old_host_activity_detected.at[host_idx].set(False),
        host_exploit_detected=harness.jax_state.host_exploit_detected.at[host_idx].set(False),
        old_host_exploit_detected=harness.jax_state.old_host_exploit_detected.at[host_idx].set(False),
        host_process_creation_pids=harness.jax_state.host_process_creation_pids.at[host_idx].set(
            jnp.full_like(harness.jax_state.host_process_creation_pids[host_idx], -1)
        ),
    )

    harness._resync_ordering_divergence(harness.cyborg_env.environment_controller.state, 0)

    assert bool(harness.jax_state.host_activity_detected[host_idx])
    assert bool(harness.jax_state.old_host_activity_detected[host_idx])
    assert bool(harness.jax_state.host_exploit_detected[host_idx])
    assert bool(harness.jax_state.old_host_exploit_detected[host_idx])
    assert int(harness.jax_state.host_process_creation_pids[host_idx, 0]) == 4321
