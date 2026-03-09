import numpy as np

from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness


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
        np.asarray(harness.jax_state.detection_randoms[:1]),
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
