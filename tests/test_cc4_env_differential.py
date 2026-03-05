import pytest

pytestmark = pytest.mark.slow


class TestCC4EnvDifferential:
    """Differential tests comparing JAX subsystems against CybORG."""

    def _make_harness(self, seed=42, max_steps=500):
        from tests.differential.harness import CC4DifferentialHarness

        return CC4DifferentialHarness(seed=seed, max_steps=max_steps)

    def test_initial_state_parity(self):
        """After reset, CybORG and JAX agree on host_compromised and red_sessions."""
        harness = self._make_harness(seed=42)
        cyborg_snap, jax_snap = harness.reset()

        from tests.differential.state_comparator import (
            _ERROR_FIELDS,
            compare_snapshots,
            format_diffs,
        )

        diffs = compare_snapshots(cyborg_snap, jax_snap)
        errors = [d for d in diffs if d.field_name in _ERROR_FIELDS]
        assert len(errors) == 0, f"Initial state:\n{format_diffs(errors)}"

    def test_red_discover_scan_parity(self):
        """Red discovers a subnet then scans a host. Compare state."""
        harness = self._make_harness(seed=42)
        harness.reset()

        from jaxborg.actions.encoding import RED_DISCOVER_START, RED_SCAN_START
        from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

        start_host = int(harness.jax_const.red_start_hosts[0])
        start_subnet = int(harness.jax_const.host_subnet[start_host])

        result = harness.step_red_only(0, RED_DISCOVER_START + start_subnet)
        errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
        assert len(errors) == 0, f"After discover:\n{format_diffs(errors)}"

        result = harness.step_red_only(0, RED_SCAN_START + start_host)
        errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
        assert len(errors) == 0, f"After scan:\n{format_diffs(errors)}"

    def test_blue_response_state_parity(self):
        """After red compromises a host, blue does analyse -> remove -> restore."""
        harness = self._make_harness(seed=42)
        harness.reset()

        from jaxborg.actions.encoding import (
            RED_DISCOVER_START,
            RED_EXPLOIT_SSH_START,
            RED_SCAN_START,
            encode_blue_action,
        )
        from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

        start_host = int(harness.jax_const.red_start_hosts[0])
        start_subnet = int(harness.jax_const.host_subnet[start_host])

        harness.step_red_only(0, RED_DISCOVER_START + start_subnet)
        harness.step_red_only(0, RED_SCAN_START + start_host)
        harness.step_red_only(0, RED_EXPLOIT_SSH_START + start_host)

        blue_actions = [
            encode_blue_action("Analyse", start_host, 0, const=harness.jax_const),
            encode_blue_action("Remove", start_host, 0, const=harness.jax_const),
            encode_blue_action("Restore", start_host, 0, const=harness.jax_const),
        ]

        for action in blue_actions:
            result = harness.step_blue_only(agent_id=0, action_idx=action)
            errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
            assert len(errors) == 0, f"After blue action {action}:\n{format_diffs(errors)}"

    def test_multi_seed_initial_parity(self):
        """Multiple seeds produce matching initial states."""
        from tests.differential.state_comparator import (
            _ERROR_FIELDS,
            compare_snapshots,
        )

        seeds = [42, 123, 456]
        total_errors = 0

        for seed in seeds:
            harness = self._make_harness(seed=seed)
            cyborg_snap, jax_snap = harness.reset()
            diffs = compare_snapshots(cyborg_snap, jax_snap)
            total_errors += sum(1 for d in diffs if d.field_name in _ERROR_FIELDS)

        assert total_errors == 0, f"Had {total_errors} errors across {len(seeds)} seeds"
