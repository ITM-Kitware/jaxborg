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

    def test_initial_policy_input_parity(self):
        """After reset, matched JAX/CybORG states must produce the same blue obs and masks."""
        harness = self._make_harness(seed=42)
        harness.reset()

        from tests.differential.state_comparator import format_diffs

        diffs = harness.compare_policy_inputs()
        errors = [d for d in diffs if d.field_name in {"observation", "action_mask"}]
        assert len(errors) == 0, f"Initial policy inputs:\n{format_diffs(errors)}"

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
        """After red compromises a host, blue does analyse -> remove -> restore.

        step_red/blue_only bypass CybORG's action-space validation and duration
        system, so we only compare state on the specific host being targeted
        (green/phishing side-effects on other hosts are expected to diverge).
        """
        harness = self._make_harness(seed=42)
        harness.reset()

        from CybORG.Simulator.Actions import Sleep

        from jaxborg.actions.encoding import (
            RED_DISCOVER_START,
            RED_EXPLOIT_SSH_START,
            RED_SCAN_START,
            encode_blue_action,
        )
        from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

        start_host = int(harness.jax_const.red_start_hosts[0])
        start_subnet = int(harness.jax_const.host_subnet[start_host])
        host_label = f"host_{start_host}"
        agent_host_labels = {f"red_{a}_host_{start_host}" for a in range(6)}

        harness.step_red_only(0, RED_DISCOVER_START + start_subnet)
        harness.step_red_only(0, RED_SCAN_START + start_host)
        # ExploitRemoteService has duration=4 in CybORG; step_red_only applies
        # it immediately in JAX but CybORG needs 4 steps to complete.
        harness.step_red_only(0, RED_EXPLOIT_SSH_START + start_host)
        for _ in range(3):
            harness.cyborg_env.step(agent="red_agent_0", action=Sleep(), skip_valid_action_check=True)

        blue_actions = [
            encode_blue_action("Analyse", start_host, 0, const=harness.jax_const),
            encode_blue_action("Remove", start_host, 0, const=harness.jax_const),
            encode_blue_action("Restore", start_host, 0, const=harness.jax_const),
        ]

        for action in blue_actions:
            result = harness.step_blue_only(agent_id=0, action_idx=action)
            errors = [
                d
                for d in result.diffs
                if d.field_name in _ERROR_FIELDS and d.host_or_agent in (host_label, *agent_host_labels)
            ]
            assert len(errors) == 0, f"After blue action {action}:\n{format_diffs(errors)}"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_multi_seed_initial_parity(self, seed):
        """Multiple seeds produce matching initial states."""
        from tests.differential.state_comparator import (
            _ERROR_FIELDS,
            compare_snapshots,
            format_diffs,
        )

        harness = self._make_harness(seed=seed)
        cyborg_snap, jax_snap = harness.reset()
        diffs = compare_snapshots(cyborg_snap, jax_snap)
        errors = [d for d in diffs if d.field_name in _ERROR_FIELDS]
        assert len(errors) == 0, f"seed={seed}:\n{format_diffs(errors)}"

    def test_policy_input_parity_after_five_full_steps(self):
        """Obs and masks should stay aligned through several matched full steps."""
        harness = self._make_harness(seed=42)
        harness.reset()

        from tests.differential.state_comparator import format_diffs

        for step in range(5):
            result = harness.full_step()
            errors = [d for d in result.diffs if d.field_name in {"observation", "action_mask"}]
            assert len(errors) == 0, f"Policy inputs after full step {step + 1}:\n{format_diffs(errors)}"
