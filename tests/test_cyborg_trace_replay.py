"""CybORG trace replay test.

Records a full CybORG episode trace (internal state at each step),
then replays the exact same action sequence in JaxBorg.
Compares ALL available state fields at every step — catches divergences
in fields the harness doesn't normally extract.
"""

import pytest

from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import (
    _ERROR_FIELDS,
    _WARNING_FIELDS,
)


class TestCyborgTraceReplay:
    """Record CybORG traces and replay in JAX, comparing exhaustively."""

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_full_episode_replay(self, seed):
        """Run a full differential episode and record all diffs per step.

        This is more thorough than the fuzzer because it:
        1. Checks ALL fields (errors + warnings + new R1 fields)
        2. Accumulates all diffs across the entire episode (doesn't stop on first)
        3. Reports a comprehensive divergence timeline
        """
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=100,
            sync_green_rng=True,
            check_obs=True,
            check_masks=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        all_divergences: dict[str, list[tuple[int, str]]] = {}
        error_count = 0

        for t in range(100):
            result = harness.full_step()

            for d in result.diffs:
                field = d.field_name
                if field not in all_divergences:
                    all_divergences[field] = []
                all_divergences[field].append((t, f"{d.host_or_agent}: cyborg={d.cyborg_value} jax={d.jax_value}"))
                if field in _ERROR_FIELDS:
                    error_count += 1

        # Report
        if all_divergences:
            lines = [f"Trace replay seed={seed}: {len(all_divergences)} field types diverged"]
            for field, occurrences in sorted(all_divergences.items()):
                severity = "ERROR" if field in _ERROR_FIELDS else "WARN"
                lines.append(f"\n  [{severity}] {field}: {len(occurrences)} occurrences")
                for step, detail in occurrences[:5]:
                    lines.append(f"    step {step}: {detail}")
                if len(occurrences) > 5:
                    lines.append(f"    ... and {len(occurrences) - 5} more")
            report = "\n".join(lines)

            if error_count > 0:
                pytest.fail(report)
            else:
                # Only warnings — print but don't fail
                print(f"\n{report}")

    @pytest.mark.parametrize("seed", [42, 7])
    def test_reward_accumulation_parity(self, seed):
        """Compare cumulative rewards across entire episode."""
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=100,
            sync_green_rng=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        cyborg_total = 0.0
        jax_total = 0.0

        for t in range(100):
            result = harness.full_step()
            cyborg_total += result.cyborg_rewards.get("total", 0.0)
            jax_total += result.jax_rewards.get("total", 0.0)

        diff = abs(cyborg_total - jax_total)
        assert diff < 1e-3, (
            f"Cumulative reward drift at seed={seed}: cyborg={cyborg_total:.4f} jax={jax_total:.4f} diff={diff:.6f}"
        )


class TestTraceFieldCoverage:
    """Verify the comparator covers critical state fields."""

    def test_state_comparator_field_coverage(self):
        """Check that important CC4State fields are in ERROR or WARNING sets."""
        critical_fields = {
            "host_compromised",
            "red_sessions",
            "red_privilege",
            "host_services",
            "host_service_reliability",
            "blocked_zones",
            "mission_phase",
            "rewards",
            "observation",
            "fsm_host_states",
            "red_session_count",
            "blue_suspicious_pids",
        }

        covered = _ERROR_FIELDS | _WARNING_FIELDS
        uncovered = critical_fields - covered
        assert not uncovered, f"Critical fields not covered by comparator: {uncovered}"


class TestWarningFieldFrequency:
    """Report how often warning fields diverge to prioritize promotion."""

    @pytest.mark.parametrize("seed", [42, 7])
    def test_warning_divergence_frequency(self, seed):
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=50,
            sync_green_rng=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        warning_counts: dict[str, int] = {}
        for t in range(50):
            result = harness.full_step()
            for d in result.diffs:
                if d.field_name in _WARNING_FIELDS:
                    warning_counts[d.field_name] = warning_counts.get(d.field_name, 0) + 1

        if warning_counts:
            print(f"\nWarning field divergence frequency (seed={seed}, 50 steps):")
            for field, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
                print(f"  {field}: {count} occurrences ({count / 50 * 100:.0f}% of steps)")
