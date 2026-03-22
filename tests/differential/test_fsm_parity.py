import pytest

from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import compare_fast


class TestFsmParity:
    def test_fsm_states_match_at_reset(self):
        harness = CC4DifferentialHarness(seed=42, max_steps=30, sync_green_rng=True, strip_inactive_knowledge=True)
        harness.reset()

        diffs = compare_fast(harness.cyborg_env, harness.jax_state, harness.jax_const, harness.mappings)
        fsm_diffs = [d for d in diffs if d.field_name == "fsm_host_states"]

        if fsm_diffs:
            lines = []
            for d in fsm_diffs:
                lines.append(f"  {d.host_or_agent} cyborg={d.cyborg_value} jax={d.jax_value}")
            msg = f"FSM divergences at reset ({len(fsm_diffs)} diffs):\n" + "\n".join(lines)
            pytest.fail(msg)

    @pytest.mark.xfail(
        reason="Env gap: FSM eligible set includes start host (K=0) that CybORG hasn't observed",
        strict=False,
    )
    def test_fsm_states_tracked_through_episode(self):
        harness = CC4DifferentialHarness(seed=42, max_steps=30, sync_green_rng=True, strip_inactive_knowledge=True)
        harness.reset()

        fsm_divergences = []
        for t in range(30):
            result = harness.full_step()
            fsm_diffs = [d for d in result.diffs if d.field_name == "fsm_host_states"]
            if fsm_diffs:
                fsm_divergences.append((t, fsm_diffs))

        # Report all FSM divergences
        if fsm_divergences:
            lines = []
            for step, diffs in fsm_divergences:
                for d in diffs:
                    lines.append(f"  step {step}: {d.host_or_agent} cyborg={d.cyborg_value} jax={d.jax_value}")
            msg = f"FSM divergences found in {len(fsm_divergences)} steps:\n" + "\n".join(lines)
            pytest.fail(msg)

    def test_session_counts_tracked(self):
        harness = CC4DifferentialHarness(seed=42, max_steps=30, sync_green_rng=True, strip_inactive_knowledge=True)
        harness.reset()

        session_divergences = []
        for t in range(30):
            result = harness.full_step()
            session_diffs = [d for d in result.diffs if d.field_name == "red_session_count"]
            if session_diffs:
                session_divergences.append((t, session_diffs))

        if session_divergences:
            lines = []
            for step, diffs in session_divergences:
                for d in diffs:
                    lines.append(f"  step {step}: {d.host_or_agent} cyborg={d.cyborg_value} jax={d.jax_value}")
            msg = f"red_session_count divergences found in {len(session_divergences)} steps:\n" + "\n".join(lines)
            pytest.fail(msg)

    def test_suspicious_pids_tracked(self):
        harness = CC4DifferentialHarness(seed=42, max_steps=30, sync_green_rng=True, strip_inactive_knowledge=True)
        harness.reset()

        pids_divergences = []
        for t in range(30):
            result = harness.full_step()
            pids_diffs = [d for d in result.diffs if d.field_name == "blue_suspicious_pids"]
            if pids_diffs:
                pids_divergences.append((t, pids_diffs))

        if pids_divergences:
            lines = []
            for step, diffs in pids_divergences:
                for d in diffs:
                    lines.append(f"  step {step}: {d.host_or_agent} cyborg={d.cyborg_value} jax={d.jax_value}")
            msg = f"blue_suspicious_pids divergences found in {len(pids_divergences)} steps:\n" + "\n".join(lines)
            pytest.fail(msg)
