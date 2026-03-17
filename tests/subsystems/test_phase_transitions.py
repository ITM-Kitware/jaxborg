import jax
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import MISSION_PHASES
from jaxborg.rewards import advance_mission_phase
from jaxborg.state import create_initial_state


class TestAdvanceMissionPhase:
    def test_starts_at_phase_0(self, jax_const):
        state = create_initial_state()
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 0

    def test_phase_0_before_boundary_1(self, jax_const):
        boundary = int(jax_const.phase_boundaries[1])
        state = create_initial_state().replace(time=boundary - 1)
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 0

    def test_phase_1_at_boundary_1(self, jax_const):
        boundary = int(jax_const.phase_boundaries[1])
        state = create_initial_state().replace(time=boundary)
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 1

    def test_phase_1_just_after_boundary_1(self, jax_const):
        boundary = int(jax_const.phase_boundaries[1])
        state = create_initial_state().replace(time=boundary + 1)
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 1

    def test_phase_2_at_boundary_2(self, jax_const):
        boundary = int(jax_const.phase_boundaries[2])
        state = create_initial_state().replace(time=boundary)
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 2

    def test_phase_2_at_end(self, jax_const):
        state = create_initial_state().replace(time=499)
        state = advance_mission_phase(state, jax_const)
        assert int(state.mission_phase) == 2

    def test_all_steps_monotonic(self, jax_const):
        prev_phase = -1
        for t in range(500):
            state = create_initial_state().replace(time=t)
            state = advance_mission_phase(state, jax_const)
            phase = int(state.mission_phase)
            assert phase >= prev_phase, f"phase decreased at step {t}"
            prev_phase = phase

    def test_phase_boundaries_500_steps(self, jax_const):
        boundaries = np.array(jax_const.phase_boundaries)
        assert int(boundaries[0]) == 0
        assert int(boundaries[1]) == 167
        assert int(boundaries[2]) == 334

    def test_jit_compatible(self, jax_const):
        state = create_initial_state().replace(time=200)
        jitted = jax.jit(advance_mission_phase)
        result = jitted(state, jax_const)
        assert int(result.mission_phase) == 1


class TestPhaseTransitionsMatchCybORG:
    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_phase_boundaries_match_cyborg(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        cyborg_state = cyborg_env.environment_controller.state
        cyborg_phases = cyborg_state.scenario.mission_phases

        expected_boundaries = [0]
        cumul = 0
        for phase_len in cyborg_phases:
            cumul += phase_len
            expected_boundaries.append(cumul)

        jax_boundaries = np.array(const.phase_boundaries)
        for i in range(MISSION_PHASES):
            assert int(jax_boundaries[i]) == expected_boundaries[i], (
                f"phase boundary {i} mismatch: JAX={int(jax_boundaries[i])} CybORG={expected_boundaries[i]}"
            )

    def test_phase_transitions_match_cyborg_over_episode(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        cyborg_state = cyborg_env.environment_controller.state
        cyborg_phases = cyborg_state.scenario.mission_phases

        cyborg_phase_at_step = []
        for p, phase_len in enumerate(cyborg_phases):
            for _ in range(phase_len):
                cyborg_phase_at_step.append(p)

        for t in range(min(500, len(cyborg_phase_at_step))):
            state = create_initial_state().replace(time=t)
            state = advance_mission_phase(state, const)
            jax_phase = int(state.mission_phase)
            cyborg_phase = cyborg_phase_at_step[t]
            assert jax_phase == cyborg_phase, f"step={t}: JAX phase={jax_phase} CybORG phase={cyborg_phase}"


class TestPhaseDifferential:
    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_phase_boundaries_via_harness(self, cyborg_env):
        """Use the differential harness to compare phase at each boundary step."""
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        boundaries = np.array(const.phase_boundaries)

        for phase_idx in range(MISSION_PHASES):
            step = int(boundaries[phase_idx])
            state = create_initial_state().replace(time=step)
            state = advance_mission_phase(state, const)
            assert int(state.mission_phase) == phase_idx, (
                f"At step {step}, expected phase {phase_idx}, got {int(state.mission_phase)}"
            )
