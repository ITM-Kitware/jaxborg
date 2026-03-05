import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.duration import process_blue_with_duration, process_red_with_duration
from jaxborg.actions.encoding import (
    BLUE_ACTION_DURATIONS,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
    RED_ACTION_DURATIONS,
    RED_EXPLOIT_SSH_START,
    RED_PRIVESC_START,
    RED_SLEEP,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.env import CC4Env


@pytest.fixture(scope="module")
def env_and_state():
    key = jax.random.PRNGKey(42)
    env = CC4Env()
    obs, env_state = env.reset(key)
    return env, obs, env_state


class TestDurationLookupRed:
    def test_duration_lookup_red(self):
        expected = [1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 1, 3, 2, 2, 1]
        for i, val in enumerate(expected):
            assert int(RED_ACTION_DURATIONS[i]) == val, (
                f"RED_ACTION_DURATIONS[{i}] = {int(RED_ACTION_DURATIONS[i])}, expected {val}"
            )


class TestDurationLookupBlue:
    def test_duration_lookup_blue(self):
        expected = [1, 1, 2, 3, 5, 2, 1, 1]
        for i, val in enumerate(expected):
            assert int(BLUE_ACTION_DURATIONS[i]) == val, (
                f"BLUE_ACTION_DURATIONS[{i}] = {int(BLUE_ACTION_DURATIONS[i])}, expected {val}"
            )


class TestDuration1ExecutesImmediately:
    def test_duration_1_executes_immediately(self, env_and_state):
        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        key = jax.random.PRNGKey(0)
        new_state = jax.jit(process_red_with_duration, static_argnums=(2,))(state, const, 0, RED_SLEEP, key)
        assert int(new_state.red_pending_ticks[0]) == 0


class TestDuration4ExploitDeferred:
    def test_duration_4_exploit_deferred(self, env_and_state):
        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_hosts = jnp.where(state.red_sessions[0] & const.host_active, size=GLOBAL_MAX_HOSTS)[0]
        target_host = int(agent_hosts[0])
        exploit_action = RED_EXPLOIT_SSH_START + target_host

        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))

        key = jax.random.PRNGKey(10)

        s1 = process_jit(state, const, 0, exploit_action, key)
        assert int(s1.red_pending_ticks[0]) == 3
        assert int(s1.red_pending_action[0]) == exploit_action

        sessions_after_s1 = bool(s1.red_sessions[0, target_host])

        different_action = RED_SLEEP
        s2 = process_jit(s1, const, 0, different_action, key)
        assert int(s2.red_pending_ticks[0]) == 2

        s3 = process_jit(s2, const, 0, different_action, key)
        assert int(s3.red_pending_ticks[0]) == 1

        s4 = process_jit(s3, const, 0, different_action, key)
        assert int(s4.red_pending_ticks[0]) == 0

        sessions_before = bool(state.red_sessions[0, target_host])
        sessions_after_s3 = bool(s3.red_sessions[0, target_host])
        assert sessions_after_s1 == sessions_before
        assert sessions_after_s3 == sessions_before


class TestBusyAgentIgnoresNewActions:
    def test_busy_agent_ignores_new_actions(self, env_and_state):
        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_hosts = jnp.where(state.red_sessions[0] & const.host_active, size=GLOBAL_MAX_HOSTS)[0]
        target_host = int(agent_hosts[0])
        exploit_action = RED_EXPLOIT_SSH_START + target_host

        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))
        key = jax.random.PRNGKey(10)

        s1 = process_jit(state, const, 0, exploit_action, key)
        assert int(s1.red_pending_ticks[0]) == 3
        stored_action = int(s1.red_pending_action[0])

        different_action = RED_PRIVESC_START + target_host
        s2 = process_jit(s1, const, 0, different_action, key)
        assert int(s2.red_pending_action[0]) == stored_action
        assert int(s2.red_pending_ticks[0]) == 2


class TestBlueRestoreDuration5:
    def test_blue_restore_duration_5(self, env_and_state):
        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        host_indices = jnp.where(const.host_active, size=GLOBAL_MAX_HOSTS)[0]
        target_host = int(host_indices[0])
        restore_action = BLUE_RESTORE_START + target_host

        process_jit = jax.jit(process_blue_with_duration, static_argnums=(2,))

        s1 = process_jit(state, const, 0, restore_action)
        assert int(s1.blue_pending_ticks[0]) == 4

        s2 = process_jit(s1, const, 0, BLUE_SLEEP)
        assert int(s2.blue_pending_ticks[0]) == 3

        s3 = process_jit(s2, const, 0, BLUE_SLEEP)
        assert int(s3.blue_pending_ticks[0]) == 2

        s4 = process_jit(s3, const, 0, BLUE_SLEEP)
        assert int(s4.blue_pending_ticks[0]) == 1

        s5 = process_jit(s4, const, 0, BLUE_SLEEP)
        assert int(s5.blue_pending_ticks[0]) == 0


class TestEnvStepUsesDuration:
    def test_env_step_uses_duration(self, env_and_state):
        env, obs, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_hosts = jnp.where(state.red_sessions[0] & const.host_active, size=GLOBAL_MAX_HOSTS)[0]
        target_host = int(agent_hosts[0])
        exploit_action = RED_EXPLOIT_SSH_START + target_host

        actions = {}
        for i in range(NUM_BLUE_AGENTS):
            actions[f"blue_{i}"] = jnp.int32(BLUE_SLEEP)
        for i in range(NUM_RED_AGENTS):
            actions[f"red_{i}"] = jnp.int32(RED_SLEEP)
        actions["red_0"] = jnp.int32(exploit_action)

        key = jax.random.PRNGKey(99)
        _, new_env_state, _, _, _ = env.step_env(key, env_state, actions)
        assert int(new_env_state.state.red_pending_ticks[0]) == 3


def _get_cyborg_remaining_ticks(controller, agent_name):
    """Get CybORG remaining_ticks for an agent, or 0 if idle."""
    aip = controller.actions_in_progress.get(agent_name)
    if aip is None:
        return 0
    return aip["remaining_ticks"]


class TestFsmRedEnvDurationTicks:
    """Test JAX-native duration tick tracking via FsmRedCC4Env (training code path)."""

    @pytest.fixture(scope="class")
    def fsm_env_and_state(self):
        from jaxborg.fsm_red_env import FsmRedCC4Env

        env = FsmRedCC4Env(num_steps=100)
        key = jax.random.PRNGKey(42)
        obs, env_state = env.reset(key)
        return env, env_state

    def test_exploit_tick_countdown(self, fsm_env_and_state):
        """Run FsmRedCC4Env until an exploit fires, verify tick countdown 3→2→1→0."""
        env, env_state = fsm_env_and_state
        key = jax.random.PRNGKey(42)

        saw_exploit_countdown = False
        for step in range(100):
            key, subkey = jax.random.split(key)
            actions = {f"blue_{b}": jnp.int32(BLUE_SLEEP) for b in range(NUM_BLUE_AGENTS)}
            obs, env_state, _, dones, _ = env.step_env(subkey, env_state, actions)

            for r in range(NUM_RED_AGENTS):
                ticks = int(env_state.state.red_pending_ticks[r])
                if ticks == 3:
                    s = env_state
                    countdown = [3]
                    for _ in range(3):
                        key, subkey = jax.random.split(key)
                        _, s, _, _, _ = env.step_env(subkey, s, actions)
                        countdown.append(int(s.state.red_pending_ticks[r]))
                    assert countdown == [3, 2, 1, 0], f"Expected [3,2,1,0], got {countdown}"
                    saw_exploit_countdown = True
                    break
            if saw_exploit_countdown:
                break

        assert saw_exploit_countdown, "No exploit (duration=4) seen in 100 steps"

    def test_blue_restore_tick_countdown(self, fsm_env_and_state):
        """Submit Blue Restore via FsmRedCC4Env, verify ticks count 4→3→2→1→0."""
        env, env_state = fsm_env_and_state
        key = jax.random.PRNGKey(43)

        active = env_state.const.host_active
        blue_hosts = env_state.const.blue_agent_hosts[0] & active
        target = int(jnp.argmax(blue_hosts))
        restore_action = BLUE_RESTORE_START + target

        countdown = []
        for step in range(6):
            key, subkey = jax.random.split(key)
            actions = {f"blue_{b}": jnp.int32(BLUE_SLEEP) for b in range(NUM_BLUE_AGENTS)}
            if step == 0:
                actions["blue_0"] = jnp.int32(restore_action)
            obs, env_state, _, _, _ = env.step_env(subkey, env_state, actions)
            countdown.append(int(env_state.state.blue_pending_ticks[0]))

        assert countdown == [4, 3, 2, 1, 0, 0], f"Expected [4,3,2,1,0,0], got {countdown}"

    def test_busy_agent_pending_ticks_nonzero_across_steps(self, fsm_env_and_state):
        """While red agent has pending exploit, new actions submitted are ignored."""
        env, env_state = fsm_env_and_state
        key = jax.random.PRNGKey(44)

        actions = {f"blue_{b}": jnp.int32(BLUE_SLEEP) for b in range(NUM_BLUE_AGENTS)}

        found_busy = False
        for step in range(100):
            key, subkey = jax.random.split(key)
            obs, env_state, _, _, _ = env.step_env(subkey, env_state, actions)
            for r in range(NUM_RED_AGENTS):
                ticks = int(env_state.state.red_pending_ticks[r])
                if ticks > 1:
                    stored_action = int(env_state.state.red_pending_action[r])
                    key, subkey = jax.random.split(key)
                    _, next_state, _, _, _ = env.step_env(subkey, env_state, actions)
                    assert int(next_state.state.red_pending_action[r]) == stored_action
                    assert int(next_state.state.red_pending_ticks[r]) == ticks - 1
                    found_busy = True
                    break
            if found_busy:
                break

        assert found_busy, "No busy red agent observed in 100 steps"


class TestDurationDifferential:
    """Differential tests verifying JAX duration tracking matches CybORG.

    The harness now uses process_red_with_duration / process_blue_with_duration
    (same as the training code path), so JAX pending_ticks should match CybORG
    remaining_ticks at every step.
    """

    def test_red_exploit_ticks_match_cyborg(self):
        """Verify JAX red_pending_ticks matches CybORG remaining_ticks each step."""
        pytest.importorskip("CybORG")
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        harness = CC4DifferentialHarness(seed=42, max_steps=50, sync_green_rng=True)
        harness.reset()
        controller = harness.cyborg_env.environment_controller

        errors = []
        red_deferred_seen = False

        for step in range(50):
            result = harness.full_step()
            step_errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
            if step_errors:
                errors.append((step, step_errors))

            for r in range(NUM_RED_AGENTS):
                cy_ticks = _get_cyborg_remaining_ticks(controller, f"red_agent_{r}")
                jax_ticks = int(harness.jax_state.red_pending_ticks[r])
                if cy_ticks > 0 or jax_ticks > 0:
                    red_deferred_seen = True
                    assert cy_ticks == jax_ticks, (
                        f"Step {step} red_agent_{r}: CybORG remaining_ticks={cy_ticks}, JAX pending_ticks={jax_ticks}"
                    )

        assert not errors, f"State parity errors: {errors[:3]}"
        assert red_deferred_seen, "No deferred red actions observed in 50 steps — test did not exercise duration"

    def test_blue_restore_ticks_match_cyborg(self):
        """Submit Blue Restore (duration=5), verify tick countdown matches CybORG."""
        pytest.importorskip("CybORG")
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        harness = CC4DifferentialHarness(seed=42, max_steps=20, sync_green_rng=True)
        harness.reset()
        controller = harness.cyborg_env.environment_controller

        active = harness.jax_const.host_active
        blue_hosts = harness.jax_const.blue_agent_hosts[0] & active
        target = int(jnp.argmax(blue_hosts))

        errors = []
        jax_ticks_per_step = []

        for step in range(10):
            if step == 0:
                blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
                blue_actions[0] = BLUE_RESTORE_START + target
            else:
                blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}

            result = harness.full_step(blue_actions=blue_actions)
            step_errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
            if step_errors:
                errors.append((step, step_errors))

            cy_ticks = _get_cyborg_remaining_ticks(controller, "blue_agent_0")
            jax_ticks = int(harness.jax_state.blue_pending_ticks[0])
            jax_ticks_per_step.append(jax_ticks)
            assert cy_ticks == jax_ticks, (
                f"Step {step} blue_agent_0: CybORG remaining_ticks={cy_ticks}, JAX pending_ticks={jax_ticks}"
            )

        assert not errors, f"Blue restore parity errors: {errors[:3]}"
        assert jax_ticks_per_step[0] == 4, f"Step 0: expected ticks=4, got {jax_ticks_per_step[0]}"
        assert jax_ticks_per_step[4] == 0, f"Step 4: expected ticks=0 (executed), got {jax_ticks_per_step[4]}"

    def test_multi_seed_tick_parity(self):
        """Multiple seeds × 40 steps: verify tick parity for red agents at every step."""
        pytest.importorskip("CybORG")
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        total_deferred = 0
        for seed in [0, 7, 42, 99, 123]:
            harness = CC4DifferentialHarness(seed=seed, max_steps=40, sync_green_rng=True)
            harness.reset()
            controller = harness.cyborg_env.environment_controller

            errors = []
            for step in range(40):
                result = harness.full_step()
                step_errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
                if step_errors:
                    errors.append((seed, step, step_errors))
                    break

                for r in range(NUM_RED_AGENTS):
                    cy_ticks = _get_cyborg_remaining_ticks(controller, f"red_agent_{r}")
                    jax_ticks = int(harness.jax_state.red_pending_ticks[r])
                    if cy_ticks > 0 or jax_ticks > 0:
                        total_deferred += 1
                        assert cy_ticks == jax_ticks, (
                            f"Seed {seed} step {step} red_agent_{r}: CybORG={cy_ticks}, JAX={jax_ticks}"
                        )

            assert not errors, f"Parity errors at seed={seed}: {errors}"

        assert total_deferred > 0, "No deferred actions across 5 seeds × 40 steps — duration not exercised"
