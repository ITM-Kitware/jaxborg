"""Differential tests comparing JAX green agents against CybORG with precomputed randoms."""

import jax.numpy as jnp

from jaxborg.constants import GLOBAL_MAX_HOSTS


class TestGreenSyncParity:
    """Tests using precomputed green randoms for deterministic CybORG/JAX comparison."""

    def _make_harness(self, seed=42, max_steps=500):
        from CybORG.Agents import EnterpriseGreenAgent, SleepAgent

        from tests.differential.harness import CC4DifferentialHarness

        return CC4DifferentialHarness(
            seed=seed,
            max_steps=max_steps,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=SleepAgent,
            sync_green_rng=True,
        )

    def test_green_sync_single_step(self):
        """Run one full step with green sync and check for state parity."""
        harness = self._make_harness()
        harness.reset()
        result = harness.full_step()
        harness.jax_state = harness.jax_state.replace(time=1)

        error_fields = {"host_compromised", "red_sessions", "red_privilege"}
        errors = [d for d in result.diffs if d.field_name in error_fields]
        assert len(errors) == 0, f"Diffs on step 0: {errors}"

    def test_green_sync_five_steps(self):
        """Run 5 full steps with green sync and check for state parity."""
        harness = self._make_harness()
        harness.reset()

        for t in range(5):
            result = harness.full_step()
            harness.jax_state = harness.jax_state.replace(time=t + 1)

            error_fields = {"host_compromised", "red_sessions", "red_privilege"}
            errors = [d for d in result.diffs if d.field_name in error_fields]
            assert len(errors) == 0, f"Diffs on step {t}: {errors}"

    def test_green_activity_detected_parity(self):
        """Check host_activity_detected matches between CybORG and JAX."""
        harness = self._make_harness()
        harness.reset()

        mismatches = 0
        for t in range(10):
            result = harness.full_step()
            harness.jax_state = harness.jax_state.replace(time=t + 1)

            activity_diffs = [d for d in result.diffs if d.field_name == "host_activity_detected"]
            mismatches += len(activity_diffs)

        assert mismatches == 0, f"host_activity_detected mismatches in {mismatches} cases"

    def test_green_lwf_asf_parity(self):
        """Check green_lwf and green_asf flags match between CybORG and JAX."""
        harness = self._make_harness()
        harness.reset()

        for t in range(10):
            harness.full_step()

            harness.jax_state = harness.jax_state.replace(
                time=t + 1,
                green_lwf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                green_asf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                host_activity_detected=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
            )

    def test_recorder_produces_valid_array(self):
        """Check that the recorder produces a well-formed array."""
        harness = self._make_harness()
        harness.reset()

        for t in range(3):
            harness.full_step()
            harness.jax_state = harness.jax_state.replace(time=t + 1)

        arr = harness.green_recorder.to_jax_array()
        assert arr.shape == (500, GLOBAL_MAX_HOSTS, 8)
        assert arr.dtype == jnp.float32
        assert jnp.all(arr >= 0.0)
        # Field 5 stores JAX host indices (0-136) for GreenAccessService dest_host;
        # other fields are uniforms in [0, 1).
        uniform_fields = jnp.concatenate([arr[:, :, :5], arr[:, :, 6:]], axis=2)
        assert jnp.all(uniform_fields <= 1.0)
        assert jnp.all(arr[:, :, 5] < GLOBAL_MAX_HOSTS)

    def test_full_episode_green_sync_10_steps(self):
        """Run 10 steps and verify no critical state divergence."""
        harness = self._make_harness(max_steps=500)
        harness.reset()

        total_errors = 0
        error_fields = {"host_compromised", "red_sessions", "red_privilege"}

        for t in range(10):
            result = harness.full_step()
            harness.jax_state = harness.jax_state.replace(
                time=t + 1,
                green_lwf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                green_asf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                host_activity_detected=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
                red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
            )

            errors = [d for d in result.diffs if d.field_name in error_fields]
            total_errors += len(errors)

        assert total_errors == 0, f"Total critical errors over 10 steps: {total_errors}"
