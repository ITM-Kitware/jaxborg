import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.duration import process_blue_with_duration
from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    encode_blue_action,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import (
    BLUE_ACTION_HOST_SLOTS,
    BLUE_MAX_OBSERVED_SUBNETS,
    BLUE_TRAFFIC_SLOTS,
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
)
from jaxborg.topology import build_topology


class TestActionMaskShape:
    def test_mask_shape(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert mask.dtype == jnp.bool_


class TestSleepMonitorAlwaysValid:
    def test_sleep_always_true(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        assert mask[0]

    def test_monitor_always_true(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        assert mask[1]


class TestHostBasedActions:
    def test_only_agent_subnets_valid(self):
        """Slots for relative subnets that are unused (-1) are masked."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)

        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            agent_obs_subnets = np.array(const.blue_obs_subnets[agent_id])

            for rel_idx in range(BLUE_MAX_OBSERVED_SUBNETS):
                sid = agent_obs_subnets[rel_idx]
                for slot in range(OBS_HOSTS_PER_SUBNET):
                    flat_slot = rel_idx * OBS_HOSTS_PER_SUBNET + slot
                    action_idx = BLUE_ANALYSE_START + flat_slot
                    if sid == -1:
                        assert not mask[action_idx], (
                            f"agent={agent_id} rel_idx={rel_idx} slot={slot} should be masked (unused subnet)"
                        )

    def test_empty_obs_slots_masked(self):
        """Slots where obs_host_map == GLOBAL_MAX_HOSTS are masked.
        Router slots (slot == OBS_HOSTS_PER_SUBNET - 1) are also masked,
        matching CybORG's BlueFlatWrapper which excludes routers.
        """
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)

        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            agent_obs_subnets = np.array(const.blue_obs_subnets[agent_id])

            for rel_idx in range(BLUE_MAX_OBSERVED_SUBNETS):
                sid = agent_obs_subnets[rel_idx]
                if sid == -1:
                    continue
                for slot in range(OBS_HOSTS_PER_SUBNET):
                    h = int(const.obs_host_map[sid, slot])
                    flat_slot = rel_idx * OBS_HOSTS_PER_SUBNET + slot
                    action_idx = BLUE_ANALYSE_START + flat_slot
                    is_router = slot == OBS_HOSTS_PER_SUBNET - 1
                    if h == GLOBAL_MAX_HOSTS or is_router:
                        assert not mask[action_idx], f"Slot rel={rel_idx} slot={slot} should be masked"
                    else:
                        assert mask[action_idx], f"Valid host rel={rel_idx} slot={slot} should be unmasked"

    def test_same_mask_for_analyse_remove_restore(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            for slot in range(BLUE_ACTION_HOST_SLOTS):
                a = bool(mask[BLUE_ANALYSE_START + slot])
                r = bool(mask[BLUE_REMOVE_START + slot])
                s = bool(mask[BLUE_RESTORE_START + slot])
                assert a == r == s, f"Mismatch at slot {slot}: analyse={a}, remove={r}, restore={s}"

    def test_decoy_mask_respects_initial_service_compatibility(self):
        """With collapsed action space, a decoy slot is valid if ANY type is compatible."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            for slot in range(BLUE_ACTION_HOST_SLOTS):
                host_valid = bool(mask[BLUE_ANALYSE_START + slot])
                # Tomcat is always compatible, so any valid host slot should have decoy enabled
                expected = host_valid
                actual = bool(mask[BLUE_DECOY_START + slot])
                assert actual == expected, f"Decoy slot {slot} mismatch: expected={expected}, actual={actual}"


class TestTrafficActions:
    def test_self_loops_masked(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            agent_obs_subnets = np.array(const.blue_obs_subnets[agent_id])
            for src in range(NUM_SUBNETS):
                for rel_dst in range(BLUE_MAX_OBSERVED_SUBNETS):
                    abs_dst = agent_obs_subnets[rel_dst]
                    if abs_dst == -1:
                        continue
                    idx = BLUE_BLOCK_TRAFFIC_START + src * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
                    if src == abs_dst:
                        assert not mask[idx], f"Self-loop src={src} dst={abs_dst} should be masked"

    def test_uncontrolled_dst_masked(self):
        """Unused relative dst slots (-1) should be masked."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            agent_obs_subnets = np.array(const.blue_obs_subnets[agent_id])
            for src in range(NUM_SUBNETS):
                for rel_dst in range(BLUE_MAX_OBSERVED_SUBNETS):
                    abs_dst = agent_obs_subnets[rel_dst]
                    idx_block = BLUE_BLOCK_TRAFFIC_START + src * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
                    if abs_dst == -1 or src == abs_dst:
                        assert not mask[idx_block], (
                            f"Block src={src} rel_dst={rel_dst} (abs={abs_dst}) should be masked"
                        )
                    else:
                        assert mask[idx_block], f"Block src={src} rel_dst={rel_dst} (abs={abs_dst}) should be valid"

    def test_block_allow_same_mask(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            block_slice = mask[BLUE_BLOCK_TRAFFIC_START : BLUE_BLOCK_TRAFFIC_START + BLUE_TRAFFIC_SLOTS]
            allow_slice = mask[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + BLUE_TRAFFIC_SLOTS]
            np.testing.assert_array_equal(block_slice, allow_slice)


class TestJITCompatibility:
    def test_jit_compiles(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        jitted = jax.jit(compute_blue_action_mask, static_argnums=(1,))
        mask = jitted(const, 0)
        assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert mask[0]


class TestBusyBlueMask:
    def test_busy_agent_sleep_only(self):
        """Mask forces Sleep-only during pending multi-tick action (matches CybORG behavior)."""
        from jaxborg.env import CC4Env

        env = CC4Env(num_steps=100)
        _, env_state = env.reset(jax.random.PRNGKey(42))
        const = env_state.const
        state = env_state.state

        active = np.array(const.host_active, dtype=bool)
        controllable = (
            np.array(const.blue_agent_hosts[0], dtype=bool) & active & ~np.array(const.host_is_router, dtype=bool)
        )
        target_host = int(np.flatnonzero(controllable)[0])
        pending_action = encode_blue_action("Restore", target_host, 0, const=const)

        idle_mask = np.array(compute_blue_action_mask(const, 0, state), dtype=bool)
        assert idle_mask.sum() > 2, "idle mask should have multiple valid actions"

        busy_state = process_blue_with_duration(state, const, 0, pending_action)
        busy_mask = np.array(compute_blue_action_mask(const, 0, busy_state), dtype=bool)

        assert int(busy_state.blue_pending_ticks[0]) == 4
        assert busy_mask[0] is np.True_, "Sleep must be valid when busy"
        assert busy_mask.sum() == 1, "only Sleep should be valid when busy"


class TestWithRealTopology:
    def test_mask_from_build_topology(self):
        key = jax.random.PRNGKey(42)
        const = build_topology(key, num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
            assert mask[0]
            assert mask[1]
            num_valid = int(mask.sum())
            assert num_valid > 2, f"Agent {agent_id} has only {num_valid} valid actions"
            assert num_valid < BLUE_ALLOW_TRAFFIC_END
