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
    ACTION_HOST_SLOTS,
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
        """Slots for subnets not controlled by the agent are masked."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)  # blue_agent_0 controls 1 subnet
        agent_subnets = np.array(const.blue_agent_subnets[0])

        for sid in range(NUM_SUBNETS):
            for slot in range(OBS_HOSTS_PER_SUBNET):
                flat_slot = sid * OBS_HOSTS_PER_SUBNET + slot
                action_idx = BLUE_ANALYSE_START + flat_slot
                if not agent_subnets[sid]:
                    assert not mask[action_idx], f"sid={sid} slot={slot} should be masked"

    def test_empty_obs_slots_masked(self):
        """Slots where obs_host_map == GLOBAL_MAX_HOSTS are masked.
        Router slots (slot == OBS_HOSTS_PER_SUBNET - 1) are also masked,
        matching CybORG's BlueFlatWrapper which excludes routers.
        """
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        agent_subnets = np.array(const.blue_agent_subnets[0])

        for sid in range(NUM_SUBNETS):
            if not agent_subnets[sid]:
                continue
            for slot in range(OBS_HOSTS_PER_SUBNET):
                h = int(const.obs_host_map[sid, slot])
                flat_slot = sid * OBS_HOSTS_PER_SUBNET + slot
                action_idx = BLUE_ANALYSE_START + flat_slot
                is_router = slot == OBS_HOSTS_PER_SUBNET - 1
                if h == GLOBAL_MAX_HOSTS or is_router:
                    assert not mask[action_idx], f"Slot sid={sid} slot={slot} should be masked"
                else:
                    assert mask[action_idx], f"Valid host sid={sid} slot={slot} should be unmasked"

    def test_same_mask_for_analyse_remove_restore(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        for slot in range(ACTION_HOST_SLOTS):
            a = bool(mask[BLUE_ANALYSE_START + slot])
            r = bool(mask[BLUE_REMOVE_START + slot])
            s = bool(mask[BLUE_RESTORE_START + slot])
            assert a == r == s, f"Mismatch at slot {slot}: analyse={a}, remove={r}, restore={s}"

    def test_decoy_mask_respects_initial_service_compatibility(self):
        """With collapsed action space, a decoy slot is valid if ANY type is compatible."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        for slot in range(ACTION_HOST_SLOTS):
            host_valid = bool(mask[BLUE_ANALYSE_START + slot])
            sid = slot // OBS_HOSTS_PER_SUBNET
            subslot = slot % OBS_HOSTS_PER_SUBNET
            int(const.obs_host_map[sid, subslot])

            # Tomcat is always compatible, so any valid host slot should have decoy enabled
            expected = host_valid
            actual = bool(mask[BLUE_DECOY_START + slot])
            assert actual == expected, f"Decoy slot {slot} mismatch: expected={expected}, actual={actual}"


class TestTrafficActions:
    def test_self_loops_masked(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        for s in range(NUM_SUBNETS):
            idx = BLUE_BLOCK_TRAFFIC_START + s * NUM_SUBNETS + s
            assert not mask[idx], f"Self-loop src={s} dst={s} should be masked"

    def test_uncontrolled_dst_masked(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        agent_subnets = np.array(const.blue_agent_subnets[0])

        for src in range(NUM_SUBNETS):
            for dst in range(NUM_SUBNETS):
                idx_block = BLUE_BLOCK_TRAFFIC_START + src * NUM_SUBNETS + dst
                if not agent_subnets[dst] or src == dst:
                    assert not mask[idx_block], f"Block src={src} dst={dst} should be masked"
                else:
                    assert mask[idx_block], f"Block src={src} dst={dst} should be valid"

    def test_allow_traffic_masked_when_nothing_blocked(self):
        """AllowTraffic is a no-op when nothing is blocked, so it should be masked."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        # Without state (reset-time): nothing blocked → AllowTraffic all False
        mask = compute_blue_action_mask(const, 0)
        n = NUM_SUBNETS * NUM_SUBNETS
        allow_slice = mask[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + n]
        assert not allow_slice.any(), "AllowTraffic should be all-masked when nothing is blocked"

    def test_allow_traffic_unmasked_when_zone_blocked(self):
        """AllowTraffic should be valid for zones that are actually blocked."""
        from jaxborg.env import CC4Env

        env = CC4Env(num_steps=100)
        _, env_state = env.reset(jax.random.PRNGKey(42))
        const = env_state.const
        state = env_state.state

        # Find a (src, dst) pair where agent 0 controls dst and src != dst
        agent_subnets = np.array(const.blue_agent_subnets[0])
        src, dst = -1, -1
        for s in range(NUM_SUBNETS):
            for d in range(NUM_SUBNETS):
                if agent_subnets[d] and s != d:
                    src, dst = s, d
                    break
            if src >= 0:
                break
        assert src >= 0, "No valid traffic pair for agent 0"

        # Initially: AllowTraffic for this pair should be masked (not blocked)
        mask_before = compute_blue_action_mask(const, 0, state)
        allow_idx = BLUE_ALLOW_TRAFFIC_START + src * NUM_SUBNETS + dst
        assert not mask_before[allow_idx], "AllowTraffic should be masked when zone is not blocked"

        # Block the zone, then AllowTraffic should be unmasked
        blocked_state = state.replace(blocked_zones=state.blocked_zones.at[dst, src].set(True))
        mask_after = compute_blue_action_mask(const, 0, blocked_state)
        assert mask_after[allow_idx], "AllowTraffic should be valid when zone is blocked"

        # BlockTraffic should still be valid in both cases
        block_idx = BLUE_BLOCK_TRAFFIC_START + src * NUM_SUBNETS + dst
        assert mask_before[block_idx], "BlockTraffic should be valid when not blocked"
        assert mask_after[block_idx], "BlockTraffic should still be valid when blocked"


class TestJITCompatibility:
    def test_jit_compiles(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        jitted = jax.jit(compute_blue_action_mask, static_argnums=(1,))
        mask = jitted(const, 0)
        assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert mask[0]


class TestBusyBlueMask:
    def test_busy_agent_only_exposes_pending_action(self):
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

        busy_state = process_blue_with_duration(state, const, 0, pending_action)
        mask = np.array(compute_blue_action_mask(const, 0, busy_state), dtype=bool)

        assert int(busy_state.blue_pending_ticks[0]) == 4
        assert mask.sum() == 1
        assert mask[pending_action]


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
