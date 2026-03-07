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
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
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
        """Slots where obs_host_map == GLOBAL_MAX_HOSTS are masked."""
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
                if h == GLOBAL_MAX_HOSTS:
                    assert not mask[action_idx], f"Empty slot sid={sid} slot={slot} should be masked"
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
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        for slot in range(ACTION_HOST_SLOTS):
            host_valid = bool(mask[BLUE_ANALYSE_START + slot])
            sid = slot // OBS_HOSTS_PER_SUBNET
            subslot = slot % OBS_HOSTS_PER_SUBNET
            host_idx = int(const.obs_host_map[sid, subslot])

            expected_by_type = {
                DECOY_IDS["HarakaSMPT"]: host_valid
                and host_idx != GLOBAL_MAX_HOSTS
                and not bool(const.initial_services[host_idx, SERVICE_IDS["SMTP"]]),
                DECOY_IDS["Apache"]: host_valid
                and host_idx != GLOBAL_MAX_HOSTS
                and not bool(const.initial_services[host_idx, SERVICE_IDS["APACHE2"]]),
                DECOY_IDS["Tomcat"]: host_valid,
                DECOY_IDS["Vsftpd"]: host_valid,
            }

            for decoy_type, expected in expected_by_type.items():
                offset = BLUE_DECOY_START + decoy_type * ACTION_HOST_SLOTS
                actual = bool(mask[offset + slot])
                assert actual == expected, f"Decoy type {decoy_type} slot {slot} mismatch"


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

    def test_block_allow_same_mask(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(const, 0)
        n = NUM_SUBNETS * NUM_SUBNETS
        block_slice = mask[BLUE_BLOCK_TRAFFIC_START : BLUE_BLOCK_TRAFFIC_START + n]
        allow_slice = mask[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + n]
        np.testing.assert_array_equal(block_slice, allow_slice)


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
