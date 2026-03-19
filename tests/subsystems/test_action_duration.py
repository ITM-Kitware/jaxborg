import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.duration import process_blue_with_duration, process_red_with_duration
from jaxborg.actions.encoding import (
    BLUE_ACTION_DURATIONS,
    BLUE_SLEEP,
    RED_ACTION_DURATIONS,
    RED_EXPLOIT_SSH_START,
    RED_PRIVESC_START,
    RED_SLEEP,
    encode_blue_action,
)
from jaxborg.actions.red_common import apply_red_session_check
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS, SERVICE_IDS
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
        expected = [1, 1, 2, 3, 5, 1, 1, 1]
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

        host_indices = jnp.where(const.host_active & const.host_is_server, size=GLOBAL_MAX_HOSTS)[0]
        target_host = int(host_indices[0])
        restore_action = encode_blue_action("Restore", target_host, 0, const=const)

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
        for step in range(30):
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

        assert saw_exploit_countdown, "No exploit (duration=4) seen in 30 steps"

    def test_blue_restore_tick_countdown(self, fsm_env_and_state):
        """Submit Blue Restore via FsmRedCC4Env, verify ticks count 4→3→2→1→0."""
        env, env_state = fsm_env_and_state
        key = jax.random.PRNGKey(43)

        active = env_state.const.host_active
        blue_hosts = env_state.const.blue_agent_hosts[0] & active & ~env_state.const.host_is_router
        target = int(jnp.argmax(blue_hosts))
        restore_action = encode_blue_action("Restore", target, 0, const=env_state.const)

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
        for step in range(30):
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

        assert found_busy, "No busy red agent observed in 30 steps"


class TestSessionBindingFollowsUpdatedAnchor:
    """Regression: scan actions with SESSION_BINDING source must re-evaluate
    the source host from the current anchor, not the stale host recorded at
    queue time.

    CybORG binds abstract scan actions to session 0 (a session ID). When
    RedSessionCheck promotes a new session into slot 0, the scan follows the
    updated session. JAX must mirror this by re-evaluating the bound source
    from the current red_scan_anchor_host at execution time.
    """

    def test_session_binding_follows_anchor_change(self, env_and_state):
        """When session 0 moves to a new host between queuing and execution,
        the scan should execute using the NEW session 0 host, not the stale
        one recorded at queue time.

        This mirrors CybORG's behavior where abstract scan actions are bound
        to session 0 by ID, and RedSessionCheck can promote a new session
        into slot 0 between steps.  The forced_primary_host parameter
        communicates CybORG's actual session 0 host to JAX.

        Setup:
        - Queue a stealth scan with SESSION_BINDING source on host_a
        - Simulate session 0 moving to host_b (RedSessionCheck promotion)
        - Remove the session from host_a
        - Pass forced_primary_host=host_b to simulate CybORG session 0
        - The scan should execute using host_b as source

        Asserted field: detection_random_index (proves scan executed and
        consumed the detection random).
        """
        from jaxborg.actions.encoding import RED_STEALTH_SCAN_START
        from jaxborg.actions.pending_source import PENDING_SOURCE_KIND_SESSION_BINDING

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        key = jax.random.PRNGKey(77)

        start_host = int(const.red_start_hosts[agent_id])
        host_a = start_host  # initial anchor
        target_subnet = int(const.host_subnet[host_a])

        # Find host_b: a different active host in the same subnet
        host_b = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != host_a:
                host_b = h
                break
        if host_b < 0:
            pytest.skip("Need at least 2 hosts in subnet for this test")

        # Find target to scan in the same subnet
        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if (
                bool(const.host_active[h])
                and int(const.host_subnet[h]) == target_subnet
                and h != host_a
                and h != host_b
            ):
                target_host = h
                break
        if target_host < 0:
            pytest.skip("No scan target in same subnet")

        # Set up initial state with host_a as the abstract anchor session
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, host_a].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, host_a].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(host_a)),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
        )

        scan_action = RED_STEALTH_SCAN_START + target_host
        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))

        # Queue the stealth scan (duration=3). Source should bind to host_a.
        s1 = process_jit(state, const, agent_id, scan_action, key)
        assert int(s1.red_pending_ticks[agent_id]) == 2  # 3 - 1 = 2
        assert int(s1.red_pending_source_kind[agent_id]) in (
            int(PENDING_SOURCE_KIND_SESSION_BINDING),
            1,  # PENDING_SOURCE_KIND_HOST is also acceptable
        )

        # Simulate anchor changing to host_b (as RedSessionCheck would do)
        s1 = s1.replace(
            red_sessions=s1.red_sessions.at[agent_id, host_b].set(True),
            red_session_is_abstract=s1.red_session_is_abstract.at[agent_id, host_b].set(True),
            red_scan_anchor_host=s1.red_scan_anchor_host.at[agent_id].set(jnp.int32(host_b)),
        )
        # Remove session from host_a so the stale binding would fail
        s1 = s1.replace(
            red_sessions=s1.red_sessions.at[agent_id, host_a].set(False),
            red_session_is_abstract=s1.red_session_is_abstract.at[agent_id, host_a].set(False),
        )

        # Enable detection random sync so we can verify consumption
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.99)),
        )
        s1 = s1.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        # Tick down: 2 -> 1 (forced_primary_host=host_b simulates CybORG session 0)
        s2 = process_jit(s1, const, agent_id, RED_SLEEP, key, forced_primary_host=jnp.int32(host_b))
        assert int(s2.red_pending_ticks[agent_id]) == 1

        # Tick down: 1 -> 0 (should execute using host_b via forced_primary_host)
        s3 = process_jit(s2, const, agent_id, RED_SLEEP, key, forced_primary_host=jnp.int32(host_b))
        assert int(s3.red_pending_ticks[agent_id]) == 0

        # The scan must have consumed one detection random (stealth scan
        # always rolls for detection when it succeeds). If the source was
        # the stale host_a (no session), the scan would skip entirely and
        # detection_random_index would remain 0.
        assert int(s3.detection_random_index) == 1, (
            "Stealth scan did not consume a detection random — the scan failed to use the updated session 0 host"
        )


class TestDiscoverDeceptionFailsAfterSessionDestroyed:
    """Regression: DiscoverDeception must fail when session 0 is destroyed by
    blue restore between queuing and execution.

    CybORG's DiscoverDeception.execute() checks ``self.session not in
    state.sessions[self.agent]`` (line 70).  When blue restore clears all red
    sessions on the host where session 0 lived, the action returns
    Observation(False) without consuming any detection randoms.

    JAX previously let DiscoverDeception succeed because
    ``select_bound_source_host`` returned the recomputed anchor (a different
    valid host promoted by ``recompute_scan_anchor_hosts`` inside blue
    restore).  The fix checks session 0 validity via ``forced_primary_host``.
    """

    def test_deception_blocked_when_session0_destroyed(self, env_and_state):
        """Queue DiscoverDeception, destroy session 0 via blue restore before
        execution, verify no detection randoms are consumed (action blocked).

        Setup:
        - Red agent has session on host_a (anchor / session 0)
        - Also has session on host_b (fallback; so recompute_scan_anchor_hosts
          will promote host_b after blue restore clears host_a)
        - Queue DiscoverDeception targeting host_c (duration=2)
        - Simulate blue restore: clear sessions on host_a, recompute anchor
        - Execute with forced_primary_host=host_a (CybORG's pre-step session 0)
        - Since red_sessions[agent, host_a] is False, forced_source_valid=False
        - deception_source_host = forced_primary_host = host_a
        - deception_source_valid = False -> can_execute = False
        - detection_random_index stays at 0 (no randoms consumed)
        """
        from jaxborg.actions.encoding import RED_DISCOVER_DECEPTION_START

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        key = jax.random.PRNGKey(99)

        start_host = int(const.red_start_hosts[agent_id])
        host_a = start_host  # session 0 host
        target_subnet = int(const.host_subnet[host_a])

        # Find host_b: fallback session in same subnet
        host_b = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != host_a:
                host_b = h
                break
        if host_b < 0:
            pytest.skip("Need at least 2 hosts in subnet")

        # Find host_c: target for DiscoverDeception
        host_c = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and h != host_a and h != host_b:
                host_c = h
                break
        if host_c < 0:
            pytest.skip("Need a third host for deception target")

        # Set up: agent has sessions on host_a (anchor) and host_b (fallback)
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, host_a].set(True).at[agent_id, host_b].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, host_a]
            .set(True)
            .at[agent_id, host_b]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(host_a)),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, host_c].set(True),
        )

        deception_action = RED_DISCOVER_DECEPTION_START + host_c
        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))

        # Queue DiscoverDeception (duration=2): ticks go from 2 to 1
        s1 = process_jit(state, const, agent_id, deception_action, key)
        assert int(s1.red_pending_ticks[agent_id]) == 1  # 2 - 1 = 1

        # Simulate blue restore destroying session on host_a
        # This mimics what apply_blue_restore does: clear sessions, recompute anchor
        s1 = s1.replace(
            red_sessions=s1.red_sessions.at[agent_id, host_a].set(False),
            red_session_is_abstract=s1.red_session_is_abstract.at[agent_id, host_a].set(False),
            # Anchor moves to host_b (recompute_scan_anchor_hosts promotes fallback)
            red_scan_anchor_host=s1.red_scan_anchor_host.at[agent_id].set(jnp.int32(host_b)),
        )

        # Enable detection random tracking (detection_randoms lives on CC4Const)
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.99)).at[1].set(jnp.float32(0.99)),
        )
        s1 = s1.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        # Execute: forced_primary_host=host_a tells JAX that CybORG's session 0
        # was on host_a before this step.  Since red_sessions[agent, host_a] is
        # now False, forced_source_valid=False.  The fix gates DiscoverDeception
        # on forced_primary_host validity, so the action should be blocked.
        s2 = process_jit(s1, const, agent_id, RED_SLEEP, key, forced_primary_host=jnp.int32(host_a))
        assert int(s2.red_pending_ticks[agent_id]) == 0  # timer expired

        # No detection randoms consumed => action was blocked
        assert int(s2.detection_random_index) == 0, (
            f"DiscoverDeception consumed {int(s2.detection_random_index)} detection randoms "
            f"but should have been blocked because session 0 on host {host_a} was destroyed"
        )


class TestScanFailsWhenSession0NotAbstract:
    """Regression: scans must fail when session 0 is a regular Session (not
    RedAbstractSession).

    CybORG's DiscoverNetworkServices.execute() checks
    ``isinstance(session_0, RedAbstractSession)`` (line 67).  When
    RedSessionCheck promotes a non-abstract session into slot 0, the host
    may still have abstract sessions, but session 0 itself is not abstract
    and scans must fail.

    JAX previously used the per-host ``red_session_is_abstract`` flag which
    is True if ANY session on the host is abstract, masking a non-abstract
    session 0.  The fix uses the per-agent ``red_primary_is_abstract`` flag
    which tracks session 0's type exactly.
    """

    def test_scan_blocked_when_primary_not_abstract(self, env_and_state):
        """Scan should fail when red_primary_is_abstract=False, even if the
        host has abstract sessions.

        Setup:
        - Red agent has both abstract and non-abstract sessions on host_a
        - red_primary_is_abstract[agent] = False (session 0 is concrete)
        - red_session_is_abstract[agent, host_a] = True (host has abstracts)
        - Submit immediate AggressiveServiceDiscovery (duration=1)
        - Scan should be blocked -> detection_random_index stays at 0
        """
        from jaxborg.actions.encoding import RED_AGGRESSIVE_SCAN_START

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        key = jax.random.PRNGKey(88)

        start_host = int(const.red_start_hosts[agent_id])
        host_a = start_host
        target_subnet = int(const.host_subnet[host_a])

        # Find a target host in the same subnet
        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != host_a:
                target_host = h
                break
        if target_host < 0:
            pytest.skip("No scan target in same subnet")

        # Set up: host_a has sessions, abstract flag True, but primary is NOT abstract
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, host_a].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, host_a].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(host_a)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(False),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
        )

        # Enable detection random tracking (detection_randoms lives on CC4Const)
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.99)),
        )
        state = state.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        scan_action = RED_AGGRESSIVE_SCAN_START + target_host
        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))

        # AggressiveServiceDiscovery has duration=1, executes immediately
        s1 = process_jit(state, const, agent_id, scan_action, key, forced_primary_host=jnp.int32(host_a))
        assert int(s1.red_pending_ticks[agent_id]) == 0  # executed (or blocked) immediately

        # Scan must NOT consume detection randoms because session 0 is not abstract
        assert int(s1.detection_random_index) == 0, (
            f"AggressiveServiceDiscovery consumed {int(s1.detection_random_index)} detection random(s) "
            f"but should have been blocked because session 0 is not abstract"
        )


class TestScanFailsWhenSession0Missing:
    """Regression: scans must fail when CybORG session 0 is missing, even if
    other abstract sessions still exist on the anchor host.

    CybORG's DiscoverNetworkServices.execute() first looks up
    ``state.sessions[agent][0]`` and returns failure immediately when slot 0
    is absent. In parity mode the harness passes ``forced_primary_host=-1`` to
    mean "session 0 was missing before red execution this step".
    """

    def test_scan_blocked_when_primary_session_missing(self, env_and_state):
        from jaxborg.actions.encoding import RED_AGGRESSIVE_SCAN_START

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        blue_id = 0
        key = jax.random.PRNGKey(89)

        source_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and bool(const.blue_agent_hosts[blue_id, h]):
                source_host = h
                break
        if source_host < 0:
            pytest.skip("No host covered by blue_agent_0")
        target_subnet = int(const.host_subnet[source_host])

        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != source_host:
                target_host = h
                break
        if target_host < 0:
            pytest.skip("No scan target in same subnet")

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, source_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, source_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(source_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.01)),
        )

        scan_action = RED_AGGRESSIVE_SCAN_START + target_host
        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))

        s1 = process_jit(state, const, agent_id, scan_action, key, forced_primary_host=jnp.int32(-1))
        assert int(s1.red_pending_ticks[agent_id]) == 0
        assert int(s1.detection_random_index) == 0, (
            f"AggressiveServiceDiscovery consumed {int(s1.detection_random_index)} detection random(s) "
            f"but should have failed because CybORG session 0 was missing"
        )
        assert int(s1.red_activity_this_step[target_host]) == 0


class TestScanFailsWhenRemoveKillsPrimarySession:
    """Regression: same-step blue Remove must invalidate the current primary
    session even when another abstract session survives on the same host.

    This mirrors the seed-28 failure where CybORG removed session 0 on the
    source host, the queued scan ran before RedSessionCheck, and only then was
    another surviving abstract session promoted back into slot 0.
    """

    def test_pending_scan_blocked_after_remove_kills_primary_only(self, env_and_state):
        from jaxborg.actions.blue_remove import apply_blue_remove
        from jaxborg.actions.encoding import RED_AGGRESSIVE_SCAN_START

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        blue_id = 0
        key = jax.random.PRNGKey(90)

        source_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and bool(const.blue_agent_hosts[blue_id, h]):
                source_host = h
                break
        if source_host < 0:
            pytest.skip("No host covered by blue_agent_0")
        target_subnet = int(const.host_subnet[source_host])

        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != source_host:
                target_host = h
                break
        if target_host < 0:
            pytest.skip("No scan target in same subnet")

        primary_pid = jnp.int32(101)
        other_pid = jnp.int32(202)
        suspicious = jnp.full((GLOBAL_MAX_HOSTS, state.blue_suspicious_pids.shape[2]), -1, dtype=jnp.int32)
        suspicious = suspicious.at[source_host, 0].set(primary_pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, source_host].set(True),
            red_session_count=state.red_session_count.at[agent_id, source_host].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, source_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(source_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_primary_pid=state.red_primary_pid.at[agent_id].set(primary_pid),
            red_session_pids=state.red_session_pids.at[agent_id, source_host, 0]
            .set(primary_pid)
            .at[agent_id, source_host, 1]
            .set(other_pid),
            red_session_abstract_pids=state.red_session_abstract_pids.at[agent_id, source_host, 0]
            .set(primary_pid)
            .at[agent_id, source_host, 1]
            .set(other_pid),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_id].set(suspicious),
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.01)),
        )

        after_remove = apply_blue_remove(state, const, blue_id, source_host)
        assert int(after_remove.red_session_count[agent_id, source_host]) == 1
        assert int(after_remove.red_primary_pid[agent_id]) == -1
        assert int(after_remove.red_scan_anchor_host[agent_id]) == -1

        scan_action = RED_AGGRESSIVE_SCAN_START + target_host
        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))
        s1 = process_jit(after_remove, const, agent_id, scan_action, key)
        assert int(s1.red_pending_ticks[agent_id]) == 0
        assert int(s1.detection_random_index) == 0
        assert int(s1.red_activity_this_step[target_host]) == 0


class TestExploitFailsWhenRemoveKillsPrimarySession:
    """Regression: queued exploit must fail when blue Remove kills session 0.

    CybORG's ExploitRemoteService executes through ``state.sessions[agent][0]``.
    If blue Remove kills session 0 on the source host before the exploit
    finishes, the queued exploit must not fall back to another surviving
    abstract session on that host.
    """

    def test_pending_exploit_blocked_after_remove_kills_primary_only(self, env_and_state):
        from jaxborg.actions.blue_remove import apply_blue_remove

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        blue_id = 0
        key = jax.random.PRNGKey(91)

        source_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and bool(const.blue_agent_hosts[blue_id, h]):
                source_host = h
                break
        if source_host < 0:
            pytest.skip("No host covered by blue_agent_0")
        target_subnet = int(const.host_subnet[source_host])

        ssh_idx = SERVICE_IDS["SSHD"]
        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if not bool(const.host_active[h]):
                continue
            if int(const.host_subnet[h]) != target_subnet or h == source_host:
                continue
            if not bool(const.initial_services[h, ssh_idx]):
                continue
            if not bool(const.host_has_bruteforceable_user[h]):
                continue
            target_host = h
            break
        if target_host < 0:
            pytest.skip("No same-subnet SSH target with a bruteforceable user")

        primary_pid = jnp.int32(101)
        other_pid = jnp.int32(202)
        suspicious = jnp.full((GLOBAL_MAX_HOSTS, state.blue_suspicious_pids.shape[2]), -1, dtype=jnp.int32)
        suspicious = suspicious.at[source_host, 0].set(primary_pid)

        state = state.replace(
            host_services=jnp.array(const.initial_services),
            red_sessions=state.red_sessions.at[agent_id, source_host].set(True),
            red_session_count=state.red_session_count.at[agent_id, source_host].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, source_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(source_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_primary_pid=state.red_primary_pid.at[agent_id].set(primary_pid),
            red_session_pids=state.red_session_pids.at[agent_id, source_host, 0]
            .set(primary_pid)
            .at[agent_id, source_host, 1]
            .set(other_pid),
            red_session_abstract_pids=state.red_session_abstract_pids.at[agent_id, source_host, 0]
            .set(primary_pid)
            .at[agent_id, source_host, 1]
            .set(other_pid),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[agent_id, target_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[agent_id, target_host, source_host].set(True),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_id].set(suspicious),
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        after_remove = apply_blue_remove(state, const, blue_id, source_host)
        assert int(after_remove.red_session_count[agent_id, source_host]) == 1
        assert int(after_remove.red_primary_pid[agent_id]) == -1
        assert int(after_remove.red_scan_anchor_host[agent_id]) == -1

        exploit_action = RED_EXPLOIT_SSH_START + target_host
        pending_state = after_remove.replace(
            red_pending_ticks=after_remove.red_pending_ticks.at[agent_id].set(1),
            red_pending_action=after_remove.red_pending_action.at[agent_id].set(exploit_action),
            red_pending_key=after_remove.red_pending_key.at[agent_id].set(jnp.asarray(key, dtype=jnp.uint32)),
        )

        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))
        s1 = process_jit(
            pending_state,
            const,
            agent_id,
            RED_SLEEP,
            key,
            forced_primary_host=jnp.int32(source_host),
            forced_primary_pid=primary_pid,
        )

        assert int(s1.red_pending_ticks[agent_id]) == 0
        assert int(s1.red_session_count[agent_id, target_host]) == 0
        assert not bool(s1.red_sessions[agent_id, target_host])
        assert int(s1.red_activity_this_step[target_host]) == 0
        assert int(s1.detection_random_index) == 0


class TestSessionBindingToleratesPrimaryPidRowLag:
    """Regression: pending session-bound scans should still execute when the
    live CybORG session-0 PID matches JAX's tracked primary PID but the
    per-host PID row has not caught up yet."""

    def test_pending_scan_executes_when_primary_pid_matches_but_pid_row_is_stale(self, env_and_state):
        from jaxborg.actions.encoding import RED_STEALTH_SCAN_START
        from jaxborg.actions.pending_source import PENDING_SOURCE_KIND_SESSION_BINDING

        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        key = jax.random.PRNGKey(123)

        source_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[source_host])

        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != source_host:
                target_host = h
                break
        if target_host < 0:
            pytest.skip("Need a same-subnet scan target")

        primary_pid = jnp.int32(101)
        stale_pid_a = jnp.int32(201)
        stale_pid_b = jnp.int32(202)

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, source_host].set(True),
            red_session_count=state.red_session_count.at[agent_id, source_host].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, source_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(source_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_primary_pid=state.red_primary_pid.at[agent_id].set(primary_pid),
            red_session_pids=state.red_session_pids.at[agent_id, source_host, 0]
            .set(stale_pid_a)
            .at[agent_id, source_host, 1]
            .set(stale_pid_b),
            red_session_abstract_pids=state.red_session_abstract_pids.at[agent_id, source_host, 0]
            .set(stale_pid_a)
            .at[agent_id, source_host, 1]
            .set(stale_pid_b),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
            red_pending_ticks=state.red_pending_ticks.at[agent_id].set(1),
            red_pending_action=state.red_pending_action.at[agent_id].set(RED_STEALTH_SCAN_START + target_host),
            red_pending_source_kind=state.red_pending_source_kind.at[agent_id].set(PENDING_SOURCE_KIND_SESSION_BINDING),
            red_pending_source_host=state.red_pending_source_host.at[agent_id].set(jnp.int32(source_host)),
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )
        const = const.replace(
            use_detection_randoms=jnp.array(True),
            detection_randoms=const.detection_randoms.at[0].set(jnp.float32(0.99)),
        )

        process_jit = jax.jit(process_red_with_duration, static_argnums=(2,))
        new_state = process_jit(
            state,
            const,
            agent_id,
            RED_SLEEP,
            key,
            forced_primary_host=jnp.int32(source_host),
            forced_primary_pid=primary_pid,
        )

        assert bool(new_state.red_scanned_hosts[agent_id, target_host]), (
            "Queued stealth scan should execute from the forced session-0 host "
            "when red_primary_pid matches CybORG even if red_session_pids lags"
        )
        assert int(new_state.detection_random_index) == 1


class TestSessionCheckToleratesPrimaryPidRowLag:
    """Regression: scan memory should survive a session-check pass when the
    tracked primary PID matches CybORG but the host PID row is stale."""

    def test_session_check_preserves_scan_memory_when_primary_pid_matches_but_pid_row_is_stale(self, env_and_state):
        _, _, env_state = env_and_state
        state = env_state.state
        const = env_state.const

        agent_id = 0
        key = jax.random.PRNGKey(124)

        source_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[source_host])

        target_host = -1
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.host_active[h]) and int(const.host_subnet[h]) == target_subnet and h != source_host:
                target_host = h
                break
        if target_host < 0:
            pytest.skip("Need a same-subnet scan target")

        primary_pid = jnp.int32(101)
        stale_pid_a = jnp.int32(201)
        stale_pid_b = jnp.int32(202)

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, source_host].set(True),
            red_session_count=state.red_session_count.at[agent_id, source_host].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, source_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(source_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_primary_pid=state.red_primary_pid.at[agent_id].set(primary_pid),
            red_session_pids=state.red_session_pids.at[agent_id, source_host, 0]
            .set(stale_pid_a)
            .at[agent_id, source_host, 1]
            .set(stale_pid_b),
            red_session_abstract_pids=state.red_session_abstract_pids.at[agent_id, source_host, 0]
            .set(stale_pid_a)
            .at[agent_id, source_host, 1]
            .set(stale_pid_b),
            red_scanned_hosts=state.red_scanned_hosts.at[agent_id, target_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[agent_id, target_host, source_host].set(True),
        )

        new_state = apply_red_session_check(
            state,
            const,
            agent_id,
            key,
            forced_primary_host=jnp.int32(source_host),
            forced_primary_pid=primary_pid,
        )

        assert bool(new_state.red_scanned_hosts[agent_id, target_host]), (
            "Session check should not clear scan memory when the forced session-0 "
            "PID matches red_primary_pid but the host PID row lags"
        )
        assert bool(new_state.red_scanned_source_hosts[agent_id, target_host, source_host])
        assert int(new_state.red_primary_pid[agent_id]) == int(primary_pid)


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
        blue_hosts = harness.jax_const.blue_agent_hosts[0] & active & ~harness.jax_const.host_is_router
        target = int(jnp.argmax(blue_hosts))

        errors = []
        jax_ticks_per_step = []

        for step in range(10):
            if step == 0:
                blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
                blue_actions[0] = encode_blue_action("Restore", target, 0, const=harness.jax_const)
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

    def test_blue_decoy_ticks_match_cyborg(self):
        """Submit Blue Decoy, verify CybORG and JAX both execute it in one step."""
        pytest.importorskip("CybORG")
        from CybORG.Agents import SleepAgent

        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=10,
            blue_cls=SleepAgent,
            green_cls=SleepAgent,
            red_cls=SleepAgent,
            sync_green_rng=True,
        )
        harness.reset()
        controller = harness.cyborg_env.environment_controller

        active = harness.jax_const.host_active
        blue_hosts = harness.jax_const.blue_agent_hosts[0] & active & ~harness.jax_const.host_is_router
        target = int(jnp.argmax(blue_hosts))

        blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
        blue_actions[0] = encode_blue_action("DeployDecoy", target, 0, const=harness.jax_const)

        result = harness.full_step(blue_actions=blue_actions)
        step_errors = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]

        cy_ticks = _get_cyborg_remaining_ticks(controller, "blue_agent_0")
        jax_ticks = int(harness.jax_state.blue_pending_ticks[0])

        assert not step_errors, f"Blue decoy parity errors: {step_errors[:3]}"
        assert cy_ticks == 0, f"CybORG decoy should execute immediately, got remaining_ticks={cy_ticks}"
        assert jax_ticks == cy_ticks, (
            f"Blue decoy duration mismatch: CybORG remaining_ticks={cy_ticks}, JAX pending_ticks={jax_ticks}"
        )

    def test_multi_seed_tick_parity(self):
        """Multiple seeds × 40 steps: verify tick parity for red agents at every step."""
        pytest.importorskip("CybORG")
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        total_deferred = 0
        for seed in [0, 42, 99]:
            harness = CC4DifferentialHarness(seed=seed, max_steps=20, sync_green_rng=True)
            harness.reset()
            controller = harness.cyborg_env.environment_controller

            errors = []
            for step in range(20):
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

        assert total_deferred > 0, "No deferred actions across 3 seeds × 20 steps — duration not exercised"
