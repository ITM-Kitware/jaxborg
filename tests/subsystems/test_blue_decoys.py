import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import DecoyApache
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import DecoyHarakaSMPT
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import DeployDecoy
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.duration import process_blue_with_duration
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_DECOY,
    BLUE_DECOY_START,
    BLUE_SLEEP,
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.actions.pids import host_current_max_pid
from jaxborg.constants import (
    DECOY_IDS,
    DECOY_NAMES,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from jaxborg.state import create_initial_state
from tests.differential.state_comparator import compare_fast

_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture(scope="module")
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(
        host_services=jnp.array(const.initial_services),
        host_max_pid=const.host_initial_max_pid,
    )
    return state


def _find_host_in_subnet(const, subnet_name, exclude_router=True):
    from jaxborg.constants import SUBNET_IDS

    sid = SUBNET_IDS[subnet_name]
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if int(const.host_subnet[h]) != sid:
            continue
        if exclude_router and bool(const.host_is_router[h]):
            continue
        return h
    return None


def _find_blue_for_host(const, host):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host]):
            return b
    return None


HARAKA_IDX = DECOY_IDS["HarakaSMPT"]
APACHE_IDX = DECOY_IDS["Apache"]
TOMCAT_IDX = DECOY_IDS["Tomcat"]
VSFTPD_IDX = DECOY_IDS["Vsftpd"]


class TestBlueDecoyEncoding:
    def test_encode_decoy_roundtrip(self, jax_const):
        # Use a host that's in agent 0's observed subnets
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A", exclude_router=True)
        assert target is not None
        idx = encode_blue_action("DeployDecoy", target, 0, const=jax_const)
        action_type, target_host, decoy_type, *_ = decode_blue_action(idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == target
        assert int(decoy_type) == -1  # type selected at execution time

    def test_all_decoy_names_map_to_same_index(self, jax_const):
        """All legacy decoy names and generic DeployDecoy map to the same action index."""
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A", exclude_router=True)
        assert target is not None
        decoy_names = [
            "DeployDecoy",
            "DeployDecoy_HarakaSMPT",
            "DeployDecoy_Apache",
            "DeployDecoy_Tomcat",
            "DeployDecoy_Vsftpd",
        ]
        indices = [encode_blue_action(name, target, 0, const=jax_const) for name in decoy_names]
        assert len(set(indices)) == 1, f"Expected all same index, got {indices}"
        action_type, target_host, decoy_type, *_ = decode_blue_action(indices[0], 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == target
        assert int(decoy_type) == -1


class TestApplyBlueDecoy:
    def test_deploy_decoy_sets_flag(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_decoy(state, jax_const, blue_idx, target, TOMCAT_IDX)
        assert bool(new_state.host_decoys[target, TOMCAT_IDX])
        assert not bool(new_state.host_decoys[target, HARAKA_IDX])

    def test_deploy_multiple_decoys(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        state = apply_blue_decoy(state, jax_const, blue_idx, target, TOMCAT_IDX)
        state = apply_blue_decoy(state, jax_const, blue_idx, target, HARAKA_IDX)
        assert bool(state.host_decoys[target, TOMCAT_IDX])
        assert bool(state.host_decoys[target, HARAKA_IDX])

    def test_deploy_on_uncovered_host_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        uncovering_blue = None
        for b in range(NUM_BLUE_AGENTS):
            if not bool(jax_const.blue_agent_hosts[b, target]):
                uncovering_blue = b
                break
        assert uncovering_blue is not None

        new_state = apply_blue_decoy(state, jax_const, uncovering_blue, target, TOMCAT_IDX)
        assert not bool(new_state.host_decoys[target, TOMCAT_IDX])

    def test_does_not_affect_other_hosts(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None and other is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_decoy(state, jax_const, blue_idx, target, TOMCAT_IDX)
        np.testing.assert_array_equal(
            np.array(new_state.host_decoys[other]),
            np.array(state.host_decoys[other]),
        )

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_decoy, static_argnums=(2, 3, 4))
        new_state = jitted(state, jax_const, blue_idx, target, TOMCAT_IDX)
        assert bool(new_state.host_decoys[target, TOMCAT_IDX])

    def test_apache_decoy_is_noop_when_port_80_already_in_use(self, jax_const):
        state = _make_jax_state(jax_const)
        apache_hosts = np.where(np.array(jax_const.initial_services[:, SERVICE_IDS["APACHE2"]], dtype=bool))[0]
        target = next((int(h) for h in apache_hosts if not bool(jax_const.host_is_router[h])), None)
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_decoy(state, jax_const, blue_idx, target, APACHE_IDX)
        assert not bool(new_state.host_decoys[target, APACHE_IDX])


class TestDecoyViaDispatch:
    def test_decoy_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        action_idx = encode_blue_action("DeployDecoy", target, blue_idx, const=jax_const)
        new_state = _jit_apply_blue(state, jax_const, blue_idx, action_idx)
        # With random selection, at least one decoy type should be deployed
        assert bool(new_state.host_decoys[target].any())


class TestDecoyRedeploymentOrphanPid:
    """Regression: redeploying the same decoy type leaves an orphaned process
    in CybORG (old PID stays in host.processes).  JAX must track the orphan
    so recompute_host_max_pid includes it after blue Remove kills the decoy."""

    def test_redeploy_same_type_tracks_orphan_max_pid(self, jax_const):
        """Redeploying vsftpd twice creates an orphan; recompute_host_max_pid
        must return a value >= the orphan PID."""
        from jaxborg.actions.pids import recompute_host_max_pid

        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None
        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        # First deployment
        state = apply_blue_decoy(state, jax_const, blue_idx, target, VSFTPD_IDX)
        first_pid = int(state.host_decoy_process_pids[target, VSFTPD_IDX])
        assert first_pid >= 0
        assert int(state.host_orphaned_decoy_max_pid[target]) == 0

        # Advance time so second deployment gets a different PID delta
        state = state.replace(time=state.time + 1)

        # Second deployment of same type — first PID becomes orphaned
        state = apply_blue_decoy(state, jax_const, blue_idx, target, VSFTPD_IDX)
        second_pid = int(state.host_decoy_process_pids[target, VSFTPD_IDX])
        assert second_pid > first_pid
        assert int(state.host_orphaned_decoy_max_pid[target]) == first_pid

        # Simulate blue_remove killing the decoy: clear the active decoy PID
        # and recompute. The orphan PID must still be accounted for.
        cleared_decoy_pids = state.host_decoy_process_pids.at[target, VSFTPD_IDX].set(-1)
        state = state.replace(host_decoy_process_pids=cleared_decoy_pids)
        recomputed = int(recompute_host_max_pid(state, jax_const, target, state.red_session_pids))
        assert recomputed >= first_pid, f"recompute_host_max_pid={recomputed} should be >= orphan PID {first_pid}"

    def test_restore_clears_orphan_max_pid(self, jax_const):
        """Blue restore resets the host — orphaned decoy PIDs must be cleared."""
        from jaxborg.actions.blue_restore import apply_blue_restore

        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None
        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        # Deploy, advance, redeploy to create orphan
        state = apply_blue_decoy(state, jax_const, blue_idx, target, VSFTPD_IDX)
        state = state.replace(time=state.time + 1)
        state = apply_blue_decoy(state, jax_const, blue_idx, target, VSFTPD_IDX)
        assert int(state.host_orphaned_decoy_max_pid[target]) > 0

        # Restore clears everything
        state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert int(state.host_orphaned_decoy_max_pid[target]) == 0


class TestBlueActionOrder:
    def test_action_order_is_analyse_remove_restore_decoy(self):
        from jaxborg.actions.encoding import (
            BLUE_ANALYSE_START,
            BLUE_REMOVE_START,
            BLUE_RESTORE_START,
        )

        assert BLUE_ANALYSE_START < BLUE_REMOVE_START < BLUE_RESTORE_START < BLUE_DECOY_START


class TestDecoyCompatibilityMask:
    """Verify JAX compatibility mask matches CybORG per-factory is_host_compatible."""

    def test_vsftpd_compatibility_matches_cyborg(self):
        """JAX must check port 21 + Linux for Vsftpd, not hardcode True."""
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory

        from jaxborg.actions.blue_decoys import host_decoy_compatibility_mask

        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())
        factory = VsftpdDecoyFactory()

        jax_state = _make_jax_state(const)

        mismatches = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            hostname = sorted_hosts[h]
            cyborg_compat = factory.is_host_compatible(cy_state.hosts[hostname])
            jax_mask = host_decoy_compatibility_mask(jax_state.host_services[h], jax_state.host_decoys[h])
            jax_compat = bool(jax_mask[VSFTPD_IDX])
            if jax_compat != cyborg_compat:
                mismatches.append((hostname, cyborg_compat, jax_compat))

        assert mismatches == [], f"Vsftpd compatibility mismatch on {len(mismatches)} hosts: " + ", ".join(
            f"{h} (cyborg={c}, jax={j})" for h, c, j in mismatches[:5]
        )

    def test_all_factories_compatibility_matches_cyborg(self):
        """JAX compatibility mask must match CybORG for all 4 decoy factories."""
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory

        from jaxborg.actions.blue_decoys import host_decoy_compatibility_mask

        factories = [HarakaDecoyFactory(), ApacheDecoyFactory(), TomcatDecoyFactory(), VsftpdDecoyFactory()]

        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        jax_state = _make_jax_state(const)

        mismatches = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            hostname = sorted_hosts[h]
            jax_mask = host_decoy_compatibility_mask(jax_state.host_services[h], jax_state.host_decoys[h])
            for decoy_idx, factory in enumerate(factories):
                cyborg_compat = factory.is_host_compatible(cy_state.hosts[hostname])
                jax_compat = bool(jax_mask[decoy_idx])
                if jax_compat != cyborg_compat:
                    mismatches.append((hostname, type(factory).__name__, cyborg_compat, jax_compat))

        assert mismatches == [], f"{len(mismatches)} compatibility mismatches: " + ", ".join(
            f"{h}/{f} (cyborg={c}, jax={j})" for h, f, c, j in mismatches[:10]
        )


class TestDecoyTypeSelectionParity:
    """Verify that DeployDecoy produces the same decoy type in JAX and CybORG
    WITHOUT relying on the harness's use_blue_decoy_type_choices sync.

    The harness records CybORG's choice and injects it via
    use_blue_decoy_type_choices.  The fallback RNG (used in standalone mode)
    is stochastic per-episode and will NOT match CybORG without sync.
    These tests verify that the precomputed sync path produces correct results,
    and that the fallback path selects a valid compatible type.
    """

    @pytest.mark.parametrize("seed", [0, 5, 10, 15, 20, 25, 30, 42])
    def test_deploy_decoy_type_matches_cyborg_via_sync(self, seed):
        """Precomputed decoy type sync produces the same type CybORG chose."""
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=seed)
        const = build_const_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None
        hostname = sorted_hosts[target]

        # CybORG: execute DeployDecoy
        before_services = set(cy_state.hosts[hostname].services.keys())
        cy_action = DeployDecoy(session=0, agent=f"blue_agent_{blue_idx}", hostname=hostname)
        cy_obs = cy_action.execute(cy_state)
        assert str(cy_obs.success).upper() == "TRUE"
        after_services = set(cy_state.hosts[hostname].services.keys())
        added = after_services - before_services
        assert len(added) == 1
        service_to_decoy = {"haraka": HARAKA_IDX, "apache2": APACHE_IDX, "tomcat": TOMCAT_IDX, "vsftpd": VSFTPD_IDX}
        cyborg_decoy_type = service_to_decoy[next(iter(added))]

        # JAX: execute DeployDecoy WITH decoy type sync (precomputed path)
        jax_state = _make_jax_state(const)
        after_max = max(p.pid for p in cy_state.hosts[hostname].processes)
        before_max = const.host_initial_max_pid[target]
        pid_delta = int(after_max - before_max)
        const_for_test = const.replace(
            blue_decoy_pid_deltas=const.blue_decoy_pid_deltas.at[0, blue_idx, 0].set(pid_delta),
            use_blue_decoy_pid_deltas=jnp.array(True),
            blue_decoy_type_choices=const.blue_decoy_type_choices.at[0, blue_idx].set(cyborg_decoy_type),
            use_blue_decoy_type_choices=jnp.array(True),
        )
        new_state = apply_blue_decoy(jax_state, const_for_test, blue_idx, target, jnp.int32(-1))

        jax_decoy_placed = np.array(new_state.host_decoys[target])
        jax_types_placed = np.where(jax_decoy_placed)[0]
        assert len(jax_types_placed) == 1, f"Expected exactly 1 decoy placed, got {jax_types_placed}"
        jax_decoy_type = int(jax_types_placed[0])

        assert jax_decoy_type == cyborg_decoy_type, (
            f"Decoy type mismatch (seed={seed}): JAX placed {DECOY_NAMES[jax_decoy_type]} "
            f"(idx {jax_decoy_type}), CybORG placed {DECOY_NAMES[cyborg_decoy_type]} "
            f"(idx {cyborg_decoy_type})."
        )

    def test_fallback_rng_selects_valid_compatible_type(self):
        """Fallback RNG (no precomputed sync) selects a compatible decoy type."""
        import jax

        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        jax_state = _make_jax_state(const)

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        # Use stochastic key — different keys should produce valid results
        for seed in [0, 42, 123]:
            key = jax.random.PRNGKey(seed)
            new_state = apply_blue_decoy(jax_state, const, blue_idx, target, jnp.int32(-1), key)
            jax_decoy_placed = np.array(new_state.host_decoys[target])
            jax_types_placed = np.where(jax_decoy_placed)[0]
            assert len(jax_types_placed) == 1, f"Expected exactly 1 decoy (seed={seed}), got {jax_types_placed}"


class TestDifferentialWithCybORG:
    def test_decoy_process_pid_advances_host_pid_base_matches_cyborg(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert target is not None
        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None
        target_hostname = sorted_hosts[target]

        before_services = {str(name).split(".")[-1].lower() for name in cy_state.hosts[target_hostname].services}
        before_max = max((p.pid for p in cy_state.hosts[target_hostname].processes), default=4999)
        cy_action = DeployDecoy(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        cy_obs = cy_action.execute(cy_state)
        assert str(cy_obs.success).upper() == "TRUE"
        after_max = max((p.pid for p in cy_state.hosts[target_hostname].processes), default=4999)
        decoy_delta = after_max - before_max
        assert 1 <= decoy_delta <= 9
        after_services = {str(name).split(".")[-1].lower() for name in cy_state.hosts[target_hostname].services}
        added_services = after_services - before_services
        assert len(added_services) == 1
        added_service = next(iter(added_services))
        service_to_decoy = {
            "haraka": HARAKA_IDX,
            "apache2": APACHE_IDX,
            "tomcat": TOMCAT_IDX,
            "vsftpd": VSFTPD_IDX,
        }
        decoy_type = service_to_decoy[added_service]

        state = _make_jax_state(const)
        const = const.replace(
            blue_decoy_pid_deltas=const.blue_decoy_pid_deltas.at[0, blue_idx, 0].set(decoy_delta),
            use_blue_decoy_pid_deltas=jnp.array(True),
        )
        new_state = apply_blue_decoy(state, const, blue_idx, target, decoy_type)

        jax_after_max = int(host_current_max_pid(new_state, const, target))
        assert jax_after_max == after_max

    def test_apache_decoy_fails_when_target_already_has_apache(self):
        pytest.importorskip("CybORG")
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=0)
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state
        const = build_const_from_cyborg(cyborg_env)

        blue_idx = 2
        apache_hosts = np.where(
            np.array(const.blue_agent_hosts[blue_idx], dtype=bool)
            & np.array(const.initial_services[:, SERVICE_IDS["APACHE2"]], dtype=bool)
        )[0]
        target = int(apache_hosts[0])
        assert target == 84
        hostname = sorted(cy_state.hosts.keys())[target]

        cyborg_host = cy_state.hosts[hostname]
        before_decoys = sum(getattr(proc, "decoy_type", None) is not None for proc in cyborg_host.processes)
        cy_action = DecoyApache(session=0, agent=f"blue_agent_{blue_idx}", hostname=hostname)
        cy_obs = cy_action.execute(cy_state)
        after_decoys = sum(getattr(proc, "decoy_type", None) is not None for proc in cyborg_host.processes)

        jax_state = _make_jax_state(const)
        action_idx = encode_blue_action("DeployDecoy", target, blue_idx, const=const)
        # Force selection of Apache via precomputed tape
        const = const.replace(
            blue_decoy_type_choices=const.blue_decoy_type_choices.at[0, blue_idx].set(APACHE_IDX),
            use_blue_decoy_type_choices=jnp.array(True),
        )
        new_state = process_blue_with_duration(jax_state, const, blue_idx, action_idx)
        # DeployDecoy has duration=2; tick through pending action to execution
        assert int(new_state.blue_pending_ticks[blue_idx]) == 1
        new_state = process_blue_with_duration(new_state, const, blue_idx, BLUE_SLEEP)

        assert str(cy_obs.success).upper() == "FALSE"
        assert before_decoys == after_decoys
        assert int(new_state.blue_pending_ticks[blue_idx]) == 0
        assert not bool(new_state.host_decoys[target, APACHE_IDX])

    def test_compare_fast_recognizes_haraka_decoy_enum_service_names(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state

        target_hostname = "operational_zone_b_subnet_router"
        target = mappings.hostname_to_idx[target_hostname]
        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        cy_action = DecoyHarakaSMPT(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        cy_obs = cy_action.execute(cy_state)
        assert str(cy_obs.success).upper() == "TRUE"

        jax_state = _make_jax_state(const)
        jax_state = apply_blue_decoy(jax_state, const, blue_idx, target, HARAKA_IDX)

        diffs = compare_fast(cyborg_env, jax_state, const, mappings)
        host_decoy_diffs = [d for d in diffs if d.field_name == "host_decoys" and d.host_or_agent == f"host_{target}"]
        assert host_decoy_diffs == []
