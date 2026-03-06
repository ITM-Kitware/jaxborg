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
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.actions.pids import host_current_max_pid
from jaxborg.constants import (
    DECOY_IDS,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import build_mappings_from_cyborg
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


@pytest.fixture
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
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
    def test_encode_decoy_haraka_roundtrip(self, jax_const):
        idx = encode_blue_action("DeployDecoy_HarakaSMPT", 5, 0, const=jax_const)
        action_type, target_host, decoy_type, *_ = decode_blue_action(idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == 5
        assert int(decoy_type) == HARAKA_IDX

    def test_encode_decoy_apache_roundtrip(self, jax_const):
        idx = encode_blue_action("DeployDecoy_Apache", 10, 0, const=jax_const)
        action_type, target_host, decoy_type, *_ = decode_blue_action(idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == 10
        assert int(decoy_type) == APACHE_IDX

    def test_decode_decoy(self, jax_const):
        idx = encode_blue_action("DeployDecoy_Tomcat", 7, 0, const=jax_const)
        action_type, target_host, decoy_type, *_ = decode_blue_action(idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == 7
        assert int(decoy_type) == TOMCAT_IDX

    def test_decode_all_decoy_types(self, jax_const):
        decoy_names = ["DeployDecoy_HarakaSMPT", "DeployDecoy_Apache", "DeployDecoy_Tomcat", "DeployDecoy_Vsftpd"]
        for dtype, name in enumerate(decoy_names):
            idx = encode_blue_action(name, 3, 0, const=jax_const)
            action_type, target_host, decoy_type, *_ = decode_blue_action(idx, 0, jax_const)
            assert int(action_type) == BLUE_ACTION_TYPE_DECOY
            assert int(target_host) == 3
            assert int(decoy_type) == dtype


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

        action_idx = encode_blue_action("DeployDecoy_Tomcat", target, blue_idx, const=jax_const)
        new_state = _jit_apply_blue(state, jax_const, blue_idx, action_idx)
        assert bool(new_state.host_decoys[target, TOMCAT_IDX])


class TestBlueActionOrder:
    def test_action_order_is_analyse_remove_restore_decoy(self):
        from jaxborg.actions.encoding import (
            BLUE_ANALYSE_START,
            BLUE_REMOVE_START,
            BLUE_RESTORE_START,
        )

        assert BLUE_ANALYSE_START < BLUE_REMOVE_START < BLUE_RESTORE_START < BLUE_DECOY_START


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
        state = state.replace(
            blue_decoy_pid_deltas=state.blue_decoy_pid_deltas.at[0, blue_idx].set(decoy_delta),
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
        action_idx = encode_blue_action("DeployDecoy_Apache", target, blue_idx, const=const)
        new_state = process_blue_with_duration(jax_state, const, blue_idx, action_idx)

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
