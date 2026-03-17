import pytest

from jaxborg.actions.encoding import BLUE_MONITOR, BLUE_SLEEP, encode_blue_action
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    cyborg_blue_to_jax,
    jax_blue_to_cyborg,
    jax_blue_to_cyborg_wrapper_action,
)


@pytest.fixture
def blue_translation_context(cyborg_env):
    cyborg_env.reset()
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)
    cy_state = cyborg_env.environment_controller.state
    return const, mappings, cy_state


def _find_blue_host(const, mappings, cy_state, *, predicate=None):
    for agent_id in range(NUM_BLUE_AGENTS):
        for host_idx in range(mappings.num_hosts):
            if not bool(const.host_active[host_idx]) or bool(const.host_is_router[host_idx]):
                continue
            if not bool(const.blue_agent_hosts[agent_id, host_idx]):
                continue
            hostname = mappings.idx_to_hostname[host_idx]
            host = cy_state.hosts[hostname]
            if predicate is None or predicate(host):
                return agent_id, host_idx, hostname
    raise AssertionError("No blue-visible host matched the requested predicate")


class TestBlueActionTranslation:
    def test_sleep_roundtrip(self, blue_translation_context):
        const, mappings, _ = blue_translation_context

        action = jax_blue_to_cyborg(BLUE_SLEEP, 0, mappings, const=const)
        assert type(action).__name__ == "Sleep"
        assert cyborg_blue_to_jax(action, "blue_agent_0", mappings, const=const) == BLUE_SLEEP

    def test_monitor_roundtrip(self, blue_translation_context):
        const, mappings, _ = blue_translation_context

        action = jax_blue_to_cyborg(BLUE_MONITOR, 0, mappings, const=const)
        assert type(action).__name__ == "Monitor"
        assert cyborg_blue_to_jax(action, "blue_agent_0", mappings, const=const) == BLUE_MONITOR

    @pytest.mark.parametrize(
        ("action_name", "expected_cls"),
        [
            ("Analyse", "Analyse"),
            ("Remove", "Remove"),
            ("Restore", "Restore"),
        ],
    )
    def test_host_action_roundtrip(self, blue_translation_context, action_name, expected_cls):
        const, mappings, cy_state = blue_translation_context
        agent_id, host_idx, hostname = _find_blue_host(const, mappings, cy_state)

        action_idx = encode_blue_action(action_name, host_idx, agent_id, const=const)
        action = jax_blue_to_cyborg(action_idx, agent_id, mappings, const=const)

        assert type(action).__name__ == expected_cls
        assert action.hostname == hostname
        assert cyborg_blue_to_jax(action, f"blue_agent_{agent_id}", mappings, const=const) == action_idx

    def test_decoy_roundtrip(self, blue_translation_context):
        """With collapsed action space, all decoy actions produce generic DeployDecoy."""
        const, mappings, cy_state = blue_translation_context

        agent_id, host_idx, hostname = _find_blue_host(const, mappings, cy_state)
        action_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)

        action = jax_blue_to_cyborg(action_idx, agent_id, mappings, const=const)
        assert type(action).__name__ == "DeployDecoy"
        assert action.hostname == hostname
        assert cyborg_blue_to_jax(action, f"blue_agent_{agent_id}", mappings, const=const) == action_idx

        wrapper_action = jax_blue_to_cyborg_wrapper_action(action_idx, agent_id, mappings, const=const)
        assert type(wrapper_action).__name__ == "DeployDecoy"
        assert wrapper_action.hostname == hostname

    @pytest.mark.parametrize(
        ("action_name", "expected_cls"),
        [
            ("BlockTrafficZone", "BlockTrafficZone"),
            ("AllowTrafficZone", "AllowTrafficZone"),
        ],
    )
    def test_traffic_action_roundtrip(self, blue_translation_context, action_name, expected_cls):
        const, mappings, _ = blue_translation_context
        subnet_ids = sorted(mappings.subnet_names.keys())
        src_subnet, dst_subnet = subnet_ids[0], subnet_ids[1]

        action_idx = encode_blue_action(
            action_name,
            -1,
            0,
            src_subnet=src_subnet,
            dst_subnet=dst_subnet,
        )
        action = jax_blue_to_cyborg(action_idx, 0, mappings, const=const)

        assert type(action).__name__ == expected_cls
        assert action.from_subnet == mappings.subnet_names[src_subnet]
        assert action.to_subnet == mappings.subnet_names[dst_subnet]
        assert cyborg_blue_to_jax(action, "blue_agent_0", mappings, const=const) == action_idx
