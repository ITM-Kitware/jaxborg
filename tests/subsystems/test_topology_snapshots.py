import jax
import numpy as np

from jaxborg.parity.fsm_red_env import FsmRedCC4Env
from jaxborg.scenarios.cc4.topology import (
    TOPOLOGY_SNAPSHOT_FIELDS,
    TOPOLOGY_SNAPSHOT_FORMAT,
    build_topology,
    load_topology,
    load_topology_metadata,
    save_topology,
)


def _assert_const_fields_equal(actual, expected, fields):
    for name in fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
            err_msg=f"field mismatch: {name}",
        )


def test_topology_snapshot_round_trips_static_fields(tmp_path):
    const = build_topology(jax.random.PRNGKey(123))
    path = tmp_path / "topology.npz"

    save_topology(const, path, metadata={"source": "generated", "source_seed": 123})

    loaded = load_topology(path)
    _assert_const_fields_equal(loaded, const, TOPOLOGY_SNAPSHOT_FIELDS)

    metadata = load_topology_metadata(path)
    assert metadata["format"] == TOPOLOGY_SNAPSHOT_FORMAT
    assert metadata["format_version"] == 1
    assert metadata["source"] == "generated"
    assert metadata["source_seed"] == 123
    assert "jaxborg_git_sha" in metadata


def test_fsm_env_reset_from_snapshot_uses_saved_layout(tmp_path):
    const = build_topology(jax.random.PRNGKey(789))
    path = tmp_path / "topology.npz"
    save_topology(const, path)

    env = FsmRedCC4Env(num_steps=500, topology_path=path)
    _, env_state = env.reset(jax.random.PRNGKey(999))

    _assert_const_fields_equal(
        env_state.const,
        const,
        (
            "host_active",
            "host_subnet",
            "host_is_router",
            "host_is_server",
            "host_is_user",
            "initial_services",
            "blue_agent_hosts",
            "red_start_hosts",
            "green_agent_host",
            "green_agent_active",
            "num_hosts",
        ),
    )


def test_snapshot_path_list_is_accepted(tmp_path):
    const = build_topology(jax.random.PRNGKey(321))
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_topology(const, first)
    save_topology(const, second)

    env = FsmRedCC4Env(num_steps=500, topology_path=[first, second])
    _, env_state = env.reset(jax.random.PRNGKey(654))

    assert env._env._const_bank_size == 2
    _assert_const_fields_equal(env_state.const, const, ("host_active", "initial_services", "num_hosts"))
