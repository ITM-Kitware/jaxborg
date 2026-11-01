import jax
import numpy as np
import pytest

from jaxborg.constants import SUBNET_IDS
from jaxborg.scenarios.cc4.topology import (
    build_topology,
    load_topology,
    load_topology_metadata,
    save_topology,
)
from jaxborg.topology_banks import (
    _validate_cached_snapshot,
    materialize_topology_bank,
    parse_topology_generation,
)


def _section(cache_dir, *, op_zone_servers=3, generator="jax"):
    return {
        "topology_generation": {
            "generator": generator,
            "seed_start": 40,
            "count": 1,
            "op_zone_servers": op_zone_servers,
            "cache_dir": str(cache_dir),
        }
    }


def _server_count(const, subnet_name):
    subnet = SUBNET_IDS[subnet_name]
    return int(
        np.sum(
            np.asarray(const.host_active) & np.asarray(const.host_is_server) & (np.asarray(const.host_subnet) == subnet)
        )
    )


def test_op_zone_servers_changes_paths_without_reinterpreting_legacy_cache(tmp_path):
    legacy_section = _section(tmp_path)
    del legacy_section["topology_generation"]["op_zone_servers"]
    fixed_section = _section(tmp_path)

    legacy = parse_topology_generation(legacy_section, scope="eval", repo_root=tmp_path)
    fixed = parse_topology_generation(fixed_section, scope="eval", repo_root=tmp_path)

    assert legacy is not None
    assert fixed is not None
    assert legacy.op_zone_servers is None
    assert fixed.op_zone_servers == 3
    assert legacy.paths[0].name == "jax_seed_0000000040.snapshot.npz"
    assert fixed.paths[0].name == "jax_ops3_seed_0000000040.snapshot.npz"
    assert legacy.paths[0] != fixed.paths[0]


def test_materialized_bank_has_exact_operational_server_counts_and_provenance(tmp_path):
    paths = materialize_topology_bank(
        _section(tmp_path),
        scope="eval",
        repo_root=tmp_path,
    )

    assert len(paths) == 1
    const = load_topology(paths[0])
    assert _server_count(const, "OPERATIONAL_ZONE_A") == 3
    assert _server_count(const, "OPERATIONAL_ZONE_B") == 3
    metadata = load_topology_metadata(paths[0])
    assert metadata["source"] == "generated"
    assert metadata["source_seed"] == 40
    assert metadata["op_zone_servers"] == 3


@pytest.mark.parametrize("value", [True, 0, 7, 1.5])
def test_invalid_op_zone_server_counts_are_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="op_zone_servers"):
        parse_topology_generation(
            _section(tmp_path, op_zone_servers=value),
            scope="eval",
            repo_root=tmp_path,
        )


def test_cyborg_generation_cannot_claim_a_fixed_server_count(tmp_path):
    with pytest.raises(ValueError, match="only supported by the jax generator"):
        parse_topology_generation(
            _section(tmp_path, generator="cyborg"),
            scope="eval",
            repo_root=tmp_path,
        )


def test_cache_validation_checks_arrays_not_only_server_count_metadata(tmp_path):
    path = tmp_path / "jax_ops3_seed_0000000040.snapshot.npz"
    const = build_topology(jax.random.PRNGKey(40), op_zone_min_servers=3)
    op_a = SUBNET_IDS["OPERATIONAL_ZONE_A"]
    op_a_servers = np.flatnonzero(
        np.asarray(const.host_active) & np.asarray(const.host_is_server) & (np.asarray(const.host_subnet) == op_a)
    )
    malformed = const.replace(host_is_server=const.host_is_server.at[int(op_a_servers[-1])].set(False))
    save_topology(
        malformed,
        path,
        metadata={"source": "generated", "source_seed": 40, "op_zone_servers": 3},
    )

    with pytest.raises(ValueError, match=r"operational server counts \(2, 3\)"):
        _validate_cached_snapshot(path, source="generated", seed=40, op_zone_servers=3)
