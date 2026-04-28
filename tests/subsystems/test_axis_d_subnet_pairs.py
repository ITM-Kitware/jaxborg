"""Phase 2 axis D: per-key allowed_subnet_pairs variation tests."""

import jax
import numpy as np

from jaxborg.constants import MISSION_PHASES, NUM_SUBNETS
from jaxborg.topology import build_topology
from jaxborg.topology_numpy import (
    _build_allowed_subnet_pairs_pure,
    _required_pairs_per_phase,
    _validate_subnet_pairs,
    get_subnet_pairs_bank,
)


def test_subnet_pairs_bank_shape_and_default():
    bank = get_subnet_pairs_bank()
    assert bank.ndim == 4
    assert bank.shape[1:] == (MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS)
    assert bank.dtype == np.bool_
    np.testing.assert_array_equal(bank[0], _build_allowed_subnet_pairs_pure())


def test_all_bank_entries_pass_validator():
    bank = get_subnet_pairs_bank()
    for i in range(bank.shape[0]):
        assert _validate_subnet_pairs(bank[i]), f"bank[{i}] missed required pairs"


def test_required_pairs_present_in_default():
    default = _build_allowed_subnet_pairs_pure()
    for phase_idx, req in enumerate(_required_pairs_per_phase()):
        for si, di in req:
            assert default[phase_idx, si, di], f"phase {phase_idx} missing {(si, di)}"


def test_default_path_allowed_subnet_pairs_unchanged():
    """vary_subnet_pairs=False reproduces the legacy matrix exactly."""
    default = np.asarray(_build_allowed_subnet_pairs_pure())
    for seed in [0, 1, 7, 42, 12345]:
        c = build_topology(jax.random.PRNGKey(seed), vary_subnet_pairs=False)
        np.testing.assert_array_equal(np.asarray(c.allowed_subnet_pairs), default)
        assert int(c.subnet_pairs_bank_index) == 0


def test_vary_subnet_pairs_produces_distinct_entries():
    bank_size = get_subnet_pairs_bank().shape[0]
    if bank_size < 2:
        return  # nothing to vary; bank rejected all permutations
    indices = set()
    for seed in range(64):
        c = build_topology(jax.random.PRNGKey(seed), vary_subnet_pairs=True)
        indices.add(int(c.subnet_pairs_bank_index))
    assert len(indices) >= min(3, bank_size), f"only {len(indices)} distinct entries in 64 keys"


def test_vary_subnet_pairs_outputs_in_bank():
    bank = np.asarray(get_subnet_pairs_bank())
    for seed in range(32):
        c = build_topology(jax.random.PRNGKey(seed), vary_subnet_pairs=True)
        pairs = np.asarray(c.allowed_subnet_pairs)
        match = np.any(np.all(bank == pairs[None, ...], axis=(1, 2, 3)))
        assert match, f"seed={seed} allowed_subnet_pairs not found in bank"
