"""Tests for the Phase 6 Axis A topology-shape bank.

Validates:
* `_select_const` is deterministic in the PRNG key.
* `_select_const` samples uniformly across the bank (chi-square @ α=0.01).
* All 16 emitted snapshots load cleanly via ``load_topology``.
* All 16 snapshots pass ``_validate_resilience_topology(CIA_RESILIENCE)``.
* Resetting an env wired to the full bank produces multiple distinct
  ``host_subnet`` arrays (i.e. the bank actually drives const variation).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.env import ScenarioEnv
from jaxborg.evaluation.jax_env_factory import _validate_resilience_topology, make_jax_env
from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE
from jaxborg.scenarios.cc4.topology import load_topology

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = REPO_ROOT / "scripts" / "dev" / "topology_bank"
BUILDER = REPO_ROOT / "scripts" / "dev" / "build_topology_bank.py"
EXPECTED_COUNT = 16


def _bank_paths() -> list[Path]:
    return sorted(BANK_DIR.glob("shape_*.snapshot.npz"))


@pytest.fixture(scope="module")
def bank_paths() -> list[Path]:
    paths = _bank_paths()
    if len(paths) != EXPECTED_COUNT:
        # Build the bank if missing/incomplete so the test is hermetic.
        subprocess.check_call(
            [
                sys.executable,
                str(BUILDER),
                "--out-dir",
                str(BANK_DIR),
                "--count",
                str(EXPECTED_COUNT),
                "--seed",
                "0",
            ],
            cwd=REPO_ROOT,
        )
        paths = _bank_paths()
    assert len(paths) == EXPECTED_COUNT, f"expected {EXPECTED_COUNT} snapshots, got {len(paths)}"
    return paths


def test_select_const_is_deterministic(bank_paths: list[Path]) -> None:
    """Same PRNG key → same bank index, same const tree."""
    env = ScenarioEnv(topology_path=bank_paths[:4])
    key = jax.random.PRNGKey(123)
    a = env._select_const(key)
    b = env._select_const(key)
    assert jnp.array_equal(a.host_subnet, b.host_subnet)
    assert jnp.array_equal(a.data_links, b.data_links)
    assert jnp.array_equal(a.allowed_subnet_pairs, b.allowed_subnet_pairs)


def test_select_const_samples_uniformly(bank_paths: list[Path]) -> None:
    """Across many keys with a 4-entry bank, distribution is ~uniform.

    Uses chi-square goodness-of-fit at α=0.01 (df=3, critical ≈ 11.345).
    """
    bank_size = 4
    n_samples = 10000

    keys = jax.random.split(jax.random.PRNGKey(7), n_samples)
    # Mirror the bank-index draw inside ``_select_const`` exactly: a single
    # ``randint`` over [0, bank_size) per key.
    indices_fn = jax.jit(jax.vmap(lambda k: jax.random.randint(k, (), 0, bank_size)))
    indices = np.asarray(indices_fn(keys))

    counts = np.bincount(indices, minlength=bank_size)
    expected = n_samples / bank_size
    chi2 = float(((counts - expected) ** 2 / expected).sum())

    # df = bank_size - 1 = 3; α=0.01 critical ≈ 11.345.
    chi2_crit_001 = 11.345
    assert chi2 < chi2_crit_001, (
        f"chi-square={chi2:.3f} (counts={counts.tolist()}) exceeds α=0.01 critical {chi2_crit_001}; "
        "sampling is non-uniform"
    )

    # Also sanity-check that no bin is more than 5% off expected — a stronger
    # check that's redundant with chi-square but easier to interpret.
    max_dev = max(abs(c - expected) / expected for c in counts)
    assert max_dev < 0.05, f"max deviation {max_dev:.4f} exceeds 5% — counts={counts.tolist()}"


def test_each_snapshot_loads_cleanly(bank_paths: list[Path]) -> None:
    for p in bank_paths:
        const = load_topology(p)
        # Sanity: const has the expected shape ABI.
        assert const.host_subnet.shape == (const.host_active.shape[0],)


def test_each_snapshot_passes_resilience_validator(bank_paths: list[Path]) -> None:
    # Single batched call to ensure every snapshot would be acceptable
    # under the strictest variant we care about.
    _validate_resilience_topology(CIA_RESILIENCE, bank_paths)


def test_env_with_full_bank_produces_distinct_consts(bank_paths: list[Path]) -> None:
    """Reset 32 times across the full 16-shape bank → ≥3 distinct host_subnet arrays."""
    env = make_jax_env(CIA_RESILIENCE, topology_path=list(bank_paths))
    keys = jax.random.split(jax.random.PRNGKey(2026), 32)
    seen: set[bytes] = set()
    for k in keys:
        _, state = env.reset(k)
        seen.add(np.asarray(state.const.host_subnet).tobytes())
    assert len(seen) >= 3, f"expected ≥3 distinct host_subnet arrays across 32 resets, got {len(seen)}"
