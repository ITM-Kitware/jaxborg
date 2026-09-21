"""Training-only topology sampling across parallel environments."""

import jax
import numpy as np
import pytest

from jaxborg.training_topology_sampling import (
    sample_training_topology_indices,
    validate_training_topology_coverage,
)


def test_diversified_batch_samples_without_replacement() -> None:
    indices = np.asarray(
        sample_training_topology_indices(
            jax.random.PRNGKey(7),
            bank_size=100,
            num_envs=96,
        )
    )

    assert indices.shape == (96,)
    assert len(np.unique(indices)) == 96
    assert indices.min() >= 0
    assert indices.max() < 100

    next_indices = np.asarray(
        sample_training_topology_indices(
            jax.random.PRNGKey(9),
            bank_size=100,
            num_envs=96,
        )
    )
    assert not np.array_equal(indices, next_indices)


def test_singleton_bank_remains_fixed_topology_control() -> None:
    indices = np.asarray(
        sample_training_topology_indices(
            jax.random.PRNGKey(8),
            bank_size=1,
            num_envs=96,
        )
    )

    np.testing.assert_array_equal(indices, np.zeros(96, dtype=np.int32))


@pytest.mark.parametrize("bank_size", [2, 95])
def test_diversified_bank_must_cover_parallel_environments(bank_size: int) -> None:
    with pytest.raises(ValueError, match=r"bank_size >= NUM_ENVS"):
        validate_training_topology_coverage(bank_size, num_envs=96)


def test_bank_equal_to_parallel_environments_is_valid() -> None:
    validate_training_topology_coverage(bank_size=96, num_envs=96)
