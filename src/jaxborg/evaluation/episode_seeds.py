"""Shared, reproducible episode seeds for paired evaluation sweeps."""

from collections.abc import Sequence

EPISODE_SEED_SCHEME = "base_times_count_plus_replica_v1"


def expand_episode_seeds(seeds: Sequence[int], episodes_per_seed: int) -> list[int]:
    """Give each base seed a disjoint block, preserving seed/replicate order.

    ``seed + replicate`` overlaps for adjacent base seeds. Multiplication by
    the replicate count makes every episode distinct, while one episode per
    seed retains the supplied seeds. The same inputs pair across evaluators,
    policies, worker counts and topology banks.
    """
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
        raise ValueError("episodes_per_seed must be an integer")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive")
    if not seeds:
        raise ValueError("evaluation requires at least one episode seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be distinct")
    if (max(seeds) + 1) * episodes_per_seed > 2**32:
        raise ValueError("expanded episode seeds must fit in an unsigned 32-bit integer")
    return [seed * episodes_per_seed + replicate for seed in seeds for replicate in range(episodes_per_seed)]
