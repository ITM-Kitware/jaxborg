"""Regression test: eval episode seeds must produce unique CybORG episodes.

The old scheme (seed + ep*100 with bank_match_size=32) mapped 30 episodes
to only 8 unique CybORG seeds via cyborg_bank_seed_from_seed(), causing
identical episode repetition. The fix uses seed + ep without bank mapping.
"""

import jax

from jaxborg.scenarios.cc4.topology import cyborg_bank_index_from_key


class TestEvalSeedUniqueness:
    """Verify that 30 eval episodes produce unique seeds in both backends."""

    def test_jaxborg_bank_indices_unique_for_30_episodes(self):
        """JAXborg must select 30 distinct topology bank entries for 30 episodes."""
        seed = 0
        bank_size = 32
        indices = [int(cyborg_bank_index_from_key(jax.random.PRNGKey(seed + ep), bank_size)) for ep in range(30)]
        assert len(set(indices)) == 30, f"Only {len(set(indices))} unique bank indices for 30 episodes: {indices}"

    def test_cyborg_seeds_unique_for_30_episodes(self):
        """CybORG must receive 30 distinct seeds for 30 episodes."""
        seed = 0
        cyborg_seeds = [seed + ep for ep in range(30)]
        assert len(set(cyborg_seeds)) == 30

    def test_jaxborg_bank_index_matches_cyborg_seed(self):
        """With seed=0, JAXborg bank index k should match CybORG seed k.

        The topology bank is built from CybORG(seed=0), CybORG(seed=1), etc.
        so bank_index == cyborg_seed means both backends use the same topology.
        """
        seed = 0
        bank_size = 32
        for ep in range(30):
            jax_idx = int(cyborg_bank_index_from_key(jax.random.PRNGKey(seed + ep), bank_size))
            cyborg_seed = seed + ep
            assert jax_idx == cyborg_seed, f"ep {ep}: JAXborg bank_idx={jax_idx} != CybORG seed={cyborg_seed}"

    def test_old_scheme_had_collisions(self):
        """Document that the old ep*100 scheme produced only 8 unique bank seeds."""
        seed = 0
        bank_size = 32
        old_indices = [
            int(cyborg_bank_index_from_key(jax.random.PRNGKey(seed + ep * 100), bank_size)) for ep in range(30)
        ]
        # The old scheme mapped 30 episodes to only 8 unique bank entries
        assert len(set(old_indices)) == 8, (
            f"Old scheme unexpectedly has {len(set(old_indices))} unique indices (expected 8)"
        )
