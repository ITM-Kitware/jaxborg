"""L1 regression test: _ordered_green_hosts must place active hosts first.

A historical bug permuted ALL GLOBAL_MAX_HOSTS entries instead of only the
active portion, shuffling active green agents into positions beyond
num_green_agents where the fori_loop never reached them. The current
typed-phase path uses `_ordered_green_hosts` directly (no per-step shuffle),
but this contract still matters: the vmap green path slices the first
num_green_agents entries and trusts they're all active.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.green import _ordered_green_hosts
from jaxborg.constants import GLOBAL_MAX_HOSTS


@pytest.fixture(scope="module")
def topology():
    from jaxborg.scenarios.cc4.topology import build_topology

    return build_topology(jax.random.PRNGKey(42), num_steps=500)


class TestGreenShuffleCoversAllActive:
    """Verify that the green shuffle preserves all active green hosts."""

    def test_ordered_green_hosts_active_first(self, topology):
        """_ordered_green_hosts puts active hosts in the first num_green_agents slots."""
        const = topology
        order = _ordered_green_hosts(const)
        n = int(const.num_green_agents)
        active_set = set(int(order[i]) for i in range(n))
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.green_agent_active[h]):
                assert h in active_set, f"Active green host {h} not in first {n} entries"
        assert len(active_set) == n

    def test_shuffle_preserves_all_active_in_range(self, topology):
        """After shuffling, the first num_green_agents entries must all be
        active green hosts — no inactive padding leaked in."""
        const = topology
        n_green = int(const.num_green_agents)
        green_host_order = _ordered_green_hosts(const)

        active_hosts = set()
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.green_agent_active[h]):
                active_hosts.add(h)
        assert len(active_hosts) == n_green

        # Replicate the historical active-only shuffle contract.
        for trial in range(20):
            key_green = jax.random.PRNGKey(trial * 100)
            shuffle_key = jax.random.fold_in(key_green, 7919)
            green_shuffle_key = jax.random.fold_in(shuffle_key, 1)

            rand_keys = jax.random.uniform(green_shuffle_key, (GLOBAL_MAX_HOSTS,))
            is_active_pos = jnp.arange(GLOBAL_MAX_HOSTS) < const.num_green_agents
            shuffle_sort = jnp.where(is_active_pos, rand_keys, 2.0)
            shuffled = green_host_order[jnp.argsort(shuffle_sort)]

            # All first n_green entries must be active
            shuffled_active = set(int(shuffled[i]) for i in range(n_green))
            assert shuffled_active == active_hosts, (
                f"Trial {trial}: shuffled active set differs. "
                f"Missing: {active_hosts - shuffled_active}, "
                f"Extra: {shuffled_active - active_hosts}"
            )

    def test_buggy_permutation_drops_active_hosts(self, topology):
        """Demonstrate that the old buggy permutation of ALL GLOBAL_MAX_HOSTS
        entries causes active green hosts to be dropped from the loop range."""
        const = topology
        n_green = int(const.num_green_agents)
        green_host_order = _ordered_green_hosts(const)

        active_hosts = set()
        for h in range(GLOBAL_MAX_HOSTS):
            if bool(const.green_agent_active[h]):
                active_hosts.add(h)

        # Replicate the OLD buggy shuffle
        dropped_counts = []
        for trial in range(20):
            key_green = jax.random.PRNGKey(trial * 100)
            shuffle_key = jax.random.fold_in(key_green, 7919)
            green_shuffle_key = jax.random.fold_in(shuffle_key, 1)

            # BUG: permute ALL GLOBAL_MAX_HOSTS entries, not just active
            green_perm = jax.random.permutation(green_shuffle_key, GLOBAL_MAX_HOSTS)
            buggy_shuffled = green_host_order[green_perm[:GLOBAL_MAX_HOSTS]]

            # Check how many active hosts are in the first n_green positions
            in_range = set(int(buggy_shuffled[i]) for i in range(n_green))
            found_active = in_range & active_hosts
            dropped = n_green - len(found_active)
            dropped_counts.append(dropped)

        avg_dropped = np.mean(dropped_counts)
        # With ~62 active out of 137 total, the buggy shuffle drops ~50% on average
        assert avg_dropped > n_green * 0.2, (
            f"Expected buggy shuffle to drop >20% of active hosts on average, "
            f"but only dropped {avg_dropped:.1f}/{n_green}"
        )
