"""Regression test: eval episode seeds must produce unique CybORG envs.

History: the old eval scheme used ``cyborg_bank_seed_from_seed(seed + ep*100,
bank_match_size=32)`` which mapped 30 episodes to only 8 unique CybORG seeds
via the topology bank's ``mod bank_size`` indexing.  After the bank was
retired, ``make_cyborg_env`` takes raw ``seed`` and the dev parity rollout
loops pass ``seed + ep``.

This test pins that the seeding pattern produces distinct seeds across the
30-episode default eval horizon.  A future regression to ``seed * 100``-style
indexing would be caught here.
"""

import inspect

from scripts.dev.parity import cyborg_bridge, cyborg_rollout


def test_seed_plus_ep_unique_across_30_episodes():
    """30 episodes ⇒ 30 distinct CybORG seeds under the seed+ep scheme."""
    seed = 0
    cyborg_seeds = [seed + ep for ep in range(30)]
    assert len(set(cyborg_seeds)) == 30


def test_make_cyborg_env_accepts_distinct_seeds():
    """``make_cyborg_env(seed=k)`` must propagate ``seed`` to ``CybORG``.

    We don't actually instantiate the env (CybORG is heavyweight); inspecting
    the source is enough to catch the regression class we care about (someone
    silently re-introducing a ``seed * N`` collapse inside ``make_cyborg_env``).

    The bridge delegates to ``evaluation.cyborg_env_factory.make_cyborg_env``,
    so we check both layers: bridge passes ``seed`` unmodified, and the
    factory threads it into ``CybORG(...)``.
    """
    from jaxborg.evaluation import cyborg_env_factory

    bridge_src = inspect.getsource(cyborg_bridge.make_cyborg_env)
    factory_src = inspect.getsource(cyborg_env_factory.make_cyborg_env)

    # Bridge must pass `seed` through to the factory unmodified.
    assert "seed," in bridge_src or "seed=seed" in bridge_src
    # Factory must thread `seed` into the CybORG constructor.
    assert 'CybORG(sg, "sim", seed=seed)' in factory_src or "seed=seed" in factory_src
    # No leftover bank-index helpers anywhere.
    for src in (bridge_src, factory_src):
        assert "cyborg_bank_seed_from_seed" not in src
        assert "cyborg_bank_index_from_key" not in src


def test_eval_loops_use_seed_plus_ep():
    """Per-episode CybORG seeds must follow ``seed + ep`` (or plain ``ep``);
    the retired bank scheme used ``seed + ep * 100`` which collapsed 30
    episodes onto 8 distinct seeds via mod-bank-size indexing.
    """
    src = inspect.getsource(cyborg_rollout)
    seed_args = [
        line.strip() for line in src.splitlines() if "make_cyborg_env(" in line and "def make_cyborg_env" not in line
    ]
    assert seed_args, "no make_cyborg_env() call sites found in dev parity rollout"
    for line in seed_args:
        # Must thread `seed`/`ep` straight through — no `ep * N` collision-prone
        # multipliers.
        assert "*" not in line.split("seed=", 1)[1].split(")", 1)[0], (
            f"make_cyborg_env seed expression must not multiply (collision risk): {line}"
        )
