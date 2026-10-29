from CybORG.Agents.Wrappers import EnterpriseMAE

from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK, CIA_C


def test_stock_returns_no_role_map():
    env = make_cyborg_env(CC4_STOCK, seed=42, wrapper_class=EnterpriseMAE)
    r = reset_cyborg_env(env, CC4_STOCK, ep_seed=42)
    assert r.role_map is None
    assert r.obs is not None and r.info is not None


def test_cia_populates_role_map():
    env = make_cyborg_env(CIA_C, seed=42, wrapper_class=EnterpriseMAE)
    r = reset_cyborg_env(env, CIA_C, ep_seed=42)
    assert r.role_map is not None
    assert len(r.role_map) == 3
    assert set(r.role_map.values()) == {1, 2, 3}  # AUTH, DB, WEB


def test_cia_role_map_varies_with_ep_seed():
    env = make_cyborg_env(CIA_C, seed=42, wrapper_class=EnterpriseMAE)
    seen = set()
    for ep_seed in range(20):
        r = reset_cyborg_env(env, CIA_C, ep_seed=ep_seed)
        seen.add(tuple(sorted(r.role_map.items())))
    assert len(seen) >= 5  # not stuck on one assignment
