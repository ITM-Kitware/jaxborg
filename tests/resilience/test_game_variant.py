from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.game_variants import (
    CC4_STOCK,
    CIA_A,
    CIA_C,
    CIA_I,
    CIA_RESILIENCE,
    VARIANTS,
)


def test_cc4_stock_is_all_defaults():
    assert CC4_STOCK == GameVariant(name="cc4_stock")
    assert CC4_STOCK.red_agent == "finite_state"
    assert CC4_STOCK.op_zone_servers is None
    assert CC4_STOCK.resilience_roles is False


def test_cia_resilience_fixes_op_zones_and_enables_roles():
    assert CIA_RESILIENCE.op_zone_servers == 3
    assert CIA_RESILIENCE.resilience_roles is True
    assert CIA_RESILIENCE.red_agent == "resilience"


def test_cia_subvariants_share_topology_constraints():
    for v in (CIA_C, CIA_I, CIA_A):
        assert v.op_zone_servers == 3
        assert v.resilience_roles is True
        assert v.target_weight == 10.0


def test_variants_registry_is_complete():
    assert set(VARIANTS) == {"cc4_stock", "cia_resilience", "cia_c", "cia_i", "cia_a"}
    for name, v in VARIANTS.items():
        assert v.name == name
