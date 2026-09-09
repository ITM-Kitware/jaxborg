from __future__ import annotations

from dataclasses import dataclass

RED_REWARD_MODES = ("zero_sum", "damage")
BLUE_BLOCK_POLICIES = ("cc4", "mission_safe")


@dataclass(frozen=True)
class GameVariant:
    """Bundle of CC4 game rules. One source of truth for both JAX and CybORG envs.

    ``red_reward`` and ``blue_block_policy`` default to the values that
    reproduce stock CC4 exactly; every parity path leaves them alone.  They are
    opt-in co-training knobs, documented in ``docs/cotraining_collapse.md``.

    ``red_reward``
        ``zero_sum`` — learned Red's payoff is the exact negation of Blue's,
        including ``reward_asf`` (which only Blue can trigger) and Blue's
        Restore ``action_cost``.  ``damage`` — Red is paid only for the harm it
        can actually cause, ``-(reward_ria + reward_lwf)``.  Blue's reward is
        identical either way, so this changes nothing for a scripted Red.

    ``blue_block_policy``
        ``cc4`` — Blue may block any subnet pair, as in CybORG.
        ``mission_safe`` — BlockTraffic is masked out for pairs the current
        mission phase's comms policy permits, so Blue cannot cut traffic the
        green agents are required to carry.  This *removes* Blue actions and so
        changes the CC4 action contract; results under it are not comparable to
        stock-CC4 benchmark numbers.
    """

    name: str
    red_agent: str = "finite_state"
    target_weight: float = 5.0
    op_zone_servers: int | None = None
    resilience_roles: bool = False
    num_steps: int = 500
    red_reward: str = "zero_sum"
    blue_block_policy: str = "cc4"

    def __post_init__(self) -> None:
        if self.red_reward not in RED_REWARD_MODES:
            raise ValueError(f"red_reward must be one of {RED_REWARD_MODES}; got {self.red_reward!r}")
        if self.blue_block_policy not in BLUE_BLOCK_POLICIES:
            raise ValueError(f"blue_block_policy must be one of {BLUE_BLOCK_POLICIES}; got {self.blue_block_policy!r}")

    @property
    def is_stock_contract(self) -> bool:
        """True when Blue's action space and reward match stock CC4."""
        return self.blue_block_policy == "cc4"
