"""Backwards-compat shim — prefer ``make_fsm_red_env`` for new code.

The old ``ResilienceRedCC4Env`` and ``TargetedRedCC4Env`` classes have been
collapsed into the single :class:`FsmRedCC4Env` parameterised by a
:class:`RedSelector` and an extras factory. ``ResilienceRedCC4Env(...)``
constructs the same env via the new path; existing callers continue to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jaxborg.parity.fsm_red_env import FsmRedCC4Env, FsmRedEnvState, make_fsm_red_env

# Re-exported for callers that still type-annotate against this name.
ResilienceEnvState = FsmRedEnvState  # alias


def ResilienceRedCC4Env(  # noqa: N802 — kept for caller compat
    num_steps: int = 500,
    *,
    topology_mode: str = "generative",
    training_mode: bool = False,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    target_weight: float = 5.0,
    cia_target: str | None = None,
) -> FsmRedCC4Env:
    """Build a resilience- or CIA-targeted red env.

    Equivalent to ``make_fsm_red_env(red_agent=...)`` with the right name:
      cia_target=None  → ``red_agent="resilience"`` (target_weight applied)
      cia_target="c"   → ``red_agent="cia_c"``
      cia_target="i"   → ``red_agent="cia_i"``
      cia_target="a"   → ``red_agent="cia_a"``
    """
    if cia_target is None:
        red_agent = "resilience"
    elif cia_target in ("c", "i", "a"):
        red_agent = f"cia_{cia_target}"
    else:
        raise ValueError(f"Unknown CIA target {cia_target!r}; expected 'c', 'i', 'a', or None")

    return make_fsm_red_env(
        num_steps=num_steps,
        topology_mode=topology_mode,
        training_mode=training_mode,
        topology_path=topology_path,
        red_agent=red_agent,
        target_weight=target_weight,
    )


__all__ = ["ResilienceRedCC4Env", "ResilienceEnvState"]
