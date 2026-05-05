"""CIA-targeted CC4 environments for JAX training.

Drop-in replacements for ResilienceRedCC4Env that route red action selection
through one of the CIA-targeted FSM functions:

  TargetedRedCC4Env(cia_target="c")  →  c_red_select_actions (targets AUTH+DB)
  TargetedRedCC4Env(cia_target="i")  →  i_red_select_actions (targets AUTH+WEB)
  TargetedRedCC4Env(cia_target="a")  →  a_red_select_actions (targets AUTH+DB+WEB)
"""

from __future__ import annotations

from jaxborg.parity.resilience_red_env import ResilienceRedCC4Env
from jaxborg.scenarios.cc4.targeted_red_fsm import (
    a_red_select_actions,
    c_red_select_actions,
    i_red_select_actions,
)

_CIA_FN = {
    "c": c_red_select_actions,
    "i": i_red_select_actions,
    "a": a_red_select_actions,
}


class TargetedRedCC4Env(ResilienceRedCC4Env):
    """ResilienceRedCC4Env with a CIA-specific red action selector.

    Identical to ResilienceRedCC4Env (same state type, same topology
    randomisation) except the red FSM uses a targeted select_actions
    function that biases toward the servers relevant to one CIA component.
    """

    def __init__(self, cia_target: str, **kwargs):
        if cia_target not in _CIA_FN:
            raise ValueError(f"Unknown CIA target {cia_target!r}; expected 'c', 'i', or 'a'")
        super().__init__(**kwargs)
        self._cia_target = cia_target
        self._cia_fn = _CIA_FN[cia_target]

    def _call_red_select(self, state, const, host_resilience_role, red_keys):
        return self._cia_fn(state, const, host_resilience_role, red_keys)

    @property
    def name(self) -> str:
        return f"TargetedRedCC4_{self._cia_target.upper()}"
