"""BasePolicy interface — algorithm-agnostic.

The algorithm script picks a policy by `arch.name` from the recipe and
treats it as an opaque object that knows how to (a) initialize itself for
the chosen backend, (b) forward observations + masks, (c) declare its
buffer layout. Algorithm code never branches on `arch.name`.

Each policy file under `src/jaxborg/policies/` exports:

    JAX_FACTORY   : (action_dim, hidden_dim, hidden_layers, activation) -> flax.linen.Module
    TORCH_FACTORY : (obs_dim, action_dim, hidden_dim, hidden_layers) -> torch.nn.Module
    BUFFER_LAYOUT : str  ('flat' | 'per_agent' | 'sequence')

Two factories instead of a single `BasePolicy` class because the underlying
frameworks (Flax functional vs torch nn.Module) are too different to wrap
in a single concrete type without leaking abstractions. The contract that
*does* unify them is: the JAX module's `__call__(obs, avail_actions)`
returns `(policies.categorical.Categorical, value)`; the torch module's
`get_action_and_value(obs, mask, action=None)` returns
`(action, log_prob, entropy, value)`. Algorithm scripts on each backend
already speak their backend's framework — they don't try to be backend-
agnostic. The unifying piece is the *recipe* and the *registry*.
"""

from __future__ import annotations

import flax.linen as nn

BUFFER_LAYOUT_FLAT = "flat"
BUFFER_LAYOUT_PER_AGENT = "per_agent"
# Sequence-preserving: minibatches are whole trajectories, not shuffled rows.
BUFFER_LAYOUT_SEQUENCE = "sequence"


class CentralizedCriticPolicy(nn.Module):
    """A local actor with a separate, training-only ``critic_obs`` input.

    Implementations expose ``critic_input`` and ``critic_obs_dim``; the joint
    trainer supplies these observations on rollout, bootstrap, and PPO replay.
    Evaluation omits them and runs only the actor.
    """


class RecurrentPolicy(nn.Module):
    """Marker base for policies that carry hidden state between timesteps.

    A recurrent policy breaks the ``(obs, avail_actions) -> (pi, value)``
    contract above: it needs the previous hidden state, a per-row reset flag,
    and a time axis, and it returns the next hidden state. Rather than let
    every call site branch on that, the helpers in ``jaxborg.policies``
    (``policy_step`` / ``policy_sequence`` / ``initial_carry``) speak one
    signature for both families, and this class is what they dispatch on.

    Call sites that have *not* been converted fail loudly instead of silently
    running a memoryless policy: subclasses reject a missing time axis.
    """
