"""BasePolicy interface — algorithm-agnostic.

The algorithm script picks a policy by `arch.name` from the recipe and
treats it as an opaque object that knows how to (a) initialize itself for
the chosen backend, (b) forward observations + masks, (c) declare its
buffer layout. Algorithm code never branches on `arch.name`.

Each policy file under `src/jaxborg/policies/` exports:

    JAX_FACTORY   : (action_dim, hidden_dim, hidden_layers, activation) -> flax.linen.Module
    TORCH_FACTORY : (obs_dim, action_dim, hidden_dim, hidden_layers) -> torch.nn.Module
    BUFFER_LAYOUT : str  ('flat' | 'per_agent')

Two factories instead of a single `BasePolicy` class because the underlying
frameworks (Flax functional vs torch nn.Module) are too different to wrap
in a single concrete type without leaking abstractions. The contract that
*does* unify them is: the JAX module's `__call__(obs, avail_actions)`
returns `(distrax.Categorical, value)`; the torch module's
`get_action_and_value(obs, mask, action=None)` returns
`(action, log_prob, entropy, value)`. Algorithm scripts on each backend
already speak their backend's framework — they don't try to be backend-
agnostic. The unifying piece is the *recipe* and the *registry*.
"""

BUFFER_LAYOUT_FLAT = "flat"
BUFFER_LAYOUT_PER_AGENT = "per_agent"
