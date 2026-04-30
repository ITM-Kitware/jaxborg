"""Per-agent policies — STUB. arch.name = "per_agent".

Locks in the BUFFER_LAYOUT_PER_AGENT branch in the algorithm script so that
the algorithm doesn't accidentally hardcode flat buffer semantics. The
factories raise NotImplementedError; a future PR ships the real
implementation.

The shape we want: 5 independent (actor, critic) pairs, one per blue agent.
Each agent's gradient comes from its own samples, so the algorithm must
slice the rollout buffer by agent index before computing PPO loss.
"""

from .base import BUFFER_LAYOUT_PER_AGENT


def jax_factory(*_args, **_kwargs):
    raise NotImplementedError(
        "per_agent policy is not implemented yet (only registered as a stub). "
        "See plans/jax/cc4/prompts/recipe-refactor-plan.md §5.5."
    )


def torch_factory(*_args, **_kwargs):
    raise NotImplementedError(
        "per_agent policy is not implemented yet (only registered as a stub). "
        "See plans/jax/cc4/prompts/recipe-refactor-plan.md §5.5."
    )


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_PER_AGENT
