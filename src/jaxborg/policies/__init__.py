"""Policy registry — `arch.name` -> policy module.

Each policy module exposes JAX_FACTORY, TORCH_FACTORY, BUFFER_LAYOUT.
Algorithm scripts call `make_jax_policy(name, ...)` or
`make_torch_policy(name, ...)`; the factory returns a Flax module / torch
nn.Module ready to consume.

Adding a new architecture = new file under this package + one line in
POLICY_REGISTRY below.

Feedforward and recurrent policies have different native signatures. Rather
than make every caller branch, use the four helpers at the bottom of this
module — `policy_from_arch`, `init_policy_params`, `initial_carry`, and
`policy_step` / `policy_sequence`. They speak one signature for both families
and carry `None` around where a feedforward policy has no hidden state.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp

from . import per_agent, recurrent_actor_critic, separate_actor_critic, shared_actor_critic
from .base import RecurrentPolicy
from .categorical import Categorical

POLICY_REGISTRY: dict[str, ModuleType] = {
    "shared": shared_actor_critic,
    "separate": separate_actor_critic,
    "per_agent": per_agent,
    "recurrent": recurrent_actor_critic,
}

# `arch` keys every architecture understands; anything else is forwarded to the
# selected factory as an option, and rejected there if it does not know it.
ARCH_COMMON_KEYS = frozenset({"name", "hidden_dim", "hidden_layers", "activation"})


def _resolve(name: str) -> ModuleType:
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy arch '{name}'. Known: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]


def make_jax_policy(
    name: str,
    *,
    action_dim: int,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
    activation: str = "tanh",
    **options: Any,
):
    """Return a Flax module instance ready to .init() / .apply()."""
    try:
        return _resolve(name).JAX_FACTORY(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            **options,
        )
    except TypeError as exc:
        if not options:
            raise
        # `cell` / `trunk` under a feedforward `arch.name` is the likely cause,
        # and a bare "unexpected keyword argument" does not say which recipe
        # field is wrong. Silently dropping them would be worse: the run would
        # train an architecture the recipe did not ask for.
        raise ValueError(
            f"arch {name!r} does not accept the extra arch options {sorted(options)}; "
            f"remove them from the recipe's arch block or change arch.name ({exc})"
        ) from exc


def make_torch_policy(
    name: str,
    *,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
    **options: Any,
):
    """Return a torch.nn.Module instance with get_action_and_value/get_value."""
    return _resolve(name).TORCH_FACTORY(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        **options,
    )


def buffer_layout(name: str) -> str:
    return _resolve(name).BUFFER_LAYOUT


def policy_from_arch(arch: dict[str, Any], *, action_dim: int, backend: str = "jax", obs_dim: int | None = None):
    """Instantiate the policy a resolved `arch` mapping describes.

    One place that knows how an `arch` block maps onto a factory call, so
    architecture-specific keys (`cell`, `trunk`, ...) reach the factory instead
    of being dropped by a call site that predates them.
    """
    options = {key: value for key, value in arch.items() if key not in ARCH_COMMON_KEYS}
    common = dict(
        action_dim=action_dim,
        hidden_dim=int(arch.get("hidden_dim", 256)),
        hidden_layers=int(arch.get("hidden_layers", 2)),
    )
    if backend == "jax":
        return make_jax_policy(arch["name"], activation=arch.get("activation", "tanh"), **common, **options)
    if obs_dim is None:
        raise ValueError("torch policies need obs_dim")
    return make_torch_policy(arch["name"], obs_dim=int(obs_dim), **common, **options)


def is_recurrent(module: Any) -> bool:
    """Whether `module` carries hidden state between timesteps."""
    return isinstance(module, RecurrentPolicy)


def initial_carry(module: Any, batch_size: int):
    """Zeroed hidden state for `batch_size` sequences, or None if memoryless."""
    return module.initialize_carry(batch_size) if is_recurrent(module) else None


def init_policy_params(module: Any, rng: jax.Array, obs_dim: int):
    """Initialize parameters for either policy family from the observation width."""
    if not is_recurrent(module):
        return module.init(rng, jnp.zeros((obs_dim,), dtype=jnp.float32))
    return module.init(
        rng,
        module.initialize_carry(1),
        jnp.zeros((1, 1, obs_dim), dtype=jnp.float32),
        None,
        jnp.zeros((1, 1), dtype=jnp.bool_),
    )


def policy_step(
    module: Any,
    params: Any,
    obs: jax.Array,
    avail_actions: jax.Array | None = None,
    *,
    carry: Any = None,
    reset: jax.Array | None = None,
) -> tuple[Categorical, jax.Array, Any]:
    """One timestep for a batch of rows. Returns `(pi, value, next_carry)`.

    `obs` is `(B, obs_dim)`, `reset` is `(B,)`. For a feedforward policy the
    carry is returned unchanged (it is `None`), so callers can thread it
    unconditionally.
    """
    if not is_recurrent(module):
        pi, value = module.apply(params, obs, avail_actions)
        return pi, value, carry
    if carry is None:
        raise ValueError("a recurrent policy needs a carry; build one with policies.initial_carry(module, batch)")
    resets = jnp.zeros(obs.shape[:1], dtype=jnp.bool_) if reset is None else jnp.asarray(reset, dtype=jnp.bool_)
    next_carry, pi, value = module.apply(
        params,
        carry,
        obs[None],
        None if avail_actions is None else avail_actions[None],
        resets[None],
    )
    return Categorical(logits=pi.logits[0]), value[0], next_carry


def policy_sequence(
    module: Any,
    params: Any,
    obs: jax.Array,
    avail_actions: jax.Array | None = None,
    *,
    carry: Any = None,
    reset: jax.Array | None = None,
) -> tuple[Categorical, jax.Array, Any]:
    """Whole trajectories at once. `obs` is time-major `(T, B, obs_dim)`.

    This is the update-time counterpart of `policy_step`: it replays a stored
    rollout window from the hidden state that window began with.
    """
    if not is_recurrent(module):
        pi, value = module.apply(params, obs, avail_actions)
        return pi, value, carry
    if carry is None:
        raise ValueError("a recurrent policy needs a carry; build one with policies.initial_carry(module, batch)")
    resets = jnp.zeros(obs.shape[:2], dtype=jnp.bool_) if reset is None else jnp.asarray(reset, dtype=jnp.bool_)
    next_carry, pi, value = module.apply(params, carry, obs, avail_actions, resets)
    return pi, value, next_carry


__all__ = [
    "ARCH_COMMON_KEYS",
    "POLICY_REGISTRY",
    "buffer_layout",
    "init_policy_params",
    "initial_carry",
    "is_recurrent",
    "make_jax_policy",
    "make_torch_policy",
    "policy_from_arch",
    "policy_sequence",
    "policy_step",
]
