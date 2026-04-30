"""Policy registry — `arch.name` -> policy module.

Each policy module exposes JAX_FACTORY, TORCH_FACTORY, BUFFER_LAYOUT.
Algorithm scripts call `make_jax_policy(name, ...)` or
`make_torch_policy(name, ...)`; the factory returns a Flax module / torch
nn.Module ready to consume.

Adding a new architecture = new file under this package + one line in
POLICY_REGISTRY below.
"""

from __future__ import annotations

from types import ModuleType

from . import per_agent, separate_actor_critic, shared_actor_critic

POLICY_REGISTRY: dict[str, ModuleType] = {
    "shared": shared_actor_critic,
    "separate": separate_actor_critic,
    "per_agent": per_agent,
}


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
):
    """Return a Flax module instance ready to .init() / .apply()."""
    return _resolve(name).JAX_FACTORY(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    )


def make_torch_policy(
    name: str,
    *,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
):
    """Return a torch.nn.Module instance with get_action_and_value/get_value."""
    return _resolve(name).TORCH_FACTORY(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )


def buffer_layout(name: str) -> str:
    return _resolve(name).BUFFER_LAYOUT


__all__ = [
    "POLICY_REGISTRY",
    "make_jax_policy",
    "make_torch_policy",
    "buffer_layout",
]
