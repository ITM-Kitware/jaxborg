"""Recurrent actor-critic. arch.name = "recurrent".

`ScannedRNN` and the layer layout below are JaxMARL's own recurrent IPPO
baseline, `baselines/IPPO/ippo_rnn_smax.py` at the revision this repo pins
(`545d8f9`, see `uv.lock`). That is the masked-action variant — its
`ActorCriticRNN` takes `(obs, dones, avail_actions)` and applies
`logits - (1 - avail_actions) * 1e10`, which is exactly CC4's contract.

It is copied rather than imported because JaxMARL does not ship it. Its
`pyproject.toml` declares `packages.find include = ['jaxmarl*']`, and
`baselines/` is a sibling of `jaxmarl/` in the repository, so the installed
distribution contains only `environments/`, `wrappers/`, `viz/` and
`gridworld/`. `import jaxmarl.baselines` fails; `jaxmarl.wrappers.baselines`
(which the trainers do use, for `LogWrapper`) is an unrelated module.

Deviations from the upstream file, all deliberate:

* **`cell`.** JaxMARL's recurrent baselines are GRU-only — there is no LSTM
  IPPO in the repository. `cell: lstm` swaps in `nn.OptimizedLSTMCell`, which
  makes the carry a `(c, h)` pair, so the two lines that assume a single array
  (the reset, and `initialize_carry`) are written over a carry pytree instead.
* **Distribution.** `distrax.Categorical` -> `policies.categorical.Categorical`.
  This repo deliberately avoids distrax; see the jax section of pyproject.toml.
* **Initializer gains and activation.** Upstream hard-codes `relu` and
  `orthogonal(2)` on the head hidden layers. Here the activation is the
  recipe's `arch.activation` and the gain is `sqrt(2)`, matching `shared` and
  `separate`, so an arch ablation differs in architecture alone rather than
  also in initialization scale.
* **`hidden_layers` / `trunk`.** Upstream fixes one dense layer before the
  cell and one shared trunk. `hidden_layers` makes the encoder depth a recipe
  knob, and `trunk: separate` gives the actor and the critic their own encoder
  and hidden state — the recurrent analogue of `arch.name: separate`, which
  matters here because Blue's value term runs ~8x its actor term and a shared
  trunk lets that gradient rewrite the features the policy reads.

Why the architecture exists at all: `docs/cotraining_collapse.md` names "actor
is memoryless over transient evidence" as cause 2 of Blue's regression. Blue's
per-host evidence (`malicious_processes`, `network_connections`) ages out after
~2 steps, so a feedforward policy that Analyses a host and does not act inside
that window loses the finding.

Torch/CybORG backend is unsupported: `ippo_cyborg.py` has no sequence-
preserving minibatch path, so TORCH_FACTORY raises rather than quietly training
a different model than the recipe asks for.
"""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from .base import BUFFER_LAYOUT_SEQUENCE, RecurrentPolicy
from .categorical import Categorical as JaxCategorical

CELLS = ("gru", "lstm")
TRUNKS = ("shared", "separate")


def _make_cell(cell: str, features: int, **kwargs) -> nn.RNNCellBase:
    kind = nn.OptimizedLSTMCell if cell == "lstm" else nn.GRUCell
    return kind(features=features, **kwargs)


def _activation(name: str):
    return nn.relu if name == "relu" else nn.tanh


class ScannedRNN(nn.Module):
    """One recurrent layer scanned over a leading time axis.

    Inputs are ``(features, resets)`` of shape ``(T, B, hidden_dim)`` and
    ``(T, B)``. A set reset flag zeroes the incoming carry *before* the cell
    runs, so the first row of a new episode — or of a revived Red agent — sees
    a blank hidden state instead of the previous occupant's.
    """

    cell: str = "gru"

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # Upstream writes `self.initialize_carry(*rnn_state.shape)` here; that
        # init is zeros, and `zeros_like` over the tree is the same value while
        # also covering the LSTM's (c, h) pair.
        rnn_state = jax.tree.map(
            lambda leaf: jnp.where(resets[:, np.newaxis], jnp.zeros_like(leaf), leaf),
            rnn_state,
        )
        new_rnn_state, y = _make_cell(self.cell, ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    @nn.nowrap
    def initialize_carry(batch_size: int, hidden_size: int, cell: str = "gru"):
        # Use a dummy key since the default state init fn is just zeros.
        # `parent=None` because this also runs from the enclosing policy's own
        # `initialize_carry`, where flax would otherwise try to register the
        # cell as a submodule of a module that has no scope yet.
        return _make_cell(cell, hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class _Trunk(nn.Module):
    """Dense encoder feeding one recurrent layer."""

    hidden_dim: int
    hidden_layers: int
    activation: str
    cell: str

    @nn.compact
    def __call__(self, carry, x, resets):
        act_fn = _activation(self.activation)
        embedding = x
        for _ in range(self.hidden_layers):
            embedding = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = act_fn(embedding)
        return ScannedRNN(cell=self.cell)(carry, (embedding, resets))


class _JaxRecurrentActorCritic(RecurrentPolicy):
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 1
    activation: str = "tanh"
    cell: str = "gru"
    trunk: str = "shared"

    @nn.nowrap
    def initialize_carry(self, batch_size: int):
        """Zeroed hidden state for ``batch_size`` independent sequences.

        Pure — no parameters are touched — so it is callable on an unbound
        module, before `.init()`.
        """
        carry = ScannedRNN.initialize_carry(batch_size, self.hidden_dim, self.cell)
        if self.trunk == "separate":
            return {"actor": carry, "critic": carry}
        return carry

    @nn.compact
    def __call__(self, carry, x, avail_actions=None, resets=None):
        """``x`` is time-major ``(T, B, obs_dim)``; returns ``(carry, pi, value)``."""
        if getattr(x, "ndim", 0) != 3:
            raise ValueError(
                "recurrent policies take time-major input (T, B, obs_dim) plus a carry; "
                f"got an array of rank {getattr(x, 'ndim', None)}. Call sites should go "
                "through jaxborg.policies.policy_step / policy_sequence rather than "
                "module.apply(params, obs, mask)."
            )
        if resets is None:
            resets = jnp.zeros(x.shape[:2], dtype=jnp.bool_)

        act_fn = _activation(self.activation)
        trunk_kwargs = dict(
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            cell=self.cell,
        )
        if self.trunk == "separate":
            actor_carry, actor_embedding = _Trunk(**trunk_kwargs, name="actor_trunk")(carry["actor"], x, resets)
            critic_carry, critic_embedding = _Trunk(**trunk_kwargs, name="critic_trunk")(carry["critic"], x, resets)
            new_carry = {"actor": actor_carry, "critic": critic_carry}
        else:
            new_carry, actor_embedding = _Trunk(**trunk_kwargs, name="trunk")(carry, x, resets)
            critic_embedding = actor_embedding

        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_hidden",
        )(actor_embedding)
        actor_mean = act_fn(actor_mean)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_head",
        )(actor_mean)
        if avail_actions is not None:
            unavail_actions = 1 - avail_actions
            logits = logits - (unavail_actions * 1e10)

        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(critic_embedding)
        critic = act_fn(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_head",
        )(critic)
        return new_carry, JaxCategorical(logits=logits), jnp.squeeze(critic, axis=-1)


def jax_factory(
    action_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    *,
    cell: str = "gru",
    trunk: str = "shared",
    **unknown,
):
    if unknown:
        raise ValueError(f"unknown recurrent arch options: {sorted(unknown)}")
    if cell not in CELLS:
        raise ValueError(f"arch.cell must be one of {list(CELLS)}, got {cell!r}")
    if trunk not in TRUNKS:
        raise ValueError(f"arch.trunk must be one of {list(TRUNKS)}, got {trunk!r}")
    if hidden_layers < 1:
        raise ValueError(f"recurrent arch.hidden_layers must be >= 1, got {hidden_layers}")
    return _JaxRecurrentActorCritic(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        cell=cell,
        trunk=trunk,
    )


def torch_factory(*_args, **_kwargs):
    raise NotImplementedError(
        "arch.name 'recurrent' has no CybORG/torch backend: ippo_cyborg.py shuffles "
        "transitions across the time axis, which a sequence model cannot train on. "
        "Use the jax backend, or a feedforward arch."
    )


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_SEQUENCE
