"""PPO update-step parity: jax `_loss_fn`+optax.adam vs torch loss+torch.optim.Adam.

Given a fixed minibatch (obs, action, mask, old_logprob, advantage, target,
old_value) and matched initial params, run one PPO update on each backend
and compare:

  1. Per-parameter gradients (≤1e-5 tolerance)
  2. Post-Adam-step parameters    (≤1e-4 tolerance)

The two loss bodies are inlined faithfully from
`scripts/train/ippo_jax.py::_loss_fn` (default config, no busy mask, no
value clipping) and `scripts/train/ppo_cleanrl_cyborg.py` PPO update.

This test fails if the two loss math implementations diverge on identical
inputs — isolating the −227 pt matched-training gap from rollout
distributions, RNG, and accumulated float-op order.

Cited by: plans/jax/cc4/prompts/training-loop-parity-prompt.md (Tier 1.1).

Note on the `unbiased=False` torch.std calls: aligning torch's default
ddof=1 with jnp.std's ddof=0 is part of the fix this test enforces.
Without it, advantage normalization differs by sqrt(N/(N-1)) per minibatch
(0.4% at N=128, 0.007% at N=7500). See the matched-training-results entry
dated 2026-04-25.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
import torch.nn as nn
from flax.training.train_state import TrainState

from jaxborg.policy import SharedActorCritic

# Align with `scripts/train/ppo_cleanrl_agent.py::PPOAgent`.
HIDDEN_DIM = 32
OBS_DIM = 16
ACT_DIM = 8
BATCH = 128

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
ADAM_EPS = 1e-5


# ── Mini PPOAgent (inline to avoid full-deps import in this test) ─────────


class TinyPPOAgent(nn.Module):
    """Bit-equivalent to PPOAgent but with test-size dims."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.actor = nn.Linear(HIDDEN_DIM, ACT_DIM)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs, action_mask):
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        logits = self.actor(h)
        # Match ppo_cleanrl_agent.py masking convention.
        logits = logits + (action_mask.float() - 1.0) * 1e8
        value = self.critic(h).squeeze(-1)
        return logits, value


def _torch_to_flax_params(torch_agent: TinyPPOAgent) -> dict:
    """Convert TinyPPOAgent state_dict → SharedActorCritic params pytree.

    Flax Dense uses kernel (in, out); torch Linear uses weight (out, in).
    Flax @nn.compact ordering puts the four Dense layers as
    Dense_0..Dense_3 in source-call order: trunk1, trunk2, actor, critic.
    """
    sd = torch_agent.state_dict()

    def _kw(w_key, b_key):
        return {
            "kernel": jnp.asarray(sd[w_key].numpy().T.copy()),
            "bias": jnp.asarray(sd[b_key].numpy().copy()),
        }

    return {
        "params": {
            "Dense_0": _kw("fc1.weight", "fc1.bias"),
            "Dense_1": _kw("fc2.weight", "fc2.bias"),
            "Dense_2": _kw("actor.weight", "actor.bias"),
            "Dense_3": _kw("critic.weight", "critic.bias"),
        }
    }


def _make_minibatch(seed: int = 0):
    """Random fixed minibatch: matches what ppo_cleanrl_cyborg.py mb_* tensors look like."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((BATCH, OBS_DIM)).astype(np.float32)
    # All actions valid (mask=1 everywhere). Avoids needing to handle the
    # `1e8 vs 1e10` mask-fill convention asymmetry — that's a separate
    # documented deviation tested elsewhere.
    mask = np.ones((BATCH, ACT_DIM), dtype=np.float32)
    actions = rng.integers(low=0, high=ACT_DIM, size=BATCH).astype(np.int64)
    old_logp = rng.standard_normal(BATCH).astype(np.float32) * 0.1 - np.log(ACT_DIM)
    advantages = rng.standard_normal(BATCH).astype(np.float32)
    old_values = rng.standard_normal(BATCH).astype(np.float32) * 0.5
    targets = old_values + rng.standard_normal(BATCH).astype(np.float32) * 0.5
    return obs, actions, mask, old_logp, advantages, targets, old_values


# ── Loss bodies (inlined from each script) ─────────────────────────────────


def _torch_loss(agent, obs, actions, mask, old_logp, mb_adv_np, mb_ret_np):
    """Mirrors ppo_cleanrl_cyborg.py PPO inner-loop loss math.

    The only deviation from the script: `mb_adv.std(unbiased=False)` to
    match `jnp.std`'s ddof=0 default — see module docstring.
    """
    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(actions)
    mask_t = torch.from_numpy(mask)
    old_lp_t = torch.from_numpy(old_logp)
    mb_adv = torch.from_numpy(mb_adv_np)
    mb_ret = torch.from_numpy(mb_ret_np)

    logits, value = agent(obs_t, mask_t)
    dist = torch.distributions.Categorical(logits=logits)
    new_lp = dist.log_prob(act_t)
    ent = dist.entropy()

    # Advantage normalization (ddof=0 to match jnp.std).
    adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
    logratio = new_lp - old_lp_t
    ratio = logratio.exp()
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    vf_loss = 0.5 * ((value - mb_ret) ** 2).mean()
    entropy_loss = ent.mean()

    loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * vf_loss
    return loss


def _jax_loss(params, network, obs, actions, mask, old_logp, mb_adv, mb_targets, mb_value_old):
    """Mirrors ippo_jax.py::_loss_fn (no busy mask, no value clipping)."""
    gae = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
    pi, value = network.apply(params, obs, mask)
    log_prob = pi.log_prob(actions)

    # Unmasked (busy_masking=False) → policy_weight all 1s.
    value_loss = 0.5 * jnp.mean(jnp.square(value - mb_targets))

    ratio = jnp.exp(log_prob - old_logp)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * gae
    loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))
    entropy = pi.entropy().mean()

    return loss_actor + VF_COEF * value_loss - ENT_COEF * entropy


# ── Forward-pass parity (sanity check before the loss math) ───────────────


def test_forward_pass_parity():
    """torch and flax forward passes on identical params + inputs match.

    Compares per-action log-probs and values rather than raw logits because
    `distrax.Categorical.logits` returns log-softmax-normalized logits while
    torch's logits are raw — the difference would mask real divergence.
    """
    torch.manual_seed(0)
    agent = TinyPPOAgent()
    flax_params = _torch_to_flax_params(agent)
    network = SharedActorCritic(action_dim=ACT_DIM, hidden_dim=HIDDEN_DIM, activation="tanh")

    obs, actions, mask, *_ = _make_minibatch(seed=0)

    with torch.no_grad():
        logits_t, val_t = agent(torch.from_numpy(obs), torch.from_numpy(mask))
        dist_t = torch.distributions.Categorical(logits=logits_t)
        logp_t = dist_t.log_prob(torch.from_numpy(actions)).numpy()
        ent_t = dist_t.entropy().numpy()

    pi, val_j = network.apply(flax_params, jnp.asarray(obs), jnp.asarray(mask))
    logp_j = np.asarray(pi.log_prob(jnp.asarray(actions)))
    ent_j = np.asarray(pi.entropy())

    np.testing.assert_allclose(logp_j, logp_t, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ent_j, ent_t, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(val_j), val_t.numpy(), atol=1e-5, rtol=1e-5)


# ── Gradient + post-Adam-step parity ──────────────────────────────────────


@pytest.mark.slow
def test_ppo_minibatch_update_parity():
    """One PPO minibatch update: grads ≤1e-5, post-Adam params ≤1e-4."""
    torch.manual_seed(0)
    agent = TinyPPOAgent()
    flax_params = _torch_to_flax_params(agent)
    network = SharedActorCritic(action_dim=ACT_DIM, hidden_dim=HIDDEN_DIM, activation="tanh")

    obs, actions, mask, old_lp, mb_adv, mb_ret, mb_val_old = _make_minibatch(seed=0)

    # Torch: forward → backward → grads
    optim_torch = torch.optim.Adam(agent.parameters(), lr=LR, eps=ADAM_EPS)
    loss_t = _torch_loss(agent, obs, actions, mask, old_lp, mb_adv, mb_ret)
    optim_torch.zero_grad()
    loss_t.backward()
    torch_grads = {
        "fc1.weight": agent.fc1.weight.grad.detach().numpy().copy(),
        "fc1.bias": agent.fc1.bias.grad.detach().numpy().copy(),
        "fc2.weight": agent.fc2.weight.grad.detach().numpy().copy(),
        "fc2.bias": agent.fc2.bias.grad.detach().numpy().copy(),
        "actor.weight": agent.actor.weight.grad.detach().numpy().copy(),
        "actor.bias": agent.actor.bias.grad.detach().numpy().copy(),
        "critic.weight": agent.critic.weight.grad.detach().numpy().copy(),
        "critic.bias": agent.critic.bias.grad.detach().numpy().copy(),
    }
    # No grad clipping is applied — this test isolates loss + Adam math.
    # Clipping behavior is validated by the global-norm path test in
    # `test_adam_step_parity.py` (Adam math) and the existing matched-RNG
    # parity tests (env mechanics).
    optim_torch.step()
    torch_post = {
        "fc1.weight": agent.fc1.weight.detach().numpy().copy(),
        "fc1.bias": agent.fc1.bias.detach().numpy().copy(),
        "fc2.weight": agent.fc2.weight.detach().numpy().copy(),
        "fc2.bias": agent.fc2.bias.detach().numpy().copy(),
        "actor.weight": agent.actor.weight.detach().numpy().copy(),
        "actor.bias": agent.actor.bias.detach().numpy().copy(),
        "critic.weight": agent.critic.weight.detach().numpy().copy(),
        "critic.bias": agent.critic.bias.detach().numpy().copy(),
    }

    # JAX: same minibatch, same initial params
    tx = optax.adam(LR, eps=ADAM_EPS)
    train_state = TrainState.create(apply_fn=network.apply, params=flax_params, tx=tx)

    obs_j = jnp.asarray(obs)
    act_j = jnp.asarray(actions)
    mask_j = jnp.asarray(mask)
    old_lp_j = jnp.asarray(old_lp)
    adv_j = jnp.asarray(mb_adv)
    ret_j = jnp.asarray(mb_ret)
    val_j_old = jnp.asarray(mb_val_old)

    grad_fn = jax.value_and_grad(_jax_loss)
    _, grads = grad_fn(train_state.params, network, obs_j, act_j, mask_j, old_lp_j, adv_j, ret_j, val_j_old)
    train_state = train_state.apply_gradients(grads=grads)

    # Flax-side grads in the same shape convention as torch (transposed kernel).
    jax_grads_torch_layout = {
        "fc1.weight": np.asarray(grads["params"]["Dense_0"]["kernel"]).T,
        "fc1.bias": np.asarray(grads["params"]["Dense_0"]["bias"]),
        "fc2.weight": np.asarray(grads["params"]["Dense_1"]["kernel"]).T,
        "fc2.bias": np.asarray(grads["params"]["Dense_1"]["bias"]),
        "actor.weight": np.asarray(grads["params"]["Dense_2"]["kernel"]).T,
        "actor.bias": np.asarray(grads["params"]["Dense_2"]["bias"]),
        "critic.weight": np.asarray(grads["params"]["Dense_3"]["kernel"]).T.reshape(1, HIDDEN_DIM),
        "critic.bias": np.asarray(grads["params"]["Dense_3"]["bias"]).reshape(1),
    }

    # Gradient parity (per-tensor)
    for name, t_grad in torch_grads.items():
        j_grad = jax_grads_torch_layout[name]
        np.testing.assert_allclose(
            j_grad,
            t_grad,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"gradient mismatch on {name}",
        )

    jax_post_torch_layout = {
        "fc1.weight": np.asarray(train_state.params["params"]["Dense_0"]["kernel"]).T,
        "fc1.bias": np.asarray(train_state.params["params"]["Dense_0"]["bias"]),
        "fc2.weight": np.asarray(train_state.params["params"]["Dense_1"]["kernel"]).T,
        "fc2.bias": np.asarray(train_state.params["params"]["Dense_1"]["bias"]),
        "actor.weight": np.asarray(train_state.params["params"]["Dense_2"]["kernel"]).T,
        "actor.bias": np.asarray(train_state.params["params"]["Dense_2"]["bias"]),
        "critic.weight": np.asarray(train_state.params["params"]["Dense_3"]["kernel"]).T.reshape(1, HIDDEN_DIM),
        "critic.bias": np.asarray(train_state.params["params"]["Dense_3"]["bias"]).reshape(1),
    }

    # Post-step parameter parity (per-tensor)
    for name, t_post in torch_post.items():
        j_post = jax_post_torch_layout[name]
        np.testing.assert_allclose(
            j_post,
            t_post,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"post-Adam-step mismatch on {name}",
        )


# ── Tier 2.4 Normalization-scope unit test ────────────────────────────────


def test_advantage_normalization_ddof_audit():
    """Demonstrate the *exact* per-minibatch advantage-normalization mismatch.

    `torch.std()` defaults to `unbiased=True` (ddof=1); `jnp.std` defaults
    to ddof=0 (population). At N=7500 (matched-training minibatch size)
    the bias is sqrt(7500/7499) ≈ 1.0000667. At N=128 it is sqrt(128/127)
    ≈ 1.00394.

    This test demonstrates the bias direction (torch.std > jnp.std) and
    the magnitude. It also verifies that `unbiased=False` brings the two
    into byte-equivalence — the fix applied to `ppo_cleanrl_cyborg.py`.
    """
    rng = np.random.default_rng(0)
    for n in (128, 7500):
        x = rng.standard_normal(n).astype(np.float32)

        torch_default = torch.from_numpy(x).std().item()  # ddof=1
        torch_pop = torch.from_numpy(x).std(unbiased=False).item()  # ddof=0
        jnp_pop = float(jnp.std(jnp.asarray(x)))  # ddof=0

        # Default torch.std differs from jnp.std by the Bessel correction.
        ratio = torch_default / torch_pop
        expected = float(np.sqrt(n / (n - 1)))
        np.testing.assert_allclose(ratio, expected, rtol=1e-5)

        # `unbiased=False` torch.std matches jnp.std exactly.
        np.testing.assert_allclose(torch_pop, jnp_pop, atol=1e-6, rtol=1e-6)
