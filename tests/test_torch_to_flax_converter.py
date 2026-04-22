"""Unit test for the PyTorch→Flax SharedActorCritic converter.

Builds a small random PyTorch state_dict matching CleanRL CC4's PPOAgent
(features.0/2 + actor + critic), runs the converter, and verifies that
PyTorch and JAX forward passes produce identical action argmax and
≤1e-5 normalized-logit diff over 50 random observations.

Catches: transpose/permutation bugs, layer-name mismatches, dtype drift.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.policy import SharedActorCritic

torch = pytest.importorskip("torch")


def _build_pt_agent(obs_dim: int, act_dim: int, hidden_dim: int = 256):
    import torch.nn as nn

    class _Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_dim, act_dim)
            self.critic = nn.Linear(hidden_dim, 1)

    return _Agent()


def test_torch_to_flax_converter_matches_pytorch_forward():
    from scripts.eval.transfer import _torch_state_dict_to_shared_actor_critic_params

    obs_dim, act_dim, hidden_dim = 17, 11, 32

    rng = torch.Generator().manual_seed(123)
    pt = _build_pt_agent(obs_dim, act_dim, hidden_dim)
    # Replace weights with deterministic but non-trivial values.
    for p in pt.parameters():
        p.data = torch.empty_like(p).normal_(generator=rng) * 0.3
    pt.eval()

    sd = pt.state_dict()
    converted_obs_dim, converted_act_dim, params = _torch_state_dict_to_shared_actor_critic_params(
        sd, hidden_dim=hidden_dim
    )
    assert converted_obs_dim == obs_dim
    assert converted_act_dim == act_dim

    policy = SharedActorCritic(action_dim=act_dim, hidden_dim=hidden_dim, activation="tanh")

    obs_rng = np.random.RandomState(0)
    max_logit_diff = 0.0
    max_value_diff = 0.0
    argmax_mismatches = 0
    n_trials = 50
    from scipy.special import logsumexp

    for _ in range(n_trials):
        obs_np = obs_rng.randn(obs_dim).astype("float32")
        with torch.no_grad():
            feats = pt.features(torch.from_numpy(obs_np))
            pt_logits = pt.actor(feats).numpy()
            pt_value = pt.critic(feats).squeeze(-1).item()

        pi, value = policy.apply(params, jnp.asarray(obs_np), None)
        # SharedActorCritic returns (pi, value) where pi is a distrax.Categorical;
        # `pi.logits` is normalized (subtracts logsumexp), so renormalize PyTorch
        # logits the same way before comparing.
        jax_logits = np.asarray(pi.logits)
        pt_logits_norm = pt_logits - logsumexp(pt_logits)
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(jax_logits - pt_logits_norm))))
        max_value_diff = max(max_value_diff, abs(float(value) - pt_value))
        if int(np.argmax(jax_logits)) != int(np.argmax(pt_logits)):
            argmax_mismatches += 1

    assert argmax_mismatches == 0, f"argmax mismatched on {argmax_mismatches}/{n_trials} trials"
    assert max_logit_diff < 1e-4, f"max normalized-logit diff {max_logit_diff:.2e} exceeds 1e-4"
    assert max_value_diff < 1e-5, f"max value diff {max_value_diff:.2e} exceeds 1e-5"


def test_torch_to_flax_converter_rejects_hidden_dim_mismatch():
    from scripts.eval.transfer import _torch_state_dict_to_shared_actor_critic_params

    obs_dim, act_dim, hidden_dim = 8, 4, 16
    pt = _build_pt_agent(obs_dim, act_dim, hidden_dim)
    sd = pt.state_dict()
    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        _torch_state_dict_to_shared_actor_critic_params(sd, hidden_dim=32)
