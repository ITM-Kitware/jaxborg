"""PPOAgent network for CleanRL-style CybORG training.

Shared between `scripts/train/ppo_cleanrl_cyborg.py` (trains this arch) and
`scripts/eval/ppo_cleanrl_cyborg_eval.py` (loads its state_dict). Keeping a
single definition prevents the eval silently loading weights into a stale
architecture.
"""

import numpy as np
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    """Shared-trunk actor-critic with action masking (CleanRL-style)."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, act_dim)
        self.critic = nn.Linear(in_dim, 1)

        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def get_value(self, obs):
        features = self.features(obs)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(self, obs, action_mask, action=None):
        features = self.features(obs)
        logits = self.actor(features)
        logits = logits + (action_mask.float() - 1.0) * 1e8
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value
