"""Optional binary message head for the native CybORG feedforward policies."""

import torch
from torch import nn


class TorchMessages:
    def init_message_head(self, hidden_dim, message_dim):
        self.message_dim = message_dim
        if message_dim:
            self.message_head = nn.Linear(hidden_dim, message_dim)
            nn.init.orthogonal_(self.message_head.weight, gain=0.01)
            nn.init.constant_(self.message_head.bias, 0.0)

    def get_message_and_stats(self, obs, message=None, *, deterministic=False):
        if not self.message_dim:
            return None, None, None
        features = self.features(obs) if self.arch_name == "shared" else self.actor_features(obs)
        logits = self.message_head(features)
        distribution = torch.distributions.Bernoulli(logits=logits)
        if message is None:
            message = (logits > 0).float() if deterministic else distribution.sample()
        return message, distribution.log_prob(message).sum(-1), distribution.entropy().sum(-1)


class TorchMessageOverride(nn.Module):
    """Evaluation-only mute or foreign-sender control; actors keep their weights."""

    message_dim = 8

    def __init__(self, actor, sender=None, mute=False):
        super().__init__()
        self.actor_policy = actor
        self.sender_policy = sender
        self.mute = mute

    def deterministic_action(self, obs, mask):
        return self.actor_policy.deterministic_action(obs, mask)

    def get_action_and_value(self, obs, mask, action=None):
        return self.actor_policy.get_action_and_value(obs, mask, action)

    def get_message_and_stats(self, obs, message=None, *, deterministic=False):
        if self.mute:
            return torch.zeros(obs.shape[:-1] + (8,)), torch.zeros(obs.shape[:-1]), torch.zeros(obs.shape[:-1])
        sender = self.actor_policy if self.sender_policy is None else self.sender_policy
        return sender.get_message_and_stats(obs, message, deterministic=deterministic)
