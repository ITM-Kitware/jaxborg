"""Evaluation-only controls for muting Blue or substituting a message sender."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class MessageOverridePolicy:
    actor: object
    sender: object | None = None
    mute: bool = False
    message_dim: int = 8

    def initialize_carry(self, batch_size):
        from . import initial_carry

        return {
            "actor": initial_carry(self.actor, batch_size),
            "sender": initial_carry(self.sender, batch_size) if self.sender is not None else None,
        }

    def step(self, params, obs, avail_actions, *, carry, reset):
        from . import policy_step

        pi, value, actor_carry = policy_step(
            self.actor, params["actor"], obs, avail_actions, carry=carry["actor"], reset=reset
        )
        sender_carry = carry["sender"]
        if self.sender is not None:
            sender_pi, _, sender_carry = policy_step(
                self.sender, params["sender"], obs, avail_actions, carry=sender_carry, reset=reset
            )
            pi = pi.replace(message_logits=sender_pi.message_logits)
        if self.mute:
            pi = pi.replace(message_logits=jnp.full(obs.shape[:-1] + (8,), -1e9))
        return pi, value, {"actor": actor_carry, "sender": sender_carry}
