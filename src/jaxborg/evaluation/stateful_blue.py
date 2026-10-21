"""Evaluation-only adapter for external policies with observation memory."""

from abc import ABC, abstractmethod


class StatefulBluePolicy(ABC):
    @abstractmethod
    def initialize(self, env_state):
        """Return an environment state and fresh, per-episode policy memory."""

    @abstractmethod
    def select_actions(self, weights, env_state, key, carry, *, deterministic):
        """Return native Blue actions and updated memory using observable data."""
