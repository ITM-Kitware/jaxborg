"""Recipe configuration for the default and CIA-shaped team payoff."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    name: str = "default"
    scale: float = 1.0
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if self.name not in ("default", "shaping"):
            raise ValueError("train.reward.name must be 'default' or 'shaping'")
        if isinstance(self.scale, bool) or not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("train.reward.lambda must be finite and non-negative")
        if len(self.weights) != 3 or any(isinstance(w, bool) or not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("train.reward.weights must contain finite non-negative C, I, A weights")

    @classmethod
    def from_recipe(cls, recipe):
        raw = recipe.get("train", {}).get("reward", {})
        if not isinstance(raw, Mapping):
            raise ValueError("train.reward must be a mapping")
        unknown = set(raw) - {"name", "lambda", "weights"}
        if unknown:
            raise ValueError(f"unknown train.reward keys: {sorted(unknown)}")
        weights = raw.get("weights", {})
        if not isinstance(weights, Mapping) or set(weights) - {"C", "I", "A"}:
            raise ValueError("train.reward.weights must be a mapping with C, I, A keys")
        try:
            scale = raw.get("lambda", 1.0)
            values = tuple(weights.get(k, 1.0) for k in ("C", "I", "A"))
            if isinstance(scale, bool) or any(isinstance(v, bool) for v in values):
                raise ValueError("reward parameters must be numbers, not booleans")
            return cls(raw.get("name", "default"), float(scale), tuple(float(v) for v in values))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid train.reward: {exc}") from exc

    def as_dict(self):
        return {"name": self.name, "lambda": self.scale, "weights": dict(zip(("C", "I", "A"), self.weights))}
