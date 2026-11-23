"""Validated episode-level opponent populations for JAX cotraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

OPPONENT_NAMES = ("cotrained", "fsm", "cia_c", "cia_i", "cia_a")


@dataclass(frozen=True)
class OpponentPopulationSettings:
    percentages: tuple[float, ...]
    preserve_red_batch_size: bool = False

    @property
    def mixed(self) -> bool:
        return any(value > 0 for value in self.percentages[1:])

    def sample(self, key):
        probabilities = jnp.asarray(self.percentages, dtype=jnp.float32) / 100.0
        return jax.random.categorical(key, jnp.log(probabilities)).astype(jnp.int32)

    @classmethod
    def from_config(cls, raw) -> OpponentPopulationSettings | None:
        if raw is None:
            return None
        prefix = "train.opponent_population"
        if not isinstance(raw, dict):
            raise ValueError(f"{prefix} must be a mapping")
        unknown = set(raw) - {"enabled", "preserve_red_batch_size", "blue"}
        if unknown:
            raise ValueError(f"{prefix} has unknown settings (only Blue populations are supported): {sorted(unknown)}")
        for name in ("enabled", "preserve_red_batch_size"):
            if name in raw and not isinstance(raw[name], bool):
                raise ValueError(f"{prefix}.{name} must be a boolean")
        enabled = raw.get("enabled", True)
        if not enabled and "blue" not in raw:
            return None
        entries = raw.get("blue")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{prefix}.blue must be a nonempty list")
        percentages = dict.fromkeys(OPPONENT_NAMES, 0.0)
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"opponent", "percentage"}:
                raise ValueError(f"{prefix}.blue entries must contain only opponent and percentage")
            opponent, percentage = entry["opponent"], entry["percentage"]
            if not isinstance(opponent, str) or opponent not in percentages:
                raise ValueError(f"{prefix}: unknown opponent {opponent!r}; expected one of {OPPONENT_NAMES}")
            if opponent in seen:
                raise ValueError(f"{prefix}: duplicate opponent {opponent!r}")
            seen.add(opponent)
            if (
                isinstance(percentage, bool)
                or not isinstance(percentage, (float, int))
                or not math.isfinite(percentage)
                or percentage < 0
            ):
                raise ValueError(f"{prefix}: percentages must be finite nonnegative numbers")
            percentages[opponent] = float(percentage)
        if not math.isclose(sum(percentages.values()), 100.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{prefix}: percentages must total 100")
        return cls(tuple(percentages.values()), raw.get("preserve_red_batch_size", False)) if enabled else None
