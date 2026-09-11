"""Configuration for final Blue versus one explicitly selected cross-seed Red.

The multi-seed launcher supplies the next seed's final bundle after training
has finished. No checkpoint history or automatic run discovery is involved.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxborg.evaluation.play_priors import _parse_seeds


@dataclass(frozen=True)
class CrossSeedPlaySettings:
    enabled: bool = False
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False
    required: bool = True

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> "CrossSeedPlaySettings":
        raw = (recipe.get("eval") or {}).get("cross_seed_play", False)
        if raw is None or raw is False:
            return cls()
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, Mapping):
            raise ValueError("eval.cross_seed_play must be a boolean or mapping")
        unknown = set(raw) - {"enabled", "seeds", "episodes_per_seed", "deterministic", "required"}
        if unknown:
            raise ValueError(f"eval.cross_seed_play has unknown settings: {sorted(unknown)}")
        flags = {
            name: raw.get(name, default)
            for name, default in (("enabled", True), ("deterministic", False), ("required", True))
        }
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"eval.cross_seed_play.{name} must be a boolean")
        episodes = raw.get("episodes_per_seed", 1)
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ValueError("eval.cross_seed_play.episodes_per_seed must be a positive integer")
        try:
            seeds = _parse_seeds(raw.get("seeds", "1000-1009"))
        except ValueError as exc:
            raise ValueError(str(exc).replace("eval.play_priors", "eval.cross_seed_play")) from exc
        return cls(**flags, seeds=seeds, episodes_per_seed=episodes)


def validate_cross_seed_models(blue_model: str | Path, red_model: str | Path) -> Path:
    """Reject self-play, missing models and ambiguous training provenance."""
    from jaxborg.checkpoint import read_sidecar

    blue, red = (Path(path).expanduser().resolve() for path in (blue_model, red_model))
    for path in (blue, red):
        if not path.is_file():
            raise FileNotFoundError(f"cross-seed-play final model is missing: {path}")
        if not path.name.startswith("model_"):
            raise ValueError(f"cross-seed-play requires a final model bundle, not a periodic checkpoint: {path}")
    if blue == red:
        raise ValueError("cross-seed-play requires different Blue and Red bundles")
    if blue.suffix != red.suffix:
        raise ValueError("cross-seed-play requires Blue and Red to use the same backend")
    seeds = [read_sidecar(path).get("run", {}).get("seed") for path in (blue, red)]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("cross-seed-play requires run.seed in both model recipe sidecars")
    if seeds[0] == seeds[1]:
        raise ValueError("cross-seed-play requires different training seeds")
    return red
