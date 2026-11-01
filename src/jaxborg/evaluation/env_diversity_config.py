"""Recipe settings for comparisons scheduled after all training seeds finish."""

from collections.abc import Mapping
from dataclasses import dataclass

from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings


@dataclass(frozen=True)
class EnvDiversitySettings:
    enabled: bool = False
    baseline_recipe: str | None = None
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False

    @classmethod
    def from_recipe(cls, recipe):
        raw = (recipe.get("eval") or {}).get("env_diversity", False)
        if raw is None or raw is False:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("eval.env_diversity must be a mapping")
        unknown = set(raw) - {"enabled", "baseline_recipe", "seeds", "episodes_per_seed", "deterministic"}
        if unknown:
            raise ValueError(f"eval.env_diversity has unknown settings: {sorted(unknown)}")
        shared = {k: v for k, v in raw.items() if k != "baseline_recipe"}
        try:
            settings = CrossSeedPlaySettings.from_recipe({"eval": {"cross_seed_play": shared}})
        except ValueError as exc:
            raise ValueError(str(exc).replace("eval.cross_seed_play", "eval.env_diversity")) from exc
        baseline = raw.get("baseline_recipe")
        if settings.enabled and (not isinstance(baseline, str) or not baseline.strip()):
            raise ValueError("eval.env_diversity.baseline_recipe is required when enabled")
        if settings.enabled and baseline == recipe.get("meta", {}).get("name"):
            raise ValueError("eval.env_diversity.baseline_recipe must name a different recipe")
        return cls(settings.enabled, baseline, settings.seeds, settings.episodes_per_seed, settings.deterministic)
