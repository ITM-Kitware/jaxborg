"""Blue input layouts. The public enhanced-observation flag selects the latest layout."""

from jaxborg.constants import BLUE_OBS_SIZE

BLUE_HOST_SLOTS = 48
LEGACY_ENHANCED_OBS_SIZE = BLUE_OBS_SIZE + 4 * BLUE_HOST_SLOTS
CAGE4_ENHANCED_OBS_SIZE = BLUE_OBS_SIZE + 5 * BLUE_HOST_SLOTS
BLUE_OBSERVATION_VERSION = 2


def enhanced_obs_enabled(recipe: dict) -> bool:
    enabled = recipe.get("cage4_enhanced_obs", False)
    if not isinstance(enabled, bool):
        raise ValueError("cage4_enhanced_obs must be a YAML boolean")
    return enabled


def enhanced_obs_version(recipe: dict) -> int:
    """New recipes use v2; saved pre-versioning runs retain the v1 contract."""
    if not enhanced_obs_enabled(recipe):
        return BLUE_OBSERVATION_VERSION
    run = recipe.get("run")
    version = run.get("blue_observation_version", 1) if run is not None else BLUE_OBSERVATION_VERSION
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError("run.blue_observation_version must be 1 or 2")
    return version


def blue_obs_size(enhanced: bool = False, version: int = BLUE_OBSERVATION_VERSION) -> int:
    if version not in (1, 2):
        raise ValueError("Blue observation version must be 1 or 2")
    if not enhanced:
        return BLUE_OBS_SIZE
    return LEGACY_ENHANCED_OBS_SIZE if version == 1 else CAGE4_ENHANCED_OBS_SIZE


def recipe_blue_obs_size(recipe: dict) -> int:
    return blue_obs_size(enhanced_obs_enabled(recipe), enhanced_obs_version(recipe))


def observation_version_from_size(size: int) -> int:
    if size == LEGACY_ENHANCED_OBS_SIZE:
        return 1
    if size in (0, BLUE_OBS_SIZE, CAGE4_ENHANCED_OBS_SIZE):
        return BLUE_OBSERVATION_VERSION
    raise ValueError(f"Unsupported Blue observation size: {size}")
