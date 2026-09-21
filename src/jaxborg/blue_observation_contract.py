"""Versioned Blue policy input sizes; no simulator imports required."""

from jaxborg.constants import BLUE_OBS_SIZE

# Three observed subnets, each with the stock 16 server/user slots.
BLUE_HOST_SLOTS = 48
CAGE4_ENHANCED_OBS_SIZE = BLUE_OBS_SIZE + 4 * BLUE_HOST_SLOTS


def enhanced_obs_enabled(recipe: dict) -> bool:
    enabled = recipe.get("cage4_enhanced_obs", False)
    if not isinstance(enabled, bool):
        raise ValueError("cage4_enhanced_obs must be a YAML boolean")
    return enabled


def blue_obs_size(enhanced: bool = False) -> int:
    return CAGE4_ENHANCED_OBS_SIZE if enhanced else BLUE_OBS_SIZE
