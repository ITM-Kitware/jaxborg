"""CC4-aware CIA alignment metric — trajectory recording + post-hoc scoring."""

from jaxborg.eval.cia.cc4_cia_metric import (
    CIA_COMPOSITE_WEIGHT,
    CIAEpisodeScore,
    cc4_host_weight,
    score_episode,
    score_trajectory_file,
)
from jaxborg.eval.cia.jax_native import jax_host_weights, score_jax_episode

__all__ = [
    "CIA_COMPOSITE_WEIGHT",
    "CIAEpisodeScore",
    "cc4_host_weight",
    "jax_host_weights",
    "score_episode",
    "score_jax_episode",
    "score_trajectory_file",
]
