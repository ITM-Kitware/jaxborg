"""CC4-aware CIA alignment metric — trajectory recording + post-hoc scoring."""

from jaxborg.evaluation.cia.cc4_cia_metric import (
    CIA_COMPOSITE_WEIGHT,
    CIAEpisodeScore,
    cc4_host_weight,
    score_episode,
    score_trajectory_file,
)

__all__ = [
    "CIA_COMPOSITE_WEIGHT",
    "CIAEpisodeScore",
    "cc4_host_weight",
    "score_episode",
    "score_trajectory_file",
]
