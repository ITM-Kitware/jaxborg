"""Blue MAPPO / Red IPPO launcher using the shared JAX cotraining runtime.

Example:
    uv run python scripts/train/algorithms/mappo_jax.py --recipe cotraining_mappo --seed 42
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train.algorithms.ippo_jax import main  # noqa: E402

if __name__ == "__main__":
    main(expected_algorithm="mappo")
