"""Runtime setup shared by dev parity tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_runtime() -> Path:
    """Set import path and JAX runtime defaults before JAX is imported."""
    if os.environ.get("_JAXBORG_CYBORG_WORKER"):
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    return repo_root


ROOT = configure_runtime()
EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
DEFAULT_NUM_STEPS = 500
