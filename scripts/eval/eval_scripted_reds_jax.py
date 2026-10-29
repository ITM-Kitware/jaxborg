#!/usr/bin/env python3
"""CLI wrapper for fixed-topology scripted-Red evaluation in JAX."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _main() -> None:
    from jaxborg.evaluation.jax_scripted_red import main

    main()


if __name__ == "__main__":
    _main()
