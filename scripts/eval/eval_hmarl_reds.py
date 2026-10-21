#!/usr/bin/env python3
"""Evaluate Blue against Default, Aggressive, Stealthy and Impact Red in JAX."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

if __name__ == "__main__":
    from jaxborg.evaluation.hmarl_reds import main

    main()
