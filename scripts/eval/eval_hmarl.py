#!/usr/bin/env python3
"""Evaluate pretrained H-MARL in JAX using an evaluation-only recipe."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

if __name__ == "__main__":
    from jaxborg.pretrained.hmarl_eval import main

    main()
