#!/usr/bin/env python3
"""CLI wrapper for the trained-Blue vs scripted-Red evaluation sweep."""

from __future__ import annotations

import os

# CybORG rollout is CPU work, and one-at-a-time JAX policy inference is better
# kept off the training GPU.  Users can override this environment variable
# explicitly when invoking the module entry point instead of this wrapper.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _main() -> None:
    from jaxborg.evaluation.scripted_red import main

    main()


if __name__ == "__main__":
    _main()
