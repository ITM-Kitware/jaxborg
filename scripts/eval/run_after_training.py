#!/usr/bin/env python3
"""CLI wrapper for an ordered recipe-configured evaluation pipeline."""


def _main():
    import jax

    # Recipe validation imports simulator constants that initialize JAX. The
    # coordinator stays alive while its children run, so keep its allocations
    # off their GPU. Leave JAX_PLATFORMS in the environment for child selection.
    jax.config.update("jax_platforms", "cpu")

    from jaxborg.evaluation.post_training import main

    main()


if __name__ == "__main__":
    _main()
