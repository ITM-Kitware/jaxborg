#!/usr/bin/env bash
# Source before starting Python so separate training/evaluation jobs inherit
# the same persistent cache. Explicit caller settings always take precedence.
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jaxborg/xla}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# The cluster has seen failures writing XLA's optional per-fusion autotune
# disk cache. Ordinary JAX executable caching and GPU autotuning stay enabled.
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}"
