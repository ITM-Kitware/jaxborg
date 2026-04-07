"""Benchmark different NUM_ENVS values to find optimal GPU utilization.

Usage:
    srun --gres=gpu:1 --mem=64G -- uv run python scripts/benchmark_envs.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _default_cache = str(Path.home() / ".cache" / "jaxborg" / "xla")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.fsm_red_env import FsmRedCC4Env


def bench_envs(num_envs, num_steps=20):
    inner_env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents
    action_dim = inner_env.action_space(agents[0]).n

    key = jax.random.PRNGKey(0)
    init_keys = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(inner_env.reset)(init_keys)

    @jax.jit
    def batched_step(keys, state, act_dict):
        return jax.vmap(inner_env.step)(keys, state, act_dict)

    rng = jax.random.PRNGKey(42)
    random_actions = {agents[i]: jax.random.randint(jax.random.PRNGKey(i), (num_envs,), 0, action_dim) for i in range(num_agents)}

    # Compile
    rng, _rng = jax.random.split(rng)
    step_keys = jax.random.split(_rng, num_envs)
    t0 = time.perf_counter()
    result = batched_step(step_keys, env_state, random_actions)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0
    obs, env_state = result[:2]

    # Warmup
    for _ in range(3):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        result = batched_step(step_keys, env_state, random_actions)
        obs, env_state = result[:2]
        jax.block_until_ready(result)

    # Bench
    times = []
    for _ in range(num_steps):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        t0 = time.perf_counter()
        result = batched_step(step_keys, env_state, random_actions)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
        obs, env_state = result[:2]

    times = np.array(times)
    sps = num_envs / times
    return compile_time, times.mean(), sps.mean()


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"\n{'NUM_ENVS':>10} {'Compile':>10} {'Step (ms)':>10} {'SPS':>10} {'SPS/env':>10}")
    print("-" * 55)

    for num_envs in [256, 512, 1024, 2048, 4096]:
        try:
            compile_time, step_time, sps = bench_envs(num_envs)
            print(f"{num_envs:>10} {compile_time:>10.1f}s {step_time*1000:>10.1f} {sps:>10,.0f} {sps/num_envs:>10.2f}")
        except Exception as e:
            print(f"{num_envs:>10} FAILED: {e}")
            break


if __name__ == "__main__":
    main()
