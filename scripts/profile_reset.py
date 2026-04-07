"""Profile auto-reset cost vs step_env cost.

Usage:
    srun --gres=gpu:1 --mem=64G -- uv run python scripts/profile_reset.py
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


def profile():
    num_envs = 1024
    print(f"JAX devices: {jax.devices()}")

    inner_env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents
    action_dim = inner_env.action_space(agents[0]).n

    key = jax.random.PRNGKey(0)
    init_keys = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(inner_env.reset)(init_keys)

    # Profile step_env (no auto-reset)
    @jax.jit
    def batched_step_env(keys, state, act_dict):
        return jax.vmap(inner_env.step_env)(keys, state, act_dict)

    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    step_keys = jax.random.split(_rng, num_envs)
    random_actions = {agents[i]: jax.random.randint(jax.random.PRNGKey(i), (num_envs,), 0, action_dim) for i in range(num_agents)}

    print("Compiling step_env (no auto-reset)...")
    t0 = time.perf_counter()
    result = batched_step_env(step_keys, env_state, random_actions)
    jax.block_until_ready(result)
    print(f"  Compile + first step_env: {time.perf_counter() - t0:.2f}s")

    # Warmup
    for _ in range(3):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        result = batched_step_env(step_keys, env_state, random_actions)
        jax.block_until_ready(result)

    # Time step_env
    times = []
    for _ in range(20):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        t0 = time.perf_counter()
        result = batched_step_env(step_keys, env_state, random_actions)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    step_env_time = np.mean(times)
    print(f"  step_env: {step_env_time*1000:.1f}ms ({num_envs} envs)")

    # Profile full step (with auto-reset)
    from jaxmarl.wrappers.baselines import LogWrapper
    env = LogWrapper(inner_env)
    obs2, env_state2 = jax.vmap(env.reset)(init_keys)

    @jax.jit
    def batched_step(keys, state, act_dict):
        return jax.vmap(env.step)(keys, state, act_dict)

    print("\nCompiling full step (with auto-reset)...")
    t0 = time.perf_counter()
    result2 = batched_step(step_keys, env_state2, random_actions)
    jax.block_until_ready(result2)
    print(f"  Compile + first step: {time.perf_counter() - t0:.2f}s")

    for _ in range(3):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        result2 = batched_step(step_keys, env_state2, random_actions)
        jax.block_until_ready(result2)

    times = []
    for _ in range(20):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        t0 = time.perf_counter()
        result2 = batched_step(step_keys, env_state2, random_actions)
        jax.block_until_ready(result2)
        times.append(time.perf_counter() - t0)
    step_time = np.mean(times)
    print(f"  full step: {step_time*1000:.1f}ms ({num_envs} envs)")

    # Profile _reset_state only
    @jax.jit
    def batched_reset(state, keys):
        return jax.vmap(inner_env._env._reset_state)(state.env_state, keys)

    print("\nCompiling _reset_state...")
    reset_keys = jax.random.split(jax.random.PRNGKey(99), num_envs)
    t0 = time.perf_counter()
    reset_result = batched_reset(env_state2, reset_keys)
    jax.block_until_ready(reset_result)
    print(f"  Compile + first reset: {time.perf_counter() - t0:.2f}s")

    for _ in range(3):
        reset_result = batched_reset(env_state2, reset_keys)
        jax.block_until_ready(reset_result)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        reset_result = batched_reset(env_state2, reset_keys)
        jax.block_until_ready(reset_result)
        times.append(time.perf_counter() - t0)
    reset_time = np.mean(times)
    print(f"  reset_state: {reset_time*1000:.1f}ms ({num_envs} envs)")

    print(f"\n{'='*60}")
    print(f"PROFILING SUMMARY ({num_envs} envs)")
    print(f"{'='*60}")
    print(f"  step_env (no reset): {step_env_time*1000:.1f}ms")
    print(f"  full step (w/ reset): {step_time*1000:.1f}ms")
    print(f"  reset_state alone:    {reset_time*1000:.1f}ms")
    print(f"  reset overhead:       {(step_time - step_env_time)*1000:.1f}ms ({(step_time - step_env_time)/step_time*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    profile()
