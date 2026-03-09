"""Benchmark env step throughput at various NUM_ENVS.

Usage: CUDA_VISIBLE_DEVICES=2,3 uv run python scripts/benchmark_throughput.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

import time

import jax

from jaxborg.fsm_red_env import FsmRedCC4Env

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")


def benchmark_env_steps(num_envs: int, num_steps: int = 100, warmup_steps: int = 5):
    env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    agents = list(env.agents)

    init_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    init_obs, init_state = jax.vmap(env.reset)(init_keys)

    action_space_n = env.action_space(agents[0]).n

    @jax.jit
    def do_step(state, obs, rng):
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        actions = {}
        for i, a in enumerate(agents):
            actions[a] = jax.random.randint(
                jax.random.fold_in(act_rng, i), (num_envs,), 0, action_space_n
            )
        step_keys = jax.random.split(step_rng, num_envs)
        obs, state, rewards, dones, info = jax.vmap(env.step)(step_keys, state, actions)
        return state, obs, rng

    rng = jax.random.PRNGKey(42)
    state, obs = init_state, init_obs

    t0 = time.perf_counter()
    state, obs, rng = do_step(state, obs, rng)
    jax.block_until_ready(state)
    compile_time = time.perf_counter() - t0
    print(f"  Compile: {compile_time:.1f}s")

    for _ in range(warmup_steps):
        state, obs, rng = do_step(state, obs, rng)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for _ in range(num_steps):
        state, obs, rng = do_step(state, obs, rng)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    total_env_steps = num_steps * num_envs
    sps = total_env_steps / elapsed
    per_step_ms = elapsed / num_steps * 1000
    print(
        f"  {num_steps} iters x {num_envs} envs = {total_env_steps} steps"
        f" in {elapsed:.2f}s = {sps:.0f} sps ({per_step_ms:.1f} ms/iter)"
    )
    return sps, compile_time


def benchmark_scan_rollout(num_envs: int, scan_length: int = 100):
    env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    agents = list(env.agents)
    action_space_n = env.action_space(agents[0]).n

    init_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    init_obs, init_state = jax.vmap(env.reset)(init_keys)

    @jax.jit
    def scan_rollout(state, obs, rng):
        def _step(carry, _):
            state, obs, rng = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            actions = {}
            for i, a in enumerate(agents):
                actions[a] = jax.random.randint(
                    jax.random.fold_in(act_rng, i), (num_envs,), 0, action_space_n
                )
            step_keys = jax.random.split(step_rng, num_envs)
            obs, state, rewards, dones, info = jax.vmap(env.step)(step_keys, state, actions)
            return (state, obs, rng), rewards[agents[0]]

        (state, obs, rng), _ = jax.lax.scan(_step, (state, obs, rng), None, scan_length)
        return state, obs, rng

    rng = jax.random.PRNGKey(42)

    t0 = time.perf_counter()
    state, obs, rng = scan_rollout(init_state, init_obs, rng)
    jax.block_until_ready(state)
    compile_time = time.perf_counter() - t0
    print(f"  Scan compile ({scan_length} steps): {compile_time:.1f}s")

    t0 = time.perf_counter()
    state, obs, rng = scan_rollout(state, obs, rng)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    total = scan_length * num_envs
    sps = total / elapsed
    print(f"  Scan rollout: {total} steps in {elapsed:.2f}s = {sps:.0f} sps")
    return sps, compile_time


if __name__ == "__main__":
    print("\n=== Single-step benchmark (varying NUM_ENVS) ===")
    for ne in [1, 4, 16, 64]:
        print(f"\nNUM_ENVS={ne}:")
        benchmark_env_steps(ne, num_steps=50)

    print("\n=== Scan rollout benchmark ===")
    for ne in [4, 16, 64]:
        print(f"\nNUM_ENVS={ne}, scan_length=100:")
        benchmark_scan_rollout(ne, scan_length=100)
