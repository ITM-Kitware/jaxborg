"""Quick benchmark: test multiple NUM_ENVS without recompiling each.

Usage: CUDA_VISIBLE_DEVICES=2,3 uv run python scripts/bench_quick.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

import sys
import time

import jax

from jaxborg.fsm_red_env import FsmRedCC4Env

NUM_ITERS = 20
WARMUP = 3

print(f"JAX devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

num_envs_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 32, 128, 256]

for NUM_ENVS in num_envs_list:
    env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    agents = list(env.agents)
    action_n = env.action_space(agents[0]).n

    init_keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    init_obs, init_state = jax.vmap(env.reset)(init_keys)

    @jax.jit
    def do_step(state, rng):
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        actions = {
            a: jax.random.randint(jax.random.fold_in(act_rng, i), (NUM_ENVS,), 0, action_n)
            for i, a in enumerate(agents)
        }
        step_keys = jax.random.split(step_rng, NUM_ENVS)
        obs, state, rewards, dones, info = jax.vmap(env.step)(step_keys, state, actions)
        return state, rng

    rng = jax.random.PRNGKey(42)

    print(f"\nNUM_ENVS={NUM_ENVS}, compiling...", flush=True)
    t0 = time.perf_counter()
    state, rng = do_step(init_state, rng)
    jax.block_until_ready(state)
    print(f"  Compile: {time.perf_counter() - t0:.1f}s", flush=True)

    for _ in range(WARMUP):
        state, rng = do_step(state, rng)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        state, rng = do_step(state, rng)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    total = NUM_ITERS * NUM_ENVS
    ms_per_iter = elapsed / NUM_ITERS * 1000
    sps = total / elapsed
    print(f"  {NUM_ITERS} iters: {elapsed:.2f}s | {ms_per_iter:.1f} ms/iter | {sps:.0f} env-steps/sec")
