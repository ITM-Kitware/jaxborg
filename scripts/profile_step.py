"""Profile environment step and sub-components.

Usage:
    srun --gres=gpu:1 --mem=64G -- uv run python scripts/profile_step.py
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

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxmarl.wrappers.baselines import LogWrapper


def profile():
    num_envs = 1024
    print(f"JAX devices: {jax.devices()}")

    inner_env = FsmRedCC4Env(num_steps=500, topology_mode="pure")
    env = LogWrapper(inner_env)

    # Init
    key = jax.random.PRNGKey(0)
    init_keys = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(env.reset)(init_keys)

    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents

    # Prepare random actions
    action_dim = inner_env.action_space(agents[0]).n
    rng = jax.random.PRNGKey(42)

    # JIT compile the step function
    @jax.jit
    def batched_step(keys, state, act_dict):
        return jax.vmap(env.step)(keys, state, act_dict)

    # Warmup: compile
    rng, _rng = jax.random.split(rng)
    step_keys = jax.random.split(_rng, num_envs)
    random_actions = {agents[i]: jax.random.randint(jax.random.PRNGKey(i), (num_envs,), 0, action_dim) for i in range(num_agents)}

    print("Compiling step function...")
    t0 = time.perf_counter()
    obs, env_state, rewards, dones, info = batched_step(step_keys, env_state, random_actions)
    jax.block_until_ready(obs)
    t_compile = time.perf_counter() - t0
    print(f"  Compile + first step: {t_compile:.2f}s")

    # Now profile steady-state
    num_profile_steps = 100
    step_times = []

    for i in range(num_profile_steps):
        rng, _rng = jax.random.split(rng)
        step_keys = jax.random.split(_rng, num_envs)
        random_actions = {agents[j]: jax.random.randint(jax.random.fold_in(rng, j), (num_envs,), 0, action_dim) for j in range(num_agents)}

        t0 = time.perf_counter()
        obs, env_state, rewards, dones, info = batched_step(step_keys, env_state, random_actions)
        jax.block_until_ready(obs)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    step_times = np.array(step_times)
    sps = num_envs / step_times

    print(f"\nSteady-state ({num_profile_steps} steps, {num_envs} envs):")
    print(f"  Per-step time: {step_times.mean()*1000:.1f}ms +/- {step_times.std()*1000:.1f}ms")
    print(f"  Steps/sec: {sps.mean():,.0f} +/- {sps.std():,.0f}")
    print(f"  Per-env per-step: {step_times.mean()*1000/num_envs:.4f}ms")

    # Now profile sub-components by timing individual JITed functions
    print("\n--- Sub-component profiling ---")

    from jaxborg.observations import get_blue_obs, get_red_obs
    from jaxborg.actions.masking import compute_blue_action_mask
    from jaxborg.rewards import compute_reward_breakdown

    # Get a single env's state for profiling
    single_state = jax.tree.map(lambda x: x[0], env_state.env_state.state)
    single_const = jax.tree.map(lambda x: x[0], env_state.env_state.const)

    # Blue obs
    blue_obs_fn = jax.jit(lambda: jnp.stack([get_blue_obs(single_state, single_const, b) for b in range(NUM_BLUE_AGENTS)]))
    blue_obs_fn()  # compile
    jax.block_until_ready(blue_obs_fn())
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        jax.block_until_ready(blue_obs_fn())
        times.append(time.perf_counter() - t0)
    print(f"  Blue obs (5 agents): {np.mean(times)*1000:.2f}ms")

    # Blue obs vmapped
    vmap_blue_obs = jax.jit(jax.vmap(lambda s, c: jnp.stack([get_blue_obs(s, c, b) for b in range(NUM_BLUE_AGENTS)])))
    batch_state = env_state.env_state.state
    batch_const = env_state.env_state.const
    vmap_blue_obs(batch_state, batch_const)  # compile
    jax.block_until_ready(vmap_blue_obs(batch_state, batch_const))
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(vmap_blue_obs(batch_state, batch_const))
        times.append(time.perf_counter() - t0)
    print(f"  Blue obs vmapped ({num_envs} envs): {np.mean(times)*1000:.1f}ms")

    # Action mask
    vmap_mask = jax.jit(jax.vmap(
        jax.vmap(compute_blue_action_mask, in_axes=(None, 0, None)),
        in_axes=(0, None, 0)
    ))
    agent_ids = jnp.arange(NUM_BLUE_AGENTS)
    vmap_mask(batch_const, agent_ids, batch_state)  # compile
    jax.block_until_ready(vmap_mask(batch_const, agent_ids, batch_state))
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(vmap_mask(batch_const, agent_ids, batch_state))
        times.append(time.perf_counter() - t0)
    print(f"  Action mask ({num_envs} envs, {NUM_BLUE_AGENTS} agents): {np.mean(times)*1000:.1f}ms")


if __name__ == "__main__":
    profile()
