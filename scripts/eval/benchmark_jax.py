"""Benchmark XLA compile time and steady-state training throughput.

Usage:
    srun --gres=gpu:1 --mem=64G -- uv run python scripts/eval/benchmark_jax.py [--num-updates N] [--clear-cache]
"""

import os
import shutil
import sys
import time
from pathlib import Path

# Must be set BEFORE importing JAX.
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _default_cache = str(Path.home() / ".cache" / "jaxborg" / "xla")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse

import jax
import numpy as np


def run_benchmark(num_updates: int = 5, clear_cache: bool = False):
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if clear_cache and cache_dir and Path(cache_dir).exists():
        print(f"Clearing XLA cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"XLA cache dir: {cache_dir}")

    # Import after JAX env vars are set

    # --- Config (mirrors recipes/default.yaml; this script is a standalone benchmark, not a recipe consumer) ---
    config = {
        "NUM_ENVS": 1024,
        "NUM_STEPS": 500,
        "TOTAL_TIMESTEPS": num_updates * 500 * 1024,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "CLIP_VALUE_LOSS": False,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTOR_MAX_GRAD_NORM": 0.5,
        "CRITIC_MAX_GRAD_NORM": 50.0,
        "ACTIVATION": "tanh",
        "HIDDEN_DIM": 256,
        "HIDDEN_LAYERS": 2,
        "ANNEAL_LR": False,
        "LR": 3e-4,
        "NORM_REWARDS": True,
        "REWARD_SCALE": 1.0,
        "SEED": 0,
        "TRAINING_MODE": True,
    }

    # Use make_train from the training script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "algorithms"))
    import jax.numpy as jnp
    from ippo_jax import RewardNormState, make_train

    from jaxborg.parity.fsm_red_env import FsmRedCC4Env
    from jaxborg.policies import make_jax_policy

    print(f"\nConfig: NUM_ENVS={config['NUM_ENVS']}, NUM_STEPS={config['NUM_STEPS']}")
    print(f"Benchmark: {num_updates} updates\n")

    # --- Setup ---
    t_setup_start = time.perf_counter()
    inner_env = FsmRedCC4Env(num_steps=500)
    action_dim = inner_env.action_space(inner_env.agents[0]).n
    network = make_jax_policy(
        "shared",
        action_dim=action_dim,
        hidden_dim=config["HIDDEN_DIM"],
        hidden_layers=config["HIDDEN_LAYERS"],
        activation=config["ACTIVATION"],
    )
    env, init_obs, init_env_state, init_train_state, collect_and_update = make_train(config, network)
    t_setup = time.perf_counter() - t_setup_start
    print(f"Setup (env+network build): {t_setup:.2f}s")

    rng = jax.random.PRNGKey(1)
    rng, _rng = jax.random.split(rng)
    train_state = init_train_state(_rng)

    env_state = init_env_state
    obs = init_obs
    num_envs = config["NUM_ENVS"]
    reward_norm_state = RewardNormState(
        returns=jnp.zeros(num_envs, dtype=jnp.float32),
        mean=jnp.zeros((), dtype=jnp.float32),
        var=jnp.ones((), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )

    # --- Run updates ---
    update_times = []
    for i in range(num_updates):
        t0 = time.perf_counter()
        train_state, env_state, obs, rng, reward_norm_state, metric = collect_and_update(
            train_state, env_state, obs, rng, reward_norm_state
        )
        # Block until computation is done
        jax.block_until_ready(metric)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        update_times.append(elapsed)

        steps = config["NUM_ENVS"] * config["NUM_STEPS"]
        sps = steps / elapsed
        label = "COMPILE+RUN" if i == 0 else f"update {i}"
        print(f"  [{label}] {elapsed:.2f}s  ({sps:,.0f} steps/sec)")

    # --- Summary ---
    compile_time = update_times[0]
    if len(update_times) > 1:
        steady_times = update_times[1:]
        steady_mean = np.mean(steady_times)
        steady_std = np.std(steady_times)
        steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
        steady_sps = steps_per_update / steady_mean
    else:
        steady_mean = compile_time
        steady_std = 0.0
        steady_sps = config["NUM_ENVS"] * config["NUM_STEPS"] / compile_time

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  XLA compile + first update: {compile_time:.2f}s")
    if len(update_times) > 1:
        print(f"  Steady-state update time:   {steady_mean:.2f}s (+/- {steady_std:.2f}s)")
        print(f"  Steady-state steps/sec:     {steady_sps:,.0f}")
    print(f"  Steps per update:           {config['NUM_ENVS'] * config['NUM_STEPS']:,}")
    print("=" * 60)

    return {
        "compile_time": compile_time,
        "steady_mean": steady_mean,
        "steady_std": steady_std,
        "steady_sps": steady_sps,
        "setup_time": t_setup,
        "update_times": update_times,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-updates", type=int, default=5)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()
    run_benchmark(num_updates=args.num_updates, clear_cache=args.clear_cache)
