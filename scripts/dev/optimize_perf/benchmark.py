"""Dev-only JAXborg performance benchmark for agentic optimization loops.

Measures cold first-update time and warm steady-state throughput for the
recipe-driven JAX IPPO training path. The XLA cache is cleared on every run so
compile-time measurements are comparable. The output includes a `PERF_JSON:`
line so scripts can compare candidates without parsing human-readable text.

GPU usage should go through Slurm, for example:

    srun --gres=gpu:1 --mem=64G -- uv run python scripts/dev/optimize_perf/benchmark.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALGO_DIR = _REPO_ROOT / "scripts" / "train" / "algorithms"

# Must be set before importing JAX.
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(_REPO_ROOT / ".agent_handoff" / "optimize_perf" / "xla_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import numpy as np

for _path in (_REPO_ROOT / "src", _ALGO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _build_network(recipe: dict, config: dict):
    from jaxborg.parity.fsm_red_env import FsmRedCC4Env
    from jaxborg.policies import make_jax_policy

    env = FsmRedCC4Env(
        num_steps=500,
        topology_mode=config.get("TOPOLOGY_MODE", "generative"),
        topology_bank_size=config.get("TOPOLOGY_BANK_SIZE", 0),
        training_mode=bool(config.get("TRAINING_MODE", True)),
    )
    action_dim = env.action_space(env.agents[0]).n
    return make_jax_policy(
        recipe["arch"]["name"],
        action_dim=action_dim,
        hidden_dim=config["HIDDEN_DIM"],
        hidden_layers=config["HIDDEN_LAYERS"],
        activation=config["ACTIVATION"],
    )


def _summarize_times(update_times: list[float], steps_per_update: int) -> dict:
    if not update_times:
        raise ValueError("No update times recorded")

    first_update_time_s = float(update_times[0])
    steady_times = update_times[1:] if len(update_times) > 1 else update_times
    steady_sps_values = [float(steps_per_update / t) for t in steady_times]
    steady_update_time_mean_s = float(np.mean(steady_times))
    steady_update_time_median_s = float(np.median(steady_times))
    compile_time_estimate_s = first_update_time_s
    if len(update_times) > 1:
        compile_time_estimate_s = max(0.0, first_update_time_s - steady_update_time_median_s)

    return {
        "first_update_time_s": first_update_time_s,
        "compile_time_estimate_s": float(compile_time_estimate_s),
        "steady_update_time_mean_s": steady_update_time_mean_s,
        "steady_update_time_median_s": steady_update_time_median_s,
        "steady_update_time_std_s": float(np.std(steady_times)),
        "steady_sps_mean": float(np.mean(steady_sps_values)),
        "steady_sps_median": float(np.median(steady_sps_values)),
        "steady_sps_std": float(np.std(steady_sps_values)),
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.recipe import load as load_recipe
    from jaxborg.recipe import project_jax

    recipe = load_recipe(args.recipe)
    config = project_jax(recipe)
    config["SEED"] = args.seed
    if args.num_envs is not None:
        config["NUM_ENVS"] = args.num_envs
    if args.num_steps is not None:
        config["NUM_STEPS"] = args.num_steps
    # Optimization benchmark always uses the default generated topology path.
    # Topology is not a benchmark knob; changing it would change the workload.
    config["TOPOLOGY_MODE"] = "generative"
    config["TOPOLOGY_BANK_SIZE"] = 0
    config["TOTAL_TIMESTEPS"] = int(args.num_updates * config["NUM_ENVS"] * config["NUM_STEPS"])
    config["MLFLOW_ENABLED"] = False

    flat_batch = int(NUM_BLUE_AGENTS * config["NUM_ENVS"] * config["NUM_STEPS"])
    if flat_batch < int(config["NUM_MINIBATCHES"]):
        raise SystemExit(
            "Benchmark shape is too small for the recipe: "
            f"{NUM_BLUE_AGENTS} agents * {config['NUM_ENVS']} envs * {config['NUM_STEPS']} steps "
            f"= {flat_batch} samples, but NUM_MINIBATCHES={config['NUM_MINIBATCHES']}. "
            "Increase --num-envs or --num-steps."
        )

    cache_dir = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).expanduser()
    if cache_dir.exists():
        print(f"Clearing XLA cache: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from ippo_jax import RewardNormState, make_train

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"XLA cache dir: {cache_dir}", flush=True)
    print(
        "Benchmark: "
        f"recipe={recipe['meta']['name']} updates={args.num_updates} "
        f"num_envs={config['NUM_ENVS']} num_steps={config['NUM_STEPS']} "
        "topology=generative",
        flush=True,
    )

    network = _build_network(recipe, config)

    setup_start = time.perf_counter()
    _, init_obs, init_env_state, init_train_state, collect_and_update = make_train(config, network)
    setup_time_s = time.perf_counter() - setup_start
    print(f"Setup time: {setup_time_s:.3f}s", flush=True)

    rng = jax.random.PRNGKey(args.seed + 1)
    rng, init_rng = jax.random.split(rng)
    train_state = init_train_state(init_rng)
    env_state = init_env_state
    obs = init_obs
    reward_norm_state = RewardNormState(
        returns=jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32),
        mean=jnp.zeros((), dtype=jnp.float32),
        var=jnp.ones((), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )

    update_times: list[float] = []
    steps_per_update = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    for update_idx in range(args.num_updates):
        start = time.perf_counter()
        train_state, env_state, obs, rng, reward_norm_state, metric = collect_and_update(
            train_state, env_state, obs, rng, reward_norm_state
        )
        jax.block_until_ready(metric)
        elapsed = time.perf_counter() - start
        update_times.append(elapsed)
        label = "compile+run" if update_idx == 0 else f"warm {update_idx}"
        print(f"{label}: {elapsed:.3f}s ({steps_per_update / elapsed:,.0f} steps/sec)", flush=True)

    result = {
        "label": args.label,
        "recipe": recipe["meta"]["name"],
        "seed": args.seed,
        "num_updates": args.num_updates,
        "num_envs": int(config["NUM_ENVS"]),
        "num_steps": int(config["NUM_STEPS"]),
        "steps_per_update": steps_per_update,
        "setup_time_s": float(setup_time_s),
        "update_times_s": [float(t) for t in update_times],
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_cache_dir": str(cache_dir),
    }
    result.update(_summarize_times(update_times, steps_per_update))

    print("")
    print("PERF SUMMARY")
    print(f"steady_sps_median: {result['steady_sps_median']:.3f}")
    print(f"compile_time_estimate_s: {result['compile_time_estimate_s']:.3f}")
    print(f"first_update_time_s: {result['first_update_time_s']:.3f}")
    print("PERF_JSON: " + json.dumps(result, sort_keys=True))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="default")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-updates", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
