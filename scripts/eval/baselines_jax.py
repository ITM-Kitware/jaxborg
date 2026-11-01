"""Evaluate JAX baselines (sleep and random blue) on a recipe-driven JAX env."""

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import jax
import jax.numpy as jnp

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.recipe import resolve_eval_variant

EPISODE_LENGTH = 500


def run_sleep_episode(env, key, topology_index=None):
    if topology_index is None:
        obs, state = env.reset(key)
    else:
        obs, state = env.reset_at_topology(key, topology_index)
    actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)
        total += float(rewards["blue_0"])
    return total


def _sample_masked_uniform(key, mask):
    # mask: bool array of length action_space; pick uniformly among True entries.
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


def run_random_episode(env, key, topology_index=None):
    if topology_index is None:
        obs, state = env.reset(key)
    else:
        obs, state = env.reset_at_topology(key, topology_index)
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        key, act_key, step_key = jax.random.split(key, 3)
        masks = env.get_avail_actions(state)
        actions = {
            f"blue_{b}": _sample_masked_uniform(jax.random.fold_in(act_key, b), masks[f"blue_{b}"])
            for b in range(NUM_BLUE_AGENTS)
        }
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        total += float(rewards["blue_0"])
    return total


def _resolve_topology_paths(
    topology_path: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    """Normalize an optional topology snapshot bank for evaluation."""
    if topology_path is None:
        return ()
    if isinstance(topology_path, (str, Path)):
        paths = (topology_path,)
    else:
        paths = tuple(topology_path)
        if not paths:
            raise ValueError("topology_path must contain at least one snapshot path")
    return tuple(Path(path).expanduser().resolve() for path in paths)


def evaluate(
    policy,
    seed,
    max_eps,
    recipe_name=None,
    checkpoint=None,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    topology_sampling: str | None = None,
):
    variant = resolve_eval_variant(recipe_name=recipe_name, checkpoint=checkpoint)
    if variant.num_steps != EPISODE_LENGTH:
        variant = replace(variant, num_steps=EPISODE_LENGTH)
    if topology_path is None and (recipe_name is not None or checkpoint is not None):
        from jaxborg.checkpoint import read_sidecar
        from jaxborg.recipe import load, project_eval

        recipe = load(recipe_name) if recipe_name is not None else read_sidecar(checkpoint)
        eval_config = project_eval(recipe, materialize_topologies=True)
        topology_paths = tuple(eval_config["TOPOLOGY_BANK"])
        selected_sampling = topology_sampling or eval_config["TOPOLOGY_SAMPLING"]
    else:
        topology_paths = _resolve_topology_paths(topology_path)
        if topology_paths and (recipe_name is not None or checkpoint is not None):
            from jaxborg.checkpoint import read_sidecar
            from jaxborg.recipe import REPO_ROOT, load
            from jaxborg.topology_banks import validate_eval_topology_override

            recipe = load(recipe_name) if recipe_name is not None else read_sidecar(checkpoint)
            validate_eval_topology_override(recipe, topology_paths, repo_root=REPO_ROOT)
        selected_sampling = topology_sampling or "exhaustive"
    if selected_sampling not in ("exhaustive", "random"):
        raise ValueError("topology_sampling must be 'exhaustive' or 'random'")
    env = make_jax_env(variant, topology_path=topology_paths or None)
    run_fn = run_sleep_episode if policy == "sleep" else run_random_episode

    episode_rewards = []
    exhaustive = bool(topology_paths) and selected_sampling == "exhaustive"
    topology_indices = range(len(topology_paths)) if exhaustive else (None,)
    for topology_index in topology_indices:
        for episode_index in range(max_eps):
            key = jax.random.PRNGKey(seed + episode_index if seed is not None else episode_index)
            episode_rewards.append(run_fn(env, key, topology_index))

    print(f"variant:   {variant.name} (red_agent={variant.red_agent})")
    print(f"policy:    {policy}")
    if topology_paths:
        print(f"topologies:{len(topology_paths):>4} snapshot(s), {selected_sampling}")
        for path in topology_paths:
            print(f"           {path}")
    else:
        print("topologies: generative")
    print(f"episodes:  {len(episode_rewards)}")
    if exhaustive:
        print(f"           ({max_eps} per topology)")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")
    print(f"min:       {min(episode_rewards):.4f}")
    print(f"max:       {max(episode_rewards):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JAX baselines on a recipe-driven JAX env")
    parser.add_argument("--policy", choices=["sleep", "random"], default="sleep")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=10)
    parser.add_argument("--recipe", default=None, help="Recipe path or name (overrides --checkpoint sidecar)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .safetensors; variant auto-resolved from its sidecar if --recipe is not set",
    )
    parser.add_argument(
        "--topology-path",
        action="extend",
        nargs="+",
        default=None,
        help="Topology snapshot(s) to sample during evaluation; may be repeated for a held-out bank",
    )
    parser.add_argument(
        "--topology-sampling",
        choices=("exhaustive", "random"),
        default=None,
        help="Bank assignment (default: recipe value or exhaustive)",
    )
    args = parser.parse_args()
    evaluate(
        args.policy,
        args.seed,
        args.max_eps,
        recipe_name=args.recipe,
        checkpoint=args.checkpoint,
        topology_path=args.topology_path,
        topology_sampling=args.topology_sampling,
    )


if __name__ == "__main__":
    main()
