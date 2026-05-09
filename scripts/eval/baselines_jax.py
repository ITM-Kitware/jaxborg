"""Evaluate JAX baselines (sleep and random blue) on a recipe-driven JAX env."""

import argparse
from dataclasses import replace
from statistics import mean, stdev

import jax
import jax.numpy as jnp

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.recipe import resolve_eval_variant

EPISODE_LENGTH = 500


def run_sleep_episode(env, key):
    obs, state = env.reset(key)
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


def run_random_episode(env, key):
    obs, state = env.reset(key)
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


def evaluate(policy, seed, max_eps, recipe_name=None, checkpoint=None):
    variant = resolve_eval_variant(recipe_name=recipe_name, checkpoint=checkpoint)
    if variant.num_steps != EPISODE_LENGTH:
        variant = replace(variant, num_steps=EPISODE_LENGTH)
    env = make_jax_env(variant)
    run_fn = run_sleep_episode if policy == "sleep" else run_random_episode

    episode_rewards = []
    for ep in range(max_eps):
        key = jax.random.PRNGKey(seed + ep if seed is not None else ep)
        episode_rewards.append(run_fn(env, key))

    print(f"variant:   {variant.name} (red_agent={variant.red_agent})")
    print(f"policy:    {policy}")
    print(f"episodes:  {max_eps}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")
    print(f"min:       {min(episode_rewards):.4f}")
    print(f"max:       {max(episode_rewards):.4f}")


if __name__ == "__main__":
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
    args = parser.parse_args()
    evaluate(args.policy, args.seed, args.max_eps, recipe_name=args.recipe, checkpoint=args.checkpoint)
