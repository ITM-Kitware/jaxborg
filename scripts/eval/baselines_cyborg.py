"""Evaluate sleep and random baselines on CybORG."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from CybORG.Agents.Wrappers import BlueFlatWrapper

from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.recipe import resolve_eval_variant

EPISODE_LENGTH = 500


def make_env(variant, seed):
    return make_cyborg_env(variant, seed, wrapper_class=BlueFlatWrapper)


def run_sleep_episode(env, variant, ep_seed, _rng):
    reset_cyborg_env(env, variant, ep_seed=ep_seed)
    actions = {agent: 0 for agent in env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        _, rewards, _, _, _ = env.step(actions)
        total += mean(rewards.values())
    return total


def run_random_episode(env, variant, ep_seed, rng):
    r = reset_cyborg_env(env, variant, ep_seed=ep_seed)
    info = r.info
    masks = {agent: info[agent]["action_mask"] for agent in env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {agent: int(rng.choice(np.flatnonzero(masks[agent]))) for agent in env.agents}
        _, rewards, _, _, info = env.step(actions)
        masks = {agent: info[agent]["action_mask"] for agent in env.agents}
        total += mean(rewards.values())
    return total


def evaluate(policy, seed, max_eps, output_json=None, recipe_name=None, checkpoint=None):
    variant = resolve_eval_variant(recipe_name=recipe_name, checkpoint=checkpoint)
    if variant.num_steps != EPISODE_LENGTH:
        variant = replace(variant, num_steps=EPISODE_LENGTH)
    base_seed = 0 if seed is None else seed
    rng = np.random.default_rng(base_seed)
    run_fn = run_sleep_episode if policy == "sleep" else run_random_episode

    episode_rewards = []
    for ep in range(max_eps):
        ep_seed = base_seed + ep
        env = make_env(variant, ep_seed)
        episode_rewards.append(run_fn(env, variant, ep_seed, rng))

    print(f"variant:   {variant.name} (red_agent={variant.red_agent})")
    print(f"policy:    {policy}")
    print(f"episodes:  {max_eps}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")

    if output_json:
        payload = {
            "variant": variant.name,
            "policy": policy,
            "seed": seed,
            "episodes": max_eps,
            "mean": mean(episode_rewards),
            "stdev": stdev(episode_rewards) if len(episode_rewards) > 1 else 0.0,
            "per_episode": [float(x) for x in episode_rewards],
        }
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote:     {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sleep/random baselines on CybORG")
    parser.add_argument("--policy", choices=["sleep", "random"], default="sleep")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=100)
    parser.add_argument("--output-json", default=None, help="Write mean/stdev/per-episode to JSON")
    parser.add_argument("--recipe", default=None, help="Recipe path or name (overrides --checkpoint sidecar)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .safetensors; variant auto-resolved from its sidecar if --recipe is not set",
    )
    args = parser.parse_args()
    evaluate(
        args.policy,
        args.seed,
        args.max_eps,
        args.output_json,
        recipe_name=args.recipe,
        checkpoint=args.checkpoint,
    )
