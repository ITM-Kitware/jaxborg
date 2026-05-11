"""JAX-native eval for Phase 6 Test 2 — held-out red sweep.

Loads a JAX-trained checkpoint, builds a JAX env with the eval variant
(red overridable via ``--eval-red``), runs ``--episodes`` deterministic-
argmax rollouts vmapped, and writes a result row to a JSONL file.

The plan's eval pipeline routes via eval_recipe.py (CybORG-side), but for
held-out-generalization claims the plan also accepts JAX-internal eval
(plan §"Training & implementation" / "Eval"). This script is the
JAX-internal version: cheaper, faster, no CybORG dependency, and the
held-out signal is a JAX-internal claim regardless.

Usage:
    uv run python scripts/eval/cec_phase6_eval_jax.py \\
        --model jaxborg-exp/ippo_jax/cec_phase6_C11_seed42/model_cec_phase6_C11_seed42.safetensors \\
        --eval-red cia_c --episodes 90 --seed 1000

Result schema mirrors eval_recipe.py for downstream aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.evaluation.jax_runner import load_jax_checkpoint
from jaxborg.scenarios.cc4.game_variants import variant_for_red

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _build_eval_env(variant_name: str, *, resilience_roles: bool):
    """Build a clean canonical-config eval env for the given red.

    Eval intentionally uses NO env-diversity banks — the held-out
    generalization claim is "policy trained on diverse env handles a held-out
    red on the canonical env." Topology is the canonical generative one;
    mission profile is (1, 1, 1); phase boundaries are canonical.
    """
    variant = variant_for_red(variant_name, resilience_roles=resilience_roles)
    return variant, make_jax_env(variant)


def run_eval(
    *,
    model_path: Path,
    eval_red: str,
    episodes: int,
    seed: int,
) -> dict:
    policy, params, recipe = load_jax_checkpoint(model_path)
    train_red = recipe.get("train", {}).get("variant", "cc4_stock")
    # CIA-biased reds need resilience_roles=True for their selectors;
    # cc4_stock and fsm reds don't.
    resilience_roles = eval_red in ("cia_c", "cia_i", "cia_a", "resilience", "c", "i", "a")
    variant, env = _build_eval_env(eval_red, resilience_roles=resilience_roles)

    blue_agents = tuple(f"blue_{i}" for i in range(NUM_BLUE_AGENTS))
    num_steps = variant.num_steps

    @jax.jit
    def _run_one(key):
        reset_key, scan_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key)
        mask = env.get_avail_actions(env_state)

        def step_fn(carry, _):
            state, obs, mask, k = carry
            k, step_key = jax.random.split(k)
            obs_stack = jnp.stack([obs[a] for a in blue_agents])
            mask_stack = jnp.stack([mask[a] for a in blue_agents])

            def _fwd(o, m):
                pi, _ = policy.apply(params, o, m)
                return pi.logits

            logits = jax.vmap(_fwd)(obs_stack, mask_stack)
            acts = jnp.argmax(logits, axis=-1)
            actions = {a: acts[i] for i, a in enumerate(blue_agents)}
            new_obs, new_state, rewards, _, _ = env.step(step_key, state, actions)
            new_mask = env.get_avail_actions(new_state)
            mean_reward = jnp.stack([rewards[a] for a in blue_agents]).mean()
            return (new_state, new_obs, new_mask, k), mean_reward

        (_, _, _, _), per_step = jax.lax.scan(step_fn, (env_state, obs, mask, scan_key), None, length=num_steps)
        return per_step.sum()

    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    t0 = time.perf_counter()
    totals = jax.vmap(_run_one)(keys)
    totals.block_until_ready()
    wall = time.perf_counter() - t0
    rewards_list = [float(x) for x in np.asarray(totals)]

    return {
        "model": str(model_path),
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "trained_backend": "jax",
        "eval_env": "jax",
        "eval_red": eval_red,
        "variant": variant.name,
        "train_variant": train_red,
        "seed": seed,
        "episodes": episodes,
        "deterministic": True,
        "mean_reward": mean(rewards_list),
        "std_reward": stdev(rewards_list) if len(rewards_list) > 1 else 0.0,
        "n_episodes": len(rewards_list),
        "wall_time_s": wall,
        "git_commit": _git_commit(),
        "per_episode": rewards_list,
    }


def main():
    parser = argparse.ArgumentParser(description="JAX-native held-out red eval for Phase 6 Test 2")
    parser.add_argument("--model", required=True, help=".safetensors checkpoint with sibling recipe sidecar")
    # NOTE: "random" (CybORG's RandomSelectRedAgent) is not in the JAX red-
    # selector REGISTRY; route that through eval_recipe.py (CybORG eval).
    parser.add_argument(
        "--eval-red", required=True, choices=["fsm", "cia_c", "cia_i", "cia_a", "resilience", "sleep"]
    )
    parser.add_argument("--episodes", type=int, default=90, help="Episodes (plan default 90 for stat power)")
    parser.add_argument("--seed", type=int, default=1000, help="PRNG seed for the rollout batch")
    parser.add_argument("--output", type=str, default=None, help="Override result jsonl path")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"=== JAX eval: {model_path.name} vs {args.eval_red} ({args.episodes} eps, seed {args.seed}) ===", flush=True)
    print(f"JAX backend: {jax.default_backend()} ({jax.devices()})", flush=True)

    row = run_eval(
        model_path=model_path,
        eval_red=args.eval_red,
        episodes=args.episodes,
        seed=args.seed,
    )
    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.seed}_{args.eval_red}"
    row["eval_id"] = eval_id

    out_dir = EXP_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"phase6_{row['recipe_name']}_{model_path.stem}_{eval_id}.jsonl"
    out_path.write_text(json.dumps(row, indent=2) + "\n")

    print(f"\nmean: {row['mean_reward']:.2f} ± {row['std_reward']:.2f} (n={row['n_episodes']})", flush=True)
    print(f"wall: {row['wall_time_s']:.1f}s", flush=True)
    print(f"wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
