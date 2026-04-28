"""Verify whether structural topology variation surfaces in blue's observation.

Rolls out N episodes under two conditions:
  - gen-fixed: constant reset key → identical topology each episode
  - gen-base: varying reset keys → fresh topology each episode

For each episode, captures the blue obs vectors at every step and reports
distributional differences. Uses random blue actions so the comparison is
about env-side variation only, not policy-induced variation.

If gen-base obs distributions differ statistically from gen-fixed, the
"data_links is the structural-observable axis" claim is supported.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.fsm_red_env import FsmRedCC4Env


def random_actions(env, key, mask):
    """Sample uniform random valid blue actions per agent."""
    keys = jax.random.split(key, env.num_agents)
    n_actions = mask.shape[-1]
    acts = jnp.zeros(env.num_agents, dtype=jnp.int32)
    for i in range(env.num_agents):
        valid = jnp.where(mask[i], 1.0, 0.0)
        valid = valid / jnp.maximum(valid.sum(), 1.0)
        a = jax.random.categorical(keys[i], jnp.log(valid + 1e-12))
        acts = acts.at[i].set(a)
    return acts


def collect_obs(env, num_episodes, fixed_key, key_seed_base):
    """Roll out num_episodes; return all blue obs concatenated (N*T*A, obs_dim)."""
    obs_per_step = []
    for ep in range(num_episodes):
        if fixed_key is not None:
            reset_key = fixed_key
        else:
            reset_key = jax.random.PRNGKey(key_seed_base + ep)
        obs_d, env_state = env.reset(reset_key)
        # blue obs: dict of agent_id → array
        for t in range(50):
            arr = jnp.stack([obs_d[a] for a in env.agents])
            obs_per_step.append(np.asarray(arr))
            mask = (
                jnp.stack([env_state.action_mask[a] for a in env.agents]) if hasattr(env_state, "action_mask") else None
            )
            if mask is None:
                # fall back: sample actions uniformly across full action space
                k = jax.random.PRNGKey(ep * 1000 + t)
                act_keys = jax.random.split(k, env.num_agents)
                acts = jnp.array(
                    [jax.random.randint(ak, (), 0, env.action_space(a).n) for ak, a in zip(act_keys, env.agents)]
                )
            else:
                k = jax.random.PRNGKey(ep * 1000 + t)
                acts = random_actions(env, k, mask)
            action_dict = {a: int(acts[i]) for i, a in enumerate(env.agents)}
            step_key = jax.random.PRNGKey(ep * 7919 + t)
            obs_d, env_state, _, _, _ = env.step(step_key, env_state, action_dict)
        if (ep + 1) % 5 == 0:
            print(f"  collected ep {ep + 1}/{num_episodes}", flush=True)
    return np.concatenate(obs_per_step, axis=0)  # (eps*T*A, obs_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    env = FsmRedCC4Env(num_steps=500, topology_mode="generative", training_mode=False)

    print("collecting gen-fixed (constant key) ...", flush=True)
    fixed_key = jax.random.PRNGKey(123)
    obs_fixed = collect_obs(env, args.episodes, fixed_key, args.seed_base)

    print("collecting gen-base (varying keys) ...", flush=True)
    obs_varied = collect_obs(env, args.episodes, None, args.seed_base + 10000)

    print("\n=== obs distribution comparison ===")
    print(f"obs shape: {obs_fixed.shape}, dim={obs_fixed.shape[-1]}")
    mean_fixed = obs_fixed.mean(axis=0)
    mean_varied = obs_varied.mean(axis=0)
    std_fixed = obs_fixed.std(axis=0)
    std_varied = obs_varied.std(axis=0)

    abs_diff = np.abs(mean_fixed - mean_varied)
    pooled_std = np.sqrt((std_fixed**2 + std_varied**2) / 2 + 1e-9)
    cohen_d = abs_diff / pooled_std

    print("\nper-feature drift (top 10 by |mean_fixed - mean_varied| / pooled_std):")
    top = np.argsort(-cohen_d)[:10]
    for idx in top:
        print(
            f"  feat[{idx:4d}]  mean_fixed={mean_fixed[idx]:+.4f}  mean_varied={mean_varied[idx]:+.4f}  "
            f"std_fix={std_fixed[idx]:.4f}  std_var={std_varied[idx]:.4f}  d={cohen_d[idx]:.3f}"
        )

    print("\noverall:")
    print(f"  features with cohen_d > 0.1:  {int((cohen_d > 0.1).sum())}/{len(cohen_d)}")
    print(f"  features with cohen_d > 0.3:  {int((cohen_d > 0.3).sum())}/{len(cohen_d)}")
    print(f"  max cohen_d: {cohen_d.max():.3f}")
    print(f"  mean cohen_d: {cohen_d.mean():.3f}")


if __name__ == "__main__":
    main()
