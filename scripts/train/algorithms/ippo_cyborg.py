"""CleanRL-style IPPO on real CybORG CC4, recipe-driven.

Algorithm script — owns the rollout loop, GAE, PPO update, metrics. Network
arch is selected by `recipe.arch.name` and instantiated via
`jaxborg.policies.make_torch_policy`; the algorithm itself is arch-agnostic.

Launch:
    uv run python scripts/train/algorithms/ippo_cyborg.py --recipe singh --seed 42

Outputs (to `$JAXBORG_EXP_DIR/ippo_cyborg/<tag>/`):
    metrics.jsonl       (standardized schema)
    recipe_<tag>.yaml   (resolved recipe sidecar)
    model_<tag>.pt      (bare state_dict)
    checkpoint_<tag>.pt (full optimizer + scaler state)
"""

# ruff: noqa: E402

import argparse
import json
import os
import signal
import sys
import time
from multiprocessing import Pipe, Process
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.checkpoint import write_sidecar
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.metrics_schema import make_row
from jaxborg.mlflow_setup import start_run
from jaxborg.policies import make_torch_policy
from jaxborg.recipe import load as load_recipe
from jaxborg.recipe import project_cleanrl
from jaxborg.scenarios.cc4.game_variant import GameVariant

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE
ACT_DIM = 242


def env_worker(pipe, env_id, variant: GameVariant):
    import random as _random

    from CybORG.Agents.Wrappers import EnterpriseMAE

    from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Per-worker RNG for per-episode resilience-role seeds + env construction.
    # Distinct per worker so vmap-equivalent envs see different sequences.
    seed_rng = _random.Random(env_id)
    env = make_cyborg_env(variant, seed_rng.randrange(2**31), wrapper_class=EnterpriseMAE)

    def _reset_and_inject():
        r = reset_cyborg_env(env, variant, ep_seed=seed_rng.randrange(2**31))
        return r.obs, r.info

    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break
        if cmd == "reset":
            obs, info = _reset_and_inject()
            pipe.send((obs, info))
        elif cmd == "step":
            obs, rew, term, trunc, info = env.step(data)
            done = any(term.values()) or any(trunc.values())
            if done:
                obs, info = _reset_and_inject()
            pipe.send((obs, rew, done, info))
        elif cmd == "close":
            pipe.close()
            break


class ParallelEnvs:
    def __init__(self, num_envs, variant: GameVariant):
        self.num_envs = num_envs
        self.pipes = []
        self.procs = []
        for i in range(num_envs):
            parent_pipe, child_pipe = Pipe()
            proc = Process(target=env_worker, args=(child_pipe, i, variant), daemon=True)
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.procs.append(proc)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.pipes]
        return [r[0] for r in results], [r[1] for r in results]

    def step(self, actions_list):
        for pipe, actions in zip(self.pipes, actions_list):
            pipe.send(("step", actions))
        results = [pipe.recv() for pipe in self.pipes]
        return (
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
            [r[3] for r in results],
        )

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class RewardScaler:
    """Scale rewards by running std of discounted returns (matches JAX side)."""

    def __init__(self, num_envs, gamma, clip=10.0):
        self.gamma = gamma
        self.clip = clip
        self.returns = np.zeros(num_envs)
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def _update_stats(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def scale(self, rewards, dones):
        self.returns = self.returns * self.gamma + rewards
        self._update_stats(self.returns)
        scaled = rewards / (np.sqrt(self.var) + 1e-8)
        scaled = np.clip(scaled, -self.clip, self.clip)
        self.returns[dones] = 0.0
        return scaled


def train(args, recipe, cfg):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = args.tag or f"{recipe['meta']['name']}_seed{args.seed}"
    save_dir = EXP_DIR / "ippo_cyborg" / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    variant: GameVariant = cfg["TRAIN_VARIANT"]
    print(f"Creating {cfg['num_envs']} parallel CybORG environments (variant={variant.name})...", flush=True)
    envs = ParallelEnvs(cfg["num_envs"], variant=variant)

    agent = make_torch_policy(
        recipe["arch"]["name"],
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        hidden_dim=cfg["hidden_dim"],
        hidden_layers=cfg["hidden_layers"],
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)
    reward_scaler = RewardScaler(cfg["num_envs"], cfg["gamma"]) if cfg["norm_rewards"] else None

    num_steps = cfg["rollout_length"]
    num_envs = cfg["num_envs"]
    obs_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS, OBS_DIM))
    actions_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS), dtype=torch.long)
    logprobs_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS))
    rewards_all = torch.zeros((num_steps, num_envs))
    dones_all = torch.zeros((num_steps, num_envs))
    values_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS))
    masks_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS, ACT_DIM))

    run = start_run(recipe, backend="cyborg", seed=args.seed)
    train_run_id = run.info.run_id

    metrics_path = save_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    all_obs, all_info = envs.reset()
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []

    start_time = time.perf_counter()
    total_steps = 0
    num_updates = 0
    rollouts_collected = 0
    accum_obs, accum_act, accum_lp = [], [], []
    accum_adv, accum_ret, accum_val, accum_mask = [], [], [], []

    steps_per_update = num_envs * num_steps * cfg["num_rollouts_per_update"]
    total_updates = max(1, cfg["total_timesteps"] // steps_per_update)

    print(f"\n{'=' * 70}")
    print(f"IPPO-CybORG [{recipe['meta']['name']}] seed={args.seed}")
    print(
        f"  num_envs={num_envs} rollout_length={num_steps} "
        f"rollouts/update={cfg['num_rollouts_per_update']} "
        f"steps/update={steps_per_update:,}"
    )
    print(f"  total_timesteps={cfg['total_timesteps']:,} updates={total_updates}")
    print(
        f"  arch={recipe['arch']['name']} lr={cfg['lr']} gamma={cfg['gamma']} "
        f"epochs={cfg['num_epochs']} mb={cfg['num_minibatches']}"
    )
    print(f"{'=' * 70}\n", flush=True)

    try:
        while total_steps < cfg["total_timesteps"]:
            for step in range(num_steps):
                for env_idx in range(num_envs):
                    for i in range(NUM_AGENTS):
                        aid = AGENT_IDS[i]
                        raw_obs = all_obs[env_idx][aid].astype(np.float32)
                        raw_mask = np.array(all_info[env_idx][aid]["action_mask"], dtype=np.float32)
                        obs_buf[step, env_idx, i, : len(raw_obs)] = torch.from_numpy(raw_obs)
                        obs_buf[step, env_idx, i, len(raw_obs) :] = 0.0
                        masks_buf[step, env_idx, i, : len(raw_mask)] = torch.from_numpy(raw_mask)
                        masks_buf[step, env_idx, i, len(raw_mask) :] = 0.0

                with torch.no_grad():
                    obs_flat = obs_buf[step].reshape(-1, OBS_DIM)
                    mask_flat = masks_buf[step].reshape(-1, ACT_DIM)
                    act, lp, _, val = agent.get_action_and_value(obs_flat, mask_flat)
                    actions_buf[step] = act.reshape(num_envs, NUM_AGENTS)
                    logprobs_buf[step] = lp.reshape(num_envs, NUM_AGENTS)
                    values_buf[step] = val.reshape(num_envs, NUM_AGENTS)

                action_dicts = []
                for env_idx in range(num_envs):
                    action_dicts.append(
                        {AGENT_IDS[i]: int(actions_buf[step, env_idx, i].item()) for i in range(NUM_AGENTS)}
                    )

                all_obs, all_rew, all_done, all_info = envs.step(action_dicts)
                raw_rewards = np.array([all_rew[e][AGENT_IDS[0]] for e in range(num_envs)])
                dones = np.array(all_done, dtype=bool)

                episode_rewards += raw_rewards
                episode_lengths += 1
                for env_idx in range(num_envs):
                    if dones[env_idx]:
                        completed_rewards.append(float(episode_rewards[env_idx]))
                        completed_lengths.append(int(episode_lengths[env_idx]))
                        episode_rewards[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                if reward_scaler is not None:
                    scaled = reward_scaler.scale(raw_rewards, dones)
                    rewards_all[step] = torch.from_numpy(scaled.astype(np.float32))
                else:
                    rewards_all[step] = torch.from_numpy(raw_rewards.astype(np.float32))

                dones_all[step] = torch.from_numpy(dones.astype(np.float32))
                total_steps += num_envs

            with torch.no_grad():
                next_obs_flat = torch.zeros(num_envs * NUM_AGENTS, OBS_DIM)
                for env_idx in range(num_envs):
                    for i in range(NUM_AGENTS):
                        raw_obs = all_obs[env_idx][AGENT_IDS[i]].astype(np.float32)
                        next_obs_flat[env_idx * NUM_AGENTS + i, : len(raw_obs)] = torch.from_numpy(raw_obs)
                next_val = agent.get_value(next_obs_flat).reshape(num_envs, NUM_AGENTS)

                advantages = torch.zeros((num_steps, num_envs, NUM_AGENTS))
                lastgaelam = torch.zeros(num_envs, NUM_AGENTS)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = next_val
                    else:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = values_buf[t + 1]
                    rew_expanded = rewards_all[t].unsqueeze(-1).expand_as(values_buf[t])
                    delta = rew_expanded + cfg["gamma"] * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
                returns = advantages + values_buf

            accum_obs.append(obs_buf.reshape(-1, OBS_DIM).clone())
            accum_act.append(actions_buf.reshape(-1).clone())
            accum_lp.append(logprobs_buf.reshape(-1).clone())
            accum_adv.append(advantages.reshape(-1).clone())
            accum_ret.append(returns.reshape(-1).clone())
            accum_val.append(values_buf.reshape(-1).clone())
            accum_mask.append(masks_buf.reshape(-1, ACT_DIM).clone())
            rollouts_collected += 1
            if rollouts_collected < cfg["num_rollouts_per_update"]:
                continue

            num_updates += 1
            lr = cfg["lr"]
            if cfg["anneal_lr"]:
                frac = 1.0 - (num_updates - 1) / total_updates
                lr = max(frac * cfg["lr"], 1e-6)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            b_obs = torch.cat(accum_obs)
            b_act = torch.cat(accum_act)
            b_lp = torch.cat(accum_lp)
            b_adv = torch.cat(accum_adv)
            b_ret = torch.cat(accum_ret)
            b_val = torch.cat(accum_val)
            b_mask = torch.cat(accum_mask)
            accum_obs.clear()
            accum_act.clear()
            accum_lp.clear()
            accum_adv.clear()
            accum_ret.clear()
            accum_val.clear()
            accum_mask.clear()
            rollouts_collected = 0

            total_n = b_obs.shape[0]
            mb_size_n = total_n // cfg["num_minibatches"]

            ep_pg = ep_vf = ep_ent = ep_kl = ep_clipfrac = 0.0
            ep_pre_grad = ep_grad = 0.0
            n_mb = 0
            for _epoch in range(cfg["num_epochs"]):
                perm = torch.randperm(total_n)
                for mb_idx in range(cfg["num_minibatches"]):
                    idx = perm[mb_idx * mb_size_n : (mb_idx + 1) * mb_size_n]
                    mb_obs = b_obs[idx]
                    mb_act = b_act[idx]
                    mb_lp = b_lp[idx]
                    mb_adv = b_adv[idx]
                    mb_ret = b_ret[idx]
                    mb_mask = b_mask[idx]
                    _, new_lp, ent, new_val = agent.get_action_and_value(mb_obs, mb_mask, mb_act)
                    adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                    logratio = new_lp - mb_lp
                    ratio = logratio.exp()
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
                    entropy_loss = ent.mean()
                    loss = pg_loss - cfg["ent_coef"] * entropy_loss + cfg["vf_coef"] * vf_loss
                    optimizer.zero_grad()
                    loss.backward()
                    pre_clip = float(nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]))
                    post_clip = min(pre_clip, cfg["max_grad_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfrac = ((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item()
                    ep_pg += pg_loss.item()
                    ep_vf += vf_loss.item()
                    ep_ent += entropy_loss.item()
                    ep_kl += approx_kl
                    ep_clipfrac += clipfrac
                    ep_pre_grad += pre_clip
                    ep_grad += post_clip
                    n_mb += 1

            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed if elapsed > 0 else 0
            avg_pg = ep_pg / max(n_mb, 1)
            avg_vf = ep_vf / max(n_mb, 1)
            avg_ent = ep_ent / max(n_mb, 1)
            avg_kl = ep_kl / max(n_mb, 1)
            avg_clipfrac = ep_clipfrac / max(n_mb, 1)
            avg_pre_grad = ep_pre_grad / max(n_mb, 1)
            avg_grad = ep_grad / max(n_mb, 1)

            with torch.no_grad():
                y_var = b_ret.var(unbiased=False)
                explained_var = (
                    (1 - (b_ret - b_val).var(unbiased=False) / (y_var + 1e-8)).item() if y_var > 1e-8 else 0.0
                )

            ep_rew = float(np.mean(completed_rewards[-50:])) if completed_rewards else float("nan")
            ep_len = float(np.mean(completed_lengths[-50:])) if completed_lengths else float("nan")

            row = make_row(
                update_idx=num_updates,
                env_steps=total_steps,
                wall_time_s=elapsed,
                throughput_sps=sps,
                loss_policy=avg_pg,
                loss_value=avg_vf,
                loss_entropy=avg_ent,
                loss_total=avg_pg + cfg["vf_coef"] * avg_vf - cfg["ent_coef"] * avg_ent,
                ppo_kl_divergence=avg_kl,
                ppo_clip_fraction=avg_clipfrac,
                ppo_explained_variance=float(explained_var),
                lr=lr,
                train_episode_reward_mean=ep_rew if not np.isnan(ep_rew) else None,
                train_episode_length_mean=ep_len if not np.isnan(ep_len) else None,
                ppo_grad_norm=avg_grad,
                ppo_pre_clip_grad_norm=avg_pre_grad,
                backend_extras={
                    "cyborg.episodes_completed": len(completed_rewards),
                    "cyborg.num_rollouts_accumulated": cfg["num_rollouts_per_update"],
                },
            )
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            try:
                safe = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(safe, step=total_steps)
            except Exception:
                pass

            print(
                f"  upd {num_updates:4d} | steps {total_steps:>9,} | ep_rew {ep_rew:>8.1f} | "
                f"pg {avg_pg:.4f} vf {avg_vf:.4f} ent {avg_ent:.3f} kl {avg_kl:.4f} | "
                f"{sps:.0f} sps | {elapsed / 3600:.2f}h",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)

    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed if elapsed > 0 else 0

    ckpt = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "total_steps": total_steps,
        "num_updates": num_updates,
    }
    if reward_scaler is not None:
        ckpt["reward_scaler"] = {
            "returns": reward_scaler.returns,
            "mean": reward_scaler.mean,
            "var": reward_scaler.var,
            "count": reward_scaler.count,
        }
    torch.save(ckpt, save_dir / f"checkpoint_{tag}.pt")
    torch.save(agent.state_dict(), save_dir / f"model_{tag}.pt")
    write_sidecar(
        save_dir / f"recipe_{tag}.yaml",
        recipe,
        seed=args.seed,
        total_steps=total_steps,
        backend="cyborg",
        train_run_id=train_run_id,
    )

    final_reward = float(np.mean(completed_rewards[-50:])) if completed_rewards else float("nan")
    try:
        finals = {"final_wall_time_sec": elapsed, "final_steps_per_second": sps}
        if not np.isnan(final_reward):
            finals["final_episode_reward_mean"] = final_reward
        finals["total_episodes"] = len(completed_rewards)
        mlflow.log_metrics(finals)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(save_dir / f"recipe_{tag}.yaml"))
        mlflow.end_run()
    except Exception as e:
        print(f"MLflow finalize warning: {e}")

    metrics_file.close()
    envs.close()
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 3600:.1f}h). Final ep reward: {final_reward:.1f}")
    print(f"Saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="IPPO on CybORG, recipe-driven")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--num-rollouts-per-update",
        type=int,
        default=None,
        help="Override the buffer_size-derived value (mainly for smoke tests)",
    )
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    cfg = project_cleanrl(recipe)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.num_envs is not None:
        cfg["num_envs"] = args.num_envs
        per_rollout = cfg["num_envs"] * cfg["rollout_length"]
        cfg["num_rollouts_per_update"] = max(1, (recipe["train"]["buffer_size"] + per_rollout - 1) // per_rollout)
    if args.num_rollouts_per_update is not None:
        cfg["num_rollouts_per_update"] = args.num_rollouts_per_update

    train(args, recipe, cfg)


if __name__ == "__main__":
    main()
