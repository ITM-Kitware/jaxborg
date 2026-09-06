"""CleanRL-style IPPO on real CybORG CC4, recipe-driven.

Algorithm script — owns the rollout loop, GAE, PPO update, metrics. Network
arch is selected by `recipe.arch.name` and instantiated via
`jaxborg.policies.make_torch_policy`; the algorithm itself is arch-agnostic.

Launch:
    uv run python scripts/train/algorithms/ippo_cyborg.py --recipe singh --seed 42

Outputs (to `$JAXBORG_EXP_DIR/ippo_cyborg/<tag>/`):
    metrics.jsonl       (standardized schema)
    recipe_<tag>.yaml   (resolved recipe sidecar)
    model_<tag>.pt      (versioned one- or two-policy bundle)
    checkpoint_<tag>.pt (full optimizer + scaler state)
"""

# ruff: noqa: E402

import argparse
import copy
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.checkpoint import load_torch_policy, read_sidecar, save_torch_bundle, write_sidecar
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.cyborg_joint import POLICY_AGENT_IDS, TEAM_SPECS, CyborgJointAdapter
from jaxborg.evaluation.training_checkpoint import evaluate_training_checkpoint
from jaxborg.metrics_schema import add_team_metrics, make_row
from jaxborg.mlflow_setup import MlflowCheckpointEvaluator, start_run
from jaxborg.policies import make_torch_policy
from jaxborg.recipe import load as load_recipe
from jaxborg.recipe import (
    project_cleanrl,
    resolve_train_opponents,
    team_recipe,
    training_teams,
)
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

    def _availability_info(info):
        controller = env.env.environment_controller
        for agent_name in AGENT_IDS:
            interface = controller.agent_interfaces[agent_name]
            actor_active = bool(interface.active) and controller.actions_in_progress.get(agent_name) is None
            agent_info = info.setdefault(agent_name, {})
            agent_info["actor_active"] = actor_active
            if not actor_active:
                mask = np.zeros(len(env.action_mask(agent_name)), dtype=bool)
                sleep_idx = env.action_labels(agent_name).index("Sleep")
                mask[sleep_idx] = True
                agent_info["action_mask"] = mask
        return info

    def _reset_and_inject():
        r = reset_cyborg_env(env, variant, ep_seed=seed_rng.randrange(2**31))
        return r.obs, _availability_info(r.info)

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
            info = _availability_info(info)
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


def joint_env_worker(pipe, env_id, seed: int, variant: GameVariant):
    """Worker entry point for learned-policy matchups."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = CyborgJointAdapter(variant, seed + env_id)

    def _reset():
        return env.reset()

    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break
        if cmd == "reset":
            pipe.send(_reset())
        elif cmd == "step":
            obs, rewards, terminated, _truncated, info = env.step(data)
            done = any(terminated.values())
            if done:
                obs, info = _reset()
            pipe.send((obs, rewards, done, info))
        elif cmd == "close":
            env.close()
            pipe.close()
            break


class ParallelJointEnvs:
    """Small process-vectorized facade over :class:`CyborgJointAdapter`."""

    def __init__(self, num_envs: int, seed: int, variant: GameVariant):
        self.num_envs = num_envs
        self.pipes = []
        self.procs = []
        for env_id in range(num_envs):
            parent_pipe, child_pipe = Pipe()
            proc = Process(
                target=joint_env_worker,
                args=(child_pipe, env_id, seed, variant),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.procs.append(proc)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.pipes]
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions_list):
        for pipe, actions in zip(self.pipes, actions_list):
            pipe.send(("step", actions))
        results = [pipe.recv() for pipe in self.pipes]
        return (
            [result[0] for result in results],
            [result[1] for result in results],
            [result[2] for result in results],
            [result[3] for result in results],
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


def train_legacy(args, recipe, cfg):
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
    actor_active_buf = torch.zeros((num_steps, num_envs, NUM_AGENTS), dtype=torch.bool)

    checkpoint_evaluator = MlflowCheckpointEvaluator(recipe)
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
    accum_adv, accum_ret, accum_val, accum_mask, accum_actor_active = [], [], [], [], []

    steps_per_update = num_envs * num_steps * cfg["num_rollouts_per_update"]
    total_updates = max(1, cfg["total_timesteps"] // steps_per_update)
    checkpoint_every = int(recipe.get("cleanrl", {}).get("checkpoint_every_updates", 50))

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
                        actor_active_buf[step, env_idx, i] = bool(all_info[env_idx][aid]["actor_active"])

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
            accum_actor_active.append(actor_active_buf.reshape(-1).clone())
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
            b_actor_active = torch.cat(accum_actor_active)
            accum_obs.clear()
            accum_act.clear()
            accum_lp.clear()
            accum_adv.clear()
            accum_ret.clear()
            accum_val.clear()
            accum_mask.clear()
            accum_actor_active.clear()
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
                    mb_actor_active = b_actor_active[idx]
                    _, new_lp, ent, new_val = agent.get_action_and_value(mb_obs, mb_mask, mb_act)
                    loss, loss_parts = compute_torch_ppo_loss(
                        new_logprob=new_lp,
                        entropy=ent,
                        new_value=new_val,
                        old_logprob=mb_lp,
                        advantages=mb_adv,
                        returns=mb_ret,
                        actor_active=mb_actor_active,
                        clip_coef=cfg["clip_coef"],
                        vf_coef=cfg["vf_coef"],
                        ent_coef=cfg["ent_coef"],
                    )
                    pg_loss = loss_parts["loss_policy"]
                    vf_loss = loss_parts["loss_value"]
                    entropy_loss = loss_parts["loss_entropy"]
                    optimizer.zero_grad()
                    loss.backward()
                    pre_clip = float(nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]))
                    post_clip = min(pre_clip, cfg["max_grad_norm"])
                    optimizer.step()
                    approx_kl = float(loss_parts["ppo_kl_divergence"].detach())
                    clipfrac = float(loss_parts["ppo_clip_fraction"].detach())
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
            is_final = total_steps >= cfg["total_timesteps"]
            previous_steps = total_steps - steps_per_update
            eval_due = checkpoint_evaluator.due(previous_steps, total_steps, final=is_final)
            periodic_checkpoint = checkpoint_every > 0 and num_updates % checkpoint_every == 0
            if periodic_checkpoint or eval_due:
                checkpoint_path = save_torch_bundle(
                    save_dir / f"checkpoint_{total_steps}.pt",
                    {
                        "blue": {
                            "weights": agent.state_dict(),
                            "obs_dim": OBS_DIM,
                            "action_dim": ACT_DIM,
                            "arch": dict(recipe["arch"]),
                            "trainable": True,
                            "source": {"kind": "fresh", "seed": int(args.seed)},
                        }
                    },
                    provenance={
                        "tag": tag,
                        "seed": args.seed,
                        "total_steps": total_steps,
                        "variant": variant.name,
                        "train_teams": ["blue"],
                        "train_run_id": train_run_id,
                    },
                )
                sidecar_path = write_sidecar(
                    save_dir / f"recipe_checkpoint_{total_steps}.yaml",
                    recipe,
                    seed=args.seed,
                    total_steps=total_steps,
                    backend="cyborg",
                    train_run_id=train_run_id,
                    extra={"train_teams": ["blue"], "model": f"checkpoint_{total_steps}.pt"},
                )
                if eval_due:
                    print(
                        f"  MLflow checkpoint evaluation at step {total_steps:,}; "
                        f"evaluating {checkpoint_evaluator.settings.episodes_per_seed} episodes per seed...",
                        flush=True,
                    )
                    try:
                        checkpoint_evaluator.on_checkpoint(
                            checkpoint_path,
                            sidecar_path,
                            env_steps=total_steps,
                            evaluate_fn=lambda episodes_per_seed: evaluate_training_checkpoint(
                                checkpoint_path,
                                backend="cyborg",
                                recipe=recipe,
                                seed=args.seed,
                                episodes_per_seed=episodes_per_seed,
                            ),
                        )
                    finally:
                        if checkpoint_every <= 0:
                            checkpoint_path.unlink(missing_ok=True)
                            sidecar_path.unlink(missing_ok=True)

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
    model_path = save_torch_bundle(
        save_dir / f"model_{tag}.pt",
        {
            "blue": {
                "weights": agent.state_dict(),
                "obs_dim": OBS_DIM,
                "action_dim": ACT_DIM,
                "arch": dict(recipe["arch"]),
                "trainable": True,
                "source": {"kind": "fresh", "seed": int(args.seed)},
            }
        },
        provenance={
            "tag": tag,
            "seed": args.seed,
            "total_steps": total_steps,
            "variant": variant.name,
            "train_teams": ["blue"],
            "train_run_id": train_run_id,
        },
    )
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
        mlflow.log_artifact(str(model_path))
        mlflow.end_run()
    except Exception as e:
        print(f"MLflow finalize warning: {e}")

    metrics_file.close()
    envs.close()
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 3600:.1f}h). Final ep reward: {final_reward:.1f}")
    print(f"Saved to: {save_dir}")
    if total_steps >= cfg["total_timesteps"]:
        from jaxborg.evaluation.post_training import run_configured_evaluations_after_training

        run_configured_evaluations_after_training(model_path, recipe)


@dataclass
class TorchTeamRuntime:
    """Independent parameter-sharing policy and PPO state for one team."""

    team: str
    cfg: dict[str, Any]
    agent: nn.Module
    arch: dict[str, Any]
    trainable: bool
    source: Any
    num_envs: int
    num_steps: int
    optimizer: optim.Optimizer | None = None
    reward_scaler: RewardScaler | None = None
    rollout: dict[str, torch.Tensor] = field(default_factory=dict)
    accumulated: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    episode_rewards: np.ndarray = field(default_factory=lambda: np.empty(0))
    completed_rewards: list[float] = field(default_factory=list)

    def __post_init__(self):
        spec = TEAM_SPECS[self.team]
        self.agent_ids = spec.agent_ids
        self.obs_dim = spec.obs_dim
        self.action_dim = spec.action_dim
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        if not self.trainable:
            return
        self.rollout = {
            "obs": torch.zeros((self.num_steps, self.num_envs, len(self.agent_ids), self.obs_dim)),
            "actions": torch.zeros((self.num_steps, self.num_envs, len(self.agent_ids)), dtype=torch.long),
            "logprobs": torch.zeros((self.num_steps, self.num_envs, len(self.agent_ids))),
            "rewards": torch.zeros((self.num_steps, self.num_envs)),
            "dones": torch.zeros((self.num_steps, self.num_envs, len(self.agent_ids))),
            "values": torch.zeros((self.num_steps, self.num_envs, len(self.agent_ids))),
            # Bool masks materially reduce the Red buffer footprint (1,106 actions).
            "masks": torch.zeros(
                (self.num_steps, self.num_envs, len(self.agent_ids), self.action_dim),
                dtype=torch.bool,
            ),
            "actor_active": torch.zeros(
                (self.num_steps, self.num_envs, len(self.agent_ids)),
                dtype=torch.bool,
            ),
            "critic_active": torch.zeros(
                (self.num_steps, self.num_envs, len(self.agent_ids)),
                dtype=torch.bool,
            ),
        }
        self.accumulated = {
            key: []
            for key in (
                "obs",
                "actions",
                "logprobs",
                "advantages",
                "returns",
                "values",
                "masks",
                "actor_active",
                "critic_active",
            )
        }

    def select_actions(self, step: int, all_obs: list[dict], all_info: list[dict]) -> np.ndarray:
        obs = np.stack(
            [[all_obs[env_idx][agent] for agent in self.agent_ids] for env_idx in range(self.num_envs)]
        ).astype(np.float32)
        masks = np.stack(
            [[all_info[env_idx][agent]["action_mask"] for agent in self.agent_ids] for env_idx in range(self.num_envs)]
        ).astype(bool)
        actor_active = np.asarray(
            [
                [all_info[env_idx][agent]["actor_active"] for agent in self.agent_ids]
                for env_idx in range(self.num_envs)
            ],
            dtype=bool,
        )
        critic_active = np.asarray(
            [
                [all_info[env_idx][agent]["critic_active"] for agent in self.agent_ids]
                for env_idx in range(self.num_envs)
            ],
            dtype=bool,
        )
        obs_t = torch.from_numpy(obs)
        masks_t = torch.from_numpy(masks)
        with torch.no_grad():
            action, logprob, _entropy, value = self.agent.get_action_and_value(
                obs_t.reshape(-1, self.obs_dim),
                masks_t.reshape(-1, self.action_dim),
            )
            action = action.reshape(self.num_envs, len(self.agent_ids))

        if self.trainable:
            self.rollout["obs"][step].copy_(obs_t)
            self.rollout["masks"][step].copy_(masks_t)
            self.rollout["actor_active"][step].copy_(torch.from_numpy(actor_active))
            self.rollout["critic_active"][step].copy_(torch.from_numpy(critic_active))
            self.rollout["actions"][step].copy_(action)
            self.rollout["logprobs"][step].copy_(logprob.reshape(self.num_envs, len(self.agent_ids)))
            self.rollout["values"][step].copy_(value.reshape(self.num_envs, len(self.agent_ids)))
        return action.numpy()

    def record_rewards(
        self,
        step: int,
        rewards: list[dict],
        dones: np.ndarray,
        next_info: list[dict],
    ) -> None:
        raw = np.asarray([reward[self.agent_ids[0]] for reward in rewards], dtype=np.float64)
        self.episode_rewards += raw
        for env_idx, done in enumerate(dones):
            if done:
                self.completed_rewards.append(float(self.episode_rewards[env_idx]))
                self.episode_rewards[env_idx] = 0.0
        if not self.trainable:
            return
        scaled = self.reward_scaler.scale(raw, dones) if self.reward_scaler is not None else raw
        self.rollout["rewards"][step].copy_(torch.from_numpy(scaled.astype(np.float32)))
        active_after = np.asarray(
            [
                [next_info[env_idx][agent]["critic_active"] for agent in self.agent_ids]
                for env_idx in range(self.num_envs)
            ],
            dtype=bool,
        )
        terminal = dones[:, None] | ~active_after
        self.rollout["dones"][step].copy_(torch.from_numpy(terminal.astype(np.float32)))

    def finish_rollout(self, all_obs: list[dict], all_info: list[dict]) -> None:
        if not self.trainable:
            return
        next_obs = np.stack(
            [[all_obs[env_idx][agent] for agent in self.agent_ids] for env_idx in range(self.num_envs)]
        ).astype(np.float32)
        with torch.no_grad():
            next_value = self.agent.get_value(torch.from_numpy(next_obs).reshape(-1, self.obs_dim))
            next_value = next_value.reshape(self.num_envs, len(self.agent_ids))
            next_active = np.asarray(
                [
                    [all_info[env_idx][agent]["critic_active"] for agent in self.agent_ids]
                    for env_idx in range(self.num_envs)
                ],
                dtype=bool,
            )
            next_value = next_value * torch.from_numpy(next_active.astype(np.float32))

        advantages = torch.zeros_like(self.rollout["values"])
        last_gae = torch.zeros((self.num_envs, len(self.agent_ids)))
        for t in reversed(range(self.num_steps)):
            nonterminal = 1.0 - self.rollout["dones"][t]
            following_value = next_value if t == self.num_steps - 1 else self.rollout["values"][t + 1]
            reward = self.rollout["rewards"][t].unsqueeze(-1).expand_as(self.rollout["values"][t])
            delta = reward + self.cfg["gamma"] * following_value * nonterminal - self.rollout["values"][t]
            last_gae = delta + self.cfg["gamma"] * self.cfg["gae_lambda"] * nonterminal * last_gae
            last_gae = last_gae * self.rollout["critic_active"][t]
            advantages[t] = last_gae
        returns = advantages + self.rollout["values"]

        flat_shapes = {
            "obs": (-1, self.obs_dim),
            "actions": (-1,),
            "logprobs": (-1,),
            "advantages": (-1,),
            "returns": (-1,),
            "values": (-1,),
            "masks": (-1, self.action_dim),
            "actor_active": (-1,),
            "critic_active": (-1,),
        }
        values = {**self.rollout, "advantages": advantages, "returns": returns}
        for key, shape in flat_shapes.items():
            self.accumulated[key].append(values[key].reshape(shape).clone())

    def clear_accumulated(self) -> None:
        for values in self.accumulated.values():
            values.clear()


def compute_torch_ppo_loss(
    *,
    new_logprob: torch.Tensor,
    entropy: torch.Tensor,
    new_value: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    actor_active: torch.Tensor,
    critic_active: torch.Tensor | None = None,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO objective with actor-only active/busy filtering.

    Value loss includes active rows even while busy, so delayed actions retain
    credit, while inactive Red rows are excluded from both objectives.
    """

    valid = actor_active.bool()
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()
    if valid.any():
        valid_adv = advantages[valid]
        normalised = (valid_adv - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)
        valid_ratio = ratio[valid]
        pg_unclipped = -normalised * valid_ratio
        pg_clipped = -normalised * torch.clamp(valid_ratio, 1 - clip_coef, 1 + clip_coef)
        policy_loss = torch.maximum(pg_unclipped, pg_clipped).mean()
        entropy_loss = entropy[valid].mean()
        approx_kl = ((valid_ratio - 1) - logratio[valid]).mean()
        clip_fraction = ((valid_ratio - 1).abs() > clip_coef).float().mean()
    else:
        # Retain a differentiable zero if an unusual minibatch is entirely busy.
        policy_loss = new_logprob.sum() * 0.0
        entropy_loss = entropy.sum() * 0.0
        approx_kl = ratio.sum() * 0.0
        clip_fraction = ratio.sum() * 0.0
    critic_valid = torch.ones_like(actor_active, dtype=torch.bool) if critic_active is None else critic_active.bool()
    if critic_valid.any():
        value_loss = 0.5 * ((new_value[critic_valid] - returns[critic_valid]) ** 2).mean()
    else:
        value_loss = new_value.sum() * 0.0
    total_loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss
    return total_loss, {
        "loss_policy": policy_loss,
        "loss_value": value_loss,
        "loss_entropy": entropy_loss,
        "loss_total": total_loss,
        "ppo_kl_divergence": approx_kl,
        "ppo_clip_fraction": clip_fraction,
    }


def _ppo_update(runtime: TorchTeamRuntime, update_idx: int, total_updates: int) -> dict[str, float]:
    cfg = runtime.cfg
    batches = {key: torch.cat(value) for key, value in runtime.accumulated.items()}
    runtime.clear_accumulated()
    assert runtime.optimizer is not None

    lr = cfg["lr"]
    if cfg["anneal_lr"]:
        frac = 1.0 - (update_idx - 1) / max(total_updates, 1)
        lr = max(frac * cfg["lr"], 1e-6)
        for group in runtime.optimizer.param_groups:
            group["lr"] = lr

    totals = {
        "loss_policy": 0.0,
        "loss_value": 0.0,
        "loss_entropy": 0.0,
        "loss_total": 0.0,
        "ppo_kl_divergence": 0.0,
        "ppo_clip_fraction": 0.0,
        "ppo_grad_norm": 0.0,
        "ppo_pre_clip_grad_norm": 0.0,
    }
    count = 0
    total_rows = batches["obs"].shape[0]
    num_minibatches = min(cfg["num_minibatches"], total_rows)
    for _epoch in range(cfg["num_epochs"]):
        for idx in torch.tensor_split(torch.randperm(total_rows), num_minibatches):
            _, new_logprob, entropy, new_value = runtime.agent.get_action_and_value(
                batches["obs"][idx],
                batches["masks"][idx],
                batches["actions"][idx],
            )
            loss, parts = compute_torch_ppo_loss(
                new_logprob=new_logprob,
                entropy=entropy,
                new_value=new_value,
                old_logprob=batches["logprobs"][idx],
                advantages=batches["advantages"][idx],
                returns=batches["returns"][idx],
                actor_active=batches["actor_active"][idx],
                critic_active=batches["critic_active"][idx],
                clip_coef=cfg["clip_coef"],
                vf_coef=cfg["vf_coef"],
                ent_coef=cfg["ent_coef"],
            )
            runtime.optimizer.zero_grad()
            loss.backward()
            pre_clip = float(nn.utils.clip_grad_norm_(runtime.agent.parameters(), cfg["max_grad_norm"]))
            runtime.optimizer.step()
            for key in (
                "loss_policy",
                "loss_value",
                "loss_entropy",
                "loss_total",
                "ppo_kl_divergence",
                "ppo_clip_fraction",
            ):
                totals[key] += float(parts[key].detach())
            totals["ppo_pre_clip_grad_norm"] += pre_clip
            totals["ppo_grad_norm"] += min(pre_clip, cfg["max_grad_norm"])
            count += 1

    for key in totals:
        totals[key] /= max(count, 1)
    critic_valid = batches["critic_active"].bool()
    returns = batches["returns"][critic_valid]
    old_values = batches["values"][critic_valid]
    target_var = returns.var(unbiased=False)
    explained = 0.0
    if target_var > 1e-8:
        explained = float(1 - (returns - old_values).var(unbiased=False) / (target_var + 1e-8))
    totals["ppo_explained_variance"] = explained
    totals["lr"] = float(lr)
    return totals


def _frozen_arch(model_path: Path, entry, fallback_recipe: dict, team: str) -> dict[str, Any]:
    if entry.arch.get("name"):
        return dict(entry.arch)
    if team != "blue":
        raise ValueError(f"legacy unversioned Torch models are Blue-only: {model_path}")
    try:
        return dict(read_sidecar(model_path)["arch"])
    except (FileNotFoundError, KeyError):
        # Preserve the old Blue loader convention for historical bare weights.
        return dict(team_recipe(fallback_recipe, "blue")["arch"])


def _make_joint_runtimes(
    recipe: dict,
    cfg: dict,
    *,
    seed: int,
    num_envs: int,
    num_steps: int,
) -> dict[str, TorchTeamRuntime]:
    trainable_teams = set(training_teams(recipe))
    opponent_paths = resolve_train_opponents(recipe, backend="cyborg", exp_dir=EXP_DIR)
    runtimes: dict[str, TorchTeamRuntime] = {}
    for team in ("blue", "red"):
        spec = TEAM_SPECS[team]
        team_cfg = project_cleanrl(recipe, team=team)
        for shared_key in ("num_envs", "rollout_length", "num_rollouts_per_update", "total_timesteps"):
            team_cfg[shared_key] = cfg[shared_key]
        trainable = team in trainable_teams
        source = None
        if trainable:
            arch = dict(team_recipe(recipe, team)["arch"])
            source = {"kind": "fresh", "seed": int(seed)}
        else:
            path = opponent_paths.get(team)
            if path is None:
                raise ValueError(f"learned-policy CybORG training requires a frozen {team} opponent")
            entry = load_torch_policy(
                path,
                team,
                expected_obs_dim=spec.obs_dim,
                expected_action_dim=spec.action_dim,
            )
            arch = _frozen_arch(path, entry, recipe, team)
            source = {"path": str(path), "bundle_source": entry.source}

        agent = make_torch_policy(
            arch["name"],
            obs_dim=spec.obs_dim,
            action_dim=spec.action_dim,
            hidden_dim=int(arch.get("hidden_dim", 256)),
            hidden_layers=int(arch.get("hidden_layers", 2)),
        )
        optimizer = optim.Adam(agent.parameters(), lr=team_cfg["lr"], eps=1e-5) if trainable else None
        if not trainable:
            agent.load_state_dict(entry.weights)
            agent.eval()
            for parameter in agent.parameters():
                parameter.requires_grad_(False)
        scaler = RewardScaler(num_envs, team_cfg["gamma"]) if trainable and team_cfg["norm_rewards"] else None
        runtimes[team] = TorchTeamRuntime(
            team=team,
            cfg=team_cfg,
            agent=agent,
            arch=arch,
            trainable=trainable,
            source=source,
            num_envs=num_envs,
            num_steps=num_steps,
            optimizer=optimizer,
            reward_scaler=scaler,
        )
    return runtimes


def _save_joint_bundle(path: Path, runtimes: dict[str, TorchTeamRuntime], *, provenance: dict) -> Path:
    policies = {
        team: {
            "weights": runtime.agent.state_dict(),
            "obs_dim": runtime.obs_dim,
            "action_dim": runtime.action_dim,
            "arch": runtime.arch,
            "trainable": runtime.trainable,
            "source": runtime.source,
        }
        for team, runtime in runtimes.items()
    }
    return save_torch_bundle(path, policies, provenance=provenance)


def train_joint(args, recipe, cfg):
    """Train Red, Blue, or both against learned policies in real CybORG."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = args.tag or f"{recipe['meta']['name']}_seed{args.seed}"
    save_dir = EXP_DIR / "ippo_cyborg" / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    num_envs = cfg["num_envs"]
    num_steps = cfg["rollout_length"]
    variant: GameVariant = cfg["TRAIN_VARIANT"]
    runtimes = _make_joint_runtimes(
        recipe,
        cfg,
        seed=args.seed,
        num_envs=num_envs,
        num_steps=num_steps,
    )
    trainable = {team: runtime for team, runtime in runtimes.items() if runtime.trainable}
    envs = ParallelJointEnvs(num_envs, args.seed, variant)

    checkpoint_evaluator = MlflowCheckpointEvaluator(recipe)
    run = start_run(recipe, backend="cyborg", seed=args.seed)
    train_run_id = run.info.run_id
    metrics_path = save_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")
    all_obs, all_info = envs.reset()
    episode_lengths = np.zeros(num_envs, dtype=int)
    completed_lengths: list[int] = []

    total_steps = 0
    update_idx = 0
    rollouts_collected = 0
    steps_per_update = num_envs * num_steps * cfg["num_rollouts_per_update"]
    total_updates = max(1, cfg["total_timesteps"] // steps_per_update)
    ckpt_every = int(recipe.get("cleanrl", {}).get("checkpoint_every_updates", 50))
    start = time.perf_counter()
    print(
        f"Creating {num_envs} joint CybORG envs (variant={variant.name}, train={','.join(trainable)})...",
        flush=True,
    )

    try:
        while total_steps < cfg["total_timesteps"]:
            for step in range(num_steps):
                selected = {team: runtime.select_actions(step, all_obs, all_info) for team, runtime in runtimes.items()}
                action_dicts = []
                for env_idx in range(num_envs):
                    joint_actions: dict[str, int] = {}
                    for team, runtime in runtimes.items():
                        joint_actions.update(
                            {
                                agent: int(selected[team][env_idx, agent_idx])
                                for agent_idx, agent in enumerate(runtime.agent_ids)
                            }
                        )
                    action_dicts.append(joint_actions)
                if set(action_dicts[0]) != set(POLICY_AGENT_IDS):
                    raise RuntimeError("joint trainer did not select all 11 actions")

                all_obs, all_rewards, all_done, all_info = envs.step(action_dicts)
                dones = np.asarray(all_done, dtype=bool)
                for runtime in runtimes.values():
                    runtime.record_rewards(step, all_rewards, dones, all_info)
                episode_lengths += 1
                for env_idx, done in enumerate(dones):
                    if done:
                        completed_lengths.append(int(episode_lengths[env_idx]))
                        episode_lengths[env_idx] = 0
                total_steps += num_envs

            for runtime in trainable.values():
                runtime.finish_rollout(all_obs, all_info)
            rollouts_collected += 1
            if rollouts_collected < cfg["num_rollouts_per_update"]:
                continue
            rollouts_collected = 0
            update_idx += 1
            stats = {team: _ppo_update(runtime, update_idx, total_updates) for team, runtime in trainable.items()}

            elapsed = time.perf_counter() - start
            sps = total_steps / elapsed if elapsed else 0.0
            primary_stats = stats.get("blue", stats["red"])
            blue_rewards = runtimes["blue"].completed_rewards
            blue_return = float(np.mean(blue_rewards[-50:])) if blue_rewards else float("nan")
            ep_len = float(np.mean(completed_lengths[-50:])) if completed_lengths else float("nan")
            row = make_row(
                update_idx=update_idx,
                env_steps=total_steps,
                wall_time_s=elapsed,
                throughput_sps=sps,
                loss_policy=primary_stats["loss_policy"],
                loss_value=primary_stats["loss_value"],
                loss_entropy=primary_stats["loss_entropy"],
                loss_total=primary_stats["loss_total"],
                ppo_kl_divergence=primary_stats["ppo_kl_divergence"],
                ppo_clip_fraction=primary_stats["ppo_clip_fraction"],
                ppo_explained_variance=primary_stats["ppo_explained_variance"],
                lr=primary_stats["lr"],
                train_episode_reward_mean=blue_return if not np.isnan(blue_return) else None,
                train_episode_length_mean=ep_len if not np.isnan(ep_len) else None,
                ppo_grad_norm=primary_stats["ppo_grad_norm"],
                ppo_pre_clip_grad_norm=primary_stats["ppo_pre_clip_grad_norm"],
                backend_extras={
                    "cyborg.episodes_completed": len(completed_lengths),
                    "cyborg.train_teams": ",".join(training_teams(recipe)),
                },
            )
            for team, runtime in runtimes.items():
                team_metrics = dict(stats.get(team, {}))
                recent = runtime.completed_rewards[-50:]
                if recent:
                    team_metrics["train_episode_reward_mean"] = float(np.mean(recent))
                add_team_metrics(row, team, team_metrics)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            try:
                mlflow.log_metrics(
                    {key: float(value) for key, value in row.items() if isinstance(value, (int, float))},
                    step=total_steps,
                )
            except Exception:
                pass
            print(
                f"  upd {update_idx:4d} | steps {total_steps:>9,} | blue {blue_return:>8.1f} | "
                + " ".join(f"{team}:pg={value['loss_policy']:.4f}" for team, value in stats.items())
                + f" | {sps:.0f} sps",
                flush=True,
            )
            is_final = total_steps >= cfg["total_timesteps"]
            previous_steps = total_steps - steps_per_update
            eval_due = checkpoint_evaluator.due(previous_steps, total_steps, final=is_final)
            periodic_checkpoint = ckpt_every > 0 and update_idx % ckpt_every == 0
            if periodic_checkpoint or eval_due:
                checkpoint_path = _save_joint_bundle(
                    save_dir / f"checkpoint_{total_steps}.pt",
                    runtimes,
                    provenance={
                        "tag": tag,
                        "seed": args.seed,
                        "total_steps": total_steps,
                        "variant": variant.name,
                        "train_teams": list(training_teams(recipe)),
                        "train_run_id": train_run_id,
                    },
                )
                sidecar_path = write_sidecar(
                    save_dir / f"recipe_checkpoint_{total_steps}.yaml",
                    recipe,
                    seed=args.seed,
                    total_steps=total_steps,
                    backend="cyborg",
                    train_run_id=train_run_id,
                    extra={
                        "train_teams": list(training_teams(recipe)),
                        "model": checkpoint_path.name,
                    },
                )
                if eval_due:
                    print(
                        f"  MLflow checkpoint evaluation at step {total_steps:,}; "
                        f"evaluating {checkpoint_evaluator.settings.episodes_per_seed} episodes per seed...",
                        flush=True,
                    )
                    try:
                        checkpoint_evaluator.on_checkpoint(
                            checkpoint_path,
                            sidecar_path,
                            env_steps=total_steps,
                            evaluate_fn=lambda episodes_per_seed: evaluate_training_checkpoint(
                                checkpoint_path,
                                backend="cyborg",
                                recipe=recipe,
                                seed=args.seed,
                                episodes_per_seed=episodes_per_seed,
                            ),
                        )
                    finally:
                        if ckpt_every <= 0:
                            checkpoint_path.unlink(missing_ok=True)
                            sidecar_path.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
    finally:
        envs.close()

    elapsed = time.perf_counter() - start
    model_path = _save_joint_bundle(
        save_dir / f"model_{tag}.pt",
        runtimes,
        provenance={
            "tag": tag,
            "seed": args.seed,
            "total_steps": total_steps,
            "variant": variant.name,
            "train_teams": list(training_teams(recipe)),
            "train_run_id": train_run_id,
        },
    )
    full_checkpoint = {
        "schema_version": 1,
        "total_steps": total_steps,
        "num_updates": update_idx,
        "policies": {team: runtime.agent.state_dict() for team, runtime in runtimes.items()},
        "optimizers": {
            team: runtime.optimizer.state_dict() for team, runtime in trainable.items() if runtime.optimizer is not None
        },
        "reward_scalers": {
            team: {
                "returns": runtime.reward_scaler.returns,
                "mean": runtime.reward_scaler.mean,
                "var": runtime.reward_scaler.var,
                "count": runtime.reward_scaler.count,
            }
            for team, runtime in trainable.items()
            if runtime.reward_scaler is not None
        },
    }
    torch.save(full_checkpoint, save_dir / f"checkpoint_{tag}.pt")
    sidecar = write_sidecar(
        save_dir / f"recipe_{tag}.yaml",
        recipe,
        seed=args.seed,
        total_steps=total_steps,
        backend="cyborg",
        train_run_id=train_run_id,
        extra={"train_teams": list(training_teams(recipe))},
    )
    try:
        mlflow.log_metrics({"final_wall_time_sec": elapsed, "final_steps_per_second": total_steps / max(elapsed, 1e-8)})
        for artifact in (metrics_path, sidecar, model_path):
            mlflow.log_artifact(str(artifact))
        mlflow.end_run()
    except Exception as exc:
        print(f"MLflow finalize warning: {exc}")
    metrics_file.close()
    print(f"\nDone in {elapsed:.1f}s. Saved joint bundle to: {model_path}")
    if total_steps >= cfg["total_timesteps"]:
        from jaxborg.evaluation.post_training import run_configured_evaluations_after_training

        run_configured_evaluations_after_training(model_path, recipe)


def train(args, recipe, cfg):
    """Preserve the legacy Blue-vs-FSM path unless a learned matchup is requested."""

    teams = training_teams(recipe)
    opponents = recipe.get("train", {}).get("opponents") or {}
    if teams == ("blue",) and "red" not in opponents:
        return train_legacy(args, recipe, cfg)
    return train_joint(args, recipe, cfg)


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

    recipe = copy.deepcopy(load_recipe(args.recipe))
    if args.total_timesteps is not None:
        recipe["train"]["total_timesteps"] = int(args.total_timesteps)
    if args.num_envs is not None:
        recipe.setdefault("cleanrl", {})["num_envs"] = int(args.num_envs)
    if args.num_rollouts_per_update is not None:
        recipe.setdefault("cleanrl", {})["num_rollouts_per_update"] = int(args.num_rollouts_per_update)

    cfg = project_cleanrl(recipe)
    if args.num_envs is not None and args.num_rollouts_per_update is None:
        per_rollout = cfg["num_envs"] * cfg["rollout_length"]
        cfg["num_rollouts_per_update"] = max(1, (recipe["train"]["buffer_size"] + per_rollout - 1) // per_rollout)
        recipe.setdefault("cleanrl", {})["num_rollouts_per_update"] = cfg["num_rollouts_per_update"]

    train(args, recipe, cfg)


if __name__ == "__main__":
    main()
