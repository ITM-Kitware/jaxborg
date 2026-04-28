"""Pure-CybORG PPO baseline for CC4 with action masking.

Goal: produce the simplest credible CPU-only baseline for CC4, so JaxBorg's
GPU speedup and policy quality can be compared against a real reference point.
Not chasing SOTA — establishing a trustworthy, reproducible number.

Written in CleanRL style (single-file, minimal dependencies, readable).
Uses full parameter sharing: a single policy for all 5 agents. Agents 0-3
(single-subnet) have their observations and action masks zero-padded to match
agent 4's dimensions. Action masking prevents invalid actions.

Hyperparameter sources:
- Singh et al. (AAMAS 2025): LR=5e-5, minibatch=32768, SGD_iters=30, net=[256,256]
- CC4 Tutorial: gamma=0.85
- Cybermonic KEEP (CC4 competition): actor_lr=3e-4, critic_lr=1e-3, batch=2500, epochs=4

Key design choices:
- Reward scaling via running return std (critical for CC4's large negative rewards)
- Action masking via logit masking (set invalid logits to -1e8)
- GAE with proper episode boundary handling
- Multiple parallel CybORG environments via multiprocessing
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_cleanrl_agent import PPOAgent  # noqa: E402

from jaxborg.constants import BLUE_OBS_SIZE  # noqa: E402

# ── Environment setup ──────────────────────────────────────────────────────

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

# Unified obs/act dims (agent 4's sizes — agents 0-3 are zero-padded to match)
OBS_DIM = BLUE_OBS_SIZE  # 210
ACT_DIM = 242


def make_cyborg_env():
    """Create a fresh CybORG CC4 environment with EnterpriseMAE wrapper."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import EnterpriseMAE
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(cyborg)


# ── Parallel env worker ───────────────────────────────────────────────────


def env_worker(pipe, env_id):
    """Worker process running a single CybORG environment."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = make_cyborg_env()

    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break
        if cmd == "reset":
            obs, info = env.reset()
            pipe.send((obs, info))
        elif cmd == "step":
            obs, rew, term, trunc, info = env.step(data)
            done = any(term.values()) or any(trunc.values())
            if done:
                # Auto-reset, return the fresh obs but keep the terminal reward
                obs, info = env.reset()
            pipe.send((obs, rew, done, info))
        elif cmd == "close":
            pipe.close()
            break


class ParallelEnvs:
    """Manages multiple CybORG environments in separate processes."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.pipes = []
        self.procs = []

        for i in range(num_envs):
            parent_pipe, child_pipe = Pipe()
            proc = Process(target=env_worker, args=(child_pipe, i), daemon=True)
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.procs.append(proc)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.pipes]
        all_obs = [r[0] for r in results]
        all_info = [r[1] for r in results]
        return all_obs, all_info

    def step(self, actions_list):
        for pipe, actions in zip(self.pipes, actions_list):
            pipe.send(("step", actions))
        results = [pipe.recv() for pipe in self.pipes]
        all_obs = [r[0] for r in results]
        all_rew = [r[1] for r in results]
        all_done = [r[2] for r in results]
        all_info = [r[3] for r in results]
        return all_obs, all_rew, all_done, all_info

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


# ── Reward normalizer ─────────────────────────────────────────────────────


class RewardScaler:
    """Scale rewards by running std of discounted returns.

    This is the approach used by OpenAI baselines and many successful PPO
    implementations. It divides rewards by sqrt(Var[returns]) which keeps
    the value function targets in a reasonable range.
    """

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
        """Scale rewards by running return std."""
        self.returns = self.returns * self.gamma + rewards
        self._update_stats(self.returns)
        scaled = rewards / (np.sqrt(self.var) + 1e-8)
        scaled = np.clip(scaled, -self.clip, self.clip)
        self.returns[dones] = 0.0
        return scaled


# ── Training loop ─────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create parallel environments
    print(f"Creating {args.num_envs} parallel CybORG environments...", flush=True)
    envs = ParallelEnvs(args.num_envs)

    # Single shared policy for all 5 agents (obs/masks zero-padded to max dims)
    agent = PPOAgent(OBS_DIM, ACT_DIM, hidden_dims=(256, 256)).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Reward scaler
    reward_scaler = RewardScaler(args.num_envs, args.gamma) if args.norm_rewards else None

    # Resume from checkpoint
    resumed_steps = 0
    resumed_updates = 0
    if args.resume_tag:
        ckpt_dir = EXP_DIR / "cleanrl_ppo"
        ckpt_path = ckpt_dir / f"checkpoint_{args.resume_tag}.pt"
        if ckpt_path.exists():
            print(f"Resuming from full checkpoint {ckpt_path}...", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.load_state_dict(ckpt["agent"])
            optimizer.load_state_dict(ckpt["optimizer"])
            resumed_steps = ckpt["total_steps"]
            resumed_updates = ckpt["num_updates"]
            if reward_scaler is not None and "reward_scaler" in ckpt:
                rs = ckpt["reward_scaler"]
                reward_scaler.returns = rs["returns"]
                reward_scaler.mean = rs["mean"]
                reward_scaler.var = rs["var"]
                reward_scaler.count = rs["count"]
        else:
            # Fall back to bare model weights (no optimizer/scaler state)
            model_path = ckpt_dir / f"model_{args.resume_tag}.pt"
            print(f"No full checkpoint; loading bare weights from {model_path}...", flush=True)
            agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            # Estimate resumed steps from metrics file
            metrics_src = ckpt_dir / f"metrics_{args.resume_tag}.jsonl"
            if metrics_src.exists():
                last_line = metrics_src.read_text().strip().split("\n")[-1]
                last = json.loads(last_line)
                resumed_steps = last["steps"]
                resumed_updates = last["update"]
        print(f"  Resumed at step {resumed_steps:,}, update {resumed_updates}", flush=True)

    # Rollout storage
    num_steps = args.rollout_length

    # Storage arrays — all 5 agents use unified dims
    obs_buf = torch.zeros((num_steps, args.num_envs, NUM_AGENTS, OBS_DIM))
    actions_buf = torch.zeros((num_steps, args.num_envs, NUM_AGENTS), dtype=torch.long)
    logprobs_buf = torch.zeros((num_steps, args.num_envs, NUM_AGENTS))
    rewards_all = torch.zeros((num_steps, args.num_envs))  # shared reward (same for all agents)
    dones_all = torch.zeros((num_steps, args.num_envs))
    values_buf = torch.zeros((num_steps, args.num_envs, NUM_AGENTS))
    masks_buf = torch.zeros((num_steps, args.num_envs, NUM_AGENTS, ACT_DIM))

    # MLflow setup
    mlflow_db = EXP_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("cleanrl-ppo-baseline")
    run_name = f"cleanrl-ppo-{args.tag}" if args.tag else "cleanrl-ppo"
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "algorithm": "CleanRL-PPO-IPPO-Masked",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "rollout_length": args.rollout_length,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "num_epochs": args.num_epochs,
            "num_minibatches": args.num_minibatches,
            "clip_coef": args.clip_coef,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "norm_rewards": args.norm_rewards,
            "network": "[256, 256]",
            "shared_policy": "all_agents_shared",
            "anneal_lr": args.anneal_lr,
            "num_rollouts_per_update": args.num_rollouts_per_update,
        }
    )

    # Metrics file
    save_dir = EXP_DIR / "cleanrl_ppo"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / f"metrics_{args.tag or 'default'}.jsonl"
    metrics_file = open(metrics_path, "a" if args.resume_tag else "w")

    # Initial reset
    all_obs, all_info = envs.reset()

    # Track episode stats
    episode_rewards = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    completed_rewards = []
    completed_lengths = []

    start_time = time.perf_counter()
    total_steps = resumed_steps
    num_updates = resumed_updates
    rollouts_collected = 0
    accum_obs, accum_act, accum_lp = [], [], []
    accum_adv, accum_ret, accum_val, accum_mask = [], [], [], []

    steps_per_update = args.num_envs * args.rollout_length * args.num_rollouts_per_update
    total_updates = args.total_timesteps // steps_per_update
    # Agent-level batch size for SGD (accounts for accumulation)
    agent_batch = num_steps * args.num_envs * NUM_AGENTS * args.num_rollouts_per_update
    mb_size = agent_batch // args.num_minibatches

    print(f"\n{'=' * 70}")
    print("CleanRL PPO for CybORG CC4")
    print(f"{'=' * 70}")
    print(f"  Envs: {args.num_envs}")
    print(f"  Rollout length: {args.rollout_length} (steps per env)")
    print(f"  Rollouts per update: {args.num_rollouts_per_update}")
    print(f"  Steps per update: {steps_per_update:,} (env steps)")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Total updates: {total_updates}")
    print(f"  Agent batch: {agent_batch:,} -> {mb_size:,} per minibatch")
    print(f"  LR: {args.lr}, Gamma: {args.gamma}, GAE Lambda: {args.gae_lambda}")
    print(f"  Epochs: {args.num_epochs}, Clip: {args.clip_coef}")
    print(f"  Ent coef: {args.ent_coef}, VF coef: {args.vf_coef}")
    print(f"  Reward normalization: {args.norm_rewards}")
    print(f"  LR annealing: {args.anneal_lr}")
    print(f"{'=' * 70}\n", flush=True)

    try:
        while total_steps < args.total_timesteps:
            # ── Collect rollout ──────────────────────────────────────
            for step in range(num_steps):
                # Store observations and masks (zero-padded to unified dims)
                for env_idx in range(args.num_envs):
                    for i in range(NUM_AGENTS):
                        agent_id = AGENT_IDS[i]
                        raw_obs = all_obs[env_idx][agent_id].astype(np.float32)
                        raw_mask = np.array(all_info[env_idx][agent_id]["action_mask"], dtype=np.float32)
                        # Zero-pad shorter obs/masks to unified dims
                        obs_buf[step, env_idx, i, : len(raw_obs)] = torch.from_numpy(raw_obs)
                        obs_buf[step, env_idx, i, len(raw_obs) :] = 0.0
                        masks_buf[step, env_idx, i, : len(raw_mask)] = torch.from_numpy(raw_mask)
                        masks_buf[step, env_idx, i, len(raw_mask) :] = 0.0

                # Get actions from single policy
                with torch.no_grad():
                    obs_flat = obs_buf[step].reshape(-1, OBS_DIM)
                    mask_flat = masks_buf[step].reshape(-1, ACT_DIM)
                    act, lp, _, val = agent.get_action_and_value(obs_flat, mask_flat)
                    actions_buf[step] = act.reshape(args.num_envs, NUM_AGENTS)
                    logprobs_buf[step] = lp.reshape(args.num_envs, NUM_AGENTS)
                    values_buf[step] = val.reshape(args.num_envs, NUM_AGENTS)

                # Build action dicts and step environments
                action_dicts = []
                for env_idx in range(args.num_envs):
                    ad = {}
                    for i in range(NUM_AGENTS):
                        ad[AGENT_IDS[i]] = int(actions_buf[step, env_idx, i].item())
                    action_dicts.append(ad)

                all_obs, all_rew, all_done, all_info = envs.step(action_dicts)

                # Process rewards - use the shared team reward (same for all agents)
                raw_rewards = np.zeros(args.num_envs)
                dones = np.array(all_done, dtype=bool)
                for env_idx in range(args.num_envs):
                    raw_rewards[env_idx] = all_rew[env_idx][AGENT_IDS[0]]

                # Track episode stats
                episode_rewards += raw_rewards
                episode_lengths += 1
                for env_idx in range(args.num_envs):
                    if dones[env_idx]:
                        completed_rewards.append(episode_rewards[env_idx])
                        completed_lengths.append(episode_lengths[env_idx])
                        episode_rewards[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                # Scale rewards if requested
                if reward_scaler is not None:
                    scaled = reward_scaler.scale(raw_rewards, dones)
                    rewards_all[step] = torch.from_numpy(scaled.astype(np.float32))
                else:
                    rewards_all[step] = torch.from_numpy(raw_rewards.astype(np.float32))

                dones_all[step] = torch.from_numpy(dones.astype(np.float32))
                total_steps += args.num_envs

            # ── Compute advantages (GAE) ─────────────────────────────
            with torch.no_grad():
                # Bootstrap values for last observation (zero-padded)
                next_obs_flat = torch.zeros(args.num_envs * NUM_AGENTS, OBS_DIM)
                for env_idx in range(args.num_envs):
                    for i in range(NUM_AGENTS):
                        raw_obs = all_obs[env_idx][AGENT_IDS[i]].astype(np.float32)
                        next_obs_flat[env_idx * NUM_AGENTS + i, : len(raw_obs)] = torch.from_numpy(raw_obs)

                next_val = agent.get_value(next_obs_flat).reshape(args.num_envs, NUM_AGENTS)

                # GAE for all agents — all use the same shared reward
                advantages = torch.zeros((num_steps, args.num_envs, NUM_AGENTS))
                lastgaelam = torch.zeros(args.num_envs, NUM_AGENTS)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = next_val
                    else:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = values_buf[t + 1]
                    rew_expanded = rewards_all[t].unsqueeze(-1).expand_as(values_buf[t])
                    delta = rew_expanded + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values_buf

            # ── Accumulate rollout data ──────────────────────────────
            accum_obs.append(obs_buf.reshape(-1, OBS_DIM).clone())
            accum_act.append(actions_buf.reshape(-1).clone())
            accum_lp.append(logprobs_buf.reshape(-1).clone())
            accum_adv.append(advantages.reshape(-1).clone())
            accum_ret.append(returns.reshape(-1).clone())
            accum_val.append(values_buf.reshape(-1).clone())
            accum_mask.append(masks_buf.reshape(-1, ACT_DIM).clone())
            rollouts_collected += 1

            if rollouts_collected < args.num_rollouts_per_update:
                continue

            # ── PPO update ───────────────────────────────────────────
            num_updates += 1

            # LR annealing
            if args.anneal_lr:
                frac = 1.0 - (num_updates - 1) / total_updates
                lr = max(frac * args.lr, 1e-6)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = args.lr

            # Concatenate accumulated rollouts for minibatching
            b_obs = torch.cat(accum_obs)
            b_act = torch.cat(accum_act)
            b_lp = torch.cat(accum_lp)
            b_adv = torch.cat(accum_adv)
            b_ret = torch.cat(accum_ret)
            b_val = torch.cat(accum_val)
            b_mask = torch.cat(accum_mask)

            # Clear accumulators
            accum_obs.clear()
            accum_act.clear()
            accum_lp.clear()
            accum_adv.clear()
            accum_ret.clear()
            accum_val.clear()
            accum_mask.clear()
            rollouts_collected = 0

            total_n = b_obs.shape[0]
            mb_size_n = total_n // args.num_minibatches

            # Epoch metrics accumulators
            epoch_pg_loss = 0.0
            epoch_vf_loss = 0.0
            epoch_entropy = 0.0
            epoch_approx_kl = 0.0
            epoch_clipfrac = 0.0
            epoch_pre_clip_grad_norm = 0.0
            epoch_grad_norm = 0.0
            n_minibatches_total = 0

            for epoch in range(args.num_epochs):
                perm = torch.randperm(total_n)

                for mb_idx in range(args.num_minibatches):
                    idx = perm[mb_idx * mb_size_n : (mb_idx + 1) * mb_size_n]

                    mb_obs = b_obs[idx]
                    mb_act = b_act[idx]
                    mb_lp = b_lp[idx]
                    mb_adv = b_adv[idx]
                    mb_ret = b_ret[idx]
                    mb_mask = b_mask[idx]

                    # Forward pass
                    _, new_lp, ent, new_val = agent.get_action_and_value(mb_obs, mb_mask, mb_act)

                    # PPO loss. unbiased=False (ddof=0) to match jnp.std default
                    # used by `scripts/train/ippo_jax.py::_loss_fn`. Without it,
                    # advantage normalization differs by sqrt(N/(N-1)) per minibatch
                    # — see tests/differential/test_ppo_update_parity.py.
                    adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                    logratio = new_lp - mb_lp
                    ratio = logratio.exp()
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
                    entropy_loss = ent.mean()

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * vf_loss

                    optimizer.zero_grad()
                    loss.backward()
                    pre_clip_grad_norm = float(nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm))
                    # post-clip norm is implicit from `clip_grad_norm_`'s scale rule:
                    # min(pre_clip, max_grad_norm). Matches JAX's
                    # `grad_norm = pre_clip * min(1, max_norm/(pre_clip+1e-8))`.
                    post_clip_grad_norm = min(pre_clip_grad_norm, args.max_grad_norm)
                    optimizer.step()

                    # Track metrics
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                    epoch_pg_loss += pg_loss.item()
                    epoch_vf_loss += vf_loss.item()
                    epoch_entropy += entropy_loss.item()
                    epoch_approx_kl += approx_kl
                    epoch_clipfrac += clipfrac
                    epoch_pre_clip_grad_norm += pre_clip_grad_norm
                    epoch_grad_norm += post_clip_grad_norm
                    n_minibatches_total += 1

                # Early stopping on KL
                if args.target_kl is not None:
                    if epoch_approx_kl / n_minibatches_total > args.target_kl:
                        break

            # ── Logging ──────────────────────────────────────────────
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed if elapsed > 0 else 0

            avg_pg_loss = epoch_pg_loss / max(n_minibatches_total, 1)
            avg_vf_loss = epoch_vf_loss / max(n_minibatches_total, 1)
            avg_entropy = epoch_entropy / max(n_minibatches_total, 1)
            avg_approx_kl = epoch_approx_kl / max(n_minibatches_total, 1)
            avg_clipfrac = epoch_clipfrac / max(n_minibatches_total, 1)
            avg_pre_clip_grad_norm = epoch_pre_clip_grad_norm / max(n_minibatches_total, 1)
            avg_grad_norm = epoch_grad_norm / max(n_minibatches_total, 1)

            # Compute explained variance. unbiased=False (ddof=0) to match
            # JAX's `jnp.var` default — keeps the diagnostic comparable across
            # backends.
            with torch.no_grad():
                y_var = b_ret.var(unbiased=False)
                explained_var = (
                    (1 - (b_ret - b_val).var(unbiased=False) / (y_var + 1e-8)).item() if y_var > 1e-8 else 0.0
                )

                # Rollout-level reward & value/target stats (match
                # ippo_jax.py rollout_info schema field-for-field). Sourced
                # from the post-scaling `rewards_all` and post-update
                # `values_buf` / `b_ret`.
                rew_t = rewards_all  # (num_steps, num_envs) post-scaling
                std_step_reward = rew_t.std(unbiased=False).item()
                mean_abs_step_reward = rew_t.abs().mean().item()
                reward_min = rew_t.min().item()
                reward_max = rew_t.max().item()
                nonzero_reward_fraction = (rew_t != 0).float().mean().item()

                value_mean = b_val.mean().item()
                value_std = b_val.std(unbiased=False).item()
                target_mean = b_ret.mean().item()
                target_std = b_ret.std(unbiased=False).item()

            # Episode reward stats
            if completed_rewards:
                recent = completed_rewards[-50:]
                mean_ep_rew = np.mean(recent)
                mean_ep_len = np.mean(completed_lengths[-50:])
            else:
                mean_ep_rew = float("nan")
                mean_ep_len = float("nan")

            record = {
                "update": num_updates,
                "steps": total_steps,
                "wall_time_sec": round(elapsed, 1),
                "wall_time_hours": round(elapsed / 3600, 3),
                "steps_per_second": round(sps, 1),
                "episode_reward_mean": round(float(mean_ep_rew), 1),
                "episode_length_mean": round(float(mean_ep_len), 1),
                "episodes_completed": len(completed_rewards),
                "pg_loss": round(avg_pg_loss, 6),
                "vf_loss": round(avg_vf_loss, 6),
                "entropy": round(avg_entropy, 4),
                "approx_kl": round(avg_approx_kl, 6),
                "clipfrac": round(avg_clipfrac, 4),
                "explained_var": round(float(explained_var), 4),
                "lr": lr,
                # ── Parity instrumentation (matches ippo_jax.py schema) ──
                "pre_clip_grad_norm": round(float(avg_pre_clip_grad_norm), 6),
                "grad_norm": round(float(avg_grad_norm), 6),
                "std_step_reward": round(float(std_step_reward), 6),
                "mean_abs_step_reward": round(float(mean_abs_step_reward), 6),
                "reward_min": round(float(reward_min), 6),
                "reward_max": round(float(reward_max), 6),
                "nonzero_reward_fraction": round(float(nonzero_reward_fraction), 6),
                "value_mean": round(float(value_mean), 6),
                "value_std": round(float(value_std), 6),
                "target_mean": round(float(target_mean), 6),
                "target_std": round(float(target_std), 6),
            }
            if reward_scaler is not None:
                record["reward_norm_var"] = round(float(reward_scaler.var), 6)
                record["reward_norm_mean"] = round(float(reward_scaler.mean), 6)
                record["reward_norm_count"] = round(float(reward_scaler.count), 6)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            safe_metrics = {}
            for k, v in record.items():
                if isinstance(v, (int, float)) and k != "update":
                    fv = float(v)
                    if not (np.isnan(fv) or np.isinf(fv)):
                        safe_metrics[k] = fv
            try:
                mlflow.log_metrics(safe_metrics, step=total_steps)
            except Exception:
                pass

            print(
                f"  upd {num_updates:4d} | steps {total_steps:>9,} | "
                f"ep_rew {mean_ep_rew:>9.1f} | "
                f"pg {avg_pg_loss:>7.4f} | vf {avg_vf_loss:>8.4f} | "
                f"ent {avg_entropy:>5.3f} | kl {avg_approx_kl:.4f} | "
                f"clip {avg_clipfrac:.3f} | ev {explained_var:.4f} | "
                f"{sps:.0f} sps | {elapsed / 3600:.2f}h",
                flush=True,
            )

            # Periodic checkpoint
            if args.checkpoint_every > 0 and total_steps % args.checkpoint_every < steps_per_update:
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
                tag = args.tag or "default"
                p = save_dir / f"checkpoint_{tag}_{total_steps // 1_000_000}M.pt"
                torch.save(ckpt, p)
                print(f"  ** Checkpoint saved: {p}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)

    # ── Save and cleanup ─────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed if elapsed > 0 else 0

    # Save full checkpoint for resume
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
    ckpt_path = save_dir / f"checkpoint_{args.tag or 'default'}.pt"
    torch.save(ckpt, ckpt_path)
    # Also save bare weights for eval script compatibility
    torch.save(agent.state_dict(), save_dir / f"model_{args.tag or 'default'}.pt")

    if completed_rewards:
        final_reward = np.mean(completed_rewards[-50:])
    else:
        final_reward = float("nan")

    try:
        final_metrics = {"final_wall_time_sec": elapsed, "final_steps_per_second": sps}
        if not np.isnan(final_reward):
            final_metrics["final_episode_reward_mean"] = float(final_reward)
        final_metrics["total_episodes"] = len(completed_rewards)
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.end_run()
    except Exception as e:
        print(f"MLflow finalization warning: {e}")

    metrics_file.close()
    envs.close()

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"  Wall time: {elapsed:.1f}s ({elapsed / 3600:.1f}h)")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Throughput: {sps:.0f} sps")
    print(f"  Final ep reward (mean last 50): {final_reward:.1f}")
    print(f"  Total episodes: {len(completed_rewards)}")
    print(f"  Model saved to: {save_dir}")
    print(f"{'=' * 70}")


def _load_config_defaults(argv):
    """Pre-parse --config and return (yaml_defaults, remaining_argv).

    Values from the YAML become argparse defaults (dests in snake_case).
    Explicit CLI flags still override. Unknown keys raise.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, remaining = pre.parse_known_args(argv)
    if known.config is None:
        return {}, remaining
    import yaml

    with open(known.config) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{known.config}: top-level must be a mapping")
    return data, remaining


def main():
    import sys

    yaml_defaults, remaining = _load_config_defaults(sys.argv[1:])

    parser = argparse.ArgumentParser(description="CleanRL-style PPO for CybORG CC4")
    parser.add_argument("--config", type=str, default=None, help="YAML with defaults (CLI flags still win)")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument(
        "--rollout-length", type=int, default=500, help="Steps per rollout per env (500 = one full episode)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-epochs", type=int, default=4, help="SGD epochs per rollout")
    parser.add_argument("--num-minibatches", type=int, default=16, help="Number of minibatches per epoch")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--norm-rewards", action="store_true", default=True, help="Normalize rewards using running return stats"
    )
    parser.add_argument("--no-norm-rewards", dest="norm_rewards", action="store_false")
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument(
        "--target-kl", type=float, default=None, help="Target KL for early stopping (None = no early stopping)"
    )
    parser.add_argument(
        "--num-rollouts-per-update", type=int, default=1, help="Number of rollouts to accumulate before each PPO update"
    )
    parser.add_argument(
        "--resume-tag", type=str, default="", help="Tag of a previous run to resume from (loads checkpoint)"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, help="Save checkpoint every N env steps (0 = only at end)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")

    if yaml_defaults:
        valid = {a.dest for a in parser._actions}
        unknown = set(yaml_defaults) - valid
        if unknown:
            raise ValueError(f"{sorted(unknown)}: not valid config keys")
        parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(remaining)
    train(args)


if __name__ == "__main__":
    main()
