"""Pure-CybORG PPO baseline for CC4 with action masking.

Goal: produce the simplest credible CPU-only baseline for CC4, so JaxBorg's
GPU speedup and policy quality can be compared against a real reference point.
Not chasing SOTA — establishing a trustworthy, reproducible number.

Written in CleanRL style (single-file, minimal dependencies, readable).
Uses parameter sharing across agents 0-3 (same obs/act space) and a separate
policy for agent 4 (larger obs/act space).

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
import time
from multiprocessing import Pipe, Process
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from jaxborg.constants import BLUE_OBS_SIZE

# ── Environment setup ──────────────────────────────────────────────────────

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

# Agents 0-3 share obs/act spaces; agent 4 is different
SMALL_OBS_DIM = 92
SMALL_ACT_DIM = 82
LARGE_OBS_DIM = BLUE_OBS_SIZE
LARGE_ACT_DIM = 242


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


# ── Neural network ────────────────────────────────────────────────────────


class PPOAgent(nn.Module):
    """Actor-critic network with action masking."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Linear(in_dim, act_dim)

        # Critic head
        self.critic = nn.Linear(in_dim, 1)

        # Orthogonal initialization (CleanRL standard)
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def get_value(self, obs):
        features = self.features(obs)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(self, obs, action_mask, action=None):
        features = self.features(obs)
        logits = self.actor(features)

        # Apply action mask: set invalid actions to very negative
        logits = logits + (action_mask.float() - 1.0) * 1e8

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value


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

    # Create networks - shared policy for agents 0-3, separate for agent 4
    agent_small = PPOAgent(SMALL_OBS_DIM, SMALL_ACT_DIM, hidden_dims=(256, 256)).to(device)
    agent_large = PPOAgent(LARGE_OBS_DIM, LARGE_ACT_DIM, hidden_dims=(256, 256)).to(device)

    optimizer = optim.Adam(
        list(agent_small.parameters()) + list(agent_large.parameters()),
        lr=args.lr,
        eps=1e-5,
    )

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
            agent_small.load_state_dict(ckpt["agent_small"])
            agent_large.load_state_dict(ckpt["agent_large"])
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
            small_path = ckpt_dir / f"model_small_{args.resume_tag}.pt"
            large_path = ckpt_dir / f"model_large_{args.resume_tag}.pt"
            print(f"No full checkpoint; loading bare weights from {small_path}...", flush=True)
            agent_small.load_state_dict(torch.load(small_path, map_location=device, weights_only=True))
            agent_large.load_state_dict(torch.load(large_path, map_location=device, weights_only=True))
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

    # Storage arrays
    obs_small = torch.zeros((num_steps, args.num_envs, 4, SMALL_OBS_DIM))
    obs_large = torch.zeros((num_steps, args.num_envs, 1, LARGE_OBS_DIM))
    actions_small = torch.zeros((num_steps, args.num_envs, 4), dtype=torch.long)
    actions_large = torch.zeros((num_steps, args.num_envs, 1), dtype=torch.long)
    logprobs_small = torch.zeros((num_steps, args.num_envs, 4))
    logprobs_large = torch.zeros((num_steps, args.num_envs, 1))
    rewards_all = torch.zeros((num_steps, args.num_envs))  # shared reward (same for all agents)
    dones_all = torch.zeros((num_steps, args.num_envs))
    values_small = torch.zeros((num_steps, args.num_envs, 4))
    values_large = torch.zeros((num_steps, args.num_envs, 1))
    masks_small = torch.zeros((num_steps, args.num_envs, 4, SMALL_ACT_DIM))
    masks_large = torch.zeros((num_steps, args.num_envs, 1, LARGE_ACT_DIM))

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
            "shared_policy": "agents_0-3_shared",
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
    accum_obs_s, accum_obs_l = [], []
    accum_act_s, accum_act_l = [], []
    accum_lp_s, accum_lp_l = [], []
    accum_adv_s, accum_adv_l = [], []
    accum_ret_s, accum_ret_l = [], []
    accum_val_s, accum_val_l = [], []
    accum_mask_s, accum_mask_l = [], []

    steps_per_update = args.num_envs * args.rollout_length * args.num_rollouts_per_update
    total_updates = args.total_timesteps // steps_per_update
    # Agent-level batch size for SGD (accounts for accumulation)
    small_batch = num_steps * args.num_envs * 4 * args.num_rollouts_per_update
    large_batch = num_steps * args.num_envs * 1 * args.num_rollouts_per_update
    small_mb_size = small_batch // args.num_minibatches
    large_mb_size = large_batch // args.num_minibatches

    print(f"\n{'=' * 70}")
    print("CleanRL PPO for CybORG CC4")
    print(f"{'=' * 70}")
    print(f"  Envs: {args.num_envs}")
    print(f"  Rollout length: {args.rollout_length} (steps per env)")
    print(f"  Rollouts per update: {args.num_rollouts_per_update}")
    print(f"  Steps per update: {steps_per_update:,} (env steps)")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Total updates: {total_updates}")
    print(f"  Small agent batch: {small_batch:,} -> {small_mb_size:,} per minibatch")
    print(f"  Large agent batch: {large_batch:,} -> {large_mb_size:,} per minibatch")
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
                # Store observations and masks
                for env_idx in range(args.num_envs):
                    for i in range(4):
                        agent_id = AGENT_IDS[i]
                        obs_small[step, env_idx, i] = torch.from_numpy(all_obs[env_idx][agent_id].astype(np.float32))
                        masks_small[step, env_idx, i] = torch.from_numpy(
                            np.array(all_info[env_idx][agent_id]["action_mask"], dtype=np.float32)
                        )
                    agent_id = AGENT_IDS[4]
                    obs_large[step, env_idx, 0] = torch.from_numpy(all_obs[env_idx][agent_id].astype(np.float32))
                    masks_large[step, env_idx, 0] = torch.from_numpy(
                        np.array(all_info[env_idx][agent_id]["action_mask"], dtype=np.float32)
                    )

                # Get actions from policies
                with torch.no_grad():
                    obs_s_flat = obs_small[step].reshape(-1, SMALL_OBS_DIM)
                    mask_s_flat = masks_small[step].reshape(-1, SMALL_ACT_DIM)
                    act_s, lp_s, _, val_s = agent_small.get_action_and_value(obs_s_flat, mask_s_flat)
                    actions_small[step] = act_s.reshape(args.num_envs, 4)
                    logprobs_small[step] = lp_s.reshape(args.num_envs, 4)
                    values_small[step] = val_s.reshape(args.num_envs, 4)

                    obs_l_flat = obs_large[step].reshape(-1, LARGE_OBS_DIM)
                    mask_l_flat = masks_large[step].reshape(-1, LARGE_ACT_DIM)
                    act_l, lp_l, _, val_l = agent_large.get_action_and_value(obs_l_flat, mask_l_flat)
                    actions_large[step] = act_l.reshape(args.num_envs, 1)
                    logprobs_large[step] = lp_l.reshape(args.num_envs, 1)
                    values_large[step] = val_l.reshape(args.num_envs, 1)

                # Build action dicts and step environments
                action_dicts = []
                for env_idx in range(args.num_envs):
                    ad = {}
                    for i in range(4):
                        ad[AGENT_IDS[i]] = int(actions_small[step, env_idx, i].item())
                    ad[AGENT_IDS[4]] = int(actions_large[step, env_idx, 0].item())
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
                # Bootstrap values for last observation
                obs_s_flat = torch.zeros(args.num_envs * 4, SMALL_OBS_DIM)
                obs_l_flat = torch.zeros(args.num_envs, LARGE_OBS_DIM)
                for env_idx in range(args.num_envs):
                    for i in range(4):
                        obs_s_flat[env_idx * 4 + i] = torch.from_numpy(
                            all_obs[env_idx][AGENT_IDS[i]].astype(np.float32)
                        )
                    obs_l_flat[env_idx] = torch.from_numpy(all_obs[env_idx][AGENT_IDS[4]].astype(np.float32))

                next_val_s = agent_small.get_value(obs_s_flat).reshape(args.num_envs, 4)
                next_val_l = agent_large.get_value(obs_l_flat).reshape(args.num_envs, 1)

                # GAE for small agents - all use the same shared reward
                advantages_small = torch.zeros((num_steps, args.num_envs, 4))
                lastgaelam = torch.zeros(args.num_envs, 4)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = next_val_s
                    else:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = values_small[t + 1]
                    # All agents get the same reward
                    rew_expanded = rewards_all[t].unsqueeze(-1).expand_as(values_small[t])
                    delta = rew_expanded + args.gamma * nextvalues * nextnonterminal - values_small[t]
                    advantages_small[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

                returns_small = advantages_small + values_small

                # GAE for large agent
                advantages_large = torch.zeros((num_steps, args.num_envs, 1))
                lastgaelam = torch.zeros(args.num_envs, 1)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = next_val_l
                    else:
                        nextnonterminal = (1.0 - dones_all[t]).unsqueeze(-1)
                        nextvalues = values_large[t + 1]
                    rew_expanded = rewards_all[t].unsqueeze(-1)
                    delta = rew_expanded + args.gamma * nextvalues * nextnonterminal - values_large[t]
                    advantages_large[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

                returns_large = advantages_large + values_large

            # ── Accumulate rollout data ──────────────────────────────
            accum_obs_s.append(obs_small.reshape(-1, SMALL_OBS_DIM).clone())
            accum_obs_l.append(obs_large.reshape(-1, LARGE_OBS_DIM).clone())
            accum_act_s.append(actions_small.reshape(-1).clone())
            accum_act_l.append(actions_large.reshape(-1).clone())
            accum_lp_s.append(logprobs_small.reshape(-1).clone())
            accum_lp_l.append(logprobs_large.reshape(-1).clone())
            accum_adv_s.append(advantages_small.reshape(-1).clone())
            accum_adv_l.append(advantages_large.reshape(-1).clone())
            accum_ret_s.append(returns_small.reshape(-1).clone())
            accum_ret_l.append(returns_large.reshape(-1).clone())
            accum_val_s.append(values_small.reshape(-1).clone())
            accum_val_l.append(values_large.reshape(-1).clone())
            accum_mask_s.append(masks_small.reshape(-1, SMALL_ACT_DIM).clone())
            accum_mask_l.append(masks_large.reshape(-1, LARGE_ACT_DIM).clone())
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
            b_obs_s = torch.cat(accum_obs_s)
            b_obs_l = torch.cat(accum_obs_l)
            b_act_s = torch.cat(accum_act_s)
            b_act_l = torch.cat(accum_act_l)
            b_lp_s = torch.cat(accum_lp_s)
            b_lp_l = torch.cat(accum_lp_l)
            b_adv_s = torch.cat(accum_adv_s)
            b_adv_l = torch.cat(accum_adv_l)
            b_ret_s = torch.cat(accum_ret_s)
            b_ret_l = torch.cat(accum_ret_l)
            b_val_s = torch.cat(accum_val_s)
            b_val_l = torch.cat(accum_val_l)
            b_mask_s = torch.cat(accum_mask_s)
            b_mask_l = torch.cat(accum_mask_l)

            # Clear accumulators
            accum_obs_s.clear()
            accum_obs_l.clear()
            accum_act_s.clear()
            accum_act_l.clear()
            accum_lp_s.clear()
            accum_lp_l.clear()
            accum_adv_s.clear()
            accum_adv_l.clear()
            accum_ret_s.clear()
            accum_ret_l.clear()
            accum_val_s.clear()
            accum_val_l.clear()
            accum_mask_s.clear()
            accum_mask_l.clear()
            rollouts_collected = 0

            total_s = b_obs_s.shape[0]
            total_l = b_obs_l.shape[0]
            mb_size_s = total_s // args.num_minibatches
            mb_size_l = total_l // args.num_minibatches

            # Epoch metrics accumulators
            epoch_pg_loss = 0.0
            epoch_vf_loss = 0.0
            epoch_entropy = 0.0
            epoch_approx_kl = 0.0
            epoch_clipfrac = 0.0
            n_minibatches_total = 0

            for epoch in range(args.num_epochs):
                perm_s = torch.randperm(total_s)
                perm_l = torch.randperm(total_l)

                for mb_idx in range(args.num_minibatches):
                    idx_s = perm_s[mb_idx * mb_size_s : (mb_idx + 1) * mb_size_s]
                    idx_l = perm_l[mb_idx * mb_size_l : (mb_idx + 1) * mb_size_l]

                    # Small agents
                    mb_obs_s = b_obs_s[idx_s]
                    mb_act_s = b_act_s[idx_s]
                    mb_lp_s = b_lp_s[idx_s]
                    mb_adv_s = b_adv_s[idx_s]
                    mb_ret_s = b_ret_s[idx_s]
                    mb_mask_s = b_mask_s[idx_s]

                    # Large agent
                    mb_obs_l = b_obs_l[idx_l]
                    mb_act_l = b_act_l[idx_l]
                    mb_lp_l = b_lp_l[idx_l]
                    mb_adv_l = b_adv_l[idx_l]
                    mb_ret_l = b_ret_l[idx_l]
                    mb_mask_l = b_mask_l[idx_l]

                    # Forward pass
                    _, new_lp_s, ent_s, new_val_s = agent_small.get_action_and_value(mb_obs_s, mb_mask_s, mb_act_s)
                    _, new_lp_l, ent_l, new_val_l = agent_large.get_action_and_value(mb_obs_l, mb_mask_l, mb_act_l)

                    # ── Small agent loss ──
                    adv_s = (mb_adv_s - mb_adv_s.mean()) / (mb_adv_s.std() + 1e-8)
                    logratio_s = new_lp_s - mb_lp_s
                    ratio_s = logratio_s.exp()
                    pg_loss1_s = -adv_s * ratio_s
                    pg_loss2_s = -adv_s * torch.clamp(ratio_s, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss_s = torch.max(pg_loss1_s, pg_loss2_s).mean()
                    vf_loss_s = 0.5 * ((new_val_s - mb_ret_s) ** 2).mean()
                    entropy_s = ent_s.mean()

                    # ── Large agent loss ──
                    adv_l = (mb_adv_l - mb_adv_l.mean()) / (mb_adv_l.std() + 1e-8)
                    logratio_l = new_lp_l - mb_lp_l
                    ratio_l = logratio_l.exp()
                    pg_loss1_l = -adv_l * ratio_l
                    pg_loss2_l = -adv_l * torch.clamp(ratio_l, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss_l = torch.max(pg_loss1_l, pg_loss2_l).mean()
                    vf_loss_l = 0.5 * ((new_val_l - mb_ret_l) ** 2).mean()
                    entropy_l = ent_l.mean()

                    # Combined loss (weighted by num agents: 4 small + 1 large)
                    pg_loss = (4 * pg_loss_s + pg_loss_l) / 5
                    vf_loss = (4 * vf_loss_s + vf_loss_l) / 5
                    entropy_loss = (4 * entropy_s + entropy_l) / 5

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * vf_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(agent_small.parameters()) + list(agent_large.parameters()),
                        args.max_grad_norm,
                    )
                    optimizer.step()

                    # Track metrics
                    with torch.no_grad():
                        all_ratio = torch.cat([ratio_s, ratio_l])
                        all_logratio = torch.cat([logratio_s, logratio_l])
                        approx_kl = ((all_ratio - 1) - all_logratio).mean().item()
                        clipfrac = ((all_ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                    epoch_pg_loss += pg_loss.item()
                    epoch_vf_loss += vf_loss.item()
                    epoch_entropy += entropy_loss.item()
                    epoch_approx_kl += approx_kl
                    epoch_clipfrac += clipfrac
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

            # Compute explained variance
            with torch.no_grad():
                all_vals = torch.cat([b_val_s, b_val_l])
                all_rets = torch.cat([b_ret_s, b_ret_l])
                y_var = all_rets.var()
                explained_var = (1 - (all_rets - all_vals).var() / (y_var + 1e-8)).item() if y_var > 1e-8 else 0.0

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
            }
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
                    "agent_small": agent_small.state_dict(),
                    "agent_large": agent_large.state_dict(),
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
        "agent_small": agent_small.state_dict(),
        "agent_large": agent_large.state_dict(),
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
    torch.save(agent_small.state_dict(), save_dir / f"model_small_{args.tag or 'default'}.pt")
    torch.save(agent_large.state_dict(), save_dir / f"model_large_{args.tag or 'default'}.pt")

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


def main():
    parser = argparse.ArgumentParser(description="CleanRL-style PPO for CybORG CC4")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--rollout-length", type=int, default=500, help="Steps per rollout per env (500 = one full episode)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.85, help="Discount factor (CC4 tutorial: 0.85)")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-epochs", type=int, default=10, help="SGD epochs per rollout")
    parser.add_argument("--num-minibatches", type=int, default=8, help="Number of minibatches per epoch")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--norm-rewards", action="store_true", default=True, help="Normalize rewards using running return stats"
    )
    parser.add_argument("--no-norm-rewards", dest="norm_rewards", action="store_false")
    parser.add_argument("--anneal-lr", action="store_true", default=True)
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

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
