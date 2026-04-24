"""Evaluate a CleanRL-CybORG PPO checkpoint on CybORG CC4.

Loads bare model weights saved by `scripts/train/ppo_cleanrl_cyborg.py`
(`model_<tag>.pt`) via its PPOAgent class and rolls out N episodes on
CybORG's EnterpriseMAE env with zero-padded obs/masks (matching training).
"""

# ruff: noqa: E402

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jaxborg.constants import BLUE_OBS_SIZE

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE  # 210
ACT_DIM = 242
EPISODE_LENGTH = 500


def _import_ppo_agent():
    train_path = ROOT / "scripts" / "train" / "ppo_cleanrl_cyborg.py"
    spec = importlib.util.spec_from_file_location("ppo_cleanrl_cyborg", train_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PPOAgent


def make_env(seed):
    """Match transfer.py's per-episode seed convention (one fresh CybORG per ep)."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import EnterpriseMAE
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    return EnterpriseMAE(CybORG(sg, "sim", seed=seed))


def pad_obs_mask(obs_dict, info_dict):
    obs = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)
    mask = np.zeros((NUM_AGENTS, ACT_DIM), dtype=np.float32)
    for i, aid in enumerate(AGENT_IDS):
        raw_o = np.asarray(obs_dict[aid], dtype=np.float32)
        raw_m = np.asarray(info_dict[aid]["action_mask"], dtype=np.float32)
        obs[i, : len(raw_o)] = raw_o
        mask[i, : len(raw_m)] = raw_m
    return obs, mask


def rollout_episode(env, agent, device, deterministic=False):
    obs_d, info_d = env.reset()
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        obs, mask = pad_obs_mask(obs_d, info_d)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            if deterministic:
                # Apply mask then argmax
                features = agent.features(obs_t)
                logits = agent.actor(features) + (mask_t - 1.0) * 1e8
                act = logits.argmax(dim=-1).cpu().numpy()
            else:
                a, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                act = a.cpu().numpy()
        action_dict = {AGENT_IDS[i]: int(act[i]) for i in range(NUM_AGENTS)}
        obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_dict)
        total += float(rew_d[AGENT_IDS[0]])
        if any(term_d.values()) or any(trunc_d.values()):
            break
    return total


def evaluate(model_path, episodes, seed, deterministic, output_json):
    PPOAgent = _import_ppo_agent()
    device = torch.device("cpu")
    agent = PPOAgent(OBS_DIM, ACT_DIM)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    rewards = []
    for ep in range(episodes):
        # Match transfer.py convention: seed=seed+ep, fresh CybORG per episode
        env = make_env(seed + ep)
        r = rollout_episode(env, agent, device, deterministic=deterministic)
        rewards.append(r)
        print(f"  ep {ep + 1}/{episodes}: {r:.1f}", flush=True)

    m = mean(rewards)
    s = stdev(rewards) if len(rewards) > 1 else 0.0
    print(f"\nmodel:        {model_path}")
    print(f"episodes:     {episodes}")
    print(f"mean:         {m:.4f}")
    print(f"stdev:        {s:.4f}")

    if output_json:
        payload = {
            "model": str(model_path),
            "seed": seed,
            "episodes": episodes,
            "deterministic": deterministic,
            "mean": m,
            "stdev": s,
            "per_episode": [float(x) for x in rewards],
        }
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote:        {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CleanRL-CybORG PPO checkpoint on CybORG")
    parser.add_argument("--model", required=True, help="Path to model_<tag>.pt (bare state_dict)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.deterministic, args.output_json)
