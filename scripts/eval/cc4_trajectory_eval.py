"""Roll out a CleanRL-CybORG PPO checkpoint on CC4 and record full trajectories.

Each episode is written as a JSONL file with:
  - one `header` record (model, seed, host list)
  - 500 `step` records (per-agent action class, target host, success, reward;
    captured for blue, red, AND green agents — green records make ASF and
    other availability-event metrics derivable post-hoc without re-rolling out)
  - one `footer` record (total reward, steps)

Trajectories are then scored post-hoc by `cc4_score_trajectories.py`. Decoupling
rollout from scoring lets us re-evaluate with new alignment metrics without
re-running CybORG (CPU-bound, ~2 min/episode).
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.cyborg_runner import load_torch_policy

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE  # 210
ACT_DIM = 242
EPISODE_LENGTH = 500


def make_env(seed):
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


def _capture_agent_records(unwrap, agent_ids):
    """Snapshot last action + observation success for each agent."""
    out = {}
    for ag in agent_ids:
        try:
            actions = unwrap.get_last_action(ag) or []
            obs = unwrap.get_observation(ag)
        except Exception:
            continue
        if not actions:
            continue
        action = actions[0]
        success = None
        if isinstance(obs, dict):
            s = obs.get("success")
            if s is not None:
                success = s.name if hasattr(s, "name") else str(s)
        host = getattr(action, "hostname", None)
        ip = getattr(action, "ip_address", None)
        if host is None and ip is not None:
            try:
                host = unwrap.environment_controller.state.ip_addresses.get(ip)
            except Exception:
                host = None
        out[ag] = {
            "cls": action.__class__.__name__,
            "host": host,
            "ip": str(ip) if ip is not None else None,
            "success": success,
        }
    return out


def rollout_episode(env, agent, device, deterministic, episode_seed, model_path, out_path):
    obs_d, info_d = env.reset()
    unwrap = env.unwrapped if hasattr(env, "unwrapped") else env
    ec = unwrap.environment_controller
    hosts = list(ec.state.hosts.keys())
    # `ec.action` is empty right after reset; discover after first step.
    red_ids: list[str] = []
    blue_ids: list[str] = []
    green_ids: list[str] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w")
    header = {
        "type": "header",
        "model": str(model_path),
        "seed": episode_seed,
        "deterministic": deterministic,
        "hosts": hosts,
        "red_agents": red_ids,
        "blue_agents": blue_ids,
        "green_agents": green_ids,
        "episode_length": EPISODE_LENGTH,
    }
    f.write(json.dumps(header) + "\n")

    total = 0.0
    steps_run = 0
    for t in range(EPISODE_LENGTH):
        obs, mask = pad_obs_mask(obs_d, info_d)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            if deterministic:
                act = agent.deterministic_action(obs_t, mask_t).cpu().numpy()
            else:
                a, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                act = a.cpu().numpy()
        action_dict = {AGENT_IDS[i]: int(act[i]) for i in range(NUM_AGENTS)}
        obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_dict)
        reward = float(rew_d[AGENT_IDS[0]])
        total += reward
        steps_run += 1

        if not red_ids:
            red_ids = sorted(a for a in ec.action.keys() if a.startswith("red_agent_"))
            blue_ids = sorted(a for a in ec.action.keys() if a.startswith("blue_agent_"))
            green_ids = sorted(a for a in ec.action.keys() if a.startswith("green_agent_"))

        try:
            phase = ec.state.mission_phase
        except Exception:
            phase = None
        record = {
            "type": "step",
            "t": t,
            "phase": phase,
            "reward": reward,
            "red": _capture_agent_records(unwrap, red_ids),
            "blue": _capture_agent_records(unwrap, blue_ids),
            "green": _capture_agent_records(unwrap, green_ids),
        }
        f.write(json.dumps(record) + "\n")
        if any(term_d.values()) or any(trunc_d.values()):
            break

    footer = {
        "type": "footer",
        "total_reward": total,
        "steps": steps_run,
        "red_agents": red_ids,
        "blue_agents": blue_ids,
        "green_agents": green_ids,
    }
    f.write(json.dumps(footer) + "\n")
    f.close()
    return total, steps_run


def evaluate(model_path, episodes, seed, deterministic, output_dir, tag):
    device = torch.device("cpu")
    agent, _recipe = load_torch_policy(model_path)

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    tag = tag or Path(model_path).stem.replace("model_", "")

    rewards = []
    for ep in range(episodes):
        ep_seed = seed + ep
        env = make_env(ep_seed)
        out_path = output_dir / f"{tag}_seed{ep_seed}.jsonl"
        r, n = rollout_episode(env, agent, device, deterministic, ep_seed, model_path, out_path)
        rewards.append(r)
        print(f"  ep {ep + 1}/{episodes}: reward={r:+9.1f} steps={n}  → {out_path.name}", flush=True)

    print(f"\nmodel:        {model_path}")
    print(f"episodes:     {episodes}")
    print(f"reward mean:  {sum(rewards) / len(rewards):+.4f}")
    print(f"trajectories: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record CC4 trajectories from a CybORG PPO checkpoint")
    parser.add_argument("--model", required=True, help="Path to model_<tag>.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default=None, help="Override tag in trajectory filename")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.deterministic, args.output_dir, args.tag)
