"""Audit a CybORG-trained PPO policy: per-step actions, busy, phase, per-component reward.

Companion to `scripts/eval/ppo_cleanrl_cyborg_eval.py` — same rollout but
captures the diagnostic streams that `scripts/eval/transfer.py` already
emits for JAX-trained policies. Used by the matched-training v2 final
audit (2026-04-25) to make the JAX-trained vs CybORG-trained comparison
apples-to-apples.

Per-component (RIA / LWF / ASF) capture uses the same monkeypatch on
`BlueRewardMachine.calculate_reward` that transfer.py uses.

Output JSON schema (for one checkpoint):
{
  "model": ...,
  "seed": ...,
  "episodes": ...,
  "per_episode_reward": [...],
  "per_episode_ria": [...],
  "per_episode_lwf": [...],
  "per_episode_asf": [...],
  "per_episode_action_cost_implied": [...],
  "per_step_actions_by_agent": [[...], ...],   # (NUM_AGENTS, total_decision_steps)
  "per_step_busy_by_agent": [[...], ...],
  "per_step_phase_by_episode": [...],          # (episodes * 500,)
  "action_index_to_type": {0: "Sleep", ...},
}
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys
import types
from pathlib import Path
from statistics import mean, stdev

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


# Action-index → action-type bucket (matches the labels transfer.py uses).
# CybORG's EnterpriseMAE action space layout (per agent_4, 242 actions):
# 0: Sleep, 1: Monitor, then per-host: Analyse, Remove, Restore, DecoyEntry...,
# then per-subnet: BlockTraffic / AllowTraffic.
# The exact indices vary per-agent; transfer.py classifies via a per-agent
# mapping. For per-action audit we reduce to the same 8-bucket coarse type
# string by introspecting the CybORG action object in the env.
ACTION_TYPE_BUCKETS = ["Sleep", "Monitor", "Analyse", "Remove", "Restore",
                       "Decoy", "BlockTraffic", "AllowTraffic"]


def _classify_cyborg_action(act_obj) -> str:
    """Coarse 8-bucket name from a CybORG action instance."""
    name = type(act_obj).__name__
    if name == "Sleep":
        return "Sleep"
    if name == "Monitor":
        return "Monitor"
    if name == "Analyse":
        return "Analyse"
    if name == "Remove":
        return "Remove"
    if name == "Restore":
        return "Restore"
    if name.startswith("Deploy") or "Decoy" in name:
        return "Decoy"
    if name == "BlockTrafficZone":
        return "BlockTraffic"
    if name == "AllowTrafficZone":
        return "AllowTraffic"
    return f"Other({name})"


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


def _install_component_monkeypatch(env, log: dict):
    """Patch BlueRewardMachine.calculate_reward to accumulate RIA/LWF/ASF.

    Logic copied from scripts/eval/transfer.py:_tracked_calculate. Resets
    log to zero before installation; caller reads log after each episode.
    """
    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact as _Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService as _GAS
    from CybORG.Simulator.Actions.GreenActions import GreenLocalWork as _GLW

    inner = env.env if hasattr(env, "env") else env
    while not hasattr(inner, "environment_controller"):
        inner = inner.env  # peel wrappers
    ec = inner.environment_controller
    brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]

    def _tracked_calculate(self, current_state, action_dict, agent_observations, done, state):
        self.phase_rewards = self.get_phase_rewards(state.mission_phase)
        reward_list = []
        for agent_name, action in action_dict.items():
            if not action:
                continue
            act = action[0]
            if isinstance(act, _Impact):
                hostname = act.hostname
            elif isinstance(act, (_GAS, _GLW)):
                hostname = state.ip_addresses[act.ip_address]
            else:
                continue
            subnet_name = state.hostname_subnet_map[hostname].value
            sessions = state.sessions[agent_name].values()
            if len([s.ident for s in sessions if s.active]) > 0:
                success = agent_observations[agent_name].observations[0].data["success"]
                rz = self.phase_rewards[subnet_name]
                if "green" in agent_name and success == False:  # noqa: E712 -- numpy.bool_ identity vs equality
                    if isinstance(act, _GLW):
                        r = rz["LWF"]; reward_list.append(r); log["lwf"] += r
                    elif isinstance(act, _GAS):
                        r = rz["ASF"]; reward_list.append(r); log["asf"] += r
                elif "red" in agent_name and success and isinstance(act, _Impact):
                    r = rz["RIA"]; reward_list.append(r); log["ria"] += r
        return sum(reward_list)

    brm.calculate_reward = types.MethodType(_tracked_calculate, brm)
    return ec


def rollout_episode(env, agent, device):
    """Run one episode; return totals + per-step actions + busy + phase + components."""
    obs_d, info_d = env.reset()
    component_log = {"ria": 0.0, "lwf": 0.0, "asf": 0.0}
    ec = _install_component_monkeypatch(env, component_log)

    total = 0.0
    actions_by_agent = [[] for _ in range(NUM_AGENTS)]
    busy_by_agent = [[] for _ in range(NUM_AGENTS)]
    phase_per_step = []
    action_type_by_agent = [[] for _ in range(NUM_AGENTS)]

    for step in range(EPISODE_LENGTH):
        phase_per_step.append(int(ec.state.mission_phase))
        obs, mask = pad_obs_mask(obs_d, info_d)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            a, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
            act = a.cpu().numpy()

        for i in range(NUM_AGENTS):
            actions_by_agent[i].append(int(act[i]))
            pending = ec.actions_in_progress.get(AGENT_IDS[i])
            busy_by_agent[i].append(1 if pending and pending["remaining_ticks"] > 0 else 0)

        action_dict = {AGENT_IDS[i]: int(act[i]) for i in range(NUM_AGENTS)}
        # Capture coarse action types from the decoded CybORG action objects
        # by querying env.action_translator after the step. The wrapper exposes
        # decoded actions via env.actions_in_progress after env.step.
        obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_dict)
        for i in range(NUM_AGENTS):
            pending = ec.actions_in_progress.get(AGENT_IDS[i])
            if pending and "action" in pending:
                action_type_by_agent[i].append(_classify_cyborg_action(pending["action"]))
            else:
                action_type_by_agent[i].append("Unknown")

        total += float(rew_d[AGENT_IDS[0]])
        if any(term_d.values()) or any(trunc_d.values()):
            break

    return {
        "reward": total,
        "ria": component_log["ria"],
        "lwf": component_log["lwf"],
        "asf": component_log["asf"],
        "actions_by_agent": actions_by_agent,
        "busy_by_agent": busy_by_agent,
        "phase_per_step": phase_per_step,
        "action_type_by_agent": action_type_by_agent,
    }


def audit(model_path, episodes, seed, output_json):
    device = torch.device("cpu")
    agent, _recipe = load_torch_policy(model_path)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Capture per-agent action-index → label mapping ONCE from a fresh env so
    # the aggregator can decode coarse types reliably. EnterpriseMAE.action_labels(agent)
    # returns a list[str] of length action_space(agent).n.
    _bootstrap_env = make_env(seed)
    _bootstrap_env.reset()
    action_labels_by_agent = {
        AGENT_IDS[i]: list(_bootstrap_env.action_labels(AGENT_IDS[i]))
        for i in range(NUM_AGENTS)
    }

    per_ep_reward, per_ep_ria, per_ep_lwf, per_ep_asf = [], [], [], []
    per_step_actions = [[] for _ in range(NUM_AGENTS)]
    per_step_busy = [[] for _ in range(NUM_AGENTS)]
    per_step_action_type = [[] for _ in range(NUM_AGENTS)]
    per_step_phase = []

    for ep in range(episodes):
        env = make_env(seed + ep)
        out = rollout_episode(env, agent, device)
        per_ep_reward.append(out["reward"])
        per_ep_ria.append(out["ria"])
        per_ep_lwf.append(out["lwf"])
        per_ep_asf.append(out["asf"])
        for i in range(NUM_AGENTS):
            per_step_actions[i].extend(out["actions_by_agent"][i])
            per_step_busy[i].extend(out["busy_by_agent"][i])
            per_step_action_type[i].extend(out["action_type_by_agent"][i])
        per_step_phase.extend(out["phase_per_step"])
        print(f"  ep {ep + 1}/{episodes}: r={out['reward']:.1f}  RIA={out['ria']:.0f}  LWF={out['lwf']:.0f}  ASF={out['asf']:.0f}", flush=True)

    m = mean(per_ep_reward); s = stdev(per_ep_reward) if len(per_ep_reward) > 1 else 0.0
    print(f"\nmodel:        {model_path}")
    print(f"mean reward:  {m:.2f} ± {s:.2f}")
    print(f"mean RIA:     {mean(per_ep_ria):.1f}")
    print(f"mean LWF:     {mean(per_ep_lwf):.1f}")
    print(f"mean ASF:     {mean(per_ep_asf):.1f}")

    payload = {
        "model": str(model_path),
        "seed": seed,
        "episodes": episodes,
        "per_episode_reward": [float(x) for x in per_ep_reward],
        "per_episode_ria": [float(x) for x in per_ep_ria],
        "per_episode_lwf": [float(x) for x in per_ep_lwf],
        "per_episode_asf": [float(x) for x in per_ep_asf],
        "mean_reward": m,
        "stdev_reward": s,
        "mean_ria": float(mean(per_ep_ria)),
        "mean_lwf": float(mean(per_ep_lwf)),
        "mean_asf": float(mean(per_ep_asf)),
        "per_step_actions_by_agent": per_step_actions,        # int action index
        "per_step_busy_by_agent": per_step_busy,              # 0/1
        "per_step_action_type_by_agent": per_step_action_type, # coarse type (best-effort, has Unknowns)
        "per_step_phase": per_step_phase,
        "action_type_buckets": ACTION_TYPE_BUCKETS,
        "action_labels_by_agent": action_labels_by_agent,     # for index→label decoding
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")
    print(f"wrote:        {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit CybORG-trained policy: actions + per-component reward")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", required=True)
    a = p.parse_args()
    audit(a.model, a.episodes, a.seed, a.output_json)
