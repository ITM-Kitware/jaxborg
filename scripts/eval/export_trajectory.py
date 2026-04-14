#!/usr/bin/env python
"""Export CybORG CC4 episode trajectories as JSON for cynex visualization.

Usage:
    # Sleep agent (no defense):
    python scripts/eval/export_trajectory.py --seed 42 --num-episodes 3 \
        --output-dir ../cynex/public/data/trajectories/

    # Trained PPO policy:
    python scripts/eval/export_trajectory.py --seed 42 --num-episodes 1 \
        --model /path/to/cleanrl_ppo/model_cyborg_matched.pt \
        --output-dir ../cynex/public/data/trajectories/
"""

import argparse
import json
from datetime import datetime
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

EPISODE_LENGTH = 500
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

# Unified obs/act dims (agent 4's sizes — agents 0-3 are zero-padded to match)
OBS_DIM = 210
ACT_DIM = 242


class PPOAgent(nn.Module):
    """PPO agent matching the CleanRL training architecture."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, act_dim)
        self.critic = nn.Linear(in_dim, 1)

    def get_action(self, obs, action_mask, deterministic=False):
        features = self.features(obs)
        logits = self.actor(features)
        logits = logits + (action_mask.float() - 1.0) * 1e8
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()


def make_env(seed: int, steps: int = EPISODE_LENGTH) -> CybORG:
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )
    return CybORG(sg, "sim", seed=seed)


def make_wrapped_env(seed: int, steps: int = EPISODE_LENGTH) -> EnterpriseMAE:
    # CybORG bug: EnterpriseScenarioGenerator.determine_done() uses
    # `step_count >= steps - 1`, which terminates one step early through
    # the EnterpriseMAE wrapper (499 steps instead of 500). Raw
    # CybORG.parallel_step() doesn't hit this. Pass steps+1 to compensate.
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps + 1,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return EnterpriseMAE(cyborg)


def _clean_subnet_name(name) -> str:
    """Get clean subnet name from CybORG SUBNET enum or string.

    CybORG uses a str enum whose .value is already the clean name
    (e.g., 'restricted_zone_a_subnet').
    """
    if hasattr(name, "value"):
        return str(name.value)
    s = str(name)
    if "SUBNET." in s:
        return s.split("SUBNET.")[-1].lower() + "_subnet"
    return s


def _subnet_display_label(name) -> str:
    """Convert subnet enum/string to human label like 'Restricted Zone A'."""
    if hasattr(name, "name"):
        # Enum .name is like 'RESTRICTED_ZONE_A'
        return name.name.replace("_", " ").title()
    return _clean_subnet_name(name).removesuffix("_subnet").replace("_", " ").title()


def extract_topology(cyborg: CybORG) -> dict:
    """Extract network topology matching cynex HostInfo format."""
    state = cyborg.environment_controller.state
    topology = {}

    for hostname, host in state.hosts.items():
        # Interfaces
        interfaces = []
        for iface in host.interfaces:
            interfaces.append(
                {
                    "Interface Name": iface.name,
                    "IP Address": str(iface.ip_address),
                    "Subnet": {
                        "network_address": str(iface.subnet.network_address),
                        "netmask": str(iface.subnet.netmask),
                        "_prefixlen": iface.subnet.prefixlen,
                    },
                }
            )

        # Sessions
        sessions = []
        for agent_name, session_ids in host.sessions.items():
            for sid in session_ids:
                if agent_name not in state.sessions:
                    continue
                session_map = state.sessions[agent_name]
                if sid not in session_map:
                    continue
                session = session_map[sid]
                sessions.append(
                    {
                        "Username": session.username,
                        "ID": sid,
                        "Timeout": int(session.timeout),
                        "PID": int(session.pid),
                        "Type": session.session_type.name
                        if hasattr(session.session_type, "name")
                        else str(session.session_type),
                        "Agent": agent_name,
                    }
                )

        # Processes
        processes = []
        for proc in host.processes:
            processes.append(
                {
                    "PID": int(proc.pid),
                    "Username": proc.user if proc.user else "unknown",
                }
            )

        # Users
        users = []
        for user in host.users:
            groups = []
            for group in user.groups:
                groups.append({"GID": int(group.ident)})
            users.append(
                {
                    "Username": user.username,
                    "Groups": groups,
                }
            )

        # System info
        system_info = {
            "Hostname": hostname,
            "OSType": host.os_type.name if hasattr(host.os_type, "name") else str(host.os_type),
            "OSDistribution": host.distribution.name
            if hasattr(host.distribution, "name")
            else str(host.distribution),
            "OSVersion": host.version.name if hasattr(host.version, "name") else str(host.version),
            "Architecture": host.architecture.name
            if hasattr(host.architecture, "name")
            else str(host.architecture),
        }

        topology[hostname] = {
            "Interface": interfaces,
            "Sessions": sessions,
            "Processes": processes,
            "User Info": users,
            "System info": system_info,
        }

    return topology


def extract_subnet_metadata(cyborg: CybORG) -> dict:
    """Extract subnet metadata with NACL connections derived from link_diagram."""
    state = cyborg.environment_controller.state
    metadata = {}
    subnet_adjacency: dict[str, set[str]] = {}

    # Map raw CybORG subnet names to clean names
    raw_to_clean = {}
    for raw_name, cidr in state.subnet_name_to_cidr.items():
        clean = _clean_subnet_name(raw_name)
        raw_to_clean[raw_name] = clean
        metadata[clean] = {
            "label": _subnet_display_label(raw_name),
            "network_address": str(cidr.network_address),
            "netmask": str(cidr.netmask),
            "nacl_connections": [],
        }
        subnet_adjacency[clean] = set()

    # Derive subnet adjacency from link_diagram edges
    if state.link_diagram is not None:
        for host_a, host_b in state.link_diagram.edges():
            raw_a = state.hostname_subnet_map.get(host_a)
            raw_b = state.hostname_subnet_map.get(host_b)
            if raw_a and raw_b and raw_a != raw_b:
                clean_a = raw_to_clean.get(raw_a, raw_a)
                clean_b = raw_to_clean.get(raw_b, raw_b)
                subnet_adjacency[clean_a].add(clean_b)
                subnet_adjacency[clean_b].add(clean_a)

    for subnet_name in metadata:
        metadata[subnet_name]["nacl_connections"] = sorted(subnet_adjacency.get(subnet_name, []))

    return metadata


def get_host_compromise(state, red_agents: list[str]) -> dict[str, str]:
    """Determine compromise level for each host: NONE, USER, or PRIVILEGED."""
    compromise = {}
    for hostname, host in state.hosts.items():
        level = "NONE"
        for agent_name in red_agents:
            if agent_name not in host.sessions:
                continue
            for session_id in host.sessions[agent_name]:
                if agent_name not in state.sessions or session_id not in state.sessions[agent_name]:
                    continue
                session = state.sessions[agent_name][session_id]
                if session.has_privileged_access():
                    level = "PRIVILEGED"
                    break
                elif level != "PRIVILEGED":
                    level = "USER"
            if level == "PRIVILEGED":
                break
        compromise[hostname] = level
    return compromise


def resolve_action_host(action, state) -> str:
    """Resolve the target host of an action to a hostname string."""
    # Direct hostname
    hostname = getattr(action, "hostname", None)
    if hostname:
        return str(hostname)

    # IP address -> resolve to hostname
    ip = getattr(action, "ip_address", None)
    if ip is not None:
        if isinstance(ip, str):
            try:
                ip = IPv4Address(ip)
            except ValueError:
                return str(ip)
        return state.ip_addresses.get(ip, str(ip))

    # Subnet -> resolve to subnet name
    subnet = getattr(action, "subnet", None)
    if subnet is not None:
        if isinstance(subnet, str):
            try:
                subnet = IPv4Network(subnet)
            except ValueError:
                return str(subnet)
        raw_name = state.subnets_cidr_to_name.get(subnet)
        if raw_name is not None:
            return _clean_subnet_name(raw_name)
        return str(subnet)

    return ""


def action_to_dict(action, step: int, state, success: str = "TRUE") -> dict:
    """Convert a CybORG Action object to trajectory dict entry."""
    if action is None:
        return {"step": step, "Action": "Sleep", "Status": success, "Host": "", "Params": {}}

    action_name = type(action).__name__
    host = resolve_action_host(action, state)

    return {
        "step": step,
        "Action": action_name,
        "Status": success,
        "Host": host,
        "Params": {},
    }


def run_episode_sleep(seed: int, episode_num: int, steps: int = EPISODE_LENGTH) -> dict:
    """Run one CybORG CC4 episode with SleepAgent blue and return the trajectory dict."""
    cyborg = make_env(seed, steps)
    cyborg.reset()

    ctrl = cyborg.environment_controller
    state = ctrl.state

    blue_agents = sorted(ctrl.team_assignments.get("Blue", []))
    red_agents = sorted(ctrl.team_assignments.get("Red", []))
    green_agents = sorted(ctrl.team_assignments.get("Green", []))

    # Extract initial topology
    topology = extract_topology(cyborg)
    subnet_metadata = extract_subnet_metadata(cyborg)

    print(f"  Agents: {len(blue_agents)} blue, {len(red_agents)} red, {len(green_agents)} green")
    print(f"  Hosts: {len(topology)}, Subnets: {len(subnet_metadata)}")

    # Initialize records
    agent_actions: dict[str, list] = {a: [] for a in blue_agents + red_agents + green_agents}
    step_states = []
    cumulative_rewards = {a: 0.0 for a in blue_agents}
    actual_steps = 0

    for step in range(steps):
        # Blue agents all sleep
        blue_sleep = {agent: Sleep() for agent in blue_agents}
        obs, rewards, dones, _info = cyborg.parallel_step(blue_sleep)

        state = ctrl.state
        actual_steps = step + 1

        # Record each agent's action
        for agent in blue_agents + red_agents + green_agents:
            last = cyborg.get_last_action(agent)
            action = last[0] if isinstance(last, list) and last else last

            # Get success from observation
            success = "TRUE"
            if agent in obs and isinstance(obs[agent], dict) and "success" in obs[agent]:
                val = obs[agent]["success"]
                success = val.name if hasattr(val, "name") else str(val)

            agent_actions[agent].append(action_to_dict(action, step, state, success))

        # Record step state
        compromise = get_host_compromise(state, red_agents)

        step_rewards = {}
        for agent in blue_agents:
            if agent in rewards:
                r = rewards[agent]
                step_rewards[agent] = sum(r.values()) if isinstance(r, dict) else float(r)
            else:
                step_rewards[agent] = 0.0
            cumulative_rewards[agent] += step_rewards[agent]

        step_states.append(
            {
                "step": step,
                "mission_phase": state.mission_phase,
                "host_compromise": compromise,
                "rewards": step_rewards,
                "cumulative_reward": {a: round(v, 4) for a, v in cumulative_rewards.items()},
            }
        )

        if step % 100 == 0:
            compromised = sum(1 for v in compromise.values() if v != "NONE")
            print(f"  Step {step}: phase={state.mission_phase}, compromised={compromised}/{len(compromise)}")

        if dones.get("__all__", False):
            print(f"  Episode ended early at step {step}")
            break

    return _build_trajectory_dict(
        episode_num, seed, actual_steps, "SleepAgent",
        blue_agents, red_agents, green_agents,
        topology, subnet_metadata, agent_actions, step_states,
    )


def run_episode_policy(
    seed: int,
    episode_num: int,
    model: PPOAgent,
    deterministic: bool = False,
    steps: int = EPISODE_LENGTH,
) -> dict:
    """Run one CybORG CC4 episode with a trained PPO policy for blue."""
    env = make_wrapped_env(seed, steps)
    obs, info = env.reset()

    # Access the underlying CybORG for raw state extraction
    cyborg = env.env
    ctrl = cyborg.environment_controller
    state = ctrl.state

    blue_agents = sorted(ctrl.team_assignments.get("Blue", []))
    red_agents = sorted(ctrl.team_assignments.get("Red", []))
    green_agents = sorted(ctrl.team_assignments.get("Green", []))

    topology = extract_topology(cyborg)
    subnet_metadata = extract_subnet_metadata(cyborg)

    print(f"  Agents: {len(blue_agents)} blue, {len(red_agents)} red, {len(green_agents)} green")
    print(f"  Hosts: {len(topology)}, Subnets: {len(subnet_metadata)}")
    print(f"  Policy mode: {'deterministic' if deterministic else 'stochastic'}")

    agent_actions: dict[str, list] = {a: [] for a in blue_agents + red_agents + green_agents}
    step_states = []
    cumulative_rewards = {a: 0.0 for a in blue_agents}
    actual_steps = 0

    for step in range(steps):
        # Select blue actions using the trained policy
        actions = {}
        for i, agent_id in enumerate(AGENT_IDS):
            with torch.no_grad():
                raw_obs = obs[agent_id].astype(np.float32)
                raw_mask = np.array(info[agent_id]["action_mask"], dtype=np.float32)
                # Zero-pad to unified dims (agent 4 has full size, 0-3 are smaller)
                o = torch.zeros(1, OBS_DIM)
                o[0, : len(raw_obs)] = torch.from_numpy(raw_obs)
                m = torch.zeros(1, ACT_DIM)
                m[0, : len(raw_mask)] = torch.from_numpy(raw_mask)
                act = model.get_action(o, m, deterministic=deterministic).item()
            actions[agent_id] = act

        obs, rewards, term, trunc, info = env.step(actions)

        state = ctrl.state
        actual_steps = step + 1

        # Record each agent's action from the underlying CybORG
        for agent in blue_agents + red_agents + green_agents:
            last = cyborg.get_last_action(agent)
            action = last[0] if isinstance(last, list) and last else last

            success = "TRUE"
            # For blue agents, check the wrapper's observation for success
            raw_obs_dict = cyborg.get_observation(agent)
            if isinstance(raw_obs_dict, dict) and "success" in raw_obs_dict:
                val = raw_obs_dict["success"]
                success = val.name if hasattr(val, "name") else str(val)

            agent_actions[agent].append(action_to_dict(action, step, state, success))

        # Record step state
        compromise = get_host_compromise(state, red_agents)

        step_rewards = {}
        reward_val = rewards.get(AGENT_IDS[0], 0.0)
        # All blue agents share the same reward in CC4
        for agent in blue_agents:
            step_rewards[agent] = float(reward_val)
            cumulative_rewards[agent] += step_rewards[agent]

        step_states.append(
            {
                "step": step,
                "mission_phase": state.mission_phase,
                "host_compromise": compromise,
                "rewards": step_rewards,
                "cumulative_reward": {a: round(v, 4) for a, v in cumulative_rewards.items()},
            }
        )

        if step % 100 == 0:
            compromised = sum(1 for v in compromise.values() if v != "NONE")
            cum = cumulative_rewards[blue_agents[0]]
            print(
                f"  Step {step}: phase={state.mission_phase}, "
                f"compromised={compromised}/{len(compromise)}, cum_reward={cum:.1f}"
            )

        if any(term.values()) or any(trunc.values()):
            print(f"  Episode ended at step {step}")
            break

    return _build_trajectory_dict(
        episode_num, seed, actual_steps, "PPO",
        blue_agents, red_agents, green_agents,
        topology, subnet_metadata, agent_actions, step_states,
    )


def _build_trajectory_dict(
    episode_num, seed, actual_steps, blue_agent_name,
    blue_agents, red_agents, green_agents,
    topology, subnet_metadata, agent_actions, step_states,
) -> dict:
    """Build the V2 trajectory dict from collected episode data."""
    # Build flattened backward-compat arrays (interleaved all blue / all red)
    flat_blue = []
    flat_red = []
    for s in range(actual_steps):
        for agent in blue_agents:
            if s < len(agent_actions[agent]):
                flat_blue.append(agent_actions[agent][s])
        for agent in red_agents:
            if s < len(agent_actions[agent]):
                flat_red.append(agent_actions[agent][s])

    return {
        "format_version": "2.0",
        "challenge": "cc4",
        "episode": episode_num,
        "seed": seed,
        "total_steps": actual_steps,
        "experiment_time": datetime.now().isoformat(),
        "blue_agents": blue_agents,
        "red_agents": red_agents,
        "green_agents": green_agents,
        "network_topology": topology,
        "subnet_metadata": subnet_metadata,
        "agent_actions": agent_actions,
        "step_states": step_states,
        "metric_scores": [],
        # Backward compatibility
        "blue_agent_name": blue_agent_name,
        "red_agent_name": "FiniteStateRedAgent",
        "blue_actions": flat_blue,
        "red_actions": flat_red,
    }


def load_model(model_path: str) -> PPOAgent:
    """Load a trained PPO model from a .pt file."""
    model = PPOAgent(OBS_DIM, ACT_DIM)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Export CybORG CC4 trajectories as JSON for cynex")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to export")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="Steps per episode")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for JSON files")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PPO model .pt file")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (argmax) actions")
    parser.add_argument("--tag", type=str, default=None, help="Custom tag for output filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model) if args.model else None

    # Determine filename tag
    if args.tag:
        tag = args.tag
    elif model:
        tag = "ppo-det" if args.deterministic else "ppo"
    else:
        tag = "sleep"

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        print(f"Episode {ep} (seed={seed}):")

        if model:
            trajectory = run_episode_policy(seed, ep, model, args.deterministic, args.steps)
        else:
            trajectory = run_episode_sleep(seed, ep, args.steps)

        filename = f"cc4-{tag}-seed{seed}-E{ep}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        size_kb = filepath.stat().st_size / 1024
        print(f"  Saved: {filepath} ({size_kb:.0f} KB)")

    print("Done!")


if __name__ == "__main__":
    main()
