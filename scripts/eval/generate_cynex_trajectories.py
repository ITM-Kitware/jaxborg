#!/usr/bin/env python
"""Generate cynex V2 trajectory JSON for both JAXborg (JAX/Flax) and CybORG (PyTorch) policies.

Both policies are run through CybORG to produce consistent trajectories.

Usage:
    # CybORG PyTorch policy:
    python scripts/eval/generate_cynex_trajectories.py \
        --model-pt /path/to/model_cyborg_g99.pt \
        --tag cyborg-g99 --seed 42 --num-episodes 2 \
        --output-dir ../../cynex/public/data/trajectories/

    # JAXborg JAX/Flax policy:
    python scripts/eval/generate_cynex_trajectories.py \
        --model-jax /path/to/checkpoint_final.pkl \
        --tag jaxborg-g99 --seed 42 --num-episodes 2 \
        --output-dir ../../cynex/public/data/trajectories/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CybORG imports
# ---------------------------------------------------------------------------
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from export_trajectory import (
    EPISODE_LENGTH,
    _build_trajectory_dict,
    action_to_dict,
    compute_reward_breakdown,
    extract_subnet_metadata,
    extract_topology,
    get_host_compromise,
)

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

# ---------------------------------------------------------------------------
# PyTorch PPO agent (same as export_trajectory.py)
# ---------------------------------------------------------------------------
OBS_DIM = 210
ACT_DIM = 242


def _load_torch_model(path: str):
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    class PPOAgent(nn.Module):
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
            logits = logits + (action_mask.float() - 1.0) * 1e10
            if deterministic:
                return logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            return dist.sample()

    model = PPOAgent(OBS_DIM, ACT_DIM)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print(f"Loaded PyTorch model from {path}")
    return model


# ---------------------------------------------------------------------------
# JAX/Flax policy loading for JAX-trained policies
# ---------------------------------------------------------------------------
def _load_jax_model(path: str):
    """Load a JAXborg JAX/Flax checkpoint. Returns (policy, params, policy_kind)."""
    # Lazy imports so the script works without JAX when only using --model-pt
    import distrax
    import jax

    from jaxborg.evaluation.jax_runner import load_jax_checkpoint

    policy, params, recipe = load_jax_checkpoint(path)
    print(f"Loaded JAX checkpoint from {path} (arch={recipe['arch']['name']})")

    def _fwd(o, m):
        pi, _ = policy.apply(params, o, m)
        return pi.logits

    @jax.jit
    def batched_step(obs_stack, mask_stack, keys):
        logits = jax.vmap(_fwd)(obs_stack, mask_stack)
        actions = jax.vmap(lambda lg, k: distrax.Categorical(logits=lg).sample(seed=k))(logits, keys)
        return actions, logits

    return batched_step, params


# ---------------------------------------------------------------------------
# CybORG env creation
# ---------------------------------------------------------------------------
def make_cyborg_env(seed: int, steps: int = EPISODE_LENGTH):
    """Create a BlueFlatWrapper CybORG env."""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


# ---------------------------------------------------------------------------
# Mask translation: CybORG wrapper -> JAX action space
# ---------------------------------------------------------------------------
def _build_mask_cache(wrapper, mappings, const):
    """Precompute CybORG-to-JAX action translation tables."""

    from jaxborg.actions.encoding import BLUE_SLEEP, encode_blue_action

    controller = wrapper.env.environment_controller
    cyborg_state = controller.state
    cache = {}

    for agent_name in wrapper.possible_agents:
        agent_id = int(agent_name.split("_")[-1])
        cyborg_actions = wrapper.actions(agent_name)
        cyborg_labels = wrapper.action_labels(agent_name)
        agent_cache = []
        for action, label in zip(cyborg_actions, cyborg_labels):
            cls_name = type(action).__name__
            if label.startswith("[Padding]") or (cls_name == "Sleep" and label.startswith("[Invalid]")):
                agent_cache.append(None)
            elif cls_name == "DeployDecoy":
                if action.hostname not in mappings.hostname_to_idx:
                    agent_cache.append(None)
                else:
                    host_idx = mappings.hostname_to_idx[action.hostname]
                    jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
                    if jax_idx == BLUE_SLEEP:
                        agent_cache.append(None)
                    else:
                        agent_cache.append([jax_idx])
            else:
                jax_indices = _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state)
                agent_cache.append(jax_indices if jax_indices else None)
        cache[agent_name] = agent_cache
    return cache


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    """Translate a single CybORG action to JAX indices."""

    from jaxborg.parity.translate import cyborg_blue_to_jax

    try:
        jax_idx = cyborg_blue_to_jax(action, agent_name, mappings, const=const)
        return [jax_idx]
    except Exception:
        return None


def _get_jax_mask(wrapper, agent_name, mask_cache):
    """Get JAX-space action mask for an agent."""
    from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP

    controller = wrapper.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
        jax_mask[BLUE_SLEEP] = True
        return jax_mask

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_mask = action_space["mask"]
    agent_cache = mask_cache[agent_name]

    for slot_idx, valid in enumerate(cyborg_mask):
        if not valid:
            continue
        entry = agent_cache[slot_idx]
        if entry is None:
            continue
        for jax_idx in entry:
            jax_mask[jax_idx] = True

    return jax_mask


def _apply_traffic_filter(mask, blocked_zones, const, agent_id):
    """Filter no-op traffic actions from mask."""
    import jax.numpy as jnp

    from jaxborg.actions.encoding import (
        BLUE_ALLOW_TRAFFIC_END,
        BLUE_ALLOW_TRAFFIC_START,
        BLUE_BLOCK_TRAFFIC_END,
        BLUE_BLOCK_TRAFFIC_START,
    )
    from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, BLUE_TRAFFIC_SLOTS, NUM_SUBNETS

    offsets = jnp.arange(BLUE_TRAFFIC_SLOTS)
    src_offset = offsets // BLUE_MAX_OBSERVED_SUBNETS
    rel_dst = offsets % BLUE_MAX_OBSERVED_SUBNETS
    abs_dst = const.blue_obs_subnets[agent_id, rel_dst]
    src = jnp.where(src_offset >= abs_dst, src_offset + 1, src_offset)
    safe_dst = jnp.clip(abs_dst, 0, NUM_SUBNETS - 1)
    is_blocked = blocked_zones[safe_dst, src] & (abs_dst >= 0)

    mask = mask.at[BLUE_ALLOW_TRAFFIC_START:BLUE_ALLOW_TRAFFIC_END].set(
        mask[BLUE_ALLOW_TRAFFIC_START:BLUE_ALLOW_TRAFFIC_END] & is_blocked
    )
    mask = mask.at[BLUE_BLOCK_TRAFFIC_START:BLUE_BLOCK_TRAFFIC_END].set(
        mask[BLUE_BLOCK_TRAFFIC_START:BLUE_BLOCK_TRAFFIC_END] & ~is_blocked
    )
    return mask


def _cyborg_blocked_zones(controller):
    """Extract CybORG block state as (NUM_SUBNETS, NUM_SUBNETS) bool array."""
    import jax.numpy as jnp

    from jaxborg.constants import CYBORG_SUFFIX_TO_ID, NUM_SUBNETS

    blocked = jnp.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
    state = controller.state
    if not hasattr(state, "blocks"):
        return blocked

    for dst_name, src_list in state.blocks.items():
        dst_suffix = dst_name if isinstance(dst_name, str) else dst_name.value
        dst_id = CYBORG_SUFFIX_TO_ID.get(dst_suffix)
        if dst_id is None:
            continue
        for src_name in src_list:
            src_suffix = src_name if isinstance(src_name, str) else src_name.value
            src_id = CYBORG_SUFFIX_TO_ID.get(src_suffix)
            if src_id is not None:
                blocked = blocked.at[dst_id, src_id].set(True)
    return blocked


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------
def run_episode_torch(seed, episode_num, model, deterministic=False, steps=EPISODE_LENGTH):
    """Run CybORG episode with PyTorch PPO policy (delegates to export_trajectory)."""
    from export_trajectory import run_episode_policy

    return run_episode_policy(seed, episode_num, model, deterministic, steps)


def run_episode_jax(seed, episode_num, batched_step_fn, deterministic=False, steps=EPISODE_LENGTH):
    """Run CybORG episode with JAXborg JAX/Flax policy."""
    import jax
    import jax.numpy as jnp

    from jaxborg.parity.translate import build_mappings_from_cyborg, jax_blue_to_cyborg
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    # Create env with BlueFlatWrapper.
    wrapper = make_cyborg_env(seed, steps)
    observations, _ = wrapper.reset()

    cyborg = wrapper.env
    ctrl = cyborg.environment_controller
    state = ctrl.state

    blue_agents = sorted(ctrl.team_assignments.get("Blue", []))
    red_agents = sorted(ctrl.team_assignments.get("Red", []))
    green_agents = sorted(ctrl.team_assignments.get("Green", []))

    # Build JAXborg topology + mappings from CybORG
    const = build_const_from_cyborg(cyborg)
    mappings = build_mappings_from_cyborg(cyborg)

    topology = extract_topology(cyborg)
    subnet_metadata = extract_subnet_metadata(cyborg)

    print(f"  Agents: {len(blue_agents)} blue, {len(red_agents)} red, {len(green_agents)} green")
    print(f"  Hosts: {len(topology)}, Subnets: {len(subnet_metadata)}")

    mask_cache = _build_mask_cache(wrapper, mappings, const)

    agent_actions: dict[str, list] = {a: [] for a in blue_agents + red_agents + green_agents}
    step_states = []
    cumulative_rewards = {a: 0.0 for a in blue_agents}
    actual_steps = 0
    rng = jax.random.PRNGKey(seed)

    for step in range(steps):
        if wrapper.agents:
            rng, *_rngs = jax.random.split(rng, NUM_AGENTS + 1)
            act_keys = jnp.stack(_rngs)

            # Build JAX-space masks with traffic filtering
            blocked_zones = _cyborg_blocked_zones(ctrl)
            raw_masks = [_get_jax_mask(wrapper, a, mask_cache) for a in wrapper.agents]
            masks = jnp.stack(
                [_apply_traffic_filter(jnp.array(m), blocked_zones, const, i) for i, m in enumerate(raw_masks)]
            )
            obs_stack = jnp.stack([jnp.array(observations[a], dtype=jnp.float32) for a in wrapper.agents])

            # JAX policy inference
            actions_arr, _ = batched_step_fn(obs_stack, masks, act_keys)
            actions_np = np.asarray(actions_arr)

            # Translate JAX actions -> CybORG actions
            cyborg_actions = {}
            for agent_idx, agent_name in enumerate(wrapper.agents):
                jax_action_idx = int(actions_np[agent_idx])
                cyborg_actions[agent_name] = jax_blue_to_cyborg(jax_action_idx, agent_idx, mappings, const=const)
        else:
            cyborg_actions = {a: Sleep() for a in wrapper.possible_agents}

        # Step CybORG directly with the translated actions
        obs_raw, rews, dones, _info = cyborg.parallel_step(
            cyborg_actions,
            skip_valid_action_check=True,
        )
        # Flatten observations through the wrapper
        observations = {
            agent: wrapper.observation_change(agent, obs_raw[agent])
            for agent in wrapper.possible_agents
            if agent in obs_raw
        }
        rewards = {agent: sum(rews[agent].values()) for agent in wrapper.possible_agents if agent in rews}
        wrapper.agents = [agent for agent in wrapper.possible_agents if not (dones.get(agent, False))]

        state = ctrl.state
        actual_steps = step + 1

        # Record each agent's action from CybORG
        for agent in blue_agents + red_agents + green_agents:
            last = cyborg.get_last_action(agent)
            action = last[0] if isinstance(last, list) and last else last

            success = "TRUE"
            raw_obs_dict = cyborg.get_observation(agent)
            if isinstance(raw_obs_dict, dict) and "success" in raw_obs_dict:
                val = raw_obs_dict["success"]
                success = val.name if hasattr(val, "name") else str(val)

            agent_actions[agent].append(action_to_dict(action, step, state, success))

        # Record step state
        compromise = get_host_compromise(state, red_agents)
        breakdown = compute_reward_breakdown(cyborg, state, green_agents, red_agents)
        step_rewards = {}
        reward_val = rewards.get(AGENT_IDS[0], 0.0)
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
                "reward_breakdown": breakdown,
            }
        )

        if step % 100 == 0:
            compromised = sum(1 for v in compromise.values() if v != "NONE")
            cum = cumulative_rewards[blue_agents[0]]
            print(
                f"  Step {step}: phase={state.mission_phase}, "
                f"compromised={compromised}/{len(compromise)}, cum_reward={cum:.1f}"
            )

        if dones.get("__all__", False):
            print(f"  Episode ended at step {step}")
            break

    return _build_trajectory_dict(
        episode_num,
        seed,
        actual_steps,
        "JAXborg-PPO",
        blue_agents,
        red_agents,
        green_agents,
        topology,
        subnet_metadata,
        agent_actions,
        step_states,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate cynex V2 trajectory JSON from JAXborg or CybORG policies")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--num-episodes", type=int, default=2, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="Steps per episode")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--tag", type=str, required=True, help="Filename tag (e.g., jaxborg-g99)")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic (argmax) actions")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-pt", type=str, help="Path to PyTorch PPO model .pt file")
    group.add_argument("--model-jax", type=str, help="Path to JAXborg JAX/Flax checkpoint .pkl file")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the appropriate model
    if args.model_pt:
        torch_model = _load_torch_model(args.model_pt)
        jax_model = None
    else:
        torch_model = None
        jax_model, _ = _load_jax_model(args.model_jax)

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        print(f"\nEpisode {ep} (seed={seed}):")
        t0 = time.perf_counter()

        if torch_model is not None:
            trajectory = run_episode_torch(seed, ep, torch_model, args.deterministic, args.steps)
        else:
            trajectory = run_episode_jax(seed, ep, jax_model, args.deterministic, args.steps)

        elapsed = time.perf_counter() - t0
        filename = f"cc4-{args.tag}-seed{seed}-E{ep}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        cum_reward = trajectory["step_states"][-1]["cumulative_reward"]
        first_blue = list(cum_reward.values())[0]
        print(f"  Saved: {filepath} ({size_mb:.1f} MB, reward={first_blue:.1f}, {elapsed:.1f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
