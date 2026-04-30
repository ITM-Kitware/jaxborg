"""Evaluate a JAXborg-trained policy in pure CybORG."""

import argparse
import json
import pickle
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.SimpleAgents.FSMRedVariants import DiscoveryFSRed
from CybORG.Agents.SimpleAgents.RandomSelectRedAgent import RandomSelectRedAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

RED_AGENT_CLASSES = {
    "fsm": FiniteStateRedAgent,
    "discovery": DiscoveryFSRed,
    "random": RandomSelectRedAgent,
}

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.policy import ActorCritic, LegacyActor, SharedActorCritic
from jaxborg.topology import build_const_from_cyborg, cyborg_bank_seed_from_seed
from jaxborg.translate import (
    build_mappings_from_cyborg,
    cyborg_blue_to_jax,
    jax_blue_to_cyborg,
)

EPISODE_LENGTH = 500


def make_env(seed=None, bank_match_size=None, red_agent="fsm"):
    actual_seed = cyborg_bank_seed_from_seed(seed, bank_match_size) if bank_match_size is not None else seed
    red_cls = RED_AGENT_CLASSES[red_agent]
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_cls,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=actual_seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def load_checkpoint(path):
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    nested_params = ckpt["params"].get("params", {})

    if "actor_head" in nested_params:
        policy = ActorCritic(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "current"

    if "Dense_0" in nested_params:
        if ckpt["action_dim"] != BLUE_ALLOW_TRAFFIC_END:
            raise ValueError(
                f"Legacy checkpoint action_dim={ckpt['action_dim']} is incompatible with current action space "
                f"{BLUE_ALLOW_TRAFFIC_END}"
            )
        dense_count = sum(1 for k in nested_params if k.startswith("Dense_"))
        if dense_count >= 4:
            policy = SharedActorCritic(
                action_dim=ckpt["action_dim"],
                hidden_dim=ckpt["hidden_dim"],
                activation=ckpt["activation"],
            )
            return policy, ckpt["params"], "shared"

        policy = LegacyActor(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "legacy"

    raise ValueError(f"Unrecognized checkpoint format: nested params keys={sorted(nested_params.keys())}")


def policy_dist(policy, params, policy_kind, obs_jax, mask):
    if policy_kind == "current":
        return policy.apply(params, obs_jax, mask, method=ActorCritic.actor)
    if policy_kind == "shared":
        return policy.apply(params, obs_jax, mask, method=SharedActorCritic.actor)
    return policy.apply(params, obs_jax, mask)


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    cls_name = type(action).__name__
    agent_id = int(agent_name.split("_")[-1])

    if label.startswith("[Padding]"):
        return []

    if cls_name == "Sleep" and not label.startswith("[Invalid]"):
        return [BLUE_SLEEP]

    if cls_name == "Sleep" and label.startswith("[Invalid]"):
        return []

    if cls_name == "DeployDecoy":
        if action.hostname not in mappings.hostname_to_idx:
            return []
        host_idx = mappings.hostname_to_idx[action.hostname]
        jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
        if jax_idx == BLUE_SLEEP:
            return []
        return [jax_idx]

    try:
        jax_idx = cyborg_blue_to_jax(action, agent_name, mappings, const=const)
        if jax_idx == BLUE_SLEEP:
            return []  # host not in agent's observed subnets
        return [jax_idx]
    except (KeyError, ValueError):
        return []


def _build_action_lookup(env, agent_name, mappings, const):
    """Precompute cyborg_action_idx -> list[jax_idx] for one agent. Call once per episode."""
    controller = env.env.environment_controller
    cyborg_actions = env.actions(agent_name)
    cyborg_labels = env.action_labels(agent_name)
    cyborg_state = controller.state
    lookup = []
    for action, label in zip(cyborg_actions, cyborg_labels):
        jax_indices = _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state)
        lookup.append(jax_indices)
    return lookup


def _live_cyborg_mask_in_jax_space(env, agent_name, mappings, const, lookup=None):
    controller = env.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        # Force Sleep during pending ticks to avoid CybORG re-charging
        # action_cost for the resubmitted (silently dropped) action.
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        jax_mask[BLUE_SLEEP] = True
        return jnp.array(jax_mask)

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    cyborg_mask = env.get_action_space(agent_name)["mask"]

    if lookup is not None:
        for cyborg_idx, valid in enumerate(cyborg_mask):
            if valid:
                for jax_idx in lookup[cyborg_idx]:
                    jax_mask[jax_idx] = True
    else:
        cyborg_actions = env.actions(agent_name)
        cyborg_labels = env.action_labels(agent_name)
        cyborg_state = controller.state
        for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
            if not valid:
                continue
            for jax_idx in _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
                jax_mask[jax_idx] = True

    return jnp.array(jax_mask)


def _raw_cyborg_step_with_flat_obs(wrapper, actions, messages=None):
    """Step underlying CybORG with raw actions, then flatten blue observations via the wrapper."""
    obs, rews, dones, info = wrapper.env.parallel_step(
        actions,
        messages=messages,
        skip_valid_action_check=True,
    )

    observations = {
        agent: wrapper.observation_change(agent, obs[agent]) for agent in wrapper.possible_agents if agent in obs
    }
    rewards = {agent: sum(rews[agent].values()) for agent in wrapper.possible_agents if agent in rews}
    terminated = {agent: bool(dones[agent]) for agent in wrapper.possible_agents if agent in dones}
    truncated = terminated.copy()
    info = {agent: {"action_mask": wrapper.get_action_space(agent)["mask"]} for agent in wrapper.possible_agents}
    wrapper.agents = [agent for agent in wrapper.possible_agents if not terminated.get(agent, False)]
    return observations, rewards, terminated, truncated, info


def _policy_input_dim(params):
    """Read the first Dense kernel's input dim — works for current/shared/legacy."""
    nested = params.get("params", {})
    for key in ("Dense_0", "trunk_0", "actor_dense_0"):
        if key in nested and "kernel" in nested[key]:
            return nested[key]["kernel"].shape[0]
    # Fall back: walk and grab any first kernel
    for v in nested.values():
        if isinstance(v, dict) and "kernel" in v:
            return v["kernel"].shape[0]
    return None


def _capture_agent_records(unwrap, agent_ids):
    """Snapshot last action + observation success for each agent (for CIA scoring)."""
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


def run_episode(
    env, policy, params, policy_kind, deterministic, rng,
    obs_pad_tail=None, traj_writer=None, traj_meta=None,
):
    observations, _ = env.reset()
    inner_cyborg = env.env
    const = build_const_from_cyborg(inner_cyborg)
    mappings = build_mappings_from_cyborg(inner_cyborg)

    # Precompute action translation tables (once per episode, ~600x faster per step)
    action_lookups = {agent_name: _build_action_lookup(env, agent_name, mappings, const) for agent_name in env.agents}

    if traj_writer is not None:
        ec = inner_cyborg.environment_controller
        hosts = list(ec.state.hosts.keys())
        header = {"type": "header", "hosts": hosts, **(traj_meta or {})}
        traj_writer.write(json.dumps(header) + "\n")

    red_ids: list[str] = []
    blue_ids: list[str] = []
    green_ids: list[str] = []

    total = 0.0
    for t in range(EPISODE_LENGTH):
        actions = {}
        for agent_idx, agent_name in enumerate(env.agents):
            obs_vec = observations[agent_name]
            obs_jax = jnp.array(obs_vec, dtype=jnp.float32)
            if obs_pad_tail is not None and obs_pad_tail.size > 0:
                obs_jax = jnp.concatenate([obs_jax, obs_pad_tail])

            mask = _live_cyborg_mask_in_jax_space(env, agent_name, mappings, const, lookup=action_lookups[agent_name])

            pi = policy_dist(policy, params, policy_kind, obs_jax, mask)

            if deterministic:
                action_idx = int(jnp.argmax(pi.logits))
            else:
                rng, _rng = jax.random.split(rng)
                action_idx = int(pi.sample(seed=_rng))

            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

        observations, rewards, terminations, truncations, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
        step_reward = mean(rewards.values())
        total += step_reward

        if traj_writer is not None:
            ec = inner_cyborg.environment_controller
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
                "reward": step_reward,
                "red": _capture_agent_records(inner_cyborg, red_ids),
                "blue": _capture_agent_records(inner_cyborg, blue_ids),
                "green": _capture_agent_records(inner_cyborg, green_ids),
            }
            traj_writer.write(json.dumps(record) + "\n")

        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break

    if traj_writer is not None:
        footer = {"type": "footer", "total_reward": total, "steps": t + 1}
        traj_writer.write(json.dumps(footer) + "\n")

    return total


def evaluate(
    checkpoint_path,
    episodes,
    seed,
    deterministic,
    bank_match_size=None,
    red_agent="fsm",
    output=None,
    mission_multipliers=(0.0, 0.0, 0.0),
    trajectory_dir=None,
):
    policy, params, policy_kind = load_checkpoint(checkpoint_path)
    rng = jax.random.PRNGKey(seed if seed is not None else 0)

    # Auto-pad CybORG's 210-dim obs to whatever the policy expects.
    expected_dim = _policy_input_dim(params)
    obs_pad_tail = None
    if expected_dim is not None and expected_dim > 210:
        pad_size = expected_dim - 210
        if pad_size != len(mission_multipliers):
            raise ValueError(
                f"Policy expects {expected_dim}-dim obs (pad {pad_size}); "
                f"--mission-multipliers must have {pad_size} values"
            )
        obs_pad_tail = jnp.array(mission_multipliers, dtype=jnp.float32)
        print(f"Padding CybORG obs 210 -> {expected_dim} with {list(mission_multipliers)}")

    if trajectory_dir is not None:
        Path(trajectory_dir).mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    for ep in range(episodes):
        env = make_env(seed=seed + ep, bank_match_size=bank_match_size, red_agent=red_agent)
        rng, _rng = jax.random.split(rng)
        traj_writer = None
        if trajectory_dir is not None:
            traj_path = Path(trajectory_dir) / f"ep{ep:04d}_seed{seed + ep}.jsonl"
            traj_writer = traj_path.open("w")
        traj_meta = {
            "checkpoint": str(checkpoint_path),
            "red_agent": red_agent,
            "seed": seed + ep,
            "episode": ep,
        }
        try:
            reward = run_episode(
                env, policy, params, policy_kind, deterministic, _rng,
                obs_pad_tail=obs_pad_tail,
                traj_writer=traj_writer,
                traj_meta=traj_meta,
            )
        finally:
            if traj_writer is not None:
                traj_writer.close()
        episode_rewards.append(reward)
        print(f"Episode {ep + 1}: {reward:.4f}")

    mean_r = mean(episode_rewards)
    sd_r = stdev(episode_rewards) if len(episode_rewards) > 1 else 0.0
    print(f"\nepisodes:  {episodes}")
    print(f"mean:      {mean_r:.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {sd_r:.4f}")

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "red_agent": red_agent,
                    "episodes": episodes,
                    "rewards": episode_rewards,
                    "mean": mean_r,
                    "stdev": sd_r,
                    "seed": seed,
                    "deterministic": deterministic,
                    "bank_match_size": bank_match_size,
                },
                indent=2,
            )
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate JAXborg policy in CybORG")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--bank-match-size", type=int, default=32, help="Topology bank size for seed matching")
    parser.add_argument("--red-agent", choices=sorted(RED_AGENT_CLASSES.keys()), default="fsm")
    parser.add_argument("--output", default=None, help="Optional JSON path for summary")
    parser.add_argument(
        "--mission-multipliers",
        default="0,0,0",
        help="Comma-separated values to append to obs (matches Phase 3 mission_multipliers slots).",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="If set, write one .jsonl trajectory per episode (for post-hoc CIA scoring).",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.deterministic and args.stochastic:
        raise ValueError("Choose at most one of --deterministic or --stochastic")
    deterministic = not args.stochastic
    if args.deterministic:
        deterministic = True
    mission_mult = tuple(float(x) for x in args.mission_multipliers.split(","))
    evaluate(
        args.checkpoint,
        args.episodes,
        args.seed,
        deterministic,
        args.bank_match_size,
        red_agent=args.red_agent,
        output=args.output,
        mission_multipliers=mission_mult,
        trajectory_dir=args.trajectory_dir,
    )


if __name__ == "__main__":
    main()
