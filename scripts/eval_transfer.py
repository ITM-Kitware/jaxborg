"""Evaluate JAXborg-trained policy: JAXborg rollout, CybORG transfer, baselines, diagnostics."""

# ruff: noqa: E402

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory
from flax.linen.initializers import constant, orthogonal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from train_ippo_cc4 import ActorCritic

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
    encode_blue_action,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import (
    ACTION_HOST_SLOTS,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    OBS_HOSTS_PER_SUBNET,
)
from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxborg.observations import get_blue_obs
from jaxborg.topology import build_const_from_cyborg, cyborg_bank_seed_from_seed
from jaxborg.translate import (
    build_mappings_from_cyborg,
    cyborg_blue_to_jax,
    describe_blue_action,
    jax_blue_to_cyborg,
)
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import compare_snapshots, extract_cyborg_snapshot, extract_jax_snapshot

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()

ACTION_TYPE_NAMES = [
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "Decoy",
    "BlockTraffic",
    "AllowTraffic",
]

ACTION_TYPE_RANGES = [
    (BLUE_SLEEP, BLUE_SLEEP + 1),
    (BLUE_MONITOR, BLUE_MONITOR + 1),
    (BLUE_ANALYSE_START, BLUE_ANALYSE_START + ACTION_HOST_SLOTS),
    (BLUE_REMOVE_START, BLUE_REMOVE_START + ACTION_HOST_SLOTS),
    (BLUE_RESTORE_START, BLUE_RESTORE_START + ACTION_HOST_SLOTS),
    (BLUE_DECOY_START, BLUE_BLOCK_TRAFFIC_START),
    (BLUE_BLOCK_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_START),
    (BLUE_ALLOW_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_END),
]
DECOY_FACTORY_ACTIONS = (
    (HarakaDecoyFactory(), "DeployDecoy_HarakaSMPT"),
    (ApacheDecoyFactory(), "DeployDecoy_Apache"),
    (TomcatDecoyFactory(), "DeployDecoy_Tomcat"),
    (VsftpdDecoyFactory(), "DeployDecoy_Vsftpd"),
)


class LegacyActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        if avail_actions is not None:
            action_logits = action_logits - ((1 - avail_actions) * 1e10)

        return distrax.Categorical(logits=action_logits)


def classify_action(action_idx: int) -> int:
    for i, (start, end) in enumerate(ACTION_TYPE_RANGES):
        if start <= action_idx < end:
            return i
    return 0


def action_distribution(actions):
    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(int(a))] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


@dataclass
class StepSnapshot:
    reward: float
    cumulative_reward: float
    hosts_compromised_user: int
    hosts_compromised_priv: int
    red_sessions_total: int
    mission_phase: int


@dataclass
class EpisodeResult:
    actions_by_agent: list = field(default_factory=list)  # [agent_idx][step] = action_id
    rewards: list = field(default_factory=list)  # per-step rewards
    cumulative_reward: float = 0.0
    trajectory: list = field(default_factory=list)  # list[StepSnapshot]


def print_per_agent_action_dist(all_actions_by_agent, label="JAXborg"):
    """Print per-agent action distribution table."""
    n_agents = len(all_actions_by_agent)
    if n_agents == 0:
        return
    header = f"{'Agent':<10}"
    for name in ACTION_TYPE_NAMES:
        header += f" {name:>8}"
    print(f"\nPer-Agent Action Distribution ({label}):")
    print(header)
    print("-" * len(header))
    for agent_idx in range(n_agents):
        dist = action_distribution(all_actions_by_agent[agent_idx])
        row = f"{'blue_' + str(agent_idx):<10}"
        for pct in dist:
            row += f" {pct * 100:7.1f}%"
        print(row)


def print_trajectory_summary(trajectory, label="JAXborg ep"):
    """Print compact trajectory table sampled at key steps and phase boundaries."""
    if not trajectory:
        return
    # Collect indices to show: every 50 steps + phase transitions + last step
    shown = set()
    prev_phase = -1
    for i, snap in enumerate(trajectory):
        if i % 50 == 0 or i == len(trajectory) - 1:
            shown.add(i)
        if snap.mission_phase != prev_phase:
            shown.add(i)
            prev_phase = snap.mission_phase

    phase_labels = {1: "MissionA", 2: "MissionB"}
    print(f"\nTrajectory ({label}):")
    print(f" {'Step':>4}  {'Phase':>5}  {'Reward':>7}  {'CumRew':>8}  {'Compromised(U/P)':>17}  {'RedSessions':>11}")
    for i in sorted(shown):
        s = trajectory[i]
        marker = ""
        if i > 0 and trajectory[i].mission_phase != trajectory[i - 1].mission_phase:
            marker = f"  <- {phase_labels.get(s.mission_phase, f'Phase{s.mission_phase}')}"
        print(
            f" {i:>4}  {s.mission_phase:>5}  {s.reward:>7.1f}  {s.cumulative_reward:>8.1f}"
            f"  {s.hosts_compromised_user:>8}/{s.hosts_compromised_priv:<8}"
            f"  {s.red_sessions_total:>11}{marker}"
        )


def load_checkpoint(path):
    with open(path, "rb") as f:
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
    return policy.apply(params, obs_jax, mask)


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    """Translate a CybORG blue action into the JAX canonical action space."""
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
        host = cyborg_state.hosts[action.hostname]
        host_idx = mappings.hostname_to_idx[action.hostname]
        return [
            encode_blue_action(action_name, host_idx, agent_id, const=const)
            for factory, action_name in DECOY_FACTORY_ACTIONS
            if factory.is_host_compatible(host)
        ]

    try:
        return [cyborg_blue_to_jax(action, agent_name, mappings, const=const)]
    except (KeyError, ValueError):
        return []


def _live_cyborg_mask_in_jax_space(env, agent_name, info, mappings, const):
    """Project CybORG's live action mask into JAX canonical indices."""
    controller = env.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in _cyborg_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
        return jnp.array(jax_mask)

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    cyborg_mask = info[agent_name]["action_mask"]
    cyborg_actions = env.actions(agent_name)
    cyborg_labels = env.action_labels(agent_name)
    cyborg_state = controller.state

    for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
        if not valid:
            continue
        for jax_idx in _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
            jax_mask[jax_idx] = True

    return jnp.array(jax_mask)


def _live_blue_wrapper_mask_in_jax_space(wrapper, agent_name, mappings, const):
    """Project BlueFlatWrapper's live action mask into JAX canonical indices."""
    controller = wrapper.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in _cyborg_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
        return jnp.array(jax_mask)

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    action_space = wrapper.get_action_space(agent_name)
    cyborg_mask = action_space["mask"]
    cyborg_actions = wrapper.actions(agent_name)
    cyborg_labels = wrapper.action_labels(agent_name)
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


def make_cyborg_env(seed=42, bank_match_size=None):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    actual_seed = cyborg_bank_seed_from_seed(seed, bank_match_size) if bank_match_size is not None else seed
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=actual_seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def _inject_live_red_policy_step(env_state, recorder):
    """Inject CybORG-recorded red choice tokens for the current JAX step."""
    step_idx = int(env_state.state.time)
    step_tokens = jnp.asarray(recorder.extract_step(step_idx), dtype=jnp.float32)
    return env_state.replace(
        const=env_state.const.replace(
            red_policy_randoms=env_state.const.red_policy_randoms.at[step_idx].set(step_tokens),
            use_red_policy_randoms=jnp.array(True),
        )
    )


# --- Core rollout functions ---


def rollout_jaxborg(policy, params, policy_kind, num_episodes=3, deterministic=False, seed=0):
    env = FsmRedCC4Env(num_steps=500, topology_mode="cyborg_bank", topology_bank_size=32)
    all_actions = []
    episode_rewards = []
    episode_results = []

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        key = jax.random.PRNGKey(seed + ep * 100)
        obs, env_state = env.reset(key)

        ep_reward = np.zeros(NUM_BLUE_AGENTS)
        ep_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            actions = {}
            for agent_idx in range(NUM_BLUE_AGENTS):
                agent = f"blue_{agent_idx}"
                avail = compute_blue_action_mask(env_state.const, agent_idx, env_state.state)
                pi = policy_dist(policy, params, policy_kind, obs[agent], avail)

                if deterministic:
                    action = jnp.argmax(pi.logits)
                else:
                    action = pi.sample(seed=act_keys[agent_idx])

                actions[agent] = action
                ep_actions_by_agent[agent_idx].append(int(action))

            obs, env_state, rewards, dones, _ = env.step(step_key, env_state, actions)
            step_reward = float(np.mean([float(rewards[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]))
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)
            for i in range(NUM_BLUE_AGENTS):
                ep_reward[i] += float(rewards[f"blue_{i}"])

            # Extract trajectory snapshot from state
            st = env_state.state
            active = np.array(env_state.const.host_active, dtype=bool)
            compromised = np.array(st.host_compromised)
            ep_trajectory.append(
                StepSnapshot(
                    reward=step_reward,
                    cumulative_reward=cum_reward,
                    hosts_compromised_user=int(np.sum((compromised == COMPROMISE_USER) & active)),
                    hosts_compromised_priv=int(np.sum((compromised == COMPROMISE_PRIVILEGED) & active)),
                    red_sessions_total=int(np.sum(np.array(st.red_sessions)[:NUM_RED_AGENTS])),
                    mission_phase=int(st.mission_phase),
                )
            )

            if dones["__all__"]:
                break

        elapsed = time.perf_counter() - t0
        total = ep_reward.mean()
        print(f"  JAXborg ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")

        # Flatten per-agent actions for backward compat
        flat_actions = [a for step_actions in zip(*ep_actions_by_agent) for a in step_actions]
        all_actions.extend(flat_actions)
        episode_rewards.append(total)
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_actions_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=cum_reward,
                trajectory=ep_trajectory,
            )
        )

    return np.array(all_actions), np.array(episode_rewards), episode_results


def rollout_cyborg(policy, params, policy_kind, num_episodes=3, deterministic=False, seed=0):
    all_actions = []
    episode_rewards = []
    all_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        env = make_cyborg_env(seed=seed + ep * 100, bank_match_size=32)
        observations, _ = env.reset()
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)

        rng = jax.random.PRNGKey(seed + ep * 100)
        total = 0.0
        ep_actions = []

        for _ in range(500):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
                mask = _live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const)
                pi = policy_dist(policy, params, policy_kind, obs_jax, mask)

                if deterministic:
                    action_idx = int(jnp.argmax(pi.logits))
                else:
                    rng, _rng = jax.random.split(rng)
                    action_idx = int(pi.sample(seed=_rng))

                cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
                actions[agent_name] = cyborg_action
                ep_actions.append(action_idx)
                all_actions_by_agent[agent_idx].append(action_idx)

            observations, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
            total += mean(rewards.values())

        elapsed = time.perf_counter() - t0
        print(f"  CybORG  ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")
        all_actions.extend(ep_actions)
        episode_rewards.append(total)

    return np.array(all_actions), np.array(episode_rewards), all_actions_by_agent


def rollout_independent_transfer_synced_red(policy, params, policy_kind, num_episodes=3, deterministic=False, seed=0):
    """Run paired independent episodes with live CybORG red-choice sync into JAX.

    Blue actions are still chosen independently from each environment's own
    observations. Red stochastic choices are synced from CybORG step-by-step so
    any remaining drift is due to simulator state/observation differences rather
    than different RNG engines.
    """

    all_jax_actions = []
    all_cyborg_actions = []
    jax_rewards = []
    cyborg_rewards = []
    jax_results = []
    cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        ep_seed = seed + ep * 100

        jax_env = FsmRedCC4Env(num_steps=500, topology_mode="cyborg_bank", topology_bank_size=32)
        key = jax.random.PRNGKey(ep_seed)
        jax_obs, jax_state = jax_env.reset(key)

        cyborg_env = make_cyborg_env(seed=ep_seed, bank_match_size=32)
        cyborg_obs, _ = cyborg_env.reset()
        inner = cyborg_env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        red_recorder = RedPolicyRecorder()
        red_recorder.install(inner, mappings)

        ep_jax_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        jax_total = 0.0
        cyborg_total = 0.0
        first_action_diff = None
        first_state_diff = None

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS + 1)

            jax_blue_actions = {}
            cyborg_blue_actions = {}
            cyborg_actions = {}

            for agent_idx in range(NUM_BLUE_AGENTS):
                jax_agent = f"blue_{agent_idx}"
                cyborg_agent = f"blue_agent_{agent_idx}"

                jax_mask = compute_blue_action_mask(jax_state.const, agent_idx, jax_state.state)
                jax_pi = policy_dist(policy, params, policy_kind, jax_obs[jax_agent], jax_mask)
                if deterministic:
                    jax_action = int(jnp.argmax(jax_pi.logits))
                else:
                    jax_action = int(jax_pi.sample(seed=act_keys[agent_idx]))

                cyborg_mask = _live_blue_wrapper_mask_in_jax_space(cyborg_env, cyborg_agent, mappings, const)
                cyborg_pi = policy_dist(
                    policy,
                    params,
                    policy_kind,
                    jnp.array(cyborg_obs[cyborg_agent], dtype=jnp.float32),
                    cyborg_mask,
                )
                if deterministic:
                    cyborg_action = int(jnp.argmax(cyborg_pi.logits))
                else:
                    cyborg_action = int(cyborg_pi.sample(seed=act_keys[agent_idx]))

                jax_blue_actions[jax_agent] = jnp.int32(jax_action)
                cyborg_blue_actions[agent_idx] = cyborg_action
                cyborg_actions[cyborg_agent] = jax_blue_to_cyborg(cyborg_action, agent_idx, mappings, const=const)

                ep_jax_actions_by_agent[agent_idx].append(jax_action)
                ep_cyborg_actions_by_agent[agent_idx].append(cyborg_action)
                cyborg_actions_by_agent[agent_idx].append(cyborg_action)

            if first_action_diff is None:
                action_vec_jax = [int(jax_blue_actions[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]
                action_vec_cy = [int(cyborg_blue_actions[i]) for i in range(NUM_BLUE_AGENTS)]
                if action_vec_jax != action_vec_cy:
                    first_action_diff = (step, action_vec_jax, action_vec_cy)

            cyborg_obs, cyborg_step_rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(
                cyborg_env, actions=cyborg_actions
            )
            cyborg_step_reward = float(mean(cyborg_step_rewards.values()))
            cyborg_total += cyborg_step_reward

            jax_state = _inject_live_red_policy_step(jax_state, red_recorder)
            jax_obs, jax_state, jax_step_rewards, _, _ = jax_env.step(step_key, jax_state, jax_blue_actions)
            jax_step_reward = float(np.mean([float(jax_step_rewards[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]))
            jax_total += jax_step_reward

            if first_state_diff is None:
                diffs = compare_snapshots(
                    extract_cyborg_snapshot(inner, mappings),
                    extract_jax_snapshot(jax_state.state, jax_state.const, mappings),
                )
                if diffs:
                    first_state_diff = (step, diffs[0])

            ep_step_rewards.append(jax_step_reward)
            st = jax_state.state
            active = np.array(jax_state.const.host_active, dtype=bool)
            compromised = np.array(st.host_compromised)
            ep_trajectory.append(
                StepSnapshot(
                    reward=jax_step_reward,
                    cumulative_reward=jax_total,
                    hosts_compromised_user=int(np.sum((compromised == COMPROMISE_USER) & active)),
                    hosts_compromised_priv=int(np.sum((compromised == COMPROMISE_PRIVILEGED) & active)),
                    red_sessions_total=int(np.sum(np.array(st.red_sessions)[:NUM_RED_AGENTS])),
                    mission_phase=int(st.mission_phase),
                )
            )

        elapsed = time.perf_counter() - t0
        print(f"  Independent ep {ep + 1}: JAX={jax_total:.1f} CybORG={cyborg_total:.1f} ({elapsed:.1f}s)")
        if first_action_diff is None:
            print("    first blue action diff: none")
        else:
            step_idx, jv, cv = first_action_diff
            print(f"    first blue action diff: step {step_idx} jax={jv} cyborg={cv}")
        if first_state_diff is None:
            print("    first state diff: none")
        else:
            step_idx, diff = first_state_diff
            print(
                "    first state diff: "
                f"step {step_idx} {diff.field_name} {diff.host_or_agent} "
                f"cyborg={diff.cyborg_value} jax={diff.jax_value}"
            )

        flat_jax_actions = [a for step_actions in zip(*ep_jax_actions_by_agent) for a in step_actions]
        flat_cyborg_actions = [a for step_actions in zip(*ep_cyborg_actions_by_agent) for a in step_actions]
        all_jax_actions.extend(flat_jax_actions)
        all_cyborg_actions.extend(flat_cyborg_actions)
        jax_rewards.append(jax_total)
        cyborg_rewards.append(cyborg_total)
        jax_results.append(
            EpisodeResult(
                actions_by_agent=ep_jax_actions_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=jax_total,
                trajectory=ep_trajectory,
            )
        )

    return (
        np.array(all_jax_actions),
        np.array(jax_rewards),
        jax_results,
        np.array(all_cyborg_actions),
        np.array(cyborg_rewards),
        cyborg_actions_by_agent,
    )


def rollout_matched_transfer(policy, params, policy_kind, num_episodes=3, deterministic=False, seed=0):
    """Compare policy outputs on matched JAX/CybORG states.

    JAX-selected actions drive the synced rollout so the underlying episode stays
    matched. CybORG-selected actions are recorded from the same synced states for
    transfer diagnostics, not applied.
    """

    all_jax_actions = []
    all_cyborg_actions = []
    episode_rewards = []
    episode_results = []
    all_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        harness = CC4DifferentialHarness(seed=seed + ep * 100, check_obs=True, sync_green_rng=True)
        harness.reset()

        ep_jax_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for _ in range(500):
            jax_actions = {}
            for agent_idx in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_agent_{agent_idx}"

                jax_obs = get_blue_obs(harness.jax_state, harness.jax_const, agent_idx)
                jax_mask = compute_blue_action_mask(harness.jax_const, agent_idx, harness.jax_state)
                jax_pi = policy_dist(policy, params, policy_kind, jax_obs, jax_mask)

                cyborg_obs_dict = harness.cyborg_env.get_observation(agent_name)
                cyborg_obs = jnp.array(
                    harness._blue_wrapper.observation_change(agent_name, cyborg_obs_dict),
                    dtype=jnp.float32,
                )
                cyborg_mask = _live_blue_wrapper_mask_in_jax_space(
                    harness._blue_wrapper,
                    agent_name,
                    harness.mappings,
                    harness.jax_const,
                )
                cyborg_pi = policy_dist(policy, params, policy_kind, cyborg_obs, cyborg_mask)

                if deterministic:
                    jax_action = int(jnp.argmax(jax_pi.logits))
                    cyborg_action = int(jnp.argmax(cyborg_pi.logits))
                else:
                    raise ValueError("Matched transfer mode currently requires deterministic evaluation")

                jax_actions[agent_idx] = jax_action
                ep_jax_actions_by_agent[agent_idx].append(jax_action)
                ep_cyborg_actions_by_agent[agent_idx].append(cyborg_action)
                all_cyborg_actions_by_agent[agent_idx].append(cyborg_action)

            result = harness.full_step(blue_actions=jax_actions)
            if result.diffs:
                details = ", ".join(f"{d.field_name}:{d.host_or_agent}" for d in result.diffs[:5])
                raise RuntimeError(f"Matched transfer replay diverged at step {result.step}: {details}")

            step_reward = float(
                harness.cyborg_env.environment_controller.reward.get("Blue", {}).get("BlueRewardMachine", 0.0)
            )
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)

            st = harness.jax_state
            active = np.array(harness.jax_const.host_active, dtype=bool)
            compromised = np.array(st.host_compromised)
            ep_trajectory.append(
                StepSnapshot(
                    reward=step_reward,
                    cumulative_reward=cum_reward,
                    hosts_compromised_user=int(np.sum((compromised == COMPROMISE_USER) & active)),
                    hosts_compromised_priv=int(np.sum((compromised == COMPROMISE_PRIVILEGED) & active)),
                    red_sessions_total=int(np.sum(np.array(st.red_sessions)[:NUM_RED_AGENTS])),
                    mission_phase=int(st.mission_phase),
                )
            )

        elapsed = time.perf_counter() - t0
        print(f"  Matched ep {ep + 1}: reward={cum_reward:.1f} ({elapsed:.1f}s)")

        flat_jax_actions = [a for step_actions in zip(*ep_jax_actions_by_agent) for a in step_actions]
        flat_cyborg_actions = [a for step_actions in zip(*ep_cyborg_actions_by_agent) for a in step_actions]
        all_jax_actions.extend(flat_jax_actions)
        all_cyborg_actions.extend(flat_cyborg_actions)
        episode_rewards.append(cum_reward)
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_jax_actions_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=cum_reward,
                trajectory=ep_trajectory,
            )
        )

    rewards = np.array(episode_rewards)
    return (
        np.array(all_jax_actions),
        rewards,
        episode_results,
        np.array(all_cyborg_actions),
        rewards.copy(),
        all_cyborg_actions_by_agent,
    )


# --- Report / plotting ---


def print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards):
    jax_dist = action_distribution(jax_actions)
    cyborg_dist = action_distribution(cyborg_actions)

    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"{'Type':<14} {'JAXborg':>8} {'CybORG':>8} {'Delta':>8}")
    print("-" * 40)
    for name, jp, cp in zip(ACTION_TYPE_NAMES, jax_dist, cyborg_dist):
        delta = jp - cp
        print(f"{name:<14} {jp * 100:7.1f}% {cp * 100:7.1f}% {delta * 100:+7.1f}%")

    print("\n" + "=" * 70)
    print("EPISODE REWARD COMPARISON")
    print("=" * 70)
    print(f"{'':14} {'JAXborg':>10} {'CybORG':>10} {'Gap':>10}")
    print("-" * 46)
    for i, (jr, cr) in enumerate(zip(jax_rewards, cyborg_rewards)):
        print(f"Episode {i + 1:5d} {jr:10.1f} {cr:10.1f} {jr - cr:+10.1f}")

    jm, cm = jax_rewards.mean(), cyborg_rewards.mean()
    print("-" * 46)
    print(f"{'Mean':14} {jm:10.1f} {cm:10.1f} {jm - cm:+10.1f}")
    if len(jax_rewards) > 1:
        js, cs = stdev(jax_rewards.tolist()), stdev(cyborg_rewards.tolist())
        print(f"{'Stdev':14} {js:10.1f} {cs:10.1f}")


def save_reward_plot(jax_rewards, cyborg_rewards):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(jax_rewards) + 1)
    ax.bar(x - 0.2, jax_rewards, 0.35, label="JAXborg")
    ax.bar(x + 0.2, cyborg_rewards, 0.35, label="CybORG")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (mean across agents)")
    ax.set_title("JAXborg vs CybORG Transfer Comparison")
    ax.legend()
    ax.set_xticks(x)
    fig.tight_layout()

    out = EXP_DIR / "transfer_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def plot_action_distribution(actions, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(a)] += 1
    counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ACTION_TYPE_NAMES, counts)
    ax.set_ylabel("Fraction of Actions")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_training_curves(metrics_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, rewards, entropies = [], [], []
    with open(metrics_path) as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["steps"])
            rewards.append(record["episode_reward_mean"])
            entropies.append(record["entropy"])
    steps, rewards, entropies = np.array(steps), np.array(rewards), np.array(entropies)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, rewards, label="JAX IPPO (masked)", linewidth=2)
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Mean Per-Agent Episode Return")
    axes[0].set_title("Reward Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, entropies, label="JAX IPPO", linewidth=2)
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Policy Entropy")
    axes[1].set_title("Entropy Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- Baseline / diagnostic functions ---


def run_sleep_baseline(episodes=5):
    from CybORG.Simulator.Actions import Sleep

    totals = []
    for ep in range(episodes):
        env = make_cyborg_env(seed=ep)
        env.reset()
        total = 0.0
        for _ in range(500):
            actions = {a: Sleep() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_random_baseline(episodes=5, seed=42):
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(episodes):
        env = make_cyborg_env(seed=seed + ep)
        _, _ = env.reset()
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        total = 0.0
        for _ in range(500):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                mask = np.array(_live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const), dtype=bool)
                valid = np.where(mask)[0]
                action_idx = int(rng.choice(valid))
                actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            _, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_verbose_trace(policy, params, policy_kind, steps=20, seed=42):
    env = make_cyborg_env(seed=seed)
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    from jaxborg.actions.encoding import BLUE_ANALYSE_END
    from jaxborg.topology import BLUE_AGENT_SUBNETS, SUBNET_IDS

    print("\nMASK VALIDATION (step 0):")
    for agent_idx, agent_name in enumerate(env.agents):
        mask = np.array(_live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const), dtype=bool)
        valid_indices = np.where(mask)[0]
        agent_subnets = BLUE_AGENT_SUBNETS[agent_idx]
        agent_subnet_ids = [SUBNET_IDS[s] for s in agent_subnets]

        valid_analyse = valid_indices[(valid_indices >= BLUE_ANALYSE_START) & (valid_indices < BLUE_ANALYSE_END)]
        valid_slots = valid_analyse - BLUE_ANALYSE_START

        wrong_subnet_hosts = []
        for slot in valid_slots:
            subnet_id = int(slot // OBS_HOSTS_PER_SUBNET)
            slot_within = int(slot % OBS_HOSTS_PER_SUBNET)
            hidx = int(const.obs_host_map[subnet_id, slot_within])
            if hidx >= GLOBAL_MAX_HOSTS:
                continue
            h_subnet = int(const.host_subnet[hidx])
            if h_subnet not in agent_subnet_ids:
                hostname = mappings.idx_to_hostname.get(int(hidx), f"host_{hidx}")
                wrong_subnet_hosts.append((int(hidx), hostname, h_subnet))

        print(
            f"  {agent_name}: subnets={agent_subnets}, "
            f"valid_analyse_hosts={len(valid_analyse)}, "
            f"wrong_subnet={len(wrong_subnet_hosts)}"
        )
        if wrong_subnet_hosts:
            for hidx, hname, hsub in wrong_subnet_hosts[:5]:
                print(f"    BUG: host_idx={hidx} {hname} in subnet {hsub} allowed!")

    total = 0.0
    for step in range(steps):
        actions = {}
        step_actions_desc = []
        for agent_idx, agent_name in enumerate(env.agents):
            obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
            mask = _live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const)
            mask_np = np.array(mask, dtype=bool)

            pi = policy_dist(policy, params, policy_kind, obs_jax, mask)
            action_idx = int(jnp.argmax(pi.logits))
            is_valid = bool(mask_np[action_idx])
            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

            desc = describe_blue_action(action_idx, mappings, const=const)
            cyborg_cls = type(cyborg_action).__name__
            valid_str = "OK" if is_valid else "MASKED!"
            step_actions_desc.append(
                f"  {agent_name}: idx={action_idx:4d} [{valid_str:7s}] -> {desc:45s} -> CybORG:{cyborg_cls}"
            )

        observations, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
        step_reward = mean(rewards.values())
        total += step_reward

        if step < 10 or step % 5 == 0:
            print(f"\nStep {step}: reward={step_reward:.2f}  cumulative={total:.2f}")
            for desc in step_actions_desc:
                print(desc)

    print(f"\nVerbose trace total reward ({steps} steps): {total:.2f}")


def print_mask_summary():
    print("\n--- Action Mask Summary ---")
    env = FsmRedCC4Env(num_steps=100, topology_mode="cyborg_bank", topology_bank_size=32)
    key = jax.random.PRNGKey(42)
    _, env_state = env.reset(key)

    for agent_idx in range(NUM_BLUE_AGENTS):
        mask = np.array(compute_blue_action_mask(env_state.const, agent_idx, env_state.state))
        total_valid = mask.sum()
        by_type = []
        for name, (start, end) in zip(ACTION_TYPE_NAMES, ACTION_TYPE_RANGES):
            count = mask[start:end].sum()
            if count > 0:
                by_type.append(f"{name}={count}")
        print(f"  blue_{agent_idx}: {total_valid} valid actions: {', '.join(by_type)}")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAXborg-trained policy: rollout, transfer, baselines")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=3, help="Rollout episodes (default 3)")
    parser.add_argument("--stochastic", action="store_true", help="Sample from policy instead of argmax")
    parser.add_argument(
        "--independent-rollouts",
        action="store_true",
        help="Run independent JAX/CybORG episodes instead of matched-state transfer diagnostics",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--baselines", action="store_true", help="Run sleep + random baselines")
    parser.add_argument("--verbose", type=int, default=0, help="Step-by-step CybORG trace for N steps")
    parser.add_argument("--plot", action="store_true", help="Save action dist + training curve PNGs")
    parser.add_argument("--mask-summary", action="store_true", help="Print per-agent mask breakdown")
    args = parser.parse_args()

    deterministic = not args.stochastic

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, params, policy_kind = load_checkpoint(args.checkpoint)

    if args.mask_summary:
        print_mask_summary()

    if args.independent_rollouts:
        print("\n" + "=" * 70)
        print("INDEPENDENT ROLLOUTS")
        print("=" * 70)
        print("Blue actions are chosen independently in each env.")
        print("Red stochastic choices are synced live from CybORG into JAX.")
        (
            jax_actions,
            jax_rewards,
            jax_results,
            cyborg_actions,
            cyborg_rewards,
            cyborg_actions_by_agent,
        ) = rollout_independent_transfer_synced_red(
            policy,
            params,
            policy_kind,
            args.episodes,
            deterministic,
            seed=args.seed,
        )

        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")
        print_per_agent_action_dist(cyborg_actions_by_agent, label="CybORG")
    else:
        if not deterministic:
            raise ValueError("Matched transfer diagnostics require deterministic evaluation")
        print("\n" + "=" * 70)
        print("MATCHED TRANSFER ROLLOUT")
        print("=" * 70)
        print("Stepping synced episodes with JAX-selected actions.")
        print("CybORG actions below are the policy outputs on matched CybORG observations, not applied actions.")
        (
            jax_actions,
            jax_rewards,
            jax_results,
            cyborg_actions,
            cyborg_rewards,
            cyborg_actions_by_agent,
        ) = rollout_matched_transfer(
            policy,
            params,
            policy_kind,
            args.episodes,
            deterministic,
            seed=args.seed,
        )
        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAX Policy on JAX Obs")
        print_per_agent_action_dist(cyborg_actions_by_agent, label="JAX Policy on CybORG Obs")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"Matched ep {len(jax_results)}")

    # Comparison report
    print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards)

    # Optional: baselines
    if args.baselines:
        print("\n" + "=" * 70)
        print("SLEEP BASELINE")
        print("=" * 70)
        sleep_score = run_sleep_baseline(args.episodes)
        print(f"Sleep baseline ({args.episodes} episodes): {sleep_score:.1f}")

        print("\n" + "=" * 70)
        print("RANDOM POLICY (with JAXborg action mask)")
        print("=" * 70)
        random_score = run_random_baseline(args.episodes, seed=args.seed)
        print(f"Random policy ({args.episodes} episodes): {random_score:.1f}")

    # Optional: verbose trace
    if args.verbose > 0:
        print("\n" + "=" * 70)
        print(f"VERBOSE CYBORG TRACE ({args.verbose} steps, deterministic)")
        print("=" * 70)
        run_verbose_trace(policy, params, policy_kind, steps=args.verbose, seed=args.seed)

    # Optional: plots
    if args.plot:
        output_dir = EXP_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_action_distribution(jax_actions, "JAXborg Action Distribution", output_dir / "jax_action_dist.png")
        plot_action_distribution(cyborg_actions, "CybORG Action Distribution", output_dir / "cyborg_action_dist.png")

        try:
            save_reward_plot(jax_rewards, cyborg_rewards)
        except Exception as e:
            print(f"(Skipped reward plot: {e})")

        metrics_path = EXP_DIR / "ippo_cc4" / "metrics.jsonl"
        if metrics_path.exists():
            plot_training_curves(metrics_path, output_dir / "training_curves.png")
        else:
            print(f"No metrics file at {metrics_path}, skipping training curves")


if __name__ == "__main__":
    main()
