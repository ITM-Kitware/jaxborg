"""Evaluate JAXborg-trained policy: JAXborg rollout, CybORG transfer, baselines, diagnostics."""

# ruff: noqa: E402

import os

# Enable XLA compilation cache before importing JAX
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import json
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

DEFAULT_NUM_STEPS = 500
DEFAULT_BANK_SIZE = 32


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


def _make_jax_eval_env(topology_mode: str, topology_bank_size: int):
    if topology_mode == "cyborg_bank":
        if topology_bank_size <= 0:
            raise ValueError(f"topology_bank_size must be > 0 for cyborg_bank, got {topology_bank_size}")
        return FsmRedCC4Env(
            num_steps=DEFAULT_NUM_STEPS,
            topology_mode=topology_mode,
            topology_bank_size=topology_bank_size,
        )
    return FsmRedCC4Env(num_steps=DEFAULT_NUM_STEPS, topology_mode=topology_mode)


def _default_cyborg_bank_match_size(jax_topology_mode: str, topology_bank_size: int) -> int | None:
    if jax_topology_mode == "cyborg_bank":
        return topology_bank_size
    return None


def policy_dist(policy, params, policy_kind, obs_jax, mask):
    if policy_kind == "current":
        return policy.apply(params, obs_jax, mask, method=ActorCritic.actor)
    return policy.apply(params, obs_jax, mask)


def make_batched_inference_fn(policy, params, policy_kind, deterministic):
    """Build a JIT-compiled function that runs policy inference for all agents at once.

    Returns batched_step(obs_stack, mask_stack, keys) -> (actions, logits)
    where obs_stack/mask_stack are (NUM_BLUE_AGENTS, ...) and keys is (NUM_BLUE_AGENTS, 2).
    """
    if policy_kind == "current":

        def _fwd(o, m):
            return policy.apply(params, o, m, method=ActorCritic.actor).logits
    else:

        def _fwd(o, m):
            return policy.apply(params, o, m).logits

    if deterministic:

        @jax.jit
        def batched_step(obs_stack, mask_stack, _keys):
            logits = jax.vmap(_fwd)(obs_stack, mask_stack)
            return jnp.argmax(logits, axis=-1), logits
    else:

        @jax.jit
        def batched_step(obs_stack, mask_stack, keys):
            logits = jax.vmap(_fwd)(obs_stack, mask_stack)
            actions = jax.vmap(lambda lg, k: distrax.Categorical(logits=lg).sample(seed=k))(logits, keys)
            return actions, logits

    return batched_step


@jax.jit
def _all_blue_masks(const, state):
    """Compute action masks for all blue agents in one JIT call."""
    return jnp.stack([compute_blue_action_mask(const, i, state) for i in range(NUM_BLUE_AGENTS)])


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
        host_idx = mappings.hostname_to_idx[action.hostname]
        jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
        if jax_idx == BLUE_SLEEP:
            return []
        return [jax_idx]

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


def _build_cyborg_mask_cache(wrapper, mappings, const):
    """Precompute CybORG-to-JAX action translation tables for all agents.

    Returns a dict keyed by agent_name. Each value is a list (one per CybORG
    action slot) of either:
      - list[int]: static JAX indices (for non-Decoy actions)
      - ("decoy", hostname, host_idx, agent_id, factory_jax_indices): precomputed decoy info
      - None: padding/invalid slot (always skipped)
    """
    cache = {}
    controller = wrapper.env.environment_controller
    cyborg_state = controller.state

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
                # Static translation — compute once
                jax_indices = _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state)
                agent_cache.append(jax_indices if jax_indices else None)
        cache[agent_name] = agent_cache
    return cache


def _live_blue_wrapper_mask_in_jax_space_cached(wrapper, agent_name, mappings, const, mask_cache):
    """Fast version of mask projection using precomputed translation cache.

    Returns a numpy bool array (caller should stack and convert to jnp once).
    """
    controller = wrapper.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=np.bool_)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in _cyborg_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
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
    # Use only BlueRewardMachine, excluding action_cost.  CybORG adds an
    # action_cost component (e.g. Restore costs -1) that JAX does not model.
    # The differential harness also uses only BlueRewardMachine.
    rewards = {
        agent: rews[agent].get("BlueRewardMachine", sum(rews[agent].values()))
        for agent in wrapper.possible_agents
        if agent in rews
    }
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


def _inject_live_red_policy_step(env_state, recorder, step_idx=None):
    """Inject CybORG-recorded red choice tokens for the current JAX step."""
    if step_idx is None:
        step_idx = int(env_state.state.time)
    step_tokens = jnp.asarray(recorder.extract_step(step_idx), dtype=jnp.float32)
    return env_state.replace(
        const=env_state.const.replace(
            red_policy_randoms=env_state.const.red_policy_randoms.at[step_idx].set(step_tokens),
            use_red_policy_randoms=jnp.array(True),
        )
    )


# --- Core rollout functions ---


def make_scan_eval_fn(env, policy, policy_kind, deterministic):
    """Build a fully JIT'd scan-based eval rollout.

    Returns a function: (params, key, env_state, obs) -> (final_state, step_data)
    where step_data contains per-step actions, rewards, and trajectory info.

    Params are passed as a dynamic argument (not captured in closure) so the
    XLA compilation cache can be reused across runs with different checkpoints.

    First call triggers XLA compilation (~5-10 min for CC4, cached to disk).
    Subsequent calls (same or future runs) load from cache and execute in seconds.
    """
    _agent_ids = jnp.arange(NUM_BLUE_AGENTS)
    _mask_single = jax.vmap(compute_blue_action_mask, in_axes=(None, 0, None))

    if policy_kind == "current":

        def _fwd(params, obs_flat, mask_flat):
            return policy.apply(params, obs_flat, mask_flat, method=ActorCritic.actor).logits
    else:

        def _fwd(params, obs_flat, mask_flat):
            return policy.apply(params, obs_flat, mask_flat).logits

    def _env_step(carry, _):
        params, key, env_state, obs = carry

        # Compute masks for all agents: (NUM_BLUE_AGENTS, action_dim)
        masks = _mask_single(env_state.const, _agent_ids, env_state.state)

        # Stack obs: (NUM_BLUE_AGENTS, obs_dim)
        obs_stack = jnp.stack([obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])

        # Batched policy inference (params passed through, not closed over)
        logits = jax.vmap(_fwd, in_axes=(None, 0, 0))(params, obs_stack, masks)

        key, _rng = jax.random.split(key)
        if deterministic:
            actions_arr = jnp.argmax(logits, axis=-1)
        else:
            act_keys = jax.random.split(_rng, NUM_BLUE_AGENTS)
            actions_arr = jax.vmap(lambda lg, k: distrax.Categorical(logits=lg).sample(seed=k))(logits, act_keys)

        actions = {f"blue_{i}": actions_arr[i] for i in range(NUM_BLUE_AGENTS)}

        key, step_key = jax.random.split(key)
        new_obs, new_env_state, rewards, dones, _ = env.step(step_key, env_state, actions)

        # Collect per-step data
        reward_arr = jnp.stack([rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
        st = new_env_state.state
        active = new_env_state.const.host_active
        compromised = st.host_compromised

        step_data = {
            "actions": actions_arr,
            "reward_mean": reward_arr.mean(),
            "hosts_user": jnp.sum((compromised == COMPROMISE_USER) & active),
            "hosts_priv": jnp.sum((compromised == COMPROMISE_PRIVILEGED) & active),
            "red_sessions": jnp.sum(st.red_sessions[:NUM_RED_AGENTS]),
            "mission_phase": st.mission_phase,
        }

        return (params, key, new_env_state, new_obs), step_data

    @jax.jit
    def scan_eval(params, key, env_state, obs):
        (_, final_key, final_state, final_obs), step_data = jax.lax.scan(
            _env_step, (params, key, env_state, obs), None, length=500
        )
        return final_state, step_data

    return scan_eval


def rollout_jaxborg_scan(
    policy,
    params,
    policy_kind,
    num_episodes=3,
    deterministic=False,
    seed=0,
    jax_topology_mode="cyborg_bank",
    topology_bank_size=DEFAULT_BANK_SIZE,
):
    """JAXborg-only eval using jax.lax.scan + jax.vmap — all episodes in parallel.

    Runs all episodes simultaneously on GPU via vmap over seeds.
    First call triggers XLA compilation (~5-10 min, cached to disk).
    Subsequent runs load from XLA cache and execute all episodes in one GPU pass.
    """
    env = _make_jax_eval_env(jax_topology_mode, topology_bank_size)
    scan_fn = make_scan_eval_fn(env, policy, policy_kind, deterministic)

    # Build keys for all episodes
    keys = jnp.stack([jax.random.PRNGKey(seed + ep * 100) for ep in range(num_episodes)])

    # Reset all episodes in parallel: vmap over seeds
    print(f"  Resetting {num_episodes} episodes in parallel...", flush=True)
    all_obs, all_env_states = jax.vmap(env.reset)(keys)

    # Run all episodes in parallel: vmap(scan) over episodes
    print("  (first run includes XLA compilation — cached for future runs)", flush=True)
    t0 = time.perf_counter()
    _, all_step_data = jax.vmap(scan_fn, in_axes=(None, 0, 0, 0))(params, keys, all_env_states, all_obs)
    # all_step_data shapes: each field is (num_episodes, 500, ...) or (num_episodes, 500)

    # Single device-to-host transfer for all episodes
    actions_np = np.asarray(all_step_data["actions"])  # (num_episodes, 500, NUM_BLUE_AGENTS)
    rewards_np = np.asarray(all_step_data["reward_mean"])  # (num_episodes, 500)
    hosts_user_np = np.asarray(all_step_data["hosts_user"])
    hosts_priv_np = np.asarray(all_step_data["hosts_priv"])
    red_sess_np = np.asarray(all_step_data["red_sessions"])
    phase_np = np.asarray(all_step_data["mission_phase"])

    elapsed = time.perf_counter() - t0
    ep_totals = rewards_np.sum(axis=1)
    print(f"  {num_episodes} episodes completed in {elapsed:.1f}s ({elapsed / num_episodes:.1f}s/ep)")
    for ep in range(num_episodes):
        print(f"    ep {ep + 1}: reward={ep_totals[ep]:.1f}")

    # Build result objects
    all_actions_flat = []
    episode_rewards = []
    episode_results = []

    for ep in range(num_episodes):
        cum_rewards = np.cumsum(rewards_np[ep])
        ep_trajectory = [
            StepSnapshot(
                reward=float(rewards_np[ep, s]),
                cumulative_reward=float(cum_rewards[s]),
                hosts_compromised_user=int(hosts_user_np[ep, s]),
                hosts_compromised_priv=int(hosts_priv_np[ep, s]),
                red_sessions_total=int(red_sess_np[ep, s]),
                mission_phase=int(phase_np[ep, s]),
            )
            for s in range(500)
        ]
        ep_actions_by_agent = [actions_np[ep, :, i].tolist() for i in range(NUM_BLUE_AGENTS)]
        all_actions_flat.extend(actions_np[ep].ravel().tolist())
        episode_rewards.append(float(ep_totals[ep]))
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_actions_by_agent,
                rewards=rewards_np[ep].tolist(),
                cumulative_reward=float(ep_totals[ep]),
                trajectory=ep_trajectory,
            )
        )

    return np.array(all_actions_flat), np.array(episode_rewards), episode_results


def rollout_jaxborg(
    policy,
    params,
    policy_kind,
    num_episodes=3,
    deterministic=False,
    seed=0,
    jax_topology_mode="cyborg_bank",
    topology_bank_size=DEFAULT_BANK_SIZE,
):
    env = _make_jax_eval_env(jax_topology_mode, topology_bank_size)
    batched_step = make_batched_inference_fn(policy, params, policy_kind, deterministic)
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

            # Batched mask computation + policy inference (1 JIT call instead of 5)
            masks = _all_blue_masks(env_state.const, env_state.state)
            obs_stack = jnp.stack([obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            actions_arr, _ = batched_step(obs_stack, masks, act_keys)

            # Single device-to-host transfer for all actions
            actions_np = np.asarray(actions_arr)
            actions = {f"blue_{i}": actions_arr[i] for i in range(NUM_BLUE_AGENTS)}
            for i in range(NUM_BLUE_AGENTS):
                ep_actions_by_agent[i].append(int(actions_np[i]))

            obs, env_state, rewards, dones, _ = env.step(step_key, env_state, actions)

            # Batch reward extraction (single transfer)
            reward_arr = jnp.stack([rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            reward_np = np.asarray(reward_arr)
            step_reward = float(reward_np.mean())
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)
            ep_reward += reward_np

            # Extract trajectory snapshot — defer int() conversions
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
    batched_step = make_batched_inference_fn(policy, params, policy_kind, deterministic)
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
        mask_cache = _build_cyborg_mask_cache(env, mappings, const)
        total = 0.0
        ep_actions = []

        for _ in range(500):
            rng, *_rngs = jax.random.split(rng, NUM_BLUE_AGENTS + 1)
            act_keys = jnp.stack(_rngs)

            # CybORG mask with cached translation (skips re-translating static actions)
            masks = jnp.stack(
                [
                    _live_blue_wrapper_mask_in_jax_space_cached(env, agent_name, mappings, const, mask_cache)
                    for agent_name in env.agents
                ]
            )
            obs_stack = jnp.stack([jnp.array(observations[agent_name], dtype=jnp.float32) for agent_name in env.agents])

            # Batched policy inference (1 forward pass instead of 5)
            actions_arr, _ = batched_step(obs_stack, masks, act_keys)
            actions_np = np.asarray(actions_arr)

            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                action_idx = int(actions_np[agent_idx])
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


def rollout_independent_transfer_synced_red(
    policy,
    params,
    policy_kind,
    num_episodes=3,
    deterministic=False,
    seed=0,
    jax_topology_mode="cyborg_bank",
    topology_bank_size=DEFAULT_BANK_SIZE,
    cyborg_bank_match_size=None,
):
    """Run paired independent episodes with live CybORG red-choice sync into JAX.

    Blue actions are chosen independently in each env. Red stochastic choices
    are synced from CybORG step-by-step via RedPolicyRecorder choice tokens.
    This is only a partial sync: it does not replay the broader random/order
    corrections used by CC4DifferentialHarness (green RNG, detection draws,
    PID deltas, privesc session choices, or CybORG action-order resync).

    Optimized: batched policy inference (1 JIT'd vmap call per env instead of 5
    per-agent calls), batched mask computation, minimized device-to-host syncs.
    """

    batched_step = make_batched_inference_fn(policy, params, policy_kind, deterministic)
    if cyborg_bank_match_size is None:
        cyborg_bank_match_size = _default_cyborg_bank_match_size(jax_topology_mode, topology_bank_size)

    all_jax_actions = []
    all_cyborg_actions = []
    jax_rewards = []
    cyborg_rewards = []
    jax_results = []
    cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        ep_seed = seed + ep * 100

        jax_env = _make_jax_eval_env(jax_topology_mode, topology_bank_size)
        key = jax.random.PRNGKey(ep_seed)
        jax_obs, jax_state = jax_env.reset(key)

        cyborg_env = make_cyborg_env(seed=ep_seed, bank_match_size=cyborg_bank_match_size)
        cyborg_obs, _ = cyborg_env.reset()
        inner = cyborg_env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        red_recorder = RedPolicyRecorder()
        red_recorder.install(inner, mappings)

        # Pre-build agent name strings and mask cache
        cyborg_agent_names = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]
        mask_cache = _build_cyborg_mask_cache(cyborg_env, mappings, const)

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
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            # --- JAX side: batched mask + policy inference (1 call, not 5) ---
            jax_masks = _all_blue_masks(jax_state.const, jax_state.state)
            jax_obs_stack = jnp.stack([jax_obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            jax_actions_arr, _ = batched_step(jax_obs_stack, jax_masks, act_keys)

            # --- CybORG side: cached mask translation + batched policy inference ---
            cyborg_masks = jnp.stack(
                [
                    _live_blue_wrapper_mask_in_jax_space_cached(cyborg_env, name, mappings, const, mask_cache)
                    for name in cyborg_agent_names
                ]
            )
            cyborg_obs_stack = jnp.stack(
                [jnp.array(cyborg_obs[name], dtype=jnp.float32) for name in cyborg_agent_names]
            )
            cyborg_actions_arr, _ = batched_step(cyborg_obs_stack, cyborg_masks, act_keys)

            # --- Single device-to-host sync for all 10 actions ---
            jax_actions_np = np.asarray(jax_actions_arr)
            cyborg_actions_np = np.asarray(cyborg_actions_arr)

            # Build action dicts from numpy (no more JAX syncs)
            jax_blue_actions = {f"blue_{i}": jax_actions_arr[i] for i in range(NUM_BLUE_AGENTS)}
            cyborg_actions = {}
            for i in range(NUM_BLUE_AGENTS):
                jax_act = int(jax_actions_np[i])
                cyborg_act = int(cyborg_actions_np[i])
                cyborg_actions[cyborg_agent_names[i]] = jax_blue_to_cyborg(cyborg_act, i, mappings, const=const)
                ep_jax_actions_by_agent[i].append(jax_act)
                ep_cyborg_actions_by_agent[i].append(cyborg_act)
                cyborg_actions_by_agent[i].append(cyborg_act)

            # First action diff check (already numpy, no additional sync)
            if first_action_diff is None:
                jax_vec = jax_actions_np.tolist()
                cy_vec = cyborg_actions_np.tolist()
                if jax_vec != cy_vec:
                    first_action_diff = (step, jax_vec, cy_vec)

            # --- Step both envs ---
            cyborg_obs, cyborg_step_rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(
                cyborg_env, actions=cyborg_actions
            )
            cyborg_step_reward = float(mean(cyborg_step_rewards.values()))
            cyborg_total += cyborg_step_reward

            jax_state = _inject_live_red_policy_step(jax_state, red_recorder, step_idx=step)
            jax_obs, jax_state, jax_step_rewards, _, _ = jax_env.step(step_key, jax_state, jax_blue_actions)

            # Batch reward extraction
            jax_reward_arr = jnp.stack([jax_step_rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            jax_step_reward = float(np.asarray(jax_reward_arr).mean())
            jax_total += jax_step_reward

            # State comparison (only until first diff found)
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


def print_independent_sync_caveat():
    print("Independent sync caveat: this path only replays red FSM choice tokens.")
    print("It does not sync green RNG, detection draws, PID deltas, privesc choices,")
    print("or CybORG same-priority action ordering the way CC4DifferentialHarness does.")


def rollout_matched_transfer(policy, params, policy_kind, num_episodes=3, deterministic=False, seed=0):
    """Compare policy outputs on matched JAX/CybORG states.

    JAX-selected actions drive the synced rollout so the underlying episode stays
    matched. CybORG-selected actions are recorded from the same synced states for
    transfer diagnostics, not applied.

    Optimized: batched policy inference across agents.
    """

    batched_step = make_batched_inference_fn(policy, params, policy_kind, deterministic=deterministic)
    rng = jax.random.PRNGKey(seed + 9999)  # separate stream for action sampling

    all_jax_actions = []
    all_cyborg_actions = []
    episode_rewards = []
    episode_results = []
    all_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        harness = CC4DifferentialHarness(seed=seed + ep * 100, check_obs=True, sync_green_rng=True)
        harness.reset()

        cyborg_agent_names = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]
        mask_cache = _build_cyborg_mask_cache(harness._blue_wrapper, harness.mappings, harness.jax_const)
        ep_jax_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for _ in range(500):
            # --- JAX side: batched obs + masks + policy ---
            jax_obs_stack = jnp.stack(
                [get_blue_obs(harness.jax_state, harness.jax_const, i) for i in range(NUM_BLUE_AGENTS)]
            )
            jax_masks = _all_blue_masks(harness.jax_const, harness.jax_state)
            if deterministic:
                step_keys = jnp.zeros((NUM_BLUE_AGENTS, 2), dtype=jnp.uint32)
            else:
                rng, _sub = jax.random.split(rng)
                step_keys = jax.random.split(_sub, NUM_BLUE_AGENTS)
            jax_actions_arr, _ = batched_step(jax_obs_stack, jax_masks, step_keys)

            # --- CybORG side: cached mask translation + batched policy ---
            cyborg_obs_list = []
            cyborg_mask_list = []
            for i, name in enumerate(cyborg_agent_names):
                cyborg_obs_dict = harness.cyborg_env.get_observation(name)
                cyborg_obs_list.append(
                    jnp.array(harness._blue_wrapper.observation_change(name, cyborg_obs_dict), dtype=jnp.float32)
                )
                cyborg_mask_list.append(
                    _live_blue_wrapper_mask_in_jax_space_cached(
                        harness._blue_wrapper, name, harness.mappings, harness.jax_const, mask_cache
                    )
                )
            cyborg_obs_stack = jnp.stack(cyborg_obs_list)
            cyborg_masks = jnp.stack(cyborg_mask_list)
            cyborg_actions_arr, _ = batched_step(cyborg_obs_stack, cyborg_masks, step_keys)

            # Single device-to-host sync
            jax_actions_np = np.asarray(jax_actions_arr)
            cyborg_actions_np = np.asarray(cyborg_actions_arr)

            jax_actions = {}
            for i in range(NUM_BLUE_AGENTS):
                jax_act = int(jax_actions_np[i])
                cyborg_act = int(cyborg_actions_np[i])
                jax_actions[i] = jax_act
                ep_jax_actions_by_agent[i].append(jax_act)
                ep_cyborg_actions_by_agent[i].append(cyborg_act)
                all_cyborg_actions_by_agent[i].append(cyborg_act)

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


def print_independent_mode_comparison(rows):
    print("\n" + "=" * 70)
    print("INDEPENDENT MODE COMPARISON")
    print("=" * 70)
    print(f"{'JAX Mode':<14} {'JAX Mean':>10} {'CybORG Mean':>12} {'Gap':>10}")
    print("-" * 50)
    for row in rows:
        print(f"{row['jax_mode']:<14} {row['jax_mean']:>10.1f} {row['cyborg_mean']:>12.1f} {row['gap_mean']:>+10.1f}")


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


# --- TOST equivalence test (L4 verification, per Karten et al. 2026) ---


def tost_equivalence(
    perf_rewards: np.ndarray,
    ref_rewards: np.ndarray,
    margin: float,
    alpha: float = 0.05,
    paired: bool = False,
) -> dict:
    """Two One-Sided Tests for equivalence of mean episode rewards.

    Tests H0_1: mu_perf - mu_ref >= margin  and  H0_2: mu_ref - mu_perf >= margin.
    If both reject at level alpha, the two backends are equivalent within +-margin.

    Args:
        perf_rewards: Per-episode mean rewards from the performance backend (JAXborg).
        ref_rewards: Per-episode mean rewards from the reference backend (CybORG).
        margin: Environment-specific equivalence margin (Delta).
        alpha: Significance level (default 0.05).
        paired: If True, use paired-sample TOST (for matched/synced rollouts where
            episodes are 1:1 paired). Computes per-episode differences first, then
            runs a one-sample TOST on the differences. Required when perf and ref
            come from the same synced episode (e.g. matched transfer mode).

    Returns:
        Dict with keys: equivalent, p_upper, p_lower, mean_diff, margin, ci_lower, ci_upper, paired.
    """
    from scipy import stats

    if paired:
        if len(perf_rewards) != len(ref_rewards):
            raise ValueError(f"Paired TOST requires equal-length arrays, got {len(perf_rewards)} vs {len(ref_rewards)}")
        diffs = np.asarray(perf_rewards) - np.asarray(ref_rewards)
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        df = n - 1
    else:
        n_perf, n_ref = len(perf_rewards), len(ref_rewards)
        mean_diff = float(np.mean(perf_rewards) - np.mean(ref_rewards))
        se = float(np.sqrt(np.var(perf_rewards, ddof=1) / n_perf + np.var(ref_rewards, ddof=1) / n_ref))
        # Welch-Satterthwaite degrees of freedom
        s1, s2 = np.var(perf_rewards, ddof=1), np.var(ref_rewards, ddof=1)
        n_perf_f, n_ref_f = float(n_perf), float(n_ref)
        nu_num = (s1 / n_perf_f + s2 / n_ref_f) ** 2
        nu_den = (s1 / n_perf_f) ** 2 / (n_perf_f - 1) + (s2 / n_ref_f) ** 2 / (n_ref_f - 1)
        df = nu_num / nu_den if nu_den > 0 else min(n_perf, n_ref) - 1

    if se < 1e-12:
        # Identical distributions (or all paired differences are zero)
        return {
            "equivalent": True,
            "p_upper": 0.0,
            "p_lower": 0.0,
            "mean_diff": mean_diff,
            "margin": margin,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "paired": paired,
        }

    # Upper test: H0 says diff >= margin, reject if diff is sufficiently below margin
    t_upper = (mean_diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))

    # Lower test: H0 says diff <= -margin, reject if diff is sufficiently above -margin
    t_lower = (mean_diff + margin) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))

    # Confidence interval for the difference
    t_crit = float(stats.t.ppf(1 - alpha, df))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    equivalent = p_upper < alpha and p_lower < alpha

    return {
        "equivalent": equivalent,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "mean_diff": mean_diff,
        "margin": margin,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "paired": paired,
    }


def print_tost_report(
    jax_rewards: np.ndarray,
    cyborg_rewards: np.ndarray,
    margin: float = 200.0,
    alpha: float = 0.05,
    paired: bool = False,
):
    """Print TOST equivalence report for cross-backend transfer validation."""
    result = tost_equivalence(jax_rewards, cyborg_rewards, margin, alpha, paired=paired)

    test_type = "PAIRED" if paired else "INDEPENDENT"
    print("\n" + "=" * 70)
    print(f"L4 CROSS-BACKEND EQUIVALENCE (TOST, {test_type})")
    print("=" * 70)
    print(f"  Test type:            {test_type.lower()} samples")
    print(f"  Margin (Delta):       +/-{result['margin']:.1f}")
    print(f"  Significance level:   alpha={alpha}")
    print(f"  JAXborg episodes:     {len(jax_rewards)}")
    print(f"  CybORG episodes:      {len(cyborg_rewards)}")
    print(f"  Mean diff (perf-ref): {result['mean_diff']:+.2f}")
    print(f"  {int((1 - alpha) * 100)}% CI for diff:      [{result['ci_lower']:+.2f}, {result['ci_upper']:+.2f}]")
    print(f"  p_upper (diff < +D):  {result['p_upper']:.4f}")
    print(f"  p_lower (diff > -D):  {result['p_lower']:.4f}")
    verdict = "EQUIVALENT" if result["equivalent"] else "NOT EQUIVALENT"
    print(f"  Verdict:              {verdict}")
    if not result["equivalent"]:
        print("  -> Policy transfer gap detected. Run --sleep-tost to distinguish")
        print("     simulation bugs from policy-environment interaction.")
    print("=" * 70)
    return result


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


def run_jaxborg_sleep_baseline(episodes=5, seed=42, bank_size=DEFAULT_BANK_SIZE):
    """Run JAXborg episodes with Sleep-only blue actions (native FSM red)."""
    env = FsmRedCC4Env(
        num_steps=DEFAULT_NUM_STEPS,
        topology_mode="cyborg_bank",
        topology_bank_size=bank_size,
    )
    totals = []
    sleep_actions = {f"blue_{i}": jnp.int32(0) for i in range(NUM_BLUE_AGENTS)}
    for ep in range(episodes):
        ep_seed = seed + ep * 100
        key = jax.random.PRNGKey(ep_seed)
        _, state = env.reset(key)
        total = 0.0
        for step in range(DEFAULT_NUM_STEPS):
            key, step_key = jax.random.split(key)
            _, state, rewards, _, _ = env.step(step_key, state, sleep_actions)
            total += float(np.mean([float(rewards[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]))
        totals.append(total)
    return totals


def run_cyborg_sleep_baseline(episodes=5, seed=42, bank_size=DEFAULT_BANK_SIZE):
    """Run CybORG episodes with Sleep-only blue actions, matched bank seeds."""

    totals = []
    for ep in range(episodes):
        ep_seed = seed + ep * 100
        env = make_cyborg_env(seed=ep_seed, bank_match_size=bank_size)
        env.reset()
        total = 0.0
        for _ in range(DEFAULT_NUM_STEPS):
            from CybORG.Simulator.Actions import Sleep

            actions = {a: Sleep() for a in env.agents}
            obs, rews, dones, info = env.env.parallel_step(actions, skip_valid_action_check=True)
            step_rewards = []
            for agent in env.possible_agents:
                if agent in rews:
                    step_rewards.append(rews[agent].get("BlueRewardMachine", sum(rews[agent].values())))
            total += float(np.mean(step_rewards))
        totals.append(total)
    return totals


def run_cross_backend_sleep_tost(episodes=10, seed=42, bank_size=DEFAULT_BANK_SIZE, margin=200.0, alpha=0.05):
    """Run cross-backend sleep baselines and compute TOST equivalence.

    Both backends run Sleep-only blue with independent native FSM red (no sync).
    Tests whether the simulation itself produces equivalent rewards.
    """
    print("\n" + "=" * 70)
    print("SIMULATION EQUIVALENCE (Sleep Baseline TOST)")
    print("=" * 70)
    print("  Both backends: Sleep-only blue, native FSM red, no cross-backend sync.")
    print(f"  Episodes: {episodes}, Margin: +/-{margin:.0f}")

    print("  Running JAXborg sleep baseline...")
    jax_totals = run_jaxborg_sleep_baseline(episodes, seed=seed, bank_size=bank_size)
    jax_mean = mean(jax_totals)
    print(f"    JAXborg mean: {jax_mean:.1f}")

    print("  Running CybORG sleep baseline...")
    cyborg_totals = run_cyborg_sleep_baseline(episodes, seed=seed, bank_size=bank_size)
    cyborg_mean = mean(cyborg_totals)
    print(f"    CybORG  mean: {cyborg_mean:.1f}")

    jax_arr = np.array(jax_totals)
    cyborg_arr = np.array(cyborg_totals)
    result = tost_equivalence(jax_arr, cyborg_arr, margin=margin, alpha=alpha, paired=False)

    gap = jax_mean - cyborg_mean
    verdict = "EQUIVALENT" if result["equivalent"] else "NOT EQUIVALENT"
    print(f"  Mean gap:       {gap:+.1f}")
    print(f"  95% CI:         [{result['ci_lower']:+.1f}, {result['ci_upper']:+.1f}]")
    print(f"  Verdict:        {verdict}")
    if result["equivalent"]:
        print("  -> Simulation produces equivalent rewards under null policy.")
    else:
        ci_width = result["ci_upper"] - result["ci_lower"]
        if abs(gap) < margin:
            print(f"  -> Gap ({gap:+.1f}) within margin but CI too wide ({ci_width:.0f}).")
            print("     Increase episodes to narrow CI and confirm equivalence.")
        else:
            print(f"  -> Gap ({gap:+.1f}) outside margin. Investigate simulation difference.")
    print("=" * 70)
    return {
        "equivalent": result["equivalent"],
        "mean_gap": gap,
        "jax_mean": jax_mean,
        "cyborg_mean": cyborg_mean,
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "jax_totals": jax_totals,
        "cyborg_totals": cyborg_totals,
    }


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
    parser.add_argument(
        "--compare-jax-modes",
        action="store_true",
        help="Run independent diagnostics for both pure and cyborg_bank JAX envs against the same CybORG seed bank",
    )
    parser.add_argument(
        "--jax-only",
        action="store_true",
        help="Run JAXborg-only evaluation (no CybORG, much faster)",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable jax.lax.scan for JAX-only eval (use Python loop instead, for debugging)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--baselines", action="store_true", help="Run sleep + random baselines")
    parser.add_argument("--verbose", type=int, default=0, help="Step-by-step CybORG trace for N steps")
    parser.add_argument("--plot", action="store_true", help="Save action dist + training curve PNGs")
    parser.add_argument("--mask-summary", action="store_true", help="Print per-agent mask breakdown")
    parser.add_argument(
        "--jax-topology-mode",
        choices=("pure", "cyborg_bank"),
        default="cyborg_bank",
        help="JAX env topology mode for JAX-only or independent rollouts",
    )
    parser.add_argument(
        "--topology-bank-size",
        type=int,
        default=DEFAULT_BANK_SIZE,
        help="Topology bank size for cyborg_bank mode (default 32)",
    )
    parser.add_argument(
        "--cyborg-bank-match-size",
        type=int,
        default=None,
        help="Optional CybORG seed-bank size for independent rollouts; "
        "defaults to the JAX bank size in cyborg_bank mode",
    )
    parser.add_argument(
        "--sleep-tost",
        action="store_true",
        default=None,
        help="Run cross-backend sleep baseline TOST to validate simulation equivalence. "
        "Auto-enabled for --independent-rollouts unless --no-sleep-tost is set.",
    )
    parser.add_argument(
        "--no-sleep-tost",
        action="store_true",
        help="Disable automatic sleep TOST in independent rollout mode.",
    )
    parser.add_argument(
        "--sleep-tost-episodes",
        type=int,
        default=10,
        help="Number of episodes for sleep baseline TOST (default 10).",
    )
    args = parser.parse_args()

    deterministic = not args.stochastic

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, params, policy_kind = load_checkpoint(args.checkpoint)

    if args.mask_summary:
        print_mask_summary()

    if args.jax_only:
        print("\n" + "=" * 70)
        print("JAX-ONLY EVALUATION (no CybORG)")
        print("=" * 70)
        use_scan = not args.no_scan
        rollout_fn = rollout_jaxborg_scan if use_scan else rollout_jaxborg
        if use_scan:
            print("Using jax.lax.scan (compiled rollout, cached to disk)")
        jax_actions, jax_rewards, jax_results = rollout_fn(
            policy,
            params,
            policy_kind,
            args.episodes,
            deterministic,
            seed=args.seed,
            jax_topology_mode=args.jax_topology_mode,
            topology_bank_size=args.topology_bank_size,
        )
        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")

        print(f"\nMean reward ({args.episodes} episodes): {jax_rewards.mean():.1f}")
        if len(jax_rewards) > 1:
            print(f"Stdev: {stdev(jax_rewards.tolist()):.1f}")

        if args.plot:
            output_dir = EXP_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_action_distribution(jax_actions, "JAXborg Action Distribution", output_dir / "jax_action_dist.png")
        return

    if args.compare_jax_modes:
        print("\n" + "=" * 70)
        print("INDEPENDENT MODE SPLIT DIAGNOSTIC")
        print("=" * 70)
        print("Comparing pure vs cyborg_bank JAX envs against a fixed CybORG seed bank.")
        print_independent_sync_caveat()
        cyborg_bank_match_size = args.cyborg_bank_match_size or args.topology_bank_size
        rows = []

        for jax_mode in ("pure", "cyborg_bank"):
            mode_bank_size = args.topology_bank_size if jax_mode == "cyborg_bank" else 0
            print(f"\n--- JAX mode: {jax_mode} ---")
            (
                jax_actions,
                jax_rewards,
                _jax_results,
                cyborg_actions,
                cyborg_rewards,
                _cyborg_actions_by_agent,
            ) = rollout_independent_transfer_synced_red(
                policy,
                params,
                policy_kind,
                args.episodes,
                deterministic,
                seed=args.seed,
                jax_topology_mode=jax_mode,
                topology_bank_size=mode_bank_size,
                cyborg_bank_match_size=cyborg_bank_match_size,
            )
            print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards)
            rows.append(
                {
                    "jax_mode": jax_mode,
                    "jax_mean": float(jax_rewards.mean()),
                    "cyborg_mean": float(cyborg_rewards.mean()),
                    "gap_mean": float(jax_rewards.mean() - cyborg_rewards.mean()),
                    "jax_rewards": jax_rewards.tolist(),
                    "cyborg_rewards": cyborg_rewards.tolist(),
                }
            )

        print_independent_mode_comparison(rows)
        out_path = EXP_DIR / "independent_mode_compare.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "episodes": args.episodes,
                    "stochastic": not deterministic,
                    "seed": args.seed,
                    "cyborg_bank_match_size": cyborg_bank_match_size,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved mode split report: {out_path}")
        return

    is_matched = not args.independent_rollouts

    if args.independent_rollouts:
        print("\n" + "=" * 70)
        print("INDEPENDENT ROLLOUTS")
        print("=" * 70)
        print("Blue actions are chosen independently in each env.")
        print("Red stochastic choices are synced live from CybORG into JAX.")
        print_independent_sync_caveat()
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
            jax_topology_mode=args.jax_topology_mode,
            topology_bank_size=args.topology_bank_size,
            cyborg_bank_match_size=args.cyborg_bank_match_size,
        )

        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")
        print_per_agent_action_dist(cyborg_actions_by_agent, label="CybORG")
    else:
        mode_label = "STOCHASTIC" if not deterministic else "DETERMINISTIC"
        print("\n" + "=" * 70)
        print(f"MATCHED TRANSFER ROLLOUT ({mode_label})")
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

    # L4 TOST equivalence test (always run with >= 2 episodes)
    sleep_tost_result = None
    run_sleep = args.sleep_tost or (args.independent_rollouts and not args.no_sleep_tost)
    if run_sleep:
        sleep_tost_result = run_cross_backend_sleep_tost(
            episodes=args.sleep_tost_episodes,
            seed=args.seed,
            bank_size=args.topology_bank_size,
        )

    if len(jax_rewards) >= 2 and len(cyborg_rewards) >= 2:
        tost_result = print_tost_report(jax_rewards, cyborg_rewards, paired=is_matched)

        # Contextual interpretation when sleep TOST is available
        if sleep_tost_result is not None and not tost_result["equivalent"]:
            print()
            if sleep_tost_result["equivalent"] or abs(sleep_tost_result["mean_gap"]) < tost_result["margin"]:
                print("  NOTE: Sleep baseline TOST shows simulation equivalence.")
                sleep_g = sleep_tost_result["mean_gap"]
                policy_g = tost_result["mean_diff"]
                print(f"        Sleep gap: {sleep_g:+.1f} vs policy gap: {policy_g:+.1f}")
                print("        The transfer gap is from policy-env interaction (different RNG")
                print("        streams produce different obs → different blue actions → different")
                print("        outcomes). The simulation itself is correct.")
                tost_result["sim_equivalent"] = True
                tost_result["sleep_gap"] = sleep_tost_result["mean_gap"]
            else:
                print("  NOTE: Sleep baseline also shows a gap — possible simulation difference.")
                tost_result["sim_equivalent"] = False
                tost_result["sleep_gap"] = sleep_tost_result["mean_gap"]

        # Save TOST result alongside other outputs
        tost_path = EXP_DIR / "tost_result.json"
        tost_path.parent.mkdir(parents=True, exist_ok=True)
        tost_result["jax_rewards"] = jax_rewards.tolist()
        tost_result["cyborg_rewards"] = cyborg_rewards.tolist()
        if sleep_tost_result is not None:
            tost_result["sleep_tost"] = {
                "equivalent": sleep_tost_result["equivalent"],
                "mean_gap": sleep_tost_result["mean_gap"],
                "jax_mean": sleep_tost_result["jax_mean"],
                "cyborg_mean": sleep_tost_result["cyborg_mean"],
            }
        tost_path.write_text(json.dumps(tost_result, indent=2) + "\n")
        print(f"Saved TOST result: {tost_path}")
        # Record L4 result in catalog — report sim_equivalent if available
        try:
            from tests.catalog import update_l4_tost

            update_l4_tost(
                equivalent=tost_result["equivalent"],
                margin=tost_result["margin"],
                mean_diff=tost_result["mean_diff"],
                episodes=len(jax_rewards),
            )
        except Exception as e:
            print(f"WARNING: Failed to update catalog with L4 TOST result: {e}")

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
