"""IPPO (Independent PPO) for CC4 with FsmRedCC4Env.

Based on PureJaxRL PPO and JaxMARL's ippo_ff_cage.py.
Trains Blue agents against scripted FSM red agents using feedforward networks.
Fully JIT'd: rollout collection via jax.lax.scan + PPO update in one compiled fn.
"""

import os
from pathlib import Path

# Persistent XLA compilation cache — must be set BEFORE importing JAX.
# Default: ~/.cache/jaxborg/xla.  Override: JAX_COMPILATION_CACHE_DIR env var.
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _default_cache = str(Path.home() / ".cache" / "jaxborg" / "xla")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import json
import pickle
import time
from functools import partial
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxborg.policy import ActorCritic, SharedActorCritic


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    avail_actions: jnp.ndarray
    blue_busy: jnp.ndarray


class RewardNormState(NamedTuple):
    """Running statistics for reward normalization (Welford online update)."""

    returns: jnp.ndarray  # (num_envs,) — discounted return tracker
    mean: jnp.ndarray  # scalar — running mean of returns
    var: jnp.ndarray  # scalar — running variance of returns
    count: jnp.ndarray  # scalar — total sample count


def _subtree_global_norm(tree, *path):
    subtree = tree
    for key in path:
        if not isinstance(subtree, dict) or key not in subtree:
            return jnp.array(0.0, dtype=jnp.float32)
        subtree = subtree[key]
    return optax.global_norm(subtree)


def _clip_named_subtree(grads, key: str, max_norm: float):
    def _mapping_update(tree, updates):
        try:
            return tree.copy(updates)
        except TypeError:
            new_tree = tree.copy()
            new_tree.update(updates)
            return new_tree

    params_grads = grads.get("params")
    if params_grads is None:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return grads, zero, zero

    subtree = params_grads.get(key)
    if subtree is None:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return grads, zero, zero

    pre_norm = optax.global_norm(subtree)
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=jnp.float32) / (pre_norm + 1e-8))
    clipped_subtree = jax.tree.map(lambda x: x * scale, subtree)
    clipped_params = _mapping_update(params_grads, {key: clipped_subtree})
    clipped = _mapping_update(grads, {"params": clipped_params})
    post_norm = optax.global_norm(clipped_subtree)
    return clipped, pre_norm, post_norm


def masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    return jnp.sum(x * mask) / denom


def masked_var(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mean = masked_mean(x, mask)
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    return jnp.sum(mask * jnp.square(x - mean)) / denom


def compute_value_loss(
    value: jnp.ndarray,
    old_value: jnp.ndarray,
    targets: jnp.ndarray,
    clip_eps: float,
    clip_value_loss: bool,
) -> jnp.ndarray:
    """PPO critic loss with optional value clipping.

    Large-magnitude returns make PPO's value clipping stall the critic because the
    clipped branch becomes flat outside the +/-clip band. Keep it configurable,
    but default it off for CC4.
    """
    value_losses = jnp.square(value - targets)
    if not clip_value_loss:
        return 0.5 * value_losses.mean()

    value_pred_clipped = old_value + (value - old_value).clip(-clip_eps, clip_eps)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


class MetricsLogger:
    """Logs to both JSONL file and MLflow."""

    def __init__(self, filepath, *, mlflow_enabled: bool):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "w")
        self.mlflow_enabled = mlflow_enabled

    def log(self, metrics: dict, step: int):
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()
        if self.mlflow_enabled:
            mlflow_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating)) and k not in ("steps", "update")
            }
            mlflow.log_metrics(mlflow_metrics, step=step)

    def close(self):
        self.file.close()
        if self.mlflow_enabled:
            mlflow.log_artifact(str(self.filepath))


def make_train(config):
    """Build env, network, and a single JIT'd collect_and_update function.

    Returns (env, network, init_obs, init_env_state, init_train_state_fn,
    collect_and_update_fn).

    collect_and_update scans NUM_STEPS env steps then runs GAE + PPO update,
    all inside one XLA program.
    """
    num_envs = config.get("NUM_ENVS", 1)
    inner_env = FsmRedCC4Env(
        num_steps=500,
        topology_mode=config.get("TOPOLOGY_MODE", "pure"),
        topology_bank_size=config.get("TOPOLOGY_BANK_SIZE", 0),
        training_mode=bool(config.get("TRAINING_MODE", False)),
    )
    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents  # 5
    config["NUM_ACTORS"] = num_agents * num_envs
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
    config["MINIBATCH_SIZE"] = num_agents * num_envs * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = LogWrapper(inner_env)

    # Batched env init via vmap
    init_key = jax.random.PRNGKey(config["SEED"])
    init_keys = jax.random.split(init_key, num_envs)
    init_obs, init_env_state = jax.vmap(env.reset)(init_keys)

    network_type = config.get("NETWORK_TYPE", "separate")
    busy_masking = bool(config.get("BUSY_MASKING", True))
    grad_clip_mode = config.get("GRAD_CLIP_MODE", "per_head")
    norm_rewards = bool(config.get("NORM_REWARDS", False))

    if network_type == "shared":
        network = SharedActorCritic(
            inner_env.action_space(agents[0]).n,
            hidden_dim=config.get("HIDDEN_DIM", 256),
            activation=config["ACTIVATION"],
        )
    else:
        network = ActorCritic(
            inner_env.action_space(agents[0]).n,
            hidden_dim=config.get("HIDDEN_DIM", 256),
            activation=config["ACTIVATION"],
        )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def _init_train_state(rng):
        init_x = jnp.zeros(inner_env.observation_space(agents[0]).shape)
        network_params = network.init(rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.adam(learning_rate=linear_schedule, eps=1e-5)
        else:
            tx = optax.adam(config["LR"], eps=1e-5)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

    @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4))
    def _collect_and_update(train_state, env_state, obs, rng, reward_norm_state):
        """Scan NUM_STEPS env steps, compute GAE, run PPO epochs — all JIT'd."""

        # --- Rollout via scan (vmapped over NUM_ENVS) ---
        _agent_ids = jnp.arange(num_agents)
        _mask_over_envs = jax.vmap(compute_blue_action_mask, in_axes=(0, None, 0))
        _mask_over_agents = jax.vmap(_mask_over_envs, in_axes=(None, 0, None))

        # Seed the info accumulator with zeros matching the env's info dict shape.
        # Inner env returns (num_envs,) scalars; LogWrapper adds (num_envs, num_agents) arrays.
        _info_acc_init = {
            "reward_ria": jnp.zeros(num_envs, dtype=jnp.float32),
            "reward_lwf": jnp.zeros(num_envs, dtype=jnp.float32),
            "reward_asf": jnp.zeros(num_envs, dtype=jnp.float32),
            "action_cost": jnp.zeros(num_envs, dtype=jnp.float32),
            "impact_count": jnp.zeros(num_envs, dtype=jnp.float32),
            "green_lwf_count": jnp.zeros(num_envs, dtype=jnp.float32),
            "green_asf_count": jnp.zeros(num_envs, dtype=jnp.float32),
            "returned_episode_returns": jnp.zeros((num_envs, num_agents), dtype=jnp.float32),
            "returned_episode_lengths": jnp.zeros((num_envs, num_agents), dtype=jnp.float32),
            "returned_episode": jnp.zeros((num_envs, num_agents), dtype=jnp.float32),
        }

        def _env_step(carry, _):
            env_state, obs, rng, info_acc, rn_state = carry

            # obs/env_state have leading (num_envs,) dim
            # Stack agents: (num_envs, num_agents, obs_dim)
            obs_batch = jnp.stack([obs[a] for a in agents], axis=-2)
            busy_batch = env_state.env_state.state.blue_pending_ticks > 0
            avail_batch = _mask_over_agents(
                env_state.env_state.const,
                _agent_ids,
                env_state.env_state.state,
            ).transpose(1, 0, 2)

            rng, _rng = jax.random.split(rng)
            # Flatten (num_envs, num_agents) for network, then reshape back
            flat_obs = obs_batch.reshape(-1, obs_batch.shape[-1])
            flat_avail = avail_batch.reshape(-1, avail_batch.shape[-1])
            pi, value = network.apply(train_state.params, flat_obs, flat_avail)

            action_flat = pi.sample(seed=_rng)
            log_prob_flat = pi.log_prob(action_flat)

            action = action_flat.reshape(num_envs, num_agents)
            log_prob = log_prob_flat.reshape(num_envs, num_agents)
            value = value.reshape(num_envs, num_agents)

            env_act = {agents[i]: action[:, i] for i in range(num_agents)}

            rng, _rng = jax.random.split(rng)
            step_keys = jax.random.split(_rng, num_envs)
            new_obs, new_env_state, rewards, dones, info = jax.vmap(env.step)(step_keys, env_state, env_act)

            # Accumulate info sums in the carry instead of storing per-step.
            info_acc = jax.tree.map(lambda acc, v: acc + jnp.asarray(v, dtype=jnp.float32), info_acc, info)

            # Per-env team reward (all agents share the same reward)
            team_reward = rewards[agents[0]]  # (num_envs,)
            done_signal = dones[agents[0]]  # (num_envs,)

            if norm_rewards:
                # Update running discounted returns
                new_returns = rn_state.returns * config["GAMMA"] + team_reward

                # Welford online update of mean/var over the returns batch
                batch_mean = jnp.mean(new_returns)
                batch_var = jnp.var(new_returns)
                batch_count = jnp.array(num_envs, dtype=jnp.float32)

                delta = batch_mean - rn_state.mean
                total_count = rn_state.count + batch_count
                new_mean = rn_state.mean + delta * batch_count / total_count
                m_a = rn_state.var * rn_state.count
                m_b = batch_var * batch_count
                m2 = m_a + m_b + delta**2 * rn_state.count * batch_count / total_count
                new_var = m2 / total_count

                # Scale reward by running return std, clip to [-10, 10]
                scaled_reward = team_reward / (jnp.sqrt(new_var) + 1e-8)
                scaled_reward = jnp.clip(scaled_reward, -10.0, 10.0)

                # Reset returns on episode done
                new_returns = new_returns * (1.0 - done_signal)

                rn_state = RewardNormState(
                    returns=new_returns,
                    mean=new_mean,
                    var=new_var,
                    count=total_count,
                )
                # Stack across agents and apply REWARD_SCALE on top of normalization
                reward_out = jnp.stack([scaled_reward] * num_agents, axis=-1) * config["REWARD_SCALE"]
            else:
                reward_out = jnp.stack([rewards[a] for a in agents], axis=-1) * config["REWARD_SCALE"]

            transition = Transition(
                done=jnp.stack([dones[a] for a in agents], axis=-1),
                action=action,
                value=value,
                reward=reward_out,
                log_prob=log_prob,
                obs=obs_batch,
                avail_actions=avail_batch,
                blue_busy=busy_batch.astype(jnp.float32),
            )

            return (new_env_state, new_obs, rng, info_acc, rn_state), transition

        (env_state, obs, rng, info_sums, reward_norm_state), traj_batch = jax.lax.scan(
            _env_step, (env_state, obs, rng, _info_acc_init, reward_norm_state), None, config["NUM_STEPS"]
        )

        # --- GAE ---
        # traj_batch shapes: (NUM_STEPS, num_envs, num_agents, ...)
        last_obs_batch = jnp.stack([obs[a] for a in agents], axis=-2)
        flat_last_obs = last_obs_batch.reshape(-1, last_obs_batch.shape[-1])
        _, last_val = network.apply(train_state.params, flat_last_obs)
        last_val = last_val.reshape(num_envs, num_agents)

        def _get_advantages(gae_and_next_value, gae_inputs):
            gae, next_value = gae_and_next_value
            done, value, reward = gae_inputs
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae

        # Pass only the 3 fields needed for GAE, not the full Transition
        # (avoids moving obs/avail_actions through the scan unnecessarily).
        gae_inputs = (traj_batch.done, traj_batch.value, traj_batch.reward)
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            gae_inputs,
            reverse=True,
            unroll=8,
        )
        raw_advantages = advantages
        targets = raw_advantages + traj_batch.value
        if busy_masking:
            policy_mask = (1.0 - traj_batch.blue_busy).astype(jnp.float32)
        else:
            policy_mask = jnp.ones_like(traj_batch.blue_busy)
        # --- PPO update epochs ---
        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets, policy_mask = batch_info

                def _loss_fn(params, traj_batch, gae, targets, policy_mask):
                    # Per-minibatch advantage normalization (matches CybORG)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    pi, value = network.apply(params, traj_batch.obs, traj_batch.avail_actions)
                    log_prob = pi.log_prob(traj_batch.action)
                    policy_weight = policy_mask.astype(jnp.float32)
                    policy_count = jnp.maximum(policy_weight.sum(), 1.0)

                    value_loss = compute_value_loss(
                        value,
                        traj_batch.value,
                        targets,
                        config["CLIP_EPS"],
                        bool(config.get("CLIP_VALUE_LOSS", False)),
                    )

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    logratio = log_prob - traj_batch.log_prob
                    approx_kl = jnp.sum(policy_weight * ((ratio - 1) - logratio)) / policy_count
                    clip_frac = jnp.sum(policy_weight * (jnp.abs(ratio - 1) > config["CLIP_EPS"])) / policy_count

                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.sum(policy_weight * jnp.minimum(loss_actor1, loss_actor2)) / policy_count
                    entropy = jnp.sum(policy_weight * pi.entropy()) / policy_count

                    var_targets = jnp.var(targets)
                    explained_var = jnp.where(
                        var_targets > 0,
                        1 - jnp.var(targets - value) / var_targets,
                        0.0,
                    )

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, approx_kl, clip_frac, explained_var)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets, policy_mask)
                pre_clip_grad_norm = optax.global_norm(grads)
                if grad_clip_mode == "global":
                    # Global clip at MAX_GRAD_NORM matching torch.nn.utils.clip_grad_norm_
                    max_norm = jnp.asarray(config["MAX_GRAD_NORM"], dtype=jnp.float32)
                    scale = jnp.minimum(1.0, max_norm / (pre_clip_grad_norm + 1e-8))
                    grads = jax.tree.map(lambda x: x * scale, grads)
                    grad_norm = optax.global_norm(grads)
                    actor_grad_norm = pre_clip_grad_norm
                    actor_grad_norm_clipped = grad_norm
                    critic_grad_norm = pre_clip_grad_norm
                    critic_grad_norm_clipped = grad_norm
                else:
                    actor_clip_norm = config.get("ACTOR_MAX_GRAD_NORM", config["MAX_GRAD_NORM"])
                    critic_clip_norm = config.get("CRITIC_MAX_GRAD_NORM", config["MAX_GRAD_NORM"])
                    grads, actor_grad_norm, actor_grad_norm_clipped = _clip_named_subtree(
                        grads, "actor_head", actor_clip_norm
                    )
                    grads, critic_grad_norm, critic_grad_norm_clipped = _clip_named_subtree(
                        grads, "critic_head", critic_clip_norm
                    )
                    grad_norm = optax.global_norm(grads)
                train_state = train_state.apply_gradients(grads=grads)

                loss_info = {
                    "total_loss": total_loss[0],
                    "actor_loss": total_loss[1][1],
                    "critic_loss": total_loss[1][0],
                    "entropy": total_loss[1][2],
                    "approx_kl": total_loss[1][3],
                    "clip_frac": total_loss[1][4],
                    "explained_var": total_loss[1][5],
                    "pre_clip_grad_norm": pre_clip_grad_norm,
                    "grad_norm": grad_norm,
                    "actor_grad_norm": actor_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
                    "actor_grad_norm_clipped": actor_grad_norm_clipped,
                    "critic_grad_norm_clipped": critic_grad_norm_clipped,
                }
                return train_state, loss_info

            train_state, traj_batch, raw_advantages, targets, policy_mask, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, raw_advantages, targets, policy_mask)
            # Flatten (NUM_STEPS, num_envs, num_agents, ...) -> (batch_size, ...)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), batch)
            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
            update_state = (train_state, traj_batch, raw_advantages, targets, policy_mask, rng)
            return update_state, loss_info

        update_state = (train_state, traj_batch, raw_advantages, targets, policy_mask, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        rng = update_state[-1]

        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        # Raw episode return: sum of raw reward components over the rollout, mean over envs.
        raw_rollout_return = (
            info_sums["reward_ria"] + info_sums["reward_lwf"]
            + info_sums["reward_asf"] + info_sums["action_cost"]
        ).mean()
        # info_sums accumulated over NUM_STEPS; take mean over steps and envs.
        num_steps_f = jnp.float32(config["NUM_STEPS"])
        metric = jax.tree.map(lambda x: x.mean() / num_steps_f, info_sums)
        active_mask = policy_mask
        valid_action_count = traj_batch.avail_actions.sum(axis=-1).astype(jnp.float32)
        rollout_info = {
            "raw_rollout_return": raw_rollout_return,
            "mean_rollout_return": traj_batch.reward.sum(axis=0).mean(),
            "mean_valid_actions": valid_action_count.mean(),
            "mean_mask_uniform_entropy": jnp.log(jnp.maximum(valid_action_count, 1.0)).mean(),
            "busy_fraction": traj_batch.blue_busy.mean(),
            "decision_fraction": active_mask.mean(),
            "mean_active_valid_actions": masked_mean(valid_action_count, active_mask),
            "mean_active_mask_uniform_entropy": masked_mean(
                jnp.log(jnp.maximum(valid_action_count, 1.0)),
                active_mask,
            ),
            "nonzero_reward_fraction": (traj_batch.reward != 0).mean(),
            "mean_step_reward": traj_batch.reward.mean(),
            "std_step_reward": traj_batch.reward.std(),
            "mean_abs_step_reward": jnp.abs(traj_batch.reward).mean(),
            "reward_min": traj_batch.reward.min(),
            "reward_max": traj_batch.reward.max(),
            "value_mean": traj_batch.value.mean(),
            "value_std": traj_batch.value.std(),
            "target_mean": targets.mean(),
            "target_std": targets.std(),
            "raw_advantage_mean": raw_advantages.mean(),
            "raw_advantage_std": raw_advantages.std(),
            "raw_advantage_abs_mean": jnp.abs(raw_advantages).mean(),
        }
        if norm_rewards:
            rollout_info["reward_norm_var"] = reward_norm_state.var
            rollout_info["reward_norm_count"] = reward_norm_state.count
        metric = {**metric, **loss_info, **rollout_info}

        return train_state, env_state, obs, rng, reward_norm_state, metric

    return env, network, init_obs, init_env_state, _init_train_state, _collect_and_update


EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


@hydra.main(config_path=".", config_name="ippo_cc4", version_base=None)
def main(cfg):
    config = OmegaConf.to_container(cfg)
    mlflow_enabled = bool(config.get("MLFLOW_ENABLED", True))

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"XLA compilation cache: {cache_dir}", flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = EXP_DIR / f"ippo_cc4_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if mlflow_enabled:
        mlflow_db = EXP_DIR / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("ippo-cc4")
        mlflow.start_run(run_name="ippo-vs-fsm-red")

        mlflow.log_params(
            {
                "algorithm": "IPPO-FF",
                "seed": config["SEED"],
                "num_envs": config.get("NUM_ENVS", 1),
                "num_steps": config["NUM_STEPS"],
                "total_timesteps": config["TOTAL_TIMESTEPS"],
                "update_epochs": config["UPDATE_EPOCHS"],
                "num_minibatches": config["NUM_MINIBATCHES"],
                "learning_rate": config["LR"],
                "gamma": config["GAMMA"],
                "gae_lambda": config["GAE_LAMBDA"],
                "clip_eps": config["CLIP_EPS"],
                "ent_coef": config["ENT_COEF"],
                "vf_coef": config["VF_COEF"],
                "max_grad_norm": config["MAX_GRAD_NORM"],
                "actor_max_grad_norm": config.get("ACTOR_MAX_GRAD_NORM", config["MAX_GRAD_NORM"]),
                "critic_max_grad_norm": config.get("CRITIC_MAX_GRAD_NORM", config["MAX_GRAD_NORM"]),
                "hidden_dim": config.get("HIDDEN_DIM", 256),
                "activation": config["ACTIVATION"],
                "anneal_lr": config["ANNEAL_LR"],
                "norm_rewards": config.get("NORM_REWARDS", False),
            }
        )

    print("=" * 60, flush=True)
    print("IPPO-FF CC4 Training: Blue vs FSM Red")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num envs: {config.get('NUM_ENVS', 1)}")
    print(f"Num steps per rollout: {config['NUM_STEPS']}")
    print(f"Hidden dim: {config.get('HIDDEN_DIM', 256)}")
    print(f"Activation: {config['ACTIVATION']}")
    print(f"Topology mode: {config.get('TOPOLOGY_MODE', 'pure')}")
    print(f"Reward normalization: {config.get('NORM_REWARDS', False)}")
    print("=" * 60, flush=True)

    t_setup = time.perf_counter()
    env, network, init_obs, init_env_state, init_train_state, collect_and_update = make_train(config)
    print(f"  env+network setup: {time.perf_counter() - t_setup:.1f}s", flush=True)

    rng = jax.random.PRNGKey(config["SEED"] + 1)
    rng, _rng = jax.random.split(rng)
    train_state = init_train_state(_rng)

    env_state = init_env_state
    obs = init_obs

    num_envs = config.get("NUM_ENVS", 1)
    reward_norm_state = RewardNormState(
        returns=jnp.zeros(num_envs, dtype=jnp.float32),
        mean=jnp.zeros((), dtype=jnp.float32),
        var=jnp.ones((), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )

    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    num_updates = int(config["NUM_UPDATES"])
    num_steps = int(config["NUM_STEPS"])

    logger = MetricsLogger(save_dir / "metrics.jsonl", mlflow_enabled=mlflow_enabled)
    best_reward = float("-inf")

    start_time = time.perf_counter()
    print(f"Starting training ({num_updates} updates, fully JIT'd)...", flush=True)
    print("  (first update includes XLA compilation — may take a few minutes)", flush=True)

    for update_idx in range(num_updates):
        train_state, env_state, obs, rng, reward_norm_state, metric = collect_and_update(
            train_state, env_state, obs, rng, reward_norm_state
        )
        metric = jax.device_get(metric)

        if update_idx == 0:
            elapsed_first = time.perf_counter() - start_time
            print(f"  first update compiled + ran in {elapsed_first:.1f}s", flush=True)

        step = (update_idx + 1) * num_steps * config.get("NUM_ENVS", 1)
        raw_reward = float(metric["raw_rollout_return"])
        reward = float(metric["mean_rollout_return"])
        record = {
            "update": update_idx + 1,
            "steps": step,
            "raw_episode_reward_mean": raw_reward,
            "episode_reward_mean": reward,
            "loss": float(metric["total_loss"]),
            "policy_loss": float(metric["actor_loss"]),
            "value_loss": float(metric["critic_loss"]),
            "entropy": float(metric["entropy"]),
            "approx_kl": float(metric["approx_kl"]),
            "clip_frac": float(metric["clip_frac"]),
            "explained_var": float(metric["explained_var"]),
            "pre_clip_grad_norm": float(metric["pre_clip_grad_norm"]),
            "grad_norm": float(metric["grad_norm"]),
            "actor_grad_norm": float(metric["actor_grad_norm"]),
            "critic_grad_norm": float(metric["critic_grad_norm"]),
            "actor_grad_norm_clipped": float(metric["actor_grad_norm_clipped"]),
            "critic_grad_norm_clipped": float(metric["critic_grad_norm_clipped"]),
            "mean_valid_actions": float(metric["mean_valid_actions"]),
            "mean_mask_uniform_entropy": float(metric["mean_mask_uniform_entropy"]),
            "busy_fraction": float(metric["busy_fraction"]),
            "decision_fraction": float(metric["decision_fraction"]),
            "mean_active_valid_actions": float(metric["mean_active_valid_actions"]),
            "mean_active_mask_uniform_entropy": float(metric["mean_active_mask_uniform_entropy"]),
            "nonzero_reward_fraction": float(metric["nonzero_reward_fraction"]),
            "mean_step_reward": float(metric["mean_step_reward"]),
            "std_step_reward": float(metric["std_step_reward"]),
            "mean_abs_step_reward": float(metric["mean_abs_step_reward"]),
            "reward_min": float(metric["reward_min"]),
            "reward_max": float(metric["reward_max"]),
            "value_mean": float(metric["value_mean"]),
            "value_std": float(metric["value_std"]),
            "target_mean": float(metric["target_mean"]),
            "target_std": float(metric["target_std"]),
            "raw_advantage_mean": float(metric["raw_advantage_mean"]),
            "raw_advantage_std": float(metric["raw_advantage_std"]),
            "raw_advantage_abs_mean": float(metric["raw_advantage_abs_mean"]),
            "reward_ria": float(metric["reward_ria"]),
            "reward_lwf": float(metric["reward_lwf"]),
            "reward_asf": float(metric["reward_asf"]),
            "impact_count": float(metric["impact_count"]),
            "green_lwf_count": float(metric["green_lwf_count"]),
            "green_asf_count": float(metric["green_asf_count"]),
        }
        if "reward_norm_var" in metric:
            record["reward_norm_var"] = float(metric["reward_norm_var"])
            record["reward_norm_count"] = float(metric["reward_norm_count"])
        logger.log(record, step=step)
        if raw_reward > best_reward:
            best_reward = raw_reward

        if (update_idx + 1) % 50 == 0 or update_idx == num_updates - 1:
            elapsed = time.perf_counter() - start_time
            sps = step / elapsed
            print(
                f"  update {update_idx + 1}/{num_updates} | step {step} | reward {raw_reward:.1f} | {sps:.0f} sps",
                flush=True,
            )

        checkpoint_every = int(config.get("CHECKPOINT_EVERY_UPDATES", 500))
        if (update_idx + 1) % checkpoint_every == 0 or update_idx == num_updates - 1:
            ckpt_name = f"checkpoint_{step}.pkl" if update_idx < num_updates - 1 else "checkpoint_final.pkl"
            ckpt_path = save_dir / ckpt_name
            ckpt_data = {
                "params": train_state.params,
                "hidden_dim": config.get("HIDDEN_DIM", 256),
                "activation": config["ACTIVATION"],
                "action_dim": env.action_space(env.agents[0]).n,
            }
            if bool(config.get("NORM_REWARDS", False)):
                ckpt_data["reward_norm_state"] = {
                    "returns": jax.device_get(reward_norm_state.returns),
                    "mean": jax.device_get(reward_norm_state.mean),
                    "var": jax.device_get(reward_norm_state.var),
                    "count": jax.device_get(reward_norm_state.count),
                }
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt_data, f)
            if mlflow_enabled:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

    logger.close()

    elapsed = time.perf_counter() - start_time
    total_steps = int(config["TOTAL_TIMESTEPS"])
    sps = total_steps / elapsed

    final_return = float(metric["raw_rollout_return"])
    if mlflow_enabled:
        mlflow.log_artifact(str(config_path))
        mlflow.log_metrics(
            {
                "wall_time_sec": elapsed,
                "steps_per_second": sps,
                "best_reward": best_reward,
                "final_reward": final_return,
            }
        )
        mlflow.end_run()

    print("\nTraining complete!")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {sps:,.0f} steps/sec")
    print(f"Best returns: {best_reward:.2f}")
    print(f"Final returns: {final_return:.2f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    import sys

    if not any("hydra.run.dir" in a for a in sys.argv):
        sys.argv.append(f"hydra.run.dir={EXP_DIR}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
    if not any("hydra.job.chdir" in a for a in sys.argv):
        sys.argv.append("hydra.job.chdir=True")
    main()
