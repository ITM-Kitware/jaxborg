"""IPPO (Independent PPO) on JAX FsmRedCC4Env, recipe-driven.

Algorithm script — owns the rollout loop, GAE, PPO update, metrics. The
network architecture is selected by `recipe.arch.name` and instantiated via
`jaxborg.policies.make_jax_policy`; the algorithm itself is arch-agnostic.

Launch:
    uv run python scripts/train/algorithms/ippo_jax.py --recipe singh --seed 42

Outputs (to `$JAXBORG_EXP_DIR/ippo_jax/<tag>/`):
    metrics.jsonl              (standardized schema, see jaxborg.metrics_schema)
    recipe_<tag>.yaml          (resolved recipe sidecar)
    model_<tag>.safetensors    (params, safetensors format)
    checkpoint_*.safetensors   (periodic full checkpoints)
"""

# ruff: noqa: E402

import os
from pathlib import Path

# Persistent XLA compilation cache — must be set BEFORE importing JAX.
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _default_cache = str(Path.home() / ".cache" / "jaxborg" / "xla")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import copy
import json
import sys
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mlflow
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

# Make `import jaxborg.*` work when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.checkpoint import (
    PolicyBundleEntry,
    load_jax_policy,
    read_sidecar,
    save_jax_bundle,
    write_sidecar,
)
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.evaluation.training_checkpoint import evaluate_training_checkpoint
from jaxborg.learned_red import RED_OBS_SIZE, RED_POLICY_ACTION_DIM
from jaxborg.metrics_schema import add_team_metrics, make_row
from jaxborg.mlflow_setup import MlflowCheckpointEvaluator, start_run
from jaxborg.policies import make_jax_policy
from jaxborg.recipe import load as load_recipe
from jaxborg.recipe import (
    project_jax,
    resolve_train_opponents,
    team_recipe,
    training_teams,
)
from jaxborg.scenarios.cc4.game_variant import GameVariant
from scripts.train.algorithms.ippo_jax_joint import initial_reward_norm_state, make_joint_train


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
    returns: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def compute_value_loss(value, old_value, targets, clip_eps, clip_value_loss):
    value_losses = jnp.square(value - targets)
    if not clip_value_loss:
        return 0.5 * value_losses.mean()
    value_pred_clipped = old_value + (value - old_value).clip(-clip_eps, clip_eps)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


def make_train(config, network):
    """Build env and a single JIT'd collect_and_update fn from a flat config."""
    num_envs = config["NUM_ENVS"]
    variant: GameVariant = config["TRAIN_VARIANT"]
    topology_bank = config.get("TOPOLOGY_BANK") or None
    inner_env = make_jax_env(
        variant,
        training_mode=bool(config.get("TRAINING_MODE", True)),
        topology_path=list(topology_bank) if topology_bank else None,
        mission_bank=config.get("MISSION_BANK"),
        mission_bank_amplify=config.get("MISSION_BANK_AMPLIFY", 1.0),
        phase_boundary_bank=config.get("PHASE_BOUNDARY_BANK"),
        phase_rewards_bank=config.get("PHASE_REWARDS_BANK"),
    )
    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents
    config["NUM_ACTORS"] = num_agents * num_envs
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
    config["MINIBATCH_SIZE"] = num_agents * num_envs * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = LogWrapper(inner_env)
    init_key = jax.random.PRNGKey(config["SEED"])
    init_keys = jax.random.split(init_key, num_envs)
    init_obs, init_env_state = jax.vmap(env.reset)(init_keys)

    norm_rewards = bool(config.get("NORM_REWARDS", False))

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def _init_train_state(rng):
        init_x = jnp.zeros(inner_env.observation_space(agents[0]).shape)
        params = network.init(rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.adam(learning_rate=linear_schedule, eps=1e-5)
        else:
            tx = optax.adam(config["LR"], eps=1e-5)
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4))
    def _collect_and_update(train_state, env_state, obs, rng, reward_norm_state):
        _agent_ids = jnp.arange(num_agents)
        _mask_over_envs = jax.vmap(compute_blue_action_mask, in_axes=(0, None, 0))
        _mask_over_agents = jax.vmap(_mask_over_envs, in_axes=(None, 0, None))

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
            obs_batch = jnp.stack([obs[a] for a in agents], axis=-2)
            busy_batch = env_state.env_state.state.blue_pending_ticks > 0
            avail_batch = _mask_over_agents(env_state.env_state.const, _agent_ids, env_state.env_state.state).transpose(
                1, 0, 2
            )
            sleep_only = jnp.zeros(avail_batch.shape[-1], dtype=jnp.bool_).at[0].set(True)
            avail_batch = jnp.where(busy_batch[..., None], sleep_only, avail_batch)
            rng, _rng = jax.random.split(rng)
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
            info_acc = jax.tree.map(lambda acc, v: acc + jnp.asarray(v, dtype=jnp.float32), info_acc, info)
            team_reward = rewards[agents[0]]
            done_signal = dones[agents[0]]
            if norm_rewards:
                new_returns = rn_state.returns * config["GAMMA"] + team_reward
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
                scaled_reward = team_reward / (jnp.sqrt(new_var) + 1e-8)
                scaled_reward = jnp.clip(scaled_reward, -10.0, 10.0)
                new_returns = new_returns * (1.0 - done_signal)
                rn_state = RewardNormState(returns=new_returns, mean=new_mean, var=new_var, count=total_count)
                reward_out = jnp.stack([scaled_reward] * num_agents, axis=-1) * config.get("REWARD_SCALE", 1.0)
            else:
                reward_out = jnp.stack([rewards[a] for a in agents], axis=-1) * config.get("REWARD_SCALE", 1.0)

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

        gae_inputs = (traj_batch.done, traj_batch.value, traj_batch.reward)
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            gae_inputs,
            reverse=True,
            unroll=8,
        )
        targets = advantages + traj_batch.value
        # CybORG ignores submissions while an action is in progress. Force
        # Sleep above and omit those rows from the actor objective while still
        # retaining them in GAE/value learning for delayed-action credit.
        policy_mask = 1.0 - traj_batch.blue_busy

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets, policy_mask = batch_info

                def _loss_fn(params, traj_batch, gae, targets, policy_mask):
                    policy_weight = policy_mask.astype(jnp.float32)
                    policy_count = jnp.maximum(policy_weight.sum(), 1.0)
                    gae_mean = jnp.sum(policy_weight * gae) / policy_count
                    gae_var = jnp.sum(policy_weight * jnp.square(gae - gae_mean)) / policy_count
                    gae = (gae - gae_mean) / (jnp.sqrt(gae_var) + 1e-8)
                    pi, value = network.apply(params, traj_batch.obs, traj_batch.avail_actions)
                    log_prob = pi.log_prob(traj_batch.action)
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
                    explained_var = jnp.where(var_targets > 0, 1 - jnp.var(targets - value) / var_targets, 0.0)
                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, approx_kl, clip_frac, explained_var)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets, policy_mask)
                pre_clip = optax.global_norm(grads)
                max_norm = jnp.asarray(config["MAX_GRAD_NORM"], dtype=jnp.float32)
                scale = jnp.minimum(1.0, max_norm / (pre_clip + 1e-8))
                grads = jax.tree.map(lambda x: x * scale, grads)
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
                    "pre_clip_grad_norm": pre_clip,
                    "grad_norm": grad_norm,
                }
                return train_state, loss_info

            train_state, traj_batch, advantages, targets, policy_mask, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets, policy_mask)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), batch)
            shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled,
            )
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
            update_state = (train_state, traj_batch, advantages, targets, policy_mask, rng)
            return update_state, loss_info

        update_state = (train_state, traj_batch, advantages, targets, policy_mask, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        rng = update_state[-1]
        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

        raw_rollout_return = (
            info_sums["reward_ria"] + info_sums["reward_lwf"] + info_sums["reward_asf"] + info_sums["action_cost"]
        ).mean()

        rollout_info = {
            "raw_rollout_return": raw_rollout_return,
            "mean_rollout_return": traj_batch.reward.sum(axis=0).mean(),
        }
        metric = {**loss_info, **rollout_info}
        return train_state, env_state, obs, rng, reward_norm_state, metric

    return env, init_obs, init_env_state, _init_train_state, _collect_and_update


EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


def _network_from_arch(arch: dict, action_dim: int):
    return make_jax_policy(
        arch["name"],
        action_dim=action_dim,
        hidden_dim=int(arch.get("hidden_dim", 256)),
        hidden_layers=int(arch.get("hidden_layers", 2)),
        activation=arch.get("activation", "tanh"),
    )


def _legacy_arch_from_sidecar(path: Path, team: str) -> dict:
    sidecar = read_sidecar(path)
    return dict(team_recipe(sidecar, team)["arch"])


def _run_joint_training(args, recipe: dict, tag: str, save_dir: Path) -> None:
    """Train or freeze independent Blue/Red policies in a joint JAX rollout."""
    trainable_teams = training_teams(recipe)
    configs = {team: project_jax(recipe, team=team) for team in ("blue", "red")}
    for config in configs.values():
        config["SEED"] = args.seed

    opponent_paths = resolve_train_opponents(recipe, backend="jax", exp_dir=EXP_DIR)
    dims = {
        "blue": (BLUE_OBS_SIZE, BLUE_ALLOW_TRAFFIC_END),
        "red": (RED_OBS_SIZE, RED_POLICY_ACTION_DIM),
    }
    networks = {}
    initial_params = {}
    arches = {}
    sources = {}

    for team in ("blue", "red"):
        obs_dim, action_dim = dims[team]
        if team in trainable_teams:
            arch = dict(team_recipe(recipe, team)["arch"])
            source = {"kind": "fresh", "seed": int(args.seed)}
        else:
            if team not in opponent_paths:
                raise ValueError(f"joint training requires a learned {team} opponent model")
            path = opponent_paths[team]
            entry = load_jax_policy(
                path,
                team=team,
                expected_obs_dim=obs_dim,
                expected_action_dim=action_dim,
            )
            arch = dict(entry.arch) if entry.arch.get("name") else _legacy_arch_from_sidecar(path, team)
            initial_params[team] = entry.weights
            source = {
                "kind": "checkpoint",
                "path": str(path),
                "bundle_source": entry.source,
            }
        arches[team] = arch
        sources[team] = source
        networks[team] = _network_from_arch(arch, action_dim)

    print("=" * 60, flush=True)
    print(f"IPPO-JAX joint [{recipe['meta']['name']}] seed={args.seed}", flush=True)
    print(
        f"  trainable={','.join(trainable_teams)} num_envs={configs['blue']['NUM_ENVS']} "
        f"num_steps={configs['blue']['NUM_STEPS']} total_timesteps={configs['blue']['TOTAL_TIMESTEPS']:,}",
        flush=True,
    )
    for team in ("blue", "red"):
        print(
            f"  {team}: arch={arches[team]['name']} hidden_dim={arches[team].get('hidden_dim', 256)} "
            f"status={'trainable' if team in trainable_teams else 'frozen'}",
            flush=True,
        )
    print("=" * 60, flush=True)

    checkpoint_evaluator = MlflowCheckpointEvaluator(recipe)
    run = start_run(recipe, backend="jax", seed=args.seed)
    train_run_id = run.info.run_id
    t0 = time.perf_counter()
    env, obs, env_state, init_states, collect_and_update = make_joint_train(
        configs,
        networks,
        trainable_teams=trainable_teams,
        initial_params=initial_params,
    )
    del env
    rng = jax.random.PRNGKey(args.seed + 1)
    rng, init_key = jax.random.split(rng)
    train_states = init_states(init_key)
    reward_norm_states = {team: initial_reward_norm_state(configs[team]["NUM_ENVS"]) for team in ("blue", "red")}
    print(f"  env+networks setup: {time.perf_counter() - t0:.1f}s", flush=True)

    num_updates = int(configs["blue"]["NUM_UPDATES"])
    if num_updates < 1:
        raise ValueError("total_timesteps must cover at least one joint rollout")
    num_steps = int(configs["blue"]["NUM_STEPS"])
    num_envs = int(configs["blue"]["NUM_ENVS"])
    metrics_path = save_dir / "metrics.jsonl"
    start = time.perf_counter()
    final_metrics = None

    def save_bundle(path: Path, env_steps: int) -> None:
        policies = {
            team: PolicyBundleEntry(
                weights=train_states[team].params,
                team=team,
                obs_dim=dims[team][0],
                action_dim=dims[team][1],
                arch=arches[team],
                trainable=team in trainable_teams,
                source=sources[team],
            )
            for team in ("blue", "red")
        }
        save_jax_bundle(
            path,
            policies,
            provenance={
                "recipe": recipe["meta"]["name"],
                "seed": int(args.seed),
                "total_steps": int(env_steps),
                "train_run_id": train_run_id,
            },
        )

    print(f"Starting joint training ({num_updates} updates)...", flush=True)
    with metrics_path.open("w") as metrics_file:
        for update_idx in range(num_updates):
            train_states, env_state, obs, rng, reward_norm_states, metrics = collect_and_update(
                train_states,
                env_state,
                obs,
                rng,
                reward_norm_states,
            )
            metrics = jax.device_get(metrics)
            final_metrics = metrics
            if update_idx == 0:
                print(f"  first update compiled+ran in {time.perf_counter() - start:.1f}s", flush=True)

            env_steps = (update_idx + 1) * num_steps * num_envs
            elapsed = time.perf_counter() - start
            sps = env_steps / elapsed if elapsed > 0 else 0.0
            primary_team = "blue" if "blue" in trainable_teams else "red"
            primary = metrics[primary_team]
            blue = metrics["blue"]
            row = make_row(
                update_idx=update_idx + 1,
                env_steps=env_steps,
                wall_time_s=elapsed,
                throughput_sps=sps,
                loss_policy=float(primary["actor_loss"]),
                loss_value=float(primary["critic_loss"]),
                loss_entropy=float(primary["entropy"]),
                loss_total=float(primary["total_loss"]),
                ppo_kl_divergence=float(primary["approx_kl"]),
                ppo_clip_fraction=float(primary["clip_frac"]),
                ppo_explained_variance=float(primary["explained_var"]),
                lr=float(configs[primary_team]["LR"]),
                train_episode_reward_mean=float(blue["raw_rollout_return"]),
                ppo_grad_norm=float(primary["grad_norm"]),
                ppo_pre_clip_grad_norm=float(primary["pre_clip_grad_norm"]),
                backend_extras={"jax.game.red_return": float(metrics["game"]["red_return"])},
            )
            metric_names = {
                "actor_loss": "loss_policy",
                "critic_loss": "loss_value",
                "entropy": "loss_entropy",
                "total_loss": "loss_total",
                "approx_kl": "ppo_kl_divergence",
                "clip_frac": "ppo_clip_fraction",
                "explained_var": "ppo_explained_variance",
                "grad_norm": "ppo_grad_norm",
                "pre_clip_grad_norm": "ppo_pre_clip_grad_norm",
                "raw_rollout_return": "return",
                "mean_rollout_return": "normalized_return",
                "actor_fraction": "actor_fraction",
                "critic_fraction": "critic_fraction",
            }
            for team in ("blue", "red"):
                add_team_metrics(
                    row,
                    team,
                    {metric_names[key]: float(metrics[team][key]) for key in metric_names},
                )
                row[f"team.{team}.lr"] = float(configs[team]["LR"])
                row[f"team.{team}.trainable"] = team in trainable_teams
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            mlflow.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "update_idx"
                },
                step=env_steps,
            )
            if (update_idx + 1) % 50 == 0 or update_idx == num_updates - 1:
                print(
                    f"  upd {update_idx + 1}/{num_updates} step {env_steps:,} "
                    f"blue {metrics['game']['blue_return']:.1f} red {metrics['game']['red_return']:.1f} "
                    f"{sps:.0f} sps",
                    flush=True,
                )

            checkpoint_every = int(configs["blue"].get("CHECKPOINT_EVERY_UPDATES", 50))
            is_final = update_idx == num_updates - 1
            previous_steps = update_idx * num_steps * num_envs
            eval_due = checkpoint_evaluator.due(previous_steps, env_steps, final=is_final)
            periodic_checkpoint = checkpoint_every > 0 and (update_idx + 1) % checkpoint_every == 0
            if periodic_checkpoint or is_final or eval_due:
                name = f"model_{tag}.safetensors" if is_final else f"checkpoint_{env_steps}.safetensors"
                checkpoint_path = save_dir / name
                save_bundle(checkpoint_path, env_steps)
                sidecar_name = f"recipe_{tag}.yaml" if is_final else f"recipe_checkpoint_{env_steps}.yaml"
                sidecar_path = write_sidecar(
                    save_dir / sidecar_name,
                    recipe,
                    seed=args.seed,
                    total_steps=env_steps,
                    backend="jax",
                    train_run_id=train_run_id,
                    extra={
                        "trainable_teams": list(trainable_teams),
                        "policy_sources": sources,
                        "model": name,
                    },
                )
                if eval_due:
                    print(
                        f"  MLflow checkpoint evaluation at step {env_steps:,}; "
                        f"evaluating {checkpoint_evaluator.settings.episodes} episodes...",
                        flush=True,
                    )
                    try:
                        checkpoint_evaluator.on_checkpoint(
                            checkpoint_path,
                            sidecar_path,
                            env_steps=env_steps,
                            evaluate_fn=lambda episodes: evaluate_training_checkpoint(
                                checkpoint_path,
                                backend="jax",
                                recipe=recipe,
                                seed=args.seed,
                                episodes=episodes,
                            ),
                        )
                    finally:
                        if not is_final and checkpoint_every <= 0:
                            checkpoint_path.unlink(missing_ok=True)
                            sidecar_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    sps = int(configs["blue"]["TOTAL_TIMESTEPS"]) / elapsed if elapsed > 0 else 0.0
    final_reward = float(final_metrics["game"]["blue_return"]) if final_metrics is not None else float("nan")
    mlflow.log_metrics({"wall_time_sec": elapsed, "steps_per_second": sps, "final_reward": final_reward})
    mlflow.log_artifact(str(metrics_path))
    final_model = save_dir / f"model_{tag}.safetensors"
    sidecar = save_dir / f"recipe_{tag}.yaml"
    if final_model.exists():
        mlflow.log_artifact(str(final_model))
    if sidecar.exists():
        mlflow.log_artifact(str(sidecar))
    mlflow.end_run()
    print(f"\nDone in {elapsed:.1f}s ({sps:,.0f} sps). Blue score: {final_reward:.1f}")
    print(f"Saved to: {save_dir}")

    # Run in a fresh CPU process only after the canonical final bundle and its
    # sidecar exist and the training MLflow run has been closed.
    from jaxborg.evaluation.scripted_red import run_configured_after_training

    run_configured_after_training(final_model, recipe)


def main():
    parser = argparse.ArgumentParser(description="IPPO-FF on JAX, recipe-driven")
    parser.add_argument("--recipe", required=True, help="Recipe name (e.g. 'singh') or path to YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help="Run tag (defaults to <recipe>_seed<n>)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override recipe.train.total_timesteps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override recipe.jax.num_envs")
    args = parser.parse_args()

    recipe = copy.deepcopy(load_recipe(args.recipe))
    # Apply CLI overrides to the resolved recipe itself so the exported
    # sidecar describes the run that actually happened.
    if args.total_timesteps is not None:
        recipe["train"]["total_timesteps"] = int(args.total_timesteps)
    if args.num_envs is not None:
        recipe.setdefault("jax", {})["num_envs"] = int(args.num_envs)
    config = project_jax(recipe)
    config["SEED"] = args.seed

    tag = args.tag or f"{recipe['meta']['name']}_seed{args.seed}"
    save_dir = EXP_DIR / "ippo_jax" / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"XLA compilation cache: {cache_dir}", flush=True)

    teams = training_teams(recipe)
    learned_red = "red" in teams or bool(recipe.get("train", {}).get("opponents", {}).get("red"))
    if learned_red:
        _run_joint_training(args, recipe, tag, save_dir)
        return

    # Build a throwaway env to get action_dim for network init.
    variant: GameVariant = config["TRAIN_VARIANT"]
    topology_bank = config.get("TOPOLOGY_BANK") or None
    inner_env = make_jax_env(
        variant,
        topology_mode=config.get("TOPOLOGY_MODE", "generative"),
        topology_path=list(topology_bank) if topology_bank else None,
        mission_bank=config.get("MISSION_BANK"),
        mission_bank_amplify=config.get("MISSION_BANK_AMPLIFY", 1.0),
        phase_boundary_bank=config.get("PHASE_BOUNDARY_BANK"),
        phase_rewards_bank=config.get("PHASE_REWARDS_BANK"),
    )
    action_dim = inner_env.action_space(inner_env.agents[0]).n
    network = make_jax_policy(
        recipe["arch"]["name"],
        action_dim=action_dim,
        hidden_dim=config["HIDDEN_DIM"],
        hidden_layers=config["HIDDEN_LAYERS"],
        activation=config["ACTIVATION"],
    )

    print("=" * 60, flush=True)
    print(f"IPPO-JAX [{recipe['meta']['name']}] seed={args.seed}")
    print(
        f"  num_envs={config['NUM_ENVS']} num_steps={config['NUM_STEPS']} total_timesteps={config['TOTAL_TIMESTEPS']:,}"
    )
    print(f"  arch={recipe['arch']['name']} hidden_dim={config['HIDDEN_DIM']}")
    print("=" * 60, flush=True)

    checkpoint_evaluator = MlflowCheckpointEvaluator(recipe)
    run = start_run(recipe, backend="jax", seed=args.seed)
    train_run_id = run.info.run_id

    t0 = time.perf_counter()
    env, init_obs, init_env_state, init_train_state, collect_and_update = make_train(config, network)
    print(f"  env+network setup: {time.perf_counter() - t0:.1f}s", flush=True)

    rng = jax.random.PRNGKey(config["SEED"] + 1)
    rng, _rng = jax.random.split(rng)
    train_state = init_train_state(_rng)
    env_state = init_env_state
    obs = init_obs
    num_envs = config["NUM_ENVS"]
    reward_norm_state = RewardNormState(
        returns=jnp.zeros(num_envs, dtype=jnp.float32),
        mean=jnp.zeros((), dtype=jnp.float32),
        var=jnp.ones((), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )

    num_updates = int(config["NUM_UPDATES"])
    num_steps = int(config["NUM_STEPS"])
    metrics_path = save_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    print(f"Starting training ({num_updates} updates)...", flush=True)
    start = time.perf_counter()
    final_metric = None

    for update_idx in range(num_updates):
        train_state, env_state, obs, rng, reward_norm_state, metric = collect_and_update(
            train_state, env_state, obs, rng, reward_norm_state
        )
        metric = jax.device_get(metric)
        final_metric = metric
        if update_idx == 0:
            print(f"  first update compiled+ran in {time.perf_counter() - start:.1f}s", flush=True)

        env_steps = (update_idx + 1) * num_steps * num_envs
        elapsed = time.perf_counter() - start
        sps = env_steps / elapsed if elapsed > 0 else 0.0

        row = make_row(
            update_idx=update_idx + 1,
            env_steps=env_steps,
            wall_time_s=elapsed,
            throughput_sps=sps,
            loss_policy=float(metric["actor_loss"]),
            loss_value=float(metric["critic_loss"]),
            loss_entropy=float(metric["entropy"]),
            loss_total=float(metric["total_loss"]),
            ppo_kl_divergence=float(metric["approx_kl"]),
            ppo_clip_fraction=float(metric["clip_frac"]),
            ppo_explained_variance=float(metric["explained_var"]),
            lr=float(config["LR"]),
            train_episode_reward_mean=float(metric["raw_rollout_return"]),
            ppo_grad_norm=float(metric["grad_norm"]),
            ppo_pre_clip_grad_norm=float(metric["pre_clip_grad_norm"]),
            backend_extras={"jax.mean_rollout_return": float(metric["mean_rollout_return"])},
        )
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        mlflow.log_metrics(
            {k: v for k, v in row.items() if isinstance(v, (int, float)) and k != "update_idx"},
            step=env_steps,
        )
        if (update_idx + 1) % 50 == 0 or update_idx == num_updates - 1:
            print(
                f"  upd {update_idx + 1}/{num_updates} step {env_steps:,} "
                f"reward {row['train_episode_reward_mean']:.1f} {sps:.0f} sps",
                flush=True,
            )

        ckpt_every = int(config.get("CHECKPOINT_EVERY_UPDATES", 50))
        is_final = update_idx == num_updates - 1
        previous_steps = update_idx * num_steps * num_envs
        eval_due = checkpoint_evaluator.due(previous_steps, env_steps, final=is_final)
        periodic_checkpoint = ckpt_every > 0 and (update_idx + 1) % ckpt_every == 0
        if periodic_checkpoint or is_final or eval_due:
            ckpt_path = save_dir / (f"model_{tag}.safetensors" if is_final else f"checkpoint_{env_steps}.safetensors")
            save_jax_bundle(
                ckpt_path,
                {
                    "blue": PolicyBundleEntry(
                        weights=train_state.params,
                        team="blue",
                        obs_dim=BLUE_OBS_SIZE,
                        action_dim=action_dim,
                        arch=dict(recipe["arch"]),
                        trainable=True,
                        source={"kind": "fresh", "seed": int(args.seed)},
                    )
                },
                provenance={
                    "recipe": recipe["meta"]["name"],
                    "seed": int(args.seed),
                    "total_steps": int(env_steps),
                    "train_run_id": train_run_id,
                },
            )
            sidecar_name = f"recipe_{tag}.yaml" if is_final else f"recipe_checkpoint_{env_steps}.yaml"
            sidecar_path = write_sidecar(
                save_dir / sidecar_name,
                recipe,
                seed=args.seed,
                total_steps=env_steps,
                backend="jax",
                train_run_id=train_run_id,
                extra={"trainable_teams": ["blue"], "model": ckpt_path.name},
            )
            if eval_due:
                print(
                    f"  MLflow checkpoint evaluation at step {env_steps:,}; "
                    f"evaluating {checkpoint_evaluator.settings.episodes} episodes...",
                    flush=True,
                )
                try:
                    checkpoint_evaluator.on_checkpoint(
                        ckpt_path,
                        sidecar_path,
                        env_steps=env_steps,
                        evaluate_fn=lambda episodes: evaluate_training_checkpoint(
                            ckpt_path,
                            backend="jax",
                            recipe=recipe,
                            seed=args.seed,
                            episodes=episodes,
                        ),
                    )
                finally:
                    if not is_final and ckpt_every <= 0:
                        ckpt_path.unlink(missing_ok=True)
                        sidecar_path.unlink(missing_ok=True)

    metrics_file.close()
    elapsed = time.perf_counter() - start
    sps = int(config["TOTAL_TIMESTEPS"]) / elapsed if elapsed > 0 else 0.0
    final_reward = float(final_metric["raw_rollout_return"]) if final_metric is not None else float("nan")
    mlflow.log_metrics({"wall_time_sec": elapsed, "steps_per_second": sps, "final_reward": final_reward})
    mlflow.log_artifact(str(metrics_path))
    final_model = save_dir / f"model_{tag}.safetensors"
    if final_model.exists():
        mlflow.log_artifact(str(final_model))
    sidecar = save_dir / f"recipe_{tag}.yaml"
    if sidecar.exists():
        mlflow.log_artifact(str(sidecar))
    mlflow.end_run()

    print(f"\nDone in {elapsed:.1f}s ({sps:,.0f} sps). Final reward: {final_reward:.1f}")
    print(f"Saved to: {save_dir}")

    from jaxborg.evaluation.scripted_red import run_configured_after_training

    run_configured_after_training(final_model, recipe)


if __name__ == "__main__":
    main()
