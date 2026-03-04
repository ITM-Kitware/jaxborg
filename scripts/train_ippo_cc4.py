"""IPPO (Independent PPO) for CC4 with FsmRedCC4Env.

Based on PureJaxRL PPO and JaxMARL's ippo_ff_cage.py.
Trains Blue agents against scripted FSM red agents using feedforward networks.
Fully JIT'd: rollout collection via jax.lax.scan + PPO update in one compiled fn.
"""

import json
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.fsm_red_env import FsmRedCC4Env


class ActorCritic(nn.Module):
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
            unavail_actions = 1 - avail_actions
            action_logits = action_logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    avail_actions: jnp.ndarray
    info: jnp.ndarray


class MetricsLogger:
    """Logs to both JSONL file and MLflow."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "w")

    def log(self, metrics: dict, step: int):
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()
        mlflow_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating)) and k not in ("steps", "update")
        }
        mlflow.log_metrics(mlflow_metrics, step=step)

    def close(self):
        self.file.close()
        mlflow.log_artifact(str(self.filepath))


def make_train(config):
    """Build env, network, and a single JIT'd collect_and_update function.

    Returns (env, network, init_obs, init_env_state, init_train_state_fn,
    collect_and_update_fn).

    collect_and_update scans NUM_STEPS env steps then runs GAE + PPO update,
    all inside one XLA program.
    """
    inner_env = FsmRedCC4Env(num_steps=500)
    agents = list(inner_env.agents)
    num_agents = inner_env.num_agents  # 5
    config["NUM_ACTORS"] = num_agents
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"]
    config["MINIBATCH_SIZE"] = num_agents * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = LogWrapper(inner_env)

    # Single env init (no env-batch dimension)
    init_key = jax.random.PRNGKey(config["SEED"])
    init_obs, init_env_state = env.reset(init_key)

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
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

    @jax.jit
    def _collect_and_update(train_state, env_state, obs, rng):
        """Scan NUM_STEPS env steps, compute GAE, run PPO epochs — all JIT'd."""

        # --- Rollout via scan ---
        def _env_step(carry, _):
            env_state, obs, rng = carry

            obs_batch = jnp.stack([obs[a] for a in agents])
            avail_batch = jnp.stack([compute_blue_action_mask(env_state.env_state.const, i) for i in range(num_agents)])

            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, obs_batch, avail_batch)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            env_act = {agents[i]: action[i] for i in range(num_agents)}

            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, rewards, dones, info = env.step(_rng, env_state, env_act)

            transition = Transition(
                done=jnp.stack([dones[a] for a in agents]),
                action=action,
                value=value,
                reward=jnp.stack([rewards[a] for a in agents]),
                log_prob=log_prob,
                obs=obs_batch,
                avail_actions=avail_batch,
                info=info,
            )

            return (new_env_state, new_obs, rng), transition

        (env_state, obs, rng), traj_batch = jax.lax.scan(_env_step, (env_state, obs, rng), None, config["NUM_STEPS"])

        # --- GAE ---
        last_obs_batch = jnp.stack([obs[a] for a in agents])
        _, last_val = network.apply(train_state.params, last_obs_batch)

        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=8,
        )
        targets = advantages + traj_batch.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO update epochs ---
        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    pi, value = network.apply(params, traj_batch.obs, traj_batch.avail_actions)
                    log_prob = pi.log_prob(traj_batch.action)

                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    logratio = log_prob - traj_batch.log_prob
                    approx_kl = jnp.mean((ratio - 1) - logratio)
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    var_targets = jnp.var(targets)
                    explained_var = jnp.where(
                        var_targets > 0,
                        1 - jnp.var(targets - value) / var_targets,
                        0.0,
                    )

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, approx_kl, clip_frac, explained_var)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)

                loss_info = {
                    "total_loss": total_loss[0],
                    "actor_loss": total_loss[1][1],
                    "critic_loss": total_loss[1][0],
                    "entropy": total_loss[1][2],
                    "approx_kl": total_loss[1][3],
                    "clip_frac": total_loss[1][4],
                    "explained_var": total_loss[1][5],
                }
                return train_state, loss_info

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        rng = update_state[-1]

        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)
        metric = {**metric, **loss_info}

        return train_state, env_state, obs, rng, metric

    return env, network, init_obs, init_env_state, _init_train_state, _collect_and_update


EXP_DIR = Path(__file__).resolve().parent.parent / "jaxborg-exp"


@hydra.main(config_path="configs", config_name="ippo_cc4", version_base=None)
def main(cfg):
    config = OmegaConf.to_container(cfg)

    save_dir = EXP_DIR / "ippo_cc4"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not mlflow.get_tracking_uri().startswith("sqlite"):
        mlflow.set_tracking_uri(f"sqlite:///{save_dir / 'mlflow.db'}")
    mlflow.set_experiment("ippo-cc4")
    mlflow.start_run(run_name="ippo-vs-fsm-red")

    mlflow.log_params(
        {
            "algorithm": "IPPO-FF",
            "seed": config["SEED"],
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
            "hidden_dim": config.get("HIDDEN_DIM", 256),
            "activation": config["ACTIVATION"],
            "anneal_lr": config["ANNEAL_LR"],
        }
    )

    print("=" * 60)
    print("IPPO-FF CC4 Training: Blue vs FSM Red")
    print("=" * 60)
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num steps per rollout: {config['NUM_STEPS']}")
    print(f"Hidden dim: {config.get('HIDDEN_DIM', 256)}")
    print(f"Activation: {config['ACTIVATION']}")
    print("=" * 60)

    env, network, init_obs, init_env_state, init_train_state, collect_and_update = make_train(config)

    rng = jax.random.PRNGKey(config["SEED"] + 1)
    rng, _rng = jax.random.split(rng)
    train_state = init_train_state(_rng)

    env_state = init_env_state
    obs = init_obs

    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    num_updates = int(config["NUM_UPDATES"])
    num_steps = int(config["NUM_STEPS"])

    logger = MetricsLogger(save_dir / "metrics.jsonl")
    best_reward = float("-inf")

    start_time = time.perf_counter()
    print(f"Starting training ({num_updates} updates, fully JIT'd)...")
    print("  (first update includes XLA compilation — may take a few minutes)")

    for update_idx in range(num_updates):
        train_state, env_state, obs, rng, metric = collect_and_update(train_state, env_state, obs, rng)

        if update_idx == 0:
            elapsed_first = time.perf_counter() - start_time
            print(f"  first update compiled + ran in {elapsed_first:.1f}s")

        step = (update_idx + 1) * num_steps
        reward = float(metric["returned_episode_returns"])
        record = {
            "update": update_idx + 1,
            "steps": step,
            "episode_reward_mean": reward,
            "loss": float(metric["total_loss"]),
            "policy_loss": float(metric["actor_loss"]),
            "value_loss": float(metric["critic_loss"]),
            "entropy": float(metric["entropy"]),
            "approx_kl": float(metric["approx_kl"]),
            "clip_frac": float(metric["clip_frac"]),
            "explained_var": float(metric["explained_var"]),
        }
        logger.log(record, step=step)
        if reward > best_reward:
            best_reward = reward

        if (update_idx + 1) % 50 == 0 or update_idx == num_updates - 1:
            elapsed = time.perf_counter() - start_time
            sps = step / elapsed
            print(f"  update {update_idx + 1}/{num_updates} | step {step} | reward {reward:.1f} | {sps:.0f} sps")

    logger.close()

    elapsed = time.perf_counter() - start_time
    total_steps = int(config["TOTAL_TIMESTEPS"])
    sps = total_steps / elapsed

    params = train_state.params
    checkpoint_path = save_dir / "checkpoint_final.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "params": params,
                "hidden_dim": config.get("HIDDEN_DIM", 256),
                "activation": config["ACTIVATION"],
                "action_dim": env.action_space(env.agents[0]).n,
            },
            f,
        )

    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    mlflow.log_artifact(str(config_path))

    final_return = float(metric["returned_episode_returns"])
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
