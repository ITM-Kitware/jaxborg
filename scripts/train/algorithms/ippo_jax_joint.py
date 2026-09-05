"""Dual-team IPPO helpers for the JAX CC4 environment.

This module deliberately sits beside :mod:`ippo_jax` instead of replacing its
legacy Blue-vs-FSM rollout.  A joint rollout is selected only when a learned
Red policy is present.  Blue and Red share parameters within their own team,
but never share a network, optimizer, reward normalizer, or PPO batch.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxborg.evaluation.jax_env_factory import make_joint_jax_env

TEAMS = ("blue", "red")


class TeamTransition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    avail_actions: jax.Array
    actor_mask: jax.Array
    critic_mask: jax.Array


class RewardNormState(NamedTuple):
    returns: jax.Array
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def initial_reward_norm_state(num_envs: int) -> RewardNormState:
    return RewardNormState(
        returns=jnp.zeros(num_envs, dtype=jnp.float32),
        mean=jnp.zeros((), dtype=jnp.float32),
        var=jnp.ones((), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def _masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    weight = mask.astype(jnp.float32)
    return jnp.sum(value * weight) / jnp.maximum(weight.sum(), 1.0)


def _masked_normalize(value: jax.Array, mask: jax.Array) -> jax.Array:
    mean = _masked_mean(value, mask)
    var = _masked_mean(jnp.square(value - mean), mask)
    return (value - mean) / (jnp.sqrt(var) + 1e-8)


def _masked_value_loss(
    value: jax.Array,
    old_value: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    clip_eps: float,
    clip_value_loss: bool,
) -> jax.Array:
    losses = jnp.square(value - targets)
    if clip_value_loss:
        clipped = old_value + (value - old_value).clip(-clip_eps, clip_eps)
        losses = jnp.maximum(losses, jnp.square(clipped - targets))
    return 0.5 * _masked_mean(losses, mask)


def _normalize_reward(
    reward: jax.Array,
    done: jax.Array,
    state: RewardNormState,
    config: Mapping[str, Any],
) -> tuple[jax.Array, RewardNormState]:
    """Normalize a scalar team payoff across vectorized environments."""
    if not bool(config.get("NORM_REWARDS", False)):
        return reward * float(config.get("REWARD_SCALE", 1.0)), state

    new_returns = state.returns * float(config["GAMMA"]) + reward
    batch_mean = jnp.mean(new_returns)
    batch_var = jnp.var(new_returns)
    batch_count = jnp.asarray(reward.shape[0], dtype=jnp.float32)
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / total_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / total_count
    new_var = m2 / total_count
    scaled = jnp.clip(reward / (jnp.sqrt(new_var) + 1e-8), -10.0, 10.0)
    next_returns = new_returns * (1.0 - done.astype(jnp.float32))
    next_state = RewardNormState(next_returns, new_mean, new_var, total_count)
    return scaled * float(config.get("REWARD_SCALE", 1.0)), next_state


def _make_optimizer(config: Mapping[str, Any]):
    if bool(config.get("ANNEAL_LR", False)):
        num_updates = max(1, int(config["NUM_UPDATES"]))
        steps_per_update = int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])

        def schedule(count):
            update = count // steps_per_update
            return float(config["LR"]) * (1.0 - update / num_updates)

        return optax.adam(schedule, eps=1e-5)
    return optax.adam(float(config["LR"]), eps=1e-5)


def _make_team_updater(network, config: Mapping[str, Any]):
    """Create one PPO update function for one homogeneous team batch."""

    gamma = float(config["GAMMA"])
    gae_lambda = float(config["GAE_LAMBDA"])
    clip_eps = float(config["CLIP_EPS"])
    vf_coef = float(config["VF_COEF"])
    ent_coef = float(config["ENT_COEF"])
    max_grad_norm = float(config["MAX_GRAD_NORM"])
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", False))
    num_minibatches = int(config["NUM_MINIBATCHES"])
    update_epochs = int(config["UPDATE_EPOCHS"])

    def update(train_state, traj, last_value, rng):
        def gae_step(carry, transition):
            gae, next_value = carry
            delta = transition.reward + gamma * next_value * (1.0 - transition.done) - transition.value
            gae = delta + gamma * gae_lambda * (1.0 - transition.done) * gae
            # Inactive rows are not critic samples and may span many ticks.
            gae = gae * transition.critic_mask
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_value), last_value),
            traj,
            reverse=True,
            unroll=8,
        )
        targets = advantages + traj.value
        advantages = _masked_normalize(advantages, traj.actor_mask)

        # T x E x A -> one independent-IPPO sample axis.
        batch_size = traj.action.size
        if batch_size % num_minibatches != 0:
            raise ValueError(f"team batch ({batch_size}) must be divisible by NUM_MINIBATCHES ({num_minibatches})")
        flat_batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), traj)
        flat_advantages = advantages.reshape(batch_size)
        flat_targets = targets.reshape(batch_size)

        def epoch_step(carry, _):
            train_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            permutation = jax.random.permutation(perm_key, batch_size)
            shuffled = (
                jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), flat_batch),
                jnp.take(flat_advantages, permutation, axis=0),
                jnp.take(flat_targets, permutation, axis=0),
            )
            minibatches = jax.tree.map(
                lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]),
                shuffled,
            )

            def minibatch_step(train_state, batch):
                transitions, gae, batch_targets = batch

                def loss_fn(params):
                    pi, value = network.apply(params, transitions.obs, transitions.avail_actions)
                    log_prob = pi.log_prob(transitions.action)
                    ratio = jnp.exp(log_prob - transitions.log_prob)
                    log_ratio = log_prob - transitions.log_prob
                    actor_mask = transitions.actor_mask
                    actor_loss = -_masked_mean(
                        jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae,
                        ),
                        actor_mask,
                    )
                    entropy = _masked_mean(pi.entropy(), actor_mask)
                    value_loss = _masked_value_loss(
                        value,
                        transitions.value,
                        batch_targets,
                        transitions.critic_mask,
                        clip_eps,
                        clip_value_loss,
                    )
                    approx_kl = _masked_mean((ratio - 1.0) - log_ratio, actor_mask)
                    clip_frac = _masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32), actor_mask)
                    target_mean = _masked_mean(batch_targets, transitions.critic_mask)
                    target_var = _masked_mean(jnp.square(batch_targets - target_mean), transitions.critic_mask)
                    residual = batch_targets - value
                    residual_mean = _masked_mean(residual, transitions.critic_mask)
                    residual_var = _masked_mean(jnp.square(residual - residual_mean), transitions.critic_mask)
                    explained_var = jnp.where(target_var > 0, 1.0 - residual_var / target_var, 0.0)
                    total = actor_loss + vf_coef * value_loss - ent_coef * entropy
                    aux = {
                        "total_loss": total,
                        "actor_loss": actor_loss,
                        "critic_loss": value_loss,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clip_frac": clip_frac,
                        "explained_var": explained_var,
                    }
                    return total, aux

                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                del loss
                pre_clip = optax.global_norm(grads)
                scale = jnp.minimum(1.0, max_grad_norm / (pre_clip + 1e-8))
                grads = jax.tree.map(lambda x: x * scale, grads)
                metrics["pre_clip_grad_norm"] = pre_clip
                metrics["grad_norm"] = optax.global_norm(grads)
                return train_state.apply_gradients(grads=grads), metrics

            train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, rng), metrics

        (train_state, rng), metrics = jax.lax.scan(
            epoch_step,
            (train_state, rng),
            None,
            update_epochs,
        )
        return train_state, rng, jax.tree.map(lambda x: x.mean(), metrics)

    return update


def make_joint_train(
    team_configs: Mapping[str, dict[str, Any]],
    networks: Mapping[str, Any],
    *,
    trainable_teams: tuple[str, ...],
    initial_params: Mapping[str, Any] | None = None,
):
    """Build the joint environment and a JIT'd rollout/update function.

    Both network forward passes happen before the single call to ``env.step``.
    Frozen policies still participate in inference, but their PPO updater is
    omitted and their parameters therefore remain byte-identical.
    """

    if set(team_configs) != set(TEAMS) or set(networks) != set(TEAMS):
        raise ValueError("joint training requires Blue and Red policy runtimes")
    if not trainable_teams or not set(trainable_teams) <= set(TEAMS):
        raise ValueError(f"invalid trainable teams: {trainable_teams}")

    base = team_configs[trainable_teams[0]]
    num_envs = int(base["NUM_ENVS"])
    num_steps = int(base["NUM_STEPS"])
    topology_bank = tuple(base.get("TOPOLOGY_BANK") or ())
    for team, cfg in team_configs.items():
        if int(cfg["NUM_ENVS"]) != num_envs or int(cfg["NUM_STEPS"]) != num_steps:
            raise ValueError(f"{team} must share NUM_ENVS and NUM_STEPS in a joint rollout")
        if tuple(cfg.get("TOPOLOGY_BANK") or ()) != topology_bank:
            raise ValueError(f"{team} must share TOPOLOGY_BANK in a joint rollout")
        cfg["NUM_UPDATES"] = int(cfg["TOTAL_TIMESTEPS"]) // (num_envs * num_steps)

    env = make_joint_jax_env(
        base["TRAIN_VARIANT"],
        topology_mode=base.get("TOPOLOGY_MODE", "generative"),
        training_mode=bool(base.get("TRAINING_MODE", True)),
        topology_path=list(topology_bank) if topology_bank else None,
    )
    agents = {
        "blue": tuple(env.blue_agents),
        "red": tuple(env.red_agents),
    }
    num_agents = {team: len(names) for team, names in agents.items()}

    reset_key = jax.random.PRNGKey(int(base["SEED"]))
    reset_keys = jax.random.split(reset_key, num_envs)
    init_obs, init_env_state = jax.vmap(env.reset)(reset_keys)
    supplied_params = dict(initial_params or {})

    def init_train_states(rng):
        keys = dict(zip(TEAMS, jax.random.split(rng, len(TEAMS))))
        states = {}
        for team in TEAMS:
            cfg = team_configs[team]
            obs_shape = env.observation_space(agents[team][0]).shape
            params = supplied_params.get(team)
            if params is None:
                params = networks[team].init(keys[team], jnp.zeros(obs_shape, dtype=jnp.float32))
            states[team] = TrainState.create(
                apply_fn=networks[team].apply,
                params=params,
                tx=_make_optimizer(cfg),
            )
        return states

    updaters = {team: _make_team_updater(networks[team], team_configs[team]) for team in trainable_teams}
    info_keys = (
        "reward_ria",
        "reward_lwf",
        "reward_asf",
        "action_cost",
        "impact_count",
        "green_lwf_count",
        "green_asf_count",
    )

    # Do not donate the nested team state here. Small scalar leaves in the two
    # reward-normalizer pytrees may alias after construction, and XLA rejects
    # donating one physical buffer through two flattened arguments.
    @jax.jit
    def collect_and_update(train_states, env_state, obs, rng, reward_norm_states):
        info_init = {key: jnp.zeros(num_envs, dtype=jnp.float32) for key in info_keys}

        def env_step(carry, _):
            env_state, obs, rng, norm_states, info_sums = carry
            masks = jax.vmap(env.get_avail_actions)(env_state)
            rng, blue_key, red_key, step_key = jax.random.split(rng, 4)
            actions = {}
            transition_parts = {}

            # Both teams consume the same pre-step state before any action is
            # applied.  Keeping these forward passes together is intentional.
            for team, action_key in (("blue", blue_key), ("red", red_key)):
                names = agents[team]
                obs_batch = jnp.stack([obs[name] for name in names], axis=1)
                mask_batch = jnp.stack([masks[name] for name in names], axis=1)
                flat_obs = obs_batch.reshape((-1, obs_batch.shape[-1]))
                flat_mask = mask_batch.reshape((-1, mask_batch.shape[-1]))
                pi, value = networks[team].apply(train_states[team].params, flat_obs, flat_mask)
                flat_action = pi.sample(seed=action_key)
                flat_log_prob = pi.log_prob(flat_action)
                shape = (num_envs, num_agents[team])
                team_actions = flat_action.reshape(shape)
                for idx, name in enumerate(names):
                    actions[name] = team_actions[:, idx]
                transition_parts[team] = (
                    obs_batch,
                    mask_batch,
                    team_actions,
                    value.reshape(shape),
                    flat_log_prob.reshape(shape),
                )

            before = env_state.state
            step_keys = jax.random.split(step_key, num_envs)
            new_obs, new_env_state, rewards, dones, infos = jax.vmap(env.step)(step_keys, env_state, actions)
            info_sums = {key: info_sums[key] + jnp.asarray(infos[key], dtype=jnp.float32) for key in info_keys}
            done_env = dones["__all__"].astype(jnp.float32)
            transitions = {}
            for team in TEAMS:
                names = agents[team]
                obs_batch, mask_batch, team_actions, value, log_prob = transition_parts[team]
                raw_reward = rewards[names[0]]
                scaled_reward, next_norm = _normalize_reward(
                    raw_reward,
                    done_env,
                    norm_states[team],
                    team_configs[team],
                )
                norm_states[team] = next_norm
                reward_batch = jnp.repeat(scaled_reward[:, None], num_agents[team], axis=1)
                if team == "blue":
                    idle_before = before.blue_pending_ticks == 0
                    actor_mask = idle_before.astype(jnp.float32)
                    critic_mask = jnp.ones_like(actor_mask)
                    transition_done = jnp.repeat(done_env[:, None], num_agents[team], axis=1)
                else:
                    active_before = before.red_agent_active
                    idle_before = before.red_pending_ticks == 0
                    actor_mask = (active_before & idle_before).astype(jnp.float32)
                    critic_mask = active_before.astype(jnp.float32)
                    active_after = new_env_state.state.red_agent_active
                    transition_done = jnp.maximum(done_env[:, None], (~active_after).astype(jnp.float32))
                transitions[team] = TeamTransition(
                    done=transition_done,
                    action=team_actions,
                    value=value,
                    reward=reward_batch,
                    log_prob=log_prob,
                    obs=obs_batch,
                    avail_actions=mask_batch,
                    actor_mask=actor_mask,
                    critic_mask=critic_mask,
                )
            return (new_env_state, new_obs, rng, norm_states, info_sums), transitions

        (env_state, obs, rng, reward_norm_states, info_sums), trajectories = jax.lax.scan(
            env_step,
            (env_state, obs, rng, reward_norm_states, info_init),
            None,
            num_steps,
        )

        metrics = {}
        for team in TEAMS:
            names = agents[team]
            obs_batch = jnp.stack([obs[name] for name in names], axis=1)
            flat_obs = obs_batch.reshape((-1, obs_batch.shape[-1]))
            _, last_value = networks[team].apply(train_states[team].params, flat_obs)
            last_value = last_value.reshape((num_envs, num_agents[team]))
            if team == "red":
                last_value = last_value * env_state.state.red_agent_active.astype(jnp.float32)

            if team in trainable_teams:
                rng, update_key = jax.random.split(rng)
                state, _, team_metrics = updaters[team](train_states[team], trajectories[team], last_value, update_key)
                train_states[team] = state
            else:
                zero = jnp.zeros((), dtype=jnp.float32)
                team_metrics = {
                    "total_loss": zero,
                    "actor_loss": zero,
                    "critic_loss": zero,
                    "entropy": zero,
                    "approx_kl": zero,
                    "clip_frac": zero,
                    "explained_var": zero,
                    "pre_clip_grad_norm": zero,
                    "grad_norm": zero,
                }
            sign = 1.0 if team == "blue" else -1.0
            raw_return = (
                sign
                * (
                    info_sums["reward_ria"]
                    + info_sums["reward_lwf"]
                    + info_sums["reward_asf"]
                    + info_sums["action_cost"]
                ).mean()
            )
            team_metrics["raw_rollout_return"] = raw_return
            team_metrics["mean_rollout_return"] = trajectories[team].reward.sum(axis=0).mean()
            team_metrics["actor_fraction"] = trajectories[team].actor_mask.mean()
            team_metrics["critic_fraction"] = trajectories[team].critic_mask.mean()
            metrics[team] = team_metrics

        metrics["game"] = {
            "blue_return": metrics["blue"]["raw_rollout_return"],
            "red_return": metrics["red"]["raw_rollout_return"],
        }
        return train_states, env_state, obs, rng, reward_norm_states, metrics

    return env, init_obs, init_env_state, init_train_states, collect_and_update


__all__ = [
    "RewardNormState",
    "TeamTransition",
    "initial_reward_norm_state",
    "make_joint_train",
]
