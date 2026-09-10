"""Dual-team PPO helpers for IPPO and Blue MAPPO in the JAX CC4 environment.

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
from jaxborg.policies import (
    has_centralized_critic,
    init_policy_params,
    initial_carry,
    is_recurrent,
    policy_sequence,
    policy_step,
)

TEAMS = ("blue", "red")

# Signed payoff terms; these four sum to a team's rollout return.
REWARD_COMPONENTS = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
# Unsigned game outcomes shared by both teams.
GAME_COUNTERS = ("impact_count", "green_lwf_count", "green_asf_count")


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
    # Recurrent policies only: the hidden state was zeroed before this row was
    # acted on. Kept beside ``done`` rather than derived from it because the
    # two differ for Red -- ``done`` is real termination, while a sequence also
    # restarts when a dormant Red agent is revived by session reassignment.
    # ``None`` for feedforward archs, which have no hidden state to reset.
    reset: jax.Array | None = None
    # MAPPO only: world state from the same pre-step state as obs/value.
    critic_obs: jax.Array | None = None


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


def next_sequence_reset(team: str, episode_done: jax.Array, active_before: jax.Array | None) -> jax.Array:
    """Whether the *next* row opens a new sequence for a recurrent policy.

    Blue never goes dormant, so only termination restarts its sequence. A Red
    agent that loses every session is revived later by session reassignment on
    a different foothold: the hidden state its predecessor built describes a
    part of the network it no longer has access to, so that is misinformation
    rather than context. Keyed on the *previous* step's activity so the first
    live row after a gap is the one that starts blank -- keying it on the
    current step would carry the pre-eviction state across the gap.

    Kept out of ``done``: ``done`` is real termination and must stay that way
    for ``compute_gae`` to bootstrap Red's credit across a dormancy gap.
    """
    done = episode_done > 0
    if team == "blue":
        return done
    if active_before is None:
        raise ValueError("red's sequence reset needs the pre-step activity mask")
    return done | ~active_before


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


def compute_gae(
    traj: TeamTransition,
    last_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Semi-MDP GAE for teams whose agents go dormant mid-episode.

    ``critic_mask`` marks the rows where the agent is live and its value head
    is a trained sample.  A Red agent that loses every session goes dormant and
    is later revived by session reassignment, so dormancy is a gap inside one
    episode, not a terminal state.  Rows in a gap therefore:

    * contribute no advantage of their own (they are not decision points),
    * accumulate the team payoff into the bootstrap so the preceding live row
      sees what the gap earned, and
    * never leak their own value head into the bootstrap, because that head is
      excluded from the value loss and so is never trained.

    ``done`` must mark real episode termination only. Folding dormancy into it
    asserts a zero continuation value and hides every reward that lands after a
    Red eviction, which trains Red to be myopic.

    Teams whose ``critic_mask`` is all ones (Blue) reduce exactly to textbook
    GAE. Returns ``(advantages, targets)``.
    """

    def gae_step(carry, transition):
        gae, bootstrap = carry
        alive = 1.0 - transition.done
        live = transition.critic_mask
        carried = gamma * gae_lambda * alive * gae
        delta = transition.reward + gamma * bootstrap * alive - transition.value
        # Live rows emit a TD error; gap rows only decay what follows them, so
        # credit still crosses the gap discounted by its true length.
        gae = jnp.where(live > 0, delta + carried, carried)
        # A gap row hands back the payoff it collected instead of its own value.
        bootstrap = jnp.where(live > 0, transition.value, transition.reward + gamma * bootstrap * alive)
        return (gae, bootstrap), gae * live

    _, advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(last_value), last_value),
        traj,
        reverse=True,
        unroll=8,
    )
    return advantages, advantages + traj.value


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
    recurrent = is_recurrent(network)
    centralized = has_centralized_critic(network)

    def ppo_objective(pi, value, transitions, gae, targets):
        """Shared loss body. Every reduction is a mask-weighted mean, so it is
        indifferent to whether the batch is flat rows or (time, sequence)."""
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
            targets,
            transitions.critic_mask,
            clip_eps,
            clip_value_loss,
        )
        approx_kl = _masked_mean((ratio - 1.0) - log_ratio, actor_mask)
        clip_frac = _masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32), actor_mask)
        target_mean = _masked_mean(targets, transitions.critic_mask)
        target_var = _masked_mean(jnp.square(targets - target_mean), transitions.critic_mask)
        residual = targets - value
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

    def apply_gradients(train_state, loss_fn, params):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        del loss
        pre_clip = optax.global_norm(grads)
        if centralized:
            # JaxMARL MAPPO clips actor and critic independently. Adam's
            # elementwise moments stay separate in our combined parameter tree;
            # retaining one TrainState keeps the existing bundle ABI intact.
            norms = {
                role: optax.global_norm({k: v for k, v in grads["params"].items() if k.startswith(role + "_")})
                for role in ("actor", "critic")
            }
            scales = {role: jnp.minimum(1.0, max_grad_norm / (norm + 1e-8)) for role, norm in norms.items()}
            grads = jax.tree.map_with_path(lambda path, g: g * scales[path[1].key.split("_", 1)[0]], grads)
        else:
            scale = jnp.minimum(1.0, max_grad_norm / (pre_clip + 1e-8))
            grads = jax.tree.map(lambda x: x * scale, grads)
        metrics["pre_clip_grad_norm"] = pre_clip
        metrics["grad_norm"] = optax.global_norm(grads)
        return train_state.apply_gradients(grads=grads), metrics

    def flat_epoch(carry, batch):
        """Feedforward layout: every (t, env, agent) row is an independent sample."""
        flat_batch, flat_advantages, flat_targets, batch_size = batch
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

        def minibatch_step(train_state, minibatch):
            transitions, gae, batch_targets = minibatch

            def loss_fn(params):
                pi, value, _ = policy_step(
                    network, params, transitions.obs, transitions.avail_actions, critic_obs=transitions.critic_obs
                )
                return ppo_objective(pi, value, transitions, gae, batch_targets)

            return apply_gradients(train_state, loss_fn, train_state.params)

        train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
        return (train_state, rng), metrics

    def sequence_epoch(carry, batch):
        """Recurrent layout: a minibatch is whole trajectories, time axis intact.

        Shuffling rows would destroy the order the hidden state is defined by,
        so the permutation is over sequences (env x agent) and each minibatch is
        replayed from the hidden state its window began with.
        """
        seq_batch, seq_advantages, seq_targets, carry_batch, num_sequences = batch
        train_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, num_sequences)
        shuffled = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=1),
            (seq_batch, seq_advantages, seq_targets, carry_batch),
        )
        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(x.reshape((x.shape[0], num_minibatches, -1) + x.shape[2:]), 0, 1),
            shuffled,
        )

        def minibatch_step(train_state, minibatch):
            transitions, gae, batch_targets, init_carry = minibatch
            # The leading axis of 1 exists only so the carry rides the same
            # take/reshape as the time-major arrays; drop it before the replay.
            init_carry = jax.tree.map(lambda leaf: leaf[0], init_carry)

            def loss_fn(params):
                pi, value, _ = policy_sequence(
                    network,
                    params,
                    transitions.obs,
                    transitions.avail_actions,
                    carry=init_carry,
                    reset=transitions.reset,
                    critic_obs=transitions.critic_obs,
                )
                return ppo_objective(pi, value, transitions, gae, batch_targets)

            return apply_gradients(train_state, loss_fn, train_state.params)

        train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
        return (train_state, rng), metrics

    def update(train_state, traj, last_value, rng, init_carry=None):
        if centralized and traj.critic_obs is None:
            raise ValueError("MAPPO updates require stored critic_obs; actor-only inference is for evaluation")
        if recurrent and init_carry is None:
            raise ValueError("a recurrent team updater needs the hidden state the rollout window started from")
        advantages, targets = compute_gae(traj, last_value, gamma=gamma, gae_lambda=gae_lambda)
        advantages = _masked_normalize(advantages, traj.actor_mask)

        if recurrent:
            # T x E x A -> T x (E*A): one sequence per (env, agent), time intact.
            num_sequences = int(traj.action.shape[1] * traj.action.shape[2])
            if num_sequences % num_minibatches != 0:
                raise ValueError(
                    f"recurrent minibatching splits sequences, not rows: env x agent ({num_sequences}) "
                    f"must be divisible by NUM_MINIBATCHES ({num_minibatches})"
                )
            to_sequences = lambda x: x.reshape((x.shape[0], num_sequences) + x.shape[3:])  # noqa: E731
            batch = (
                jax.tree.map(to_sequences, traj),
                to_sequences(advantages),
                to_sequences(targets),
                jax.tree.map(lambda leaf: leaf[None], init_carry),
                num_sequences,
            )
            epoch_step = sequence_epoch
        else:
            # T x E x A -> one independent-IPPO sample axis.
            batch_size = traj.action.size
            if batch_size % num_minibatches != 0:
                raise ValueError(f"team batch ({batch_size}) must be divisible by NUM_MINIBATCHES ({num_minibatches})")
            batch = (
                jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), traj),
                advantages.reshape(batch_size),
                targets.reshape(batch_size),
                batch_size,
            )
            epoch_step = flat_epoch

        (train_state, rng), metrics = jax.lax.scan(
            lambda carry, _: epoch_step(carry, batch),
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
    if has_centralized_critic(networks["red"]):
        raise ValueError("CC4 centralized critic inputs are currently defined for Blue only; use IPPO for Red")

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
                params = init_policy_params(networks[team], keys[team], int(obs_shape[-1]))
            states[team] = TrainState.create(
                apply_fn=networks[team].apply,
                params=params,
                tx=_make_optimizer(cfg),
            )
        return states

    updaters = {team: _make_team_updater(networks[team], team_configs[team]) for team in trainable_teams}
    info_keys = REWARD_COMPONENTS + GAME_COUNTERS
    recurrent = {team: is_recurrent(networks[team]) for team in TEAMS}
    centralized = {team: has_centralized_critic(networks[team]) and team in trainable_teams for team in TEAMS}

    def critic_observations(team, env_state):
        if not centralized[team]:
            return None
        return jax.vmap(lambda s: env.get_critic_obs(s, networks[team].critic_input))(env_state)

    # One sequence per (env, agent), ordered env-major to match the flatten in
    # the rollout and the reshape in the updater.
    sequence_counts = {team: num_envs * num_agents[team] for team in TEAMS}

    # Do not donate the nested team state here. Small scalar leaves in the two
    # reward-normalizer pytrees may alias after construction, and XLA rejects
    # donating one physical buffer through two flattened arguments.
    @jax.jit
    def collect_and_update(train_states, env_state, obs, rng, reward_norm_states):
        info_init = {key: jnp.zeros(num_envs, dtype=jnp.float32) for key in info_keys}
        # Truncated BPTT with the window set to the rollout. NUM_STEPS is the
        # recipe's episode_length and every env resets on the same tick, so a
        # window boundary is an episode boundary and starting from a blank
        # hidden state loses nothing. Mid-window resets, if a future recipe
        # ever produces them, are still handled by the per-row reset flags.
        window_carries = {team: initial_carry(networks[team], sequence_counts[team]) for team in TEAMS}
        # Row 0 of a window opens a sequence, so it resets by definition.
        reset_init = {team: jnp.ones((num_envs, num_agents[team]), dtype=jnp.bool_) for team in TEAMS}

        def env_step(carry, _):
            env_state, obs, rng, norm_states, info_sums, carries, resets = carry
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
                critic_obs = critic_observations(team, env_state)
                pi, value, carries[team] = policy_step(
                    networks[team],
                    train_states[team].params,
                    flat_obs,
                    flat_mask,
                    carry=carries[team],
                    reset=resets[team].reshape(-1),
                    critic_obs=None if critic_obs is None else critic_obs.reshape((-1, critic_obs.shape[-1])),
                )
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
                    critic_obs,
                )

            before = env_state.state
            step_keys = jax.random.split(step_key, num_envs)
            new_obs, new_env_state, rewards, dones, infos = jax.vmap(env.step)(step_keys, env_state, actions)
            info_sums = {key: info_sums[key] + jnp.asarray(infos[key], dtype=jnp.float32) for key in info_keys}
            done_env = dones["__all__"].astype(jnp.float32)
            transitions = {}
            next_resets = {}
            for team in TEAMS:
                names = agents[team]
                obs_batch, mask_batch, team_actions, value, log_prob, critic_obs = transition_parts[team]
                raw_reward = rewards[names[0]]
                scaled_reward, next_norm = _normalize_reward(
                    raw_reward,
                    done_env,
                    norm_states[team],
                    team_configs[team],
                )
                norm_states[team] = next_norm
                reward_batch = jnp.repeat(scaled_reward[:, None], num_agents[team], axis=1)
                episode_done = jnp.repeat(done_env[:, None], num_agents[team], axis=1)
                if team == "blue":
                    idle_before = before.blue_pending_ticks == 0
                    actor_mask = idle_before.astype(jnp.float32)
                    critic_mask = jnp.ones_like(actor_mask)
                    transition_done = episode_done
                    next_resets[team] = next_sequence_reset(team, episode_done, None)
                else:
                    active_before = before.red_agent_active
                    idle_before = before.red_pending_ticks == 0
                    actor_mask = (active_before & idle_before).astype(jnp.float32)
                    # Dormancy is a gap, not a terminal state: an evicted Red
                    # agent is revived by session reassignment later in the same
                    # episode. `critic_mask` keeps those rows out of the losses;
                    # `done` stays real termination so compute_gae can bootstrap
                    # across the gap instead of zeroing Red's future.
                    critic_mask = active_before.astype(jnp.float32)
                    transition_done = episode_done
                    next_resets[team] = next_sequence_reset(team, episode_done, active_before)
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
                    reset=resets[team] if recurrent[team] else None,
                    critic_obs=critic_obs,
                )
            return (new_env_state, new_obs, rng, norm_states, info_sums, carries, next_resets), transitions

        (env_state, obs, rng, reward_norm_states, info_sums, carries, resets), trajectories = jax.lax.scan(
            env_step,
            (env_state, obs, rng, reward_norm_states, info_init, window_carries, reset_init),
            None,
            num_steps,
        )

        metrics = {}
        for team in TEAMS:
            names = agents[team]
            obs_batch = jnp.stack([obs[name] for name in names], axis=1)
            flat_obs = obs_batch.reshape((-1, obs_batch.shape[-1]))
            critic_obs = critic_observations(team, env_state)
            _, last_value, _ = policy_step(
                networks[team],
                train_states[team].params,
                flat_obs,
                carry=carries[team],
                reset=resets[team].reshape(-1),
                critic_obs=None if critic_obs is None else critic_obs.reshape((-1, critic_obs.shape[-1])),
            )
            last_value = last_value.reshape((num_envs, num_agents[team]))
            if team == "red":
                last_value = last_value * env_state.state.red_agent_active.astype(jnp.float32)

            if team in trainable_teams:
                rng, update_key = jax.random.split(rng)
                state, _, team_metrics = updaters[team](
                    train_states[team],
                    trajectories[team],
                    last_value,
                    update_key,
                    init_carry=window_carries[team],
                )
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
            # Signed so the four components still sum to raw_rollout_return.
            # Logging them apart separates "Red landed impacts" from "Blue
            # burned budget", which the zero-sum total cannot distinguish.
            for component in REWARD_COMPONENTS:
                team_metrics[component] = sign * info_sums[component].mean()
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
            # Unsigned game outcomes. Unlike the returns these are not
            # zero-sum, so they show absolute progress for one team without
            # the other team's decline confounding it.
            **{counter: info_sums[counter].mean() for counter in GAME_COUNTERS},
        }
        return train_states, env_state, obs, rng, reward_norm_states, metrics

    return env, init_obs, init_env_state, init_train_states, collect_and_update


__all__ = [
    "GAME_COUNTERS",
    "REWARD_COMPONENTS",
    "RewardNormState",
    "TeamTransition",
    "compute_gae",
    "initial_reward_norm_state",
    "make_joint_train",
    "next_sequence_reset",
]
