"""Read-only traffic, PPO gradient, and common-random-number fork probes."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxborg.actions import action_defs as action
from jaxborg.checkpoint_behavior import COMPONENTS, _episode
from jaxborg.policies import policy_step


def traffic_routes(const):
    """The canonical compressed source / observed destination action encoding."""
    offsets = jnp.arange(action.BLUE_TRAFFIC_SLOTS)
    destinations = const.blue_obs_subnets[:, offsets % action.BLUE_MAX_OBSERVED_SUBNETS]
    source_offsets = offsets // action.BLUE_MAX_OBSERVED_SUBNETS
    sources = source_offsets[None] + (source_offsets[None] >= destinations)
    return sources, destinations


def route_status(state):
    src, dst = traffic_routes(state.const)
    phase = state.state.mission_phase
    permitted = state.const.allowed_subnet_pairs[phase, src, dst] | state.const.allowed_subnet_pairs[phase, dst, src]
    blocked = state.state.blocked_zones[dst, src]
    reverse = state.state.blocked_zones[src, dst]
    return permitted, blocked, reverse


def make_hook(probe_envs):
    def observe(env, before, after, actions, parts, distributions, keys, infos, norms):
        del env
        permitted, blocked, reverse = jax.vmap(route_status)(before)
        _, next_blocked, _ = jax.vmap(route_status)(after)
        pi = distributions["blue"]
        probs = jax.nn.softmax(pi.logits).reshape((*parts["blue"][2].shape, -1))
        return {
            "phase": before.state.mission_phase,
            "permitted": permitted,
            "blocked": blocked,
            "reverse_blocked": reverse,
            "next_blocked": next_blocked,
            "block_probability": probs[..., action.BLUE_BLOCK_TRAFFIC_START : action.BLUE_BLOCK_TRAFFIC_END],
            "allow_probability": probs[..., action.BLUE_ALLOW_TRAFFIC_START : action.BLUE_ALLOW_TRAFFIC_END],
            "infos": {name: infos[name] for name in COMPONENTS},
            "normalizer": norms["blue"],
            "states": jax.tree.map(lambda x: x[:probe_envs], before),
            "actions": {name: value[:probe_envs] for name, value in actions.items()},
            "step_keys": jax.vmap(lambda k: jax.random.split(k)[0])(keys[:probe_envs]),
        }

    return observe


def traffic_groups(traj, observed):
    chosen = traj.action
    is_block = (chosen >= action.BLUE_BLOCK_TRAFFIC_START) & (chosen < action.BLUE_BLOCK_TRAFFIC_END)
    is_allow = (chosen >= action.BLUE_ALLOW_TRAFFIC_START) & (chosen < action.BLUE_ALLOW_TRAFFIC_END)
    offset = jnp.where(is_block, chosen - action.BLUE_BLOCK_TRAFFIC_START, chosen - action.BLUE_ALLOW_TRAFFIC_START)
    offset = jnp.clip(offset, 0, action.BLUE_TRAFFIC_SLOTS - 1)

    def take(x):
        return jnp.take_along_axis(x, offset[..., None], axis=-1)[..., 0]

    idle = traj.actor_mask > 0
    permitted, blocked = take(observed["permitted"]), take(observed["blocked"])
    return {
        "harmful_new_block": idle & is_block & permitted & ~blocked,
        "redundant_block": idle & is_block & blocked,
        "useful_allow": idle & is_allow & blocked,
        "redundant_allow": idle & is_allow & ~blocked,
        "other": idle & ~is_block & ~is_allow,
    }


def signal_summary(traj, observed, raw_advantages, normalized_advantages, targets):
    groups = traffic_groups(traj, observed)
    rows = []
    for phase in range(3):
        for name, group in groups.items():
            mask = group & (observed["phase"][..., None] == phase)
            count = mask.sum()

            def avg(x):
                return jnp.sum(jnp.where(mask, x, 0)) / jnp.maximum(count, 1)

            rows.append(
                {
                    "phase": phase,
                    "group": name,
                    "count": count,
                    "raw_advantage": avg(raw_advantages),
                    "normalized_advantage": avg(normalized_advantages),
                    "positive_advantage_fraction": avg((normalized_advantages > 0).astype(jnp.float32)),
                    "value": avg(traj.value),
                    "target": avg(targets),
                }
            )
    return rows


@partial(jax.jit, static_argnums=(0,))
def probability_scores(network, params, traj, observed):
    obs = traj.obs.reshape((-1, traj.obs.shape[-1]))
    masks = traj.avail_actions.reshape((-1, traj.avail_actions.shape[-1]))
    pi, _, _ = policy_step(network, params, obs, masks)
    probs = jax.nn.softmax(pi.logits).reshape((*traj.action.shape, -1))
    idle = traj.actor_mask
    harmful = observed["permitted"] & ~observed["blocked"]
    useful = observed["blocked"]
    out = {}
    for phase in range(3):
        weight = idle * (observed["phase"][..., None] == phase)
        for name, start, end, route_mask in (
            ("block", action.BLUE_BLOCK_TRAFFIC_START, action.BLUE_BLOCK_TRAFFIC_END, jnp.ones_like(harmful)),
            ("harmful_new_block", action.BLUE_BLOCK_TRAFFIC_START, action.BLUE_BLOCK_TRAFFIC_END, harmful),
            ("useful_allow", action.BLUE_ALLOW_TRAFFIC_START, action.BLUE_ALLOW_TRAFFIC_END, useful),
        ):
            mass = (probs[..., start:end] * route_mask).sum(-1)
            out[f"phase{phase}/{name}"] = (mass * weight).sum() / jnp.maximum(weight.sum(), 1)
    return out


def first_minibatch(trainer, traj, last_value, key, config):
    advantages, targets = trainer.compute_gae(traj, last_value, gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])
    normalized = trainer._masked_normalize(advantages, traj.actor_mask)
    _, permutation_key = jax.random.split(key)
    size = traj.action.size
    indices = jax.random.permutation(permutation_key, size)[: size // config["NUM_MINIBATCHES"]]
    flat = jax.tree.map(lambda x: x.reshape((size,) + x.shape[3:]), traj)
    mini = jax.tree.map(lambda x: x[indices], flat)
    return mini, normalized.reshape(-1)[indices], targets.reshape(-1)[indices], advantages, normalized, targets, indices


def component_replay(trainer, network, state, minibatch, advantages, targets, config, probe_traj, observed):
    """One identical minibatch, including a zero-gradient Adam momentum control."""
    before = probability_scores(network, state.params, probe_traj, observed)
    gradients, rows, states = {}, [], {}

    def score_fn(p):
        return probability_scores(network, p, probe_traj, observed)["phase2/harmful_new_block"]

    score_grad = jax.jit(jax.grad(score_fn))(state.params)
    for component in ("zero", "actor", "critic", "entropy", "full"):

        def loss_fn(params):
            pi, value, _ = policy_step(network, params, minibatch.obs, minibatch.avail_actions)
            return trainer.ppo_objective(pi, value, minibatch, advantages, targets, config, loss_component=component)

        (_, metrics), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(state.params)
        gradients[component] = grads
        norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, config["MAX_GRAD_NORM"] / (norm + 1e-8))
        changed = state.apply_gradients(grads=jax.tree.map(lambda x: x * scale, grads))
        after = probability_scores(network, changed.params, probe_traj, observed)
        dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(score_grad), jax.tree.leaves(grads)))
        states[component] = changed
        rows.append(
            {
                "component": component,
                "gradient_norm": norm,
                "sgd_harmful_direction": -dot,
                "losses": metrics,
                "before": before,
                "after": after,
                "change": {k: after[k] - before[k] for k in before},
            }
        )
    residual = jax.tree.map(
        lambda f, a, v, e: f - a - v - e,
        gradients["full"],
        gradients["actor"],
        gradients["critic"],
        gradients["entropy"],
    )
    return rows, states, float(optax.global_norm(residual))


@dataclass(frozen=True)
class BlockExcluded:
    base: object

    def apply(self, params, obs, avail_actions=None):
        pi, value = self.base.apply(params, obs, avail_actions)
        pi = pi.replace(
            logits=pi.logits.at[..., action.BLUE_BLOCK_TRAFFIC_START : action.BLUE_BLOCK_TRAFFIC_END].add(-1e10)
        )
        return pi, value


def make_evaluator(env, networks, *, batch_size=64):
    def evaluate(blue_params, red_params, seeds, *, no_block=False):
        module = BlockExcluded(networks["blue"]) if no_block else networks["blue"]
        fn = jax.jit(
            jax.vmap(
                partial(
                    _episode,
                    env=env,
                    blue_module=module,
                    red_module=networks["red"],
                    steps=500,
                ),
                in_axes=(None, None, 0),
            )
        )
        result = []
        for start in range(0, len(seeds), batch_size):
            chunk = list(seeds[start : start + batch_size])
            n = len(chunk)
            chunk.extend([chunk[-1]] * (batch_size - n))
            keys = jnp.stack([jax.random.PRNGKey(seed) for seed in chunk])
            out = jax.device_get(fn(blue_params, red_params, keys))
            result.extend({"seed": chunk[i], **{k: float(v[i]) for k, v in out.items()}} for i in range(n))
        return result

    return evaluate


def make_fork(env, networks):
    """Hold other intervention actions fixed; use common future policy/env keys."""
    names = {team: tuple(getattr(env, team + "_agents")) for team in ("blue", "red")}

    def fork(state, actions, intervention_key, forced_action, future_key, params, *, agent, discount, horizon):
        modified = dict(actions)
        # Agent is static for a compiled state/fork batch.
        modified[names["blue"][agent]] = forced_action
        obs, state, _, dones, infos = env.step_env(intervention_key, state, modified)
        totals = jnp.stack([infos[k] for k in COMPONENTS])
        discounted = totals

        def step(carry, tick):
            key, obs, state, alive, total, disc = carry

            def advance(_):
                masks = env.get_avail_actions(state)
                next_key, blue_key, red_key, step_key = jax.random.split(key, 4)
                next_actions = {}
                for team, team_key in (("blue", blue_key), ("red", red_key)):
                    batch_obs = jnp.stack([obs[n] for n in names[team]])
                    batch_mask = jnp.stack([masks[n] for n in names[team]])
                    pi, _, _ = policy_step(networks[team], params[team], batch_obs, batch_mask)
                    choices = pi.sample(seed=team_key)
                    next_actions.update({n: choices[i] for i, n in enumerate(names[team])})
                next_obs, next_state, _, done, info = env.step_env(step_key, state, next_actions)
                rewards = jnp.stack([info[k] for k in COMPONENTS])
                return (
                    next_key,
                    next_obs,
                    next_state,
                    ~done["__all__"],
                    total + rewards,
                    disc + discount**tick * rewards,
                )

            return jax.lax.cond(alive, advance, lambda _: carry, None), None

        out, _ = jax.lax.scan(
            step,
            (future_key, obs, state, ~dones["__all__"], totals, discounted),
            jnp.arange(1, horizon),
        )
        return {"undiscounted": out[-2], "discounted": out[-1]}

    return fork


def bootstrap(differences, *, seed=6500001):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "n": len(values)}
