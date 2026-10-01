"""Analyze saved traffic, action-stratum gradients, and observation aliasing without training."""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxborg.actions import action_defs as action
from jaxborg.actions.encoding import decode_blue_action
from jaxborg.blue_learning_probe import probability_scores, traffic_routes
from jaxborg.constants import NUM_BLUE_AGENTS, NUM_SUBNETS
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.observations import JAX_ID_TO_CYBORG_POS, SUBNET_BLOCK_SIZE
from jaxborg.policies import policy_from_arch, policy_step
from jaxborg.recipe import load, project_jax
from jaxborg.tracking import assigned_devices
from scripts.experiments.blue_learning_mechanism import restore_tree
from scripts.train.algorithms import ippo_jax_joint as trainer


def route_probes(root):
    calibrated = restore_tree(root / "captures/calibrated-state")
    const = jax.tree.map(lambda x: x[0], calibrated[1].const)
    src, dst = map(np.asarray, traffic_routes(const))
    inverse = np.asarray(JAX_ID_TO_CYBORG_POS)
    policy = policy_from_arch(
        {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"},
        action_dim=action.BLUE_ALLOW_TRAFFIC_END,
    )
    rows = []
    for path in sorted((root / "captures").glob("update-*-minibatch.npz")):
        update = int(path.stem.split("-")[1])
        state, traj, advantages, targets, indices, norms = restore_tree(path.with_suffix(""))
        pi, _, _ = jax.jit(lambda p, o, m: policy_step(policy, p, o, m))(state["params"], traj.obs, traj.avail_actions)
        probabilities = np.asarray(jax.nn.softmax(pi.logits))
        obs, choice, idle = np.asarray(traj.obs), np.asarray(traj.action), np.asarray(traj.actor_mask) > 0
        phase = obs[:, 0].astype(int)
        agent = np.asarray(indices) % NUM_BLUE_AGENTS
        positions = np.arange(action.BLUE_TRAFFIC_SLOTS) % action.BLUE_MAX_OBSERVED_SUBNETS
        for b in range(NUM_BLUE_AGENTS):
            for p in range(3):
                group = idle & (agent == b) & (phase == p)
                n = int(group.sum())
                if not n:
                    continue
                for route in range(action.BLUE_TRAFFIC_SLOTS):
                    source, dest = int(src[b, route]), int(dst[b, route])
                    if dest < 0:
                        continue
                    legal = np.asarray(traj.avail_actions)[group, action.BLUE_BLOCK_TRAFFIC_START + route] > 0
                    if not legal.any():
                        continue
                    blocked = obs[group, 1 + positions[route] * SUBNET_BLOCK_SIZE + NUM_SUBNETS + inverse[source]] > 0
                    permitted = bool(
                        const.allowed_subnet_pairs[p, source, dest] | const.allowed_subnet_pairs[p, dest, source]
                    )
                    block = choice[group] == action.BLUE_BLOCK_TRAFFIC_START + route
                    allow = choice[group] == action.BLUE_ALLOW_TRAFFIC_START + route
                    rows.append(
                        {
                            "update": update,
                            "phase": p,
                            "agent": b,
                            "source_subnet": source,
                            "destination_subnet": dest,
                            "idle_sample_rows": n,
                            "block_probability": float(
                                probabilities[group, action.BLUE_BLOCK_TRAFFIC_START + route].mean()
                            ),
                            "allow_probability": float(
                                probabilities[group, action.BLUE_ALLOW_TRAFFIC_START + route].mean()
                            ),
                            "mission_permitted_either_direction": permitted,
                            "blocked_fraction": float(blocked.mean()),
                            "sampled_blocks": int(block.sum()),
                            "new_blocks": int((block & ~blocked).sum()),
                            "sampled_allows": int(allow.sum()),
                            "bit_clearing_allows": int((allow & blocked).sum()),
                            "redundant_allows": int((allow & ~blocked).sum()),
                        }
                    )
    target = root / "analysis/route-phase-probes.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} route/phase summaries from captured first minibatches")


def actor_strata(root):
    calibrated = restore_tree(root / "captures/calibrated-state")
    const = jax.tree.map(lambda x: x[0], calibrated[1].const)
    sources, destinations = traffic_routes(const)
    config = json.loads((root / "effective-config.json").read_text())["blue"]
    network = policy_from_arch(
        {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"},
        action_dim=action.BLUE_ALLOW_TRAFFIC_END,
    )
    all_rows = []
    checks = []
    for path in sorted((root / "captures").glob("update-*-minibatch.npz")):
        update = int(path.stem.split("-")[1])
        state, mini, advantages, targets, indices, _ = restore_tree(path.with_suffix(""))
        n = mini.action.size
        agent = indices % NUM_BLUE_AGENTS
        src, dst = sources[agent], destinations[agent]
        phase = mini.obs[:, 0].astype(jnp.int32)
        offsets = jnp.arange(action.BLUE_TRAFFIC_SLOTS)
        positions = offsets % action.BLUE_MAX_OBSERVED_SUBNETS
        features = 1 + positions[None] * SUBNET_BLOCK_SIZE + NUM_SUBNETS + JAX_ID_TO_CYBORG_POS[src]
        blocked = jnp.take_along_axis(mini.obs, features, axis=-1) > 0
        permitted = (
            const.allowed_subnet_pairs[phase[:, None], src, dst] | const.allowed_subnet_pairs[phase[:, None], dst, src]
        )
        probe = jax.tree.map(lambda x: x.reshape((n, 1, 1) + x.shape[1:]), mini)
        observed = {
            "phase": phase[:, None],
            "permitted": permitted[:, None, None, :],
            "blocked": blocked[:, None, None, :],
        }
        scoregrad = jax.jit(
            jax.grad(lambda p: probability_scores(network, p, probe, observed)["phase2/harmful_new_block"])
        )(state["params"])

        def loss(params, adv):
            pi, value, _ = policy_step(network, params, mini.obs, mini.avail_actions)
            return trainer.ppo_objective(pi, value, mini, adv, targets, config, loss_component="actor")[0]

        gradient = jax.jit(jax.grad(loss))
        full = gradient(state["params"], advantages)
        total = jax.tree.map(jnp.zeros_like, full)
        chosen = mini.action
        idle = mini.actor_mask > 0
        block = (chosen >= action.BLUE_BLOCK_TRAFFIC_START) & (chosen < action.BLUE_BLOCK_TRAFFIC_END)
        allow = (chosen >= action.BLUE_ALLOW_TRAFFIC_START) & (chosen < action.BLUE_ALLOW_TRAFFIC_END)
        route = jnp.clip(
            jnp.where(block, chosen - action.BLUE_BLOCK_TRAFFIC_START, chosen - action.BLUE_ALLOW_TRAFFIC_START),
            0,
            action.BLUE_TRAFFIC_SLOTS - 1,
        )

        def take(v):
            return jnp.take_along_axis(v, route[:, None], axis=-1)[:, 0]

        owned, mission = take(blocked), take(permitted)
        groups = {
            "new_mission_block": block & mission & ~owned,
            "redundant_block": block & owned,
            "other_block": block & ~(mission & ~owned) & ~owned,
            "bit_clearing_allow": allow & owned,
            "redundant_allow": allow & ~owned,
            "restore": (chosen >= action.BLUE_RESTORE_START) & (chosen < action.BLUE_RESTORE_END),
            "sleep": chosen == 0,
            "monitor": chosen == 1,
            "analyse": (chosen >= action.BLUE_ANALYSE_START) & (chosen < action.BLUE_ANALYSE_END),
            "remove": (chosen >= action.BLUE_REMOVE_START) & (chosen < action.BLUE_REMOVE_END),
            "decoy": (chosen >= action.BLUE_DECOY_START) & (chosen < action.BLUE_DECOY_END),
        }
        for p in range(3):
            for group, mask in groups.items():
                mask = mask & idle & (phase == p)
                count = int(mask.sum())
                grad = gradient(state["params"], jnp.where(mask, advantages, 0))
                total = jax.tree.map(lambda a, b: a + b, total, grad)
                dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(scoregrad), jax.tree.leaves(grad)))
                norm = float(optax.global_norm(grad))
                all_rows.append(
                    {
                        "update": update,
                        "phase": p,
                        "group": group,
                        "count": count,
                        "mean_advantage": float(jnp.sum(jnp.where(mask, advantages, 0)) / max(count, 1)),
                        "positive_fraction": float(jnp.sum(mask & (advantages > 0)) / max(count, 1)),
                        "gradient_norm": norm,
                        "sgd_harmful_direction": float(-dot),
                    }
                )
        error = float(optax.global_norm(jax.tree.map(lambda a, b: a - b, total, full)))
        if error > 1e-5:
            raise ValueError("action strata failed to partition the full actor gradient")
        checks.append(
            {
                "update": update,
                "actor_group_gradient_sum_error": error,
                "full_actor_sgd_harmful_direction": float(
                    -sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(scoregrad), jax.tree.leaves(full)))
                ),
            }
        )
        print(
            "update",
            update,
            "full direction",
            checks[-1]["full_actor_sgd_harmful_direction"],
            "additivity",
            error,
            flush=True,
        )
        for r in sorted(
            [r for r in all_rows if r["update"] == update], key=lambda r: abs(r["sgd_harmful_direction"]), reverse=True
        )[:8]:
            print(r, flush=True)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "actor-gradient-strata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (out / "actor-gradient-strata-checks.json").write_text(json.dumps(checks, indent=2) + "\n")


def observation_audit(root):
    forks = sorted((root / "captures").glob("fork-*.npz"))
    if not forks:
        return
    recipe = load(str(root / "resolved-recipe.yaml"))
    cfg = project_jax(recipe, team="blue")
    env = make_joint_jax_env(cfg["EVAL_VARIANT"], topology_path=cfg["TOPOLOGY_BANK"], training_mode=False)
    states, *_ = restore_tree(root / "captures/update-5-before-full-state")
    network = policy_from_arch(
        {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"}, action_dim=242
    )
    rows = []
    for p in forks:
        state, actions, _ = restore_tree(p.with_suffix(""))
        for b in range(5):
            typ, _, _, src, dst = decode_blue_action(actions[f"blue_{b}"], b, state.const)
            if int(typ) not in (6, 7):
                continue
            original = env.get_obs(state)[f"blue_{b}"]
            changed = state.replace(
                state=state.state.replace(
                    blocked_zones=state.state.blocked_zones.at[src, dst].set(~state.state.blocked_zones[src, dst])
                )
            )
            reverse_obs = env.get_obs(changed)[f"blue_{b}"]
            busy = state.replace(
                state=state.state.replace(blue_pending_ticks=state.state.blue_pending_ticks.at[b].set(3))
            )
            busy_obs = env.get_obs(busy)[f"blue_{b}"]
            later = state.replace(state=state.state.replace(time=state.state.time + 1))
            later_obs = env.get_obs(later)[f"blue_{b}"]
            masks = env.get_avail_actions(state)[f"blue_{b}"]
            busy_mask = env.get_avail_actions(busy)[f"blue_{b}"]
            _, v, _ = policy_step(network, states["blue"]["params"], original[None], masks[None])
            _, busy_v, _ = policy_step(network, states["blue"]["params"], busy_obs[None], busy_mask[None])
            _, reverse_v, _ = policy_step(network, states["blue"]["params"], reverse_obs[None], masks[None])
            _, later_v, _ = policy_step(network, states["blue"]["params"], later_obs[None], masks[None])
            rows.append(
                {
                    "capture": p.stem,
                    "tick": int(state.state.time),
                    "agent": b,
                    "type": int(typ),
                    "source": int(src),
                    "destination": int(dst),
                    "forward_blocked": bool(state.state.blocked_zones[dst, src]),
                    "reverse_blocked": bool(state.state.blocked_zones[src, dst]),
                    "reverse_bit_visible": bool(np.any(np.asarray(original) != np.asarray(reverse_obs))),
                    "reverse_value_difference": float(reverse_v[0] - v[0]),
                    "pending_ticks_visible": bool(np.any(np.asarray(original) != np.asarray(busy_obs))),
                    "pending_ticks_value_difference": float(busy_v[0] - v[0]),
                    "busy_mask_changed": bool(np.any(np.asarray(masks) != np.asarray(busy_mask))),
                    "within_phase_clock_visible": bool(np.any(np.asarray(original) != np.asarray(later_obs))),
                    "within_phase_clock_value_difference": float(later_v[0] - v[0]),
                }
            )
    target = root / "analysis"
    target.mkdir(exist_ok=True)
    (target / "observation-alias-audit.json").write_text(
        json.dumps(
            {
                "interventions": "read-only observation audit, not an outcome comparison",
                "rows": rows,
                "conclusion": (
                    "Value inputs omit pending ticks; traffic reverse bits can be outside this agent observation. "
                    "This establishes aliasing, not a causal learning defect."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--probe", choices=("routes", "actor", "observations", "all"), default="all")
    a = p.parse_args()
    assigned_devices()
    for name, function in (("routes", route_probes), ("actor", actor_strata), ("observations", observation_audit)):
        if a.probe in (name, "all"):
            function(a.directory)


if __name__ == "__main__":
    main()
