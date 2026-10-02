"""Compare measured returns and GAE credit from the same saved legal-action forks."""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from jaxborg.actions import action_defs as action
from jaxborg.actions.encoding import decode_blue_action
from jaxborg.actions.red_policy import RED_POLICY_ACTION_DIM
from jaxborg.blue_learning_probe import COMPONENTS, bootstrap
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.policies import policy_from_arch, policy_step
from jaxborg.recipe import load, project_jax
from jaxborg.research_tracking import parameter_hash
from jaxborg.tracking import Run, assigned_devices, input_artifact
from scripts.experiments.blue_learning_mechanism import native, restore_tree, write_json


def learning_fork(env, networks, state, actions, intervention_key, agent, forced_action, key, params, norm_variance):
    names = {t: tuple(getattr(env, t + "_agents")) for t in ("blue", "red")}
    obs = env.get_obs(state)

    def blue_value(observations, masks):
        obs_batch = jnp.stack([observations[n] for n in names["blue"]])
        mask_batch = jnp.stack([masks[n] for n in names["blue"]])
        pi, value, _ = policy_step(networks["blue"], params["blue"], obs_batch, mask_batch)
        return pi, value[agent]

    pi, v0 = blue_value(obs, env.get_avail_actions(state))
    current_tick = state.state.time
    # -1 samples this agent's natural policy alternative independently while
    # preserving every other simultaneous action and the common future keys.
    initial_action_key = jax.random.fold_in(key, 700001)
    sampled = pi.sample(seed=initial_action_key)[agent]
    chosen = jnp.where(forced_action < 0, sampled, forced_action)
    legal = jnp.stack([env.get_avail_actions(state)[n] for n in names["blue"]])[agent, chosen]
    natural = jnp.stack([actions[n] for n in names["blue"]])[agent]
    _, _, _, source, destination = decode_blue_action(natural, agent, state.const)
    initially_forward = state.state.blocked_zones[destination, source]
    initially_reverse = state.state.blocked_zones[source, destination]
    modified = {n: jnp.where(agent == i, chosen, actions[n]) for i, n in enumerate(names["blue"])}
    modified.update({n: actions[n] for n in names["red"]})
    obs, state, _, done, info = env.step_env(intervention_key, state, modified)
    reward = jnp.stack([info[n] for n in COMPONENTS])
    normalized = jnp.clip(reward.sum() / (jnp.sqrt(norm_variance[current_tick]) + 1e-8), -10, 10)
    _, v1 = blue_value(obs, env.get_avail_actions(state))
    v1 = jnp.where(done["__all__"], 0, v1)
    td = normalized + 0.99 * v1 - v0
    open_now = ~(state.state.blocked_zones[destination, source] | state.state.blocked_zones[source, destination])
    carry = (
        key,
        obs,
        state,
        ~done["__all__"],
        reward,
        reward,
        normalized,
        td,
        open_now.astype(jnp.float32),
        td,
        normalized,
    )

    def step(carry, t):
        key, obs, state, alive, raw, discounted, normalized_mc, gae, open_ticks, lambda1, lambda095_reward = carry

        def advance(_):
            masks = env.get_avail_actions(state)
            key1, bkey, rkey, skey = jax.random.split(key, 4)
            bpi, value = blue_value(obs, masks)
            bactions = bpi.sample(seed=bkey)
            rpi, _, _ = policy_step(
                networks["red"],
                params["red"],
                jnp.stack([obs[n] for n in names["red"]]),
                jnp.stack([masks[n] for n in names["red"]]),
            )
            ractions = rpi.sample(seed=rkey)
            actions = {
                **{n: bactions[i] for i, n in enumerate(names["blue"])},
                **{n: ractions[i] for i, n in enumerate(names["red"])},
            }
            next_obs, next_state, _, dones, infos = env.step_env(skey, state, actions)
            r = jnp.stack([infos[n] for n in COMPONENTS])
            tick = jnp.minimum(current_tick + t, 499)
            nr = jnp.clip(r.sum() / (jnp.sqrt(norm_variance[tick]) + 1e-8), -10, 10)
            _, next_value = blue_value(next_obs, env.get_avail_actions(next_state))
            next_value = jnp.where(dones["__all__"], 0, next_value)
            delta = nr + 0.99 * next_value - value
            return (
                key1,
                next_obs,
                next_state,
                ~dones["__all__"],
                raw + r,
                discounted + 0.99**t * r,
                normalized_mc + 0.99**t * nr,
                gae + (0.99 * 0.95) ** t * delta,
                open_ticks
                + (
                    ~(
                        next_state.state.blocked_zones[destination, source]
                        | next_state.state.blocked_zones[source, destination]
                    )
                ).astype(jnp.float32),
                lambda1 + 0.99**t * delta,
                lambda095_reward + (0.99 * 0.95) ** t * nr,
            )

        return jax.lax.cond(alive, advance, lambda _: carry, None), None

    final, _ = jax.lax.scan(step, carry, jnp.arange(1, 500))
    return {
        "undiscounted": final[4],
        "discounted": final[5],
        "normalized_mc_advantage": final[6] - v0,
        "gae_advantage": final[7],
        "gae_lambda1_advantage": final[9],
        "initial_value": v0,
        "normalized_lambda095_reward_return": final[10],
        "critic_bootstrap_contribution": final[7] - final[10] + v0,
        "first_normalized_reward": normalized,
        "first_next_value": v1,
        "normalized_mc_return_after_first_action": (final[6] - normalized) / 0.99,
        "first_action": chosen,
        "route_open_ticks": final[8],
        "initial_forward_blocked": initially_forward,
        "initial_reverse_blocked": initially_reverse,
        "first_action_legal": legal,
    }


def reproduce_saved_forks(fn, params, variance, root, captures):
    # Match the existing conditional simulations before using new cohorts.
    check_inputs = []
    check_expected = []
    for i, row in enumerate(captures[:2]):
        state, actions, ikey = restore_tree(root / f"captures/fork-{i}")
        for j, seed in enumerate(row["future_seeds"]):
            check_inputs.append(
                (
                    state,
                    actions,
                    ikey,
                    jnp.int32(row["agent"]),
                    jnp.int32(row["natural_action"]),
                    jax.random.PRNGKey(seed),
                )
            )
            check_expected.append(np.asarray(row["results"]["natural"]["undiscounted"][j]))
    arrays = jax.tree.map(lambda *xs: jnp.stack(xs), *check_inputs)
    check = jax.device_get(fn(*arrays, params, variance))
    error = float(np.max(np.abs(check["undiscounted"] - np.stack(check_expected))))
    if error != 0:
        raise ValueError(f"learning fork changed saved raw outcomes: {error}")
    return [{"saved_conditional_episodes": len(check_expected), "raw_components_max_error": error}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--data-root", type=Path, help="Remap archived topology/opponent paths to a transferred input bundle"
    )
    a = p.parse_args()
    assigned_devices()
    if any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    config = yaml.safe_load(a.config.read_text())
    root = a.input_run.resolve()
    out = a.output.resolve()
    if json.loads((root / "manifest.json").read_text())["status"] != "FINISHED":
        raise ValueError("input run must finish publishing before a credit replay")
    if out.exists() and any(out.iterdir()):
        raise ValueError("retain prior outputs; choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    captures = json.loads((root / "fork-results.json").read_text())
    if not 1 <= len(captures) <= config["max_states"] <= 12 or config["futures"] != 32 or config["batch_size"] != 64:
        raise ValueError("exceeds the bounded captured-state protocol")
    if (
        config["gamma"] != 0.99
        or config["gae_lambda"] != 0.95
        or config["normalization"] != "frozen_original_rollout_variance"
    ):
        raise ValueError("unsupported credit protocol")
    state_data, _, _, _, _ = restore_tree(root / "captures/update-5-before-full-state")
    params = {t: state_data[t]["params"] for t in ("blue", "red")}
    recipe = load(str(root / "resolved-recipe.yaml"))
    if a.data_root:
        from scripts.experiments.blue_learning_mechanism import inputs

        protocol = yaml.safe_load((root / "config.yaml").read_text())
        data = inputs(protocol, a.data_root.resolve())
        topology = a.data_root.resolve() / "recipes" / f"blue_source{protocol['source_steps']}" / "topology-seed0.npz"
        recipe["train"]["topology_bank"] = recipe["eval"]["topology_bank"] = [str(topology)]
        recipe["train"]["opponents"]["red"]["path"] = str(data["original"]["model"])
    cfg = project_jax(recipe, team="blue")
    if cfg["GAMMA"] != 0.99 or cfg["GAE_LAMBDA"] != 0.95 or not cfg["NORM_REWARDS"]:
        raise ValueError("unsupported credit protocol")
    env = make_joint_jax_env(cfg["EVAL_VARIANT"], topology_path=cfg["TOPOLOGY_BANK"], training_mode=False)
    networks = {
        t: policy_from_arch(
            recipe["arch"], action_dim=action.BLUE_ALLOW_TRAFFIC_END if t == "blue" else RED_POLICY_ACTION_DIM
        )
        for t in ("blue", "red")
    }
    variance = np.asarray(json.loads((root / "captures/update-5-normalization.json").read_text())[2])
    if variance.shape != (500,) or not np.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError("invalid captured normalization profile")
    variance = jnp.asarray(variance)
    fn = jax.jit(jax.vmap(partial(learning_fork, env, networks), in_axes=(0, 0, 0, 0, 0, 0, None, None)))
    outcomes = []
    source_owner = json.loads((root / "runtime.json").read_text())["owner"]
    required = [
        root / "captures/update-5-before-full-state.npz",
        root / "captures/update-5-before-full-state.tree.pkl",
        root / "captures/update-5-normalization.json",
        root / "fork-results.json",
        root / "resolved-recipe.yaml",
    ]
    required.extend(p for p in (root / "captures").glob("fork-*") if p.suffix in (".npz", ".pkl"))
    recorded_inputs = [
        input_artifact(f"runs:/{source_owner}/diagnostic/{p.relative_to(root).as_posix()}", role=p.name)
        for p in required
    ]
    with Run(
        {"meta": {"name": "blue-learning-signal-forks"}},
        backend="jax",
        kind="comparison",
        config=config,
        inputs=recorded_inputs,
    ) as owner:
        (out / "config.yaml").write_text(yaml.safe_dump(config))
        sanity = reproduce_saved_forks(fn, params, variance, root, captures)
        write_json(
            out / "runtime.json",
            {
                "run_id": owner.run_id,
                "jax": jax.__version__,
                "devices": str(jax.devices()),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "sanity": sanity,
                "blue_parameter_hash": parameter_hash(params["blue"]),
                "frozen_red_parameter_hash": parameter_hash(params["red"]),
                "normalization": (
                    "frozen original 96-env rollout variance profile; not a counterfactual normalizer replay"
                ),
                "data_root_override": str(a.data_root.resolve()) if a.data_root else None,
            },
        )
        for cohort, start in (("discovery", config["discovery_start"]), ("confirmation", config["confirmation_start"])):
            inputs = []
            labels = []
            for i, row in enumerate(captures):
                state, actions, ikey = restore_tree(root / f"captures/fork-{i}")
                for condition, forced in (
                    ("natural", row["natural_action"]),
                    ("opposite", row["opposite_action"]),
                    ("sleep", 0),
                    ("policy", -1),
                ):
                    for k in range(config["futures"]):
                        seed = start + i * 1000 + k
                        inputs.append(
                            (state, actions, ikey, jnp.int32(row["agent"]), jnp.int32(forced), jax.random.PRNGKey(seed))
                        )
                        labels.append((i, condition, seed))
            results = []
            for begin in range(0, len(inputs), config["batch_size"]):
                chunk = inputs[begin : begin + config["batch_size"]]
                n = len(chunk)
                chunk += [chunk[-1]] * (config["batch_size"] - n)
                arrays = jax.tree.map(lambda *xs: jnp.stack(xs), *chunk)
                measured = jax.device_get(fn(*arrays, params, variance))
                if not measured["first_action_legal"].all():
                    raise ValueError("counterfactual action was outside legal policy support")
                results.extend({key: value[j] for key, value in measured.items()} for j in range(n))
            for i, row in enumerate(captures):
                by_condition = {}
                for condition in ("natural", "opposite", "sleep", "policy"):
                    items = [r for r, label in zip(results, labels) if label[0] == i and label[1] == condition]
                    by_condition[condition] = {key: np.stack([r[key] for r in items]) for key in items[0]}
                summaries = {}
                for alternative in ("opposite", "policy", "sleep"):
                    measures = {}
                    for metric in (
                        "undiscounted",
                        "discounted",
                        "normalized_mc_advantage",
                        "gae_advantage",
                        "gae_lambda1_advantage",
                        "route_open_ticks",
                        "normalized_lambda095_reward_return",
                        "critic_bootstrap_contribution",
                        "first_next_value",
                        "normalized_mc_return_after_first_action",
                    ):
                        delta = by_condition[alternative][metric] - by_condition["natural"][metric]
                        if delta.ndim > 1:
                            delta = delta.sum(-1)
                        measures[metric] = bootstrap(delta)
                    summaries[alternative + "-minus-natural"] = measures
                outcomes.append(
                    {
                        "cohort": cohort,
                        "fork": i,
                        "group": row["group"],
                        "tick": row["tick"],
                        "env": row["env"],
                        "agent": row["agent"],
                        "original_sampled_gae": row["raw_gae"],
                        "original_ppo_advantage": row["ppo_advantage"],
                        "seeds": list(range(start + i * 1000, start + i * 1000 + config["futures"])),
                        "summaries": summaries,
                        "results": native(by_condition),
                    }
                )
                print(cohort, i, row["group"], native(summaries["opposite-minus-natural"]), flush=True)
            write_json(out / "results.json", outcomes)
            max_lambda1_error = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(r["results"][c]["normalized_mc_advantage"])
                            - np.asarray(r["results"][c]["gae_lambda1_advantage"])
                        )
                    )
                )
                for r in outcomes
                for c in r["results"]
            )
            write_json(
                out / "credit-identity-check.json", {"lambda1_matches_normalized_mc_max_error": max_lambda1_error}
            )
            if max_lambda1_error > 1e-4:
                raise ValueError("lambda=1 GAE telescoping check failed")
        for path in sorted(out.rglob("*")):
            if path.is_file():
                owner.publish(path, "signal-forks/" + path.relative_to(out).as_posix())
    write_json(out / "manifest.json", owner.manifest)


if __name__ == "__main__":
    main()
