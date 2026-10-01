"""Run a bounded YAML traffic-learning diagnostic using the canonical PPO trainer."""

import argparse
import copy
import csv
import importlib.util
import json
import os
import pickle
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from jaxborg.actions import action_defs as action
from jaxborg.blue_learning_probe import (
    bootstrap,
    component_replay,
    first_minibatch,
    make_evaluator,
    make_fork,
    make_hook,
    probability_scores,
    signal_summary,
    traffic_groups,
)
from jaxborg.checkpoint import PolicyBundleEntry, load_jax_bundle, save_jax_bundle, write_sidecar
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.policies import policy_from_arch
from jaxborg.recipe import load, project_jax
from jaxborg.research_tracking import parameter_hash
from jaxborg.tracking import Run, assigned_devices, file_hash, serializable
from scripts.train.algorithms import ippo_jax_joint as trainer


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array, np.generic)):
        return np.asarray(value).tolist()
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), indent=2) + "\n")


def save_tree(path, tree):
    """Numeric leaves plus a version-pinned tree definition; no device buffers."""
    leaves, structure = jax.tree.flatten(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **{f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)})
    # This local trusted file only rebuilds the archived JAX/Flax tree structure.
    path.with_suffix(".tree.pkl").write_bytes(pickle.dumps(structure))
    write_json(
        path.with_suffix(".leaves.json"),
        [{"leaf": i, "shape": list(x.shape), "dtype": str(x.dtype)} for i, x in enumerate(leaves)],
    )


def load_reference(path):
    spec = importlib.util.spec_from_file_location("reference_joint_ppo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(config, data_root):
    rows = list(csv.DictReader((data_root / "eval/blue_diagnosis/checkpoints.csv").open()))
    step, seed = int(config["source_steps"]), int(config["training_seed"])
    wanted = {
        "warm": f"seed-{seed}-step-1920000",
        "final": f"seed-{seed}-step-final",
        "original": "original",
    }
    out = {}
    for key, label in wanted.items():
        row = next(r for r in rows if int(r["source_steps"]) == step and r["candidate"] == label)
        model = data_root / row["checkpoint"]
        if file_hash(model) != row["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {model}")
        out[key] = {"model": model, "recipe": data_root / row["recipe"], "sha256": row["sha256"]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-repository", required=True)
    args = parser.parse_args()
    devices = assigned_devices()
    if not devices or any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("This instrumented experiment requires an allocated GPU")
    config = yaml.safe_load(Path(args.config).read_text())
    if config["resources"]["partition"] != "community" or config["warm_updates"] > 20:
        raise ValueError("diagnostic resource/training bound exceeded")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh output directory; retain previous attempts")
    output.mkdir(parents=True, exist_ok=True)
    data = inputs(config, Path(args.data_root).resolve())
    recipe = load(str(data["warm"]["recipe"]))
    topology = (
        Path(args.data_root).resolve() / "recipes" / f"blue_source{config['source_steps']}" / "topology-seed0.npz"
    )
    recipe["train"]["topology_bank"] = [str(topology)]
    recipe["eval"]["topology_bank"] = [str(topology)]
    recipe["train"]["opponents"]["red"]["path"] = str(data["original"]["model"])
    configs = {team: project_jax(recipe, team=team) for team in ("blue", "red")}
    for cfg in configs.values():
        cfg["SEED"] = int(config["training_seed"])
        cfg["TOTAL_TIMESTEPS"] = config["warm_updates"] * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
    original = load_jax_bundle(data["original"]["model"])
    warm = load_jax_bundle(data["warm"]["model"])
    final = load_jax_bundle(data["final"]["model"])
    if parameter_hash(warm.policies["red"].weights) != parameter_hash(original.policies["red"].weights):
        raise ValueError("warm checkpoint opponent mismatch")
    entries = {"blue": warm.policies["blue"], "red": original.policies["red"]}
    networks = {team: policy_from_arch(entry.arch, action_dim=entry.action_dim) for team, entry in entries.items()}
    params = {team: entry.weights for team, entry in entries.items()}
    eval_env = make_joint_jax_env(configs["blue"]["EVAL_VARIANT"], topology_path=[topology], training_mode=False)
    evaluate = make_evaluator(eval_env, networks)
    reference_path = Path(args.reference_repository) / "scripts/train/algorithms/ippo_jax_joint.py"
    reference = load_reference(reference_path)
    with Run(
        {"meta": {"name": config["name"]}},
        backend="jax",
        kind="training",
        config=config,
        inputs=[{"role": k, "path": str(v["model"]), "sha256": v["sha256"]} for k, v in data.items()],
    ) as owner:
        write_json(
            output / "runtime.json",
            {
                "owner": owner.run_id,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "jax": jax.__version__,
                "jaxlib": __import__("jaxlib").__version__,
                "devices": str(jax.devices()),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "original_execution_revision": "0d577faab9ac3e0a06376e99e9592aadf128c130",
                "reference_trainer_sha256": file_hash(reference_path),
                "data_root": args.data_root,
                "inputs": native(
                    {k: {**v, "model": str(v["model"]), "recipe": str(v["recipe"])} for k, v in data.items()}
                ),
                "state_retention": "weights only; fresh Adam and reward normalizer; warm start, not exact continuation",
            },
        )
        (output / "config.yaml").write_text(yaml.safe_dump(config))
        (output / "resolved-recipe.yaml").write_text(yaml.safe_dump(recipe))
        write_json(output / "effective-config.json", serializable(configs))
        # Old cohort is only a backend/score reproduction check, not new evidence.
        archived = json.loads((Path(args.data_root) / "eval/blue_diagnosis/ippo-9600000/evaluations.json").read_text())
        expected = archived["confirmation"]["seed-11001-step-1920000"]
        reproduced = evaluate(params["blue"], params["red"], expected["per_episode_seeds"][:8])
        actual = [x["blue_return"] for x in reproduced]
        if actual != expected["per_episode_blue_returns"][:8]:
            write_json(
                output / "backend-mismatch.json",
                {"actual": actual, "expected": expected["per_episode_blue_returns"][:8]},
            )
            raise ValueError("archived GPU evaluation mismatch; diagnose before new experiments")
        write_json(output / "archived-reproduction.json", {"episodes": reproduced, "all_returns_exact": True})
        print("GPU and 8 archived episode returns verified exactly", flush=True)
        eval_seeds = list(
            range(
                config["seeds"]["validation_start"], config["seeds"]["validation_start"] + config["validation_episodes"]
            )
        )
        evaluations = {}
        for label, blue_params in (
            ("warm-0", params["blue"]),
            ("archived-final", final.policies["blue"].weights),
            ("original", original.policies["blue"].weights),
        ):
            for restricted in (False, True):
                key = label + ("-no-block" if restricted else "")
                evaluations[key] = evaluate(blue_params, params["red"], eval_seeds, no_block=restricted)
                print(key, np.mean([x["blue_return"] for x in evaluations[key]]), flush=True)
        write_json(output / "validation-episodes.json", evaluations)

        build_args = dict(trainable_teams=("blue",), initial_params=params)
        env, obs, env_state, init, collect = trainer.make_joint_train(
            copy.deepcopy(configs),
            networks,
            **build_args,
            diagnostic_hook=make_hook(config["probe_envs"]),
            capture_rollout=True,
        )
        _, _, _, _, calibrate = trainer.make_joint_train(
            copy.deepcopy(configs),
            networks,
            **build_args,
            perform_updates=False,
        )
        states = init(jax.random.PRNGKey(config["training_seed"] + 1))
        norms = {team: trainer.initial_reward_norm_state(configs[team]["NUM_ENVS"]) for team in ("blue", "red")}
        rng = jax.random.PRNGKey(config["seeds"]["warm_rollout"])
        # Re-estimate normalization from fixed warm policies before any update.
        for _ in range(config["normalization_calibration_updates"]):
            states, env_state, obs, rng, norms, _ = calibrate(states, env_state, obs, rng, norms)
        save_tree(output / "captures/calibrated-state", (states, env_state, obs, rng, norms))
        metrics_rows, gradient_rows, signals, forks = [], [], [], []
        baseline_hash = parameter_hash(states["red"].params)
        for update in range(1, config["warm_updates"] + 1):
            before = states
            before_norms = norms
            states, env_state, obs, rng, norms, metrics = collect(states, env_state, obs, rng, norms)
            diagnostic = metrics.pop("diagnostic")
            row = {
                "update": update,
                "warm_steps": update * configs["blue"]["NUM_ENVS"] * 500,
                **native(jax.device_get(metrics)),
            }
            metrics_rows.append(row)
            write_json(output / "training-metrics.json", metrics_rows)
            print("warm", update, row["blue"]["raw_rollout_return"], flush=True)
            if update in config["capture_updates"]:
                traj, observed = diagnostic["trajectories"]["blue"], diagnostic["observed"]
                key, last_value = diagnostic["update_keys"]["blue"], diagnostic["last_values"]["blue"]
                mini, adv, target, raw_adv, norm_adv, targets, indices = first_minibatch(
                    trainer, traj, last_value, key, configs["blue"]
                )
                save_tree(
                    output / f"captures/update-{update}-minibatch",
                    (before["blue"], mini, adv, target, indices, before_norms),
                )
                write_json(
                    output / f"captures/update-{update}-normalization.json",
                    native(jax.device_get(observed["normalizer"])),
                )
                # Exact updater equivalence, all 4 epochs x 16 minibatches.
                reference_state, _, _ = jax.jit(reference._make_team_updater(networks["blue"], configs["blue"]))(
                    before["blue"],
                    traj,
                    last_value,
                    key,
                )
                max_error = max(
                    float(jnp.max(jnp.abs(a - b)))
                    for a, b in zip(jax.tree.leaves(reference_state.params), jax.tree.leaves(states["blue"].params))
                )
                if max_error > 2e-6:
                    raise ValueError(f"instrumented PPO updater differs from execution revision: {max_error}")
                # Observe the same fixed phase-stratified sample for all component updates.
                probe_traj = jax.tree.map(lambda x: x[:, : config["probe_envs"]], traj)
                probe_observed = {
                    k: v[:, : config["probe_envs"]]
                    for k, v in observed.items()
                    if k in ("phase", "permitted", "blocked")
                }
                replay = partial(component_replay, trainer, networks["blue"], config=configs["blue"])
                component_rows, _, residual = replay(
                    before["blue"], mini, adv, target, probe_traj=probe_traj, observed=probe_observed
                )
                gradient_rows.append(
                    {
                        "update": update,
                        "reference_max_parameter_error": max_error,
                        "gradient_additivity_error": residual,
                        "components": native(jax.device_get(component_rows)),
                    }
                )
                write_json(output / "gradient-components.json", gradient_rows)
                signals.extend(
                    {"update": update, **native(jax.device_get(r))}
                    for r in signal_summary(traj, observed, raw_adv, norm_adv, targets)
                )
                write_json(output / "learning-signals.json", signals)
                full_scores = probability_scores(networks["blue"], states["blue"].params, probe_traj, probe_observed)
                write_json(
                    output / f"captures/update-{update}-post-full-probabilities.json",
                    native(jax.device_get(full_scores)),
                )
                if update == config["fork_capture_update"]:
                    forks = capture_forks(
                        config, output, eval_env, networks, before, traj, observed, raw_adv, norm_adv, targets
                    )
                    write_json(output / "fork-results.json", forks)
            if update % config["eval_every_updates"] == 0 or update == config["warm_updates"]:
                label = f"warm-{update}"
                evaluate_params = states["blue"].params
                evaluations[label] = evaluate(evaluate_params, states["red"].params, eval_seeds)
                evaluations[label + "-no-block"] = evaluate(
                    evaluate_params, states["red"].params, eval_seeds, no_block=True
                )
                write_json(output / "validation-episodes.json", evaluations)
                policies = {
                    team: PolicyBundleEntry(
                        weights=states[team].params,
                        team=team,
                        obs_dim=entries[team].obs_dim,
                        action_dim=entries[team].action_dim,
                        arch=entries[team].arch,
                        trainable=team == "blue",
                    )
                    for team in entries
                }
                path = output / f"checkpoints/warm-{update}.safetensors"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_jax_bundle(
                    path,
                    policies,
                    provenance={"warm_start": str(data["warm"]["model"]), "additional_steps": row["warm_steps"]},
                )
                write_sidecar(
                    path.with_suffix(".yaml"),
                    recipe,
                    seed=config["training_seed"],
                    total_steps=row["warm_steps"],
                    backend="jax",
                )
                save_tree(output / f"captures/warm-{update}-full-state", (states, env_state, obs, rng, norms))
        if parameter_hash(states["red"].params) != baseline_hash:
            raise ValueError("frozen Red parameters changed")
        # This comparison is selected in advance, independent of validation.
        confirmation_seeds = list(
            range(
                config["seeds"]["confirmation_start"],
                config["seeds"]["confirmation_start"] + config["confirmation_episodes"],
            )
        )
        confirmation = {}
        for label, blue_params in (("warm-0", params["blue"]), ("warm-final", states["blue"].params)):
            confirmation[label] = evaluate(blue_params, states["red"].params, confirmation_seeds)
            confirmation[label + "-no-block"] = evaluate(
                blue_params, states["red"].params, confirmation_seeds, no_block=True
            )
        write_json(output / "confirmation-episodes.json", confirmation)
        comparisons = {}
        for restricted in (False, True):
            suffix = "-no-block" if restricted else ""
            delta = [
                b["blue_return"] - a["blue_return"]
                for a, b in zip(confirmation["warm-0" + suffix], confirmation["warm-final" + suffix])
            ]
            comparisons["warm-final-minus-initial" + suffix] = bootstrap(delta)
        write_json(output / "confirmation-summary.json", comparisons)
        for path in sorted(output.rglob("*")):
            if path.is_file():
                owner.publish(path, "diagnostic/" + path.relative_to(output).as_posix())
        owner.export("manifest.json", output / "manifest.json")
        print("completed", comparisons, flush=True)


def capture_forks(config, output, env, networks, states, traj, observed, raw_adv, norm_adv, targets):
    groups = traffic_groups(traj, observed)
    rows = []
    for name in ("harmful_new_block", "useful_allow"):
        eligible = np.asarray(groups[name][:, : config["probe_envs"]])
        phases = np.asarray(observed["phase"][:, : config["probe_envs"]])
        eligible &= phases[..., None] == 2
        coords = np.argwhere(eligible)
        # Prespecified stratified sample spanning agents and ticks. No outcome selection.
        if len(coords):
            chosen = coords[
                np.linspace(0, len(coords) - 1, min(config["fork_states_per_group"], len(coords))).astype(int)
            ]
        else:
            chosen = []
        for tick, env_index, agent in chosen:
            state = jax.tree.map(lambda x: x[tick, env_index], observed["states"])
            actions = jax.tree.map(lambda x: x[tick, env_index], observed["actions"])
            intervention_key = observed["step_keys"][tick, env_index]
            natural_action = int(actions[f"blue_{agent}"])
            _, _, _, _, original_infos = env.step_env(intervention_key, state, actions)
            for component in ("reward_ria", "reward_lwf", "reward_asf", "action_cost"):
                actual = float(original_infos[component])
                expected = float(observed["infos"][component][tick, env_index])
                if actual != expected:
                    raise ValueError(
                        f"fork's natural intervention differs from captured transition: {component} {actual} {expected}"
                    )
            opposite = natural_action + (action.BLUE_ALLOW_TRAFFIC_START - action.BLUE_BLOCK_TRAFFIC_START) * (
                1 if name == "harmful_new_block" else -1
            )
            fork_index = len(rows)
            future_seeds = list(
                range(
                    config["seeds"]["fork_future_start"] + fork_index * 1000,
                    config["seeds"]["fork_future_start"] + fork_index * 1000 + config["fork_futures"],
                )
            )
            fn = jax.jit(
                jax.vmap(
                    partial(make_fork(env, networks), agent=int(agent), discount=0.99, horizon=500 - int(tick)),
                    in_axes=(None, None, None, 0, 0, None),
                )
            )
            # partial leaves positional: state, actions, key, forced_action, future_key, params.
            keys = jnp.stack([jax.random.PRNGKey(seed) for seed in future_seeds])
            alternatives = {"natural": natural_action, "opposite": opposite, "sleep": 0}
            results = {}
            for label, selected_action in alternatives.items():
                masks = env.get_avail_actions(state)
                if not bool(masks[f"blue_{agent}"][selected_action]):
                    raise ValueError("fork alternative must be legal")
                result = fn(
                    state,
                    actions,
                    intervention_key,
                    jnp.full(len(keys), selected_action),
                    keys,
                    {team: states[team].params for team in states},
                )
                results[label] = native(jax.device_get(result))
            row = {
                "group": name,
                "tick": int(tick),
                "env": int(env_index),
                "agent": int(agent),
                "natural_action": natural_action,
                "opposite_action": opposite,
                "value": float(traj.value[tick, env_index, agent]),
                "target": float(targets[tick, env_index, agent]),
                "raw_gae": float(raw_adv[tick, env_index, agent]),
                "ppo_advantage": float(norm_adv[tick, env_index, agent]),
                "future_seeds": future_seeds,
                "results": results,
                "paired_opposite_minus_natural": {},
            }
            for measure in ("discounted", "undiscounted"):
                diffs = np.asarray(results["opposite"][measure]) - np.asarray(results["natural"][measure])
                row["paired_opposite_minus_natural"][measure] = bootstrap(diffs.sum(-1))
            save_tree(output / f"captures/fork-{fork_index}", (state, actions, intervention_key))
            rows.append(row)
            print("fork", name, tick, row["paired_opposite_minus_natural"], flush=True)
    return rows


if __name__ == "__main__":
    main()
