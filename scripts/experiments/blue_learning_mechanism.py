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
import mlflow
import numpy as np
import yaml
from flax.training.train_state import TrainState

from jaxborg.actions import action_defs as action
from jaxborg.blue_learning_probe import (
    bootstrap,
    component_replay,
    first_minibatch,
    fork_coordinates,
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
from jaxborg.tracking import Run, assigned_devices, file_hash, input_artifact, serializable
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
    tree = jax.tree.map(
        lambda x: {"params": x.params, "opt_state": x.opt_state, "step": x.step} if isinstance(x, TrainState) else x,
        tree,
        is_leaf=lambda x: isinstance(x, TrainState),
    )
    leaves, structure = jax.tree.flatten(tree)
    leaves = [np.asarray(x) for x in leaves]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **{f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)})
    # This local trusted file only rebuilds the archived JAX/Flax tree structure.
    path.with_suffix(".tree.pkl").write_bytes(pickle.dumps(structure))
    write_json(
        path.with_suffix(".leaves.json"),
        [{"leaf": i, "shape": list(x.shape), "dtype": str(x.dtype)} for i, x in enumerate(leaves)],
    )


def restore_tree(path):
    """Read a trusted local capture with this revision's pinned JAX/Flax types."""
    arrays = np.load(path.with_suffix(".npz"))
    structure = pickle.loads(path.with_suffix(".tree.pkl").read_bytes())
    return jax.tree.unflatten(structure, [jnp.asarray(arrays[f"leaf{i}"]) for i in range(len(arrays))])


def advance_rollout_rng(rng, updates, steps):
    """Recover the trainer's parameter-independent split schedule at episode boundaries."""

    def episode(key, _):
        key, _ = jax.lax.scan(lambda k, _: (jax.random.split(k, 4)[0], None), key, None, length=steps)
        return jax.random.split(key)[0], None

    return jax.lax.scan(episode, rng, None, length=updates)[0]


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


def controlled_override(recipe, configs, protocol, previous, *, resume=False):
    """Validate the one-variable credit ablation before reusing any evidence."""
    override = protocol.get("core_override", {})
    if not override:
        if previous:
            raise ValueError("controlled-from requires the explicit credit override")
        return None
    if override != {"gae_lambda": 1.0} or not previous or resume:
        raise ValueError("only the controlled lambda-one ablation is supported")
    previous = Path(previous)
    if json.loads((previous / "manifest.json").read_text())["status"] != "FINISHED":
        raise ValueError("control must be completed")
    original = json.loads((previous / "effective-config.json").read_text())
    if serializable(configs) != original or any(configs[t]["GAE_LAMBDA"] != 0.95 for t in configs):
        raise ValueError("the control has different settings")
    old_protocol = yaml.safe_load((previous / "config.yaml").read_text())
    for key in ("source_steps", "training_seed", "warm_updates", "validation_episodes"):
        if protocol[key] != old_protocol[key]:
            raise ValueError(f"control differs in {key}")
    if protocol["seeds"]["validation_start"] != old_protocol["seeds"]["validation_start"]:
        raise ValueError("cached validation episodes use different seeds")
    if protocol["seeds"]["warm_rollout"] != old_protocol["seeds"]["warm_rollout"]:
        raise ValueError("control uses different rollout randomness")
    recipe["core"].update(override)
    for team in configs:
        configs[team]["GAE_LAMBDA"] = 1.0
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-repository", required=True)
    parser.add_argument("--resume-from", help="Trusted previous attempt with calibrated state and minibatch captures")
    parser.add_argument("--controlled-from", help="Completed matching lambda-0.95 warm start; reuse its initial state")
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
    control = controlled_override(recipe, configs, config, args.controlled_from, resume=bool(args.resume_from))
    recipe["train"]["total_timesteps"] = configs["blue"]["TOTAL_TIMESTEPS"]
    original = load_jax_bundle(data["original"]["model"])
    warm = load_jax_bundle(data["warm"]["model"])
    final = load_jax_bundle(data["final"]["model"])
    if parameter_hash(warm.policies["red"].weights) != parameter_hash(original.policies["red"].weights):
        raise ValueError("warm checkpoint opponent mismatch")
    entries = {"blue": warm.policies["blue"], "red": original.policies["red"]}
    networks = {team: policy_from_arch(entry.arch, action_dim=entry.action_dim) for team, entry in entries.items()}
    params = {team: entry.weights for team, entry in entries.items()}
    controlled_inputs = []
    if control:
        control_owner = json.loads((control / "runtime.json").read_text())["owner"]
        controlled_inputs = [
            input_artifact(f"runs:/{control_owner}/diagnostic/{p}", role=p)
            for p in (
                "captures/calibrated-state.npz",
                "captures/calibrated-state.tree.pkl",
                "checkpoints/warm-20.safetensors",
            )
        ]
    eval_env = make_joint_jax_env(configs["blue"]["EVAL_VARIANT"], topology_path=[topology], training_mode=False)
    evaluate = make_evaluator(eval_env, networks)
    reference_path = Path(args.reference_repository) / "scripts/train/algorithms/ippo_jax_joint.py"
    reference = load_reference(reference_path)
    with Run(
        recipe,
        name=config["name"],
        backend="jax",
        kind="training",
        config=configs,
        inputs=[{"role": k, "path": str(v["model"]), "sha256": v["sha256"]} for k, v in data.items()]
        + controlled_inputs,
    ) as owner:
        owner.update(diagnostic_protocol=config)
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
        baseline_policies = (
            ("warm-0", params["blue"]),
            ("archived-final", final.policies["blue"].weights),
            ("original", original.policies["blue"].weights),
        )
        if control:
            old = json.loads((control / "validation-episodes.json").read_text())
            evaluations = {
                label + suffix: old[label + suffix] for label, _ in baseline_policies for suffix in ("", "-no-block")
            }
            baseline_policies = ()
        if args.resume_from:
            evaluations = json.loads((Path(args.resume_from) / "validation-episodes.json").read_text())
            baseline_policies = ()
        for label, blue_params in baseline_policies:
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
        for _ in range(0 if args.resume_from or control else config["normalization_calibration_updates"]):
            states, env_state, obs, rng, norms, _ = calibrate(states, env_state, obs, rng, norms)
        if control:
            initial, env_state, obs, rng, norms = restore_tree(control / "captures/calibrated-state")
            for team in states:
                if parameter_hash(initial[team]["params"]) != parameter_hash(params[team]):
                    raise ValueError("calibrated control has different initial policies")
                expected = {
                    "params": states[team].params,
                    "opt_state": states[team].opt_state,
                    "step": states[team].step,
                }
                if jax.tree.structure(expected) != jax.tree.structure(initial[team]) or any(
                    not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(initial[team]))
                ):
                    raise ValueError("control is not the same fresh Adam state")
            states = {team: state.replace(**initial[team]) for team, state in states.items()}
            write_json(
                output / "controlled-start.json",
                {
                    "control_owner": control_owner,
                    "initial_policies_and_adam_exact": True,
                    "environment_rng_normalizers": "restored calibrated control",
                    "changed_core_setting": config["core_override"],
                },
            )
        metrics_rows, gradient_rows, signals, forks = [], [], [], []
        resume_update = 0
        expected_resume_mini = None
        if args.resume_from:
            previous = Path(args.resume_from).resolve()
            capture_files = list((previous / "captures").glob("update-*-minibatch.npz"))
            capture_update = max(int(p.stem.split("-")[1]) for p in capture_files)
            resume_update = capture_update - 1
            calibrated, env_state, obs, calibrated_rng, _ = restore_tree(previous / "captures/calibrated-state")
            blue, old_mini, old_adv, old_targets, old_indices, norms = restore_tree(
                previous / f"captures/update-{capture_update}-minibatch"
            )
            states = {
                team: state.replace(**(blue if team == "blue" else calibrated[team])) for team, state in states.items()
            }
            # Stock singleton topology resets deterministically after every full
            # 500-step rollout. Verify the replayed minibatch before continuing.
            if len(configs["blue"]["TOPOLOGY_BANK"]) != 1 or configs["blue"]["NUM_STEPS"] != 500:
                raise ValueError("boundary recovery requires one fixed topology and full episodes")
            rng = jax.jit(partial(advance_rollout_rng, updates=resume_update, steps=500))(calibrated_rng)
            expected_resume_mini = (old_mini, old_adv, old_targets, old_indices)
            for filename, destination in (
                ("training-metrics.json", metrics_rows),
                ("gradient-components.json", gradient_rows),
                ("learning-signals.json", signals),
            ):
                destination.extend(
                    r for r in json.loads((previous / filename).read_text()) if r["update"] <= resume_update
                )
            import shutil

            shutil.copytree(previous / "captures", output / "captures", dirs_exist_ok=True)
            write_json(
                output / "resume.json",
                {
                    "previous_attempt": str(previous),
                    "before_update": capture_update,
                    "policy_optimizer_normalizers": "restored numeric capture",
                    "environment": "singleton-topology full-episode reset state",
                    "rng": "recovered canonical split schedule; checked by replay",
                },
            )
        else:
            save_tree(output / "captures/calibrated-state", (states, env_state, obs, rng, norms))
        baseline_hash = parameter_hash(states["red"].params)
        for update in range(resume_update + 1, config["warm_updates"] + 1):
            before = states
            before_norms = norms
            if update in config["capture_updates"]:
                save_tree(output / f"captures/update-{update}-before-full-state", (states, env_state, obs, rng, norms))
            states, env_state, obs, rng, norms, metrics = collect(states, env_state, obs, rng, norms)
            diagnostic = metrics.pop("diagnostic")
            row = {
                "update": update,
                "warm_steps": update * configs["blue"]["NUM_ENVS"] * 500,
                **native(jax.device_get(metrics)),
            }
            metrics_rows.append(row)
            owner.update(
                actual_steps=row["warm_steps"],
                completed_updates=update,
                executed_training_steps=(update - resume_update) * configs["blue"]["NUM_ENVS"] * 500,
            )
            mlflow.log_metrics(
                {f"{team}/{key}": value for team in ("blue", "red", "game") for key, value in row[team].items()},
                step=row["warm_steps"],
            )
            write_json(output / "training-metrics.json", metrics_rows)
            print("warm", update, row["blue"]["raw_rollout_return"], flush=True)
            if update in config["capture_updates"]:
                traj, observed = diagnostic["trajectories"]["blue"], diagnostic["observed"]
                key, last_value = diagnostic["update_keys"]["blue"], diagnostic["last_values"]["blue"]
                mini, adv, target, raw_adv, norm_adv, targets, indices = first_minibatch(
                    trainer, traj, last_value, key, configs["blue"]
                )
                if expected_resume_mini is not None:
                    replay_error = max(
                        float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
                        for a, b in zip(
                            jax.tree.leaves((mini, adv, target, indices)), jax.tree.leaves(expected_resume_mini)
                        )
                    )
                    if replay_error > 2e-6:
                        raise ValueError(
                            f"recovered warm-start trajectory differs from saved minibatch: {replay_error}"
                        )
                    write_json(
                        output / "resume-reproduction.json",
                        {"all_minibatch_leaves_max_error": replay_error, "checked_update": update},
                    )
                    expected_resume_mini = None
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
                    train_run_id=owner.run_id,
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
        confirmation_policies = [("warm-0", params["blue"]), ("warm-final", states["blue"].params)]
        if control:
            confirmation_policies.append(
                (
                    "lambda095-final",
                    load_jax_bundle(control / "checkpoints/warm-20.safetensors").policies["blue"].weights,
                )
            )
        for label, blue_params in confirmation_policies:
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
            if control:
                comparisons["lambda1-minus-lambda095" + suffix] = bootstrap(
                    [
                        b["blue_return"] - a["blue_return"]
                        for a, b in zip(confirmation["lambda095-final" + suffix], confirmation["warm-final" + suffix])
                    ]
                )
        write_json(output / "confirmation-summary.json", comparisons)
        for path in sorted(output.rglob("*")):
            if path.is_file():
                owner.publish(path, "diagnostic/" + path.relative_to(output).as_posix())
        print("completed", comparisons, flush=True)
    write_json(output / "manifest.json", owner.manifest)


def capture_forks(config, output, env, networks, states, traj, observed, raw_adv, norm_adv, targets):
    groups = traffic_groups(traj, observed)
    rows = []
    for name in ("harmful_new_block", "useful_allow"):
        chosen = fork_coordinates(
            groups[name][:, : config["probe_envs"]],
            observed["phase"][:, : config["probe_envs"]],
            config["fork_states_per_group"],
        )
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
