"""Replay the saved CAGE 4 defender comparison on CPU."""

import argparse
import json
from functools import lru_cache
from hashlib import file_digest
from itertools import count
from pathlib import Path

import jax
import mlflow
import yaml

from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint, eligible_operational_server_indices
from jaxborg.evaluation.jax_scripted_red import evaluate_jax_scripted_reds, make_jax_env
from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, evaluate_matchup
from jaxborg.recipe import eval_variant
from jaxborg.scenarios.cc4.topology_cli import export_generated

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
CONDITIONS = ("one", "hundred")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def prepare_layouts(output):
    root = output / "layouts"
    root.mkdir(parents=True, exist_ok=False)
    familiar = PROTOCOL["familiar_topology_seed"]
    held_out = PROTOCOL["held_out"]
    seen = set()
    selected = {"familiar": [], "held-out": []}
    rejected = []
    for seed in (familiar,):
        path = root / f"seed-{seed}.npz"
        export_generated(seed, path)
        fingerprint = canonical_topology_fingerprint(path)
        if fingerprint != PROTOCOL["familiar_topology_fingerprint"]:
            raise ValueError("Regenerated familiar layout fingerprint differs")
        seen.add(fingerprint)
        selected["familiar"].append({"seed": seed, "path": path.name, "fingerprint": fingerprint})
    for seed in count(held_out["first_seed"]):
        path = root / f"seed-{seed}.npz"
        export_generated(seed, path)
        fingerprint = canonical_topology_fingerprint(path)
        if len(eligible_operational_server_indices(path)) < 3:
            rejected.append({"seed": seed, "reason": "fewer than three eligible operational servers"})
            continue
        if fingerprint in seen:
            rejected.append({"seed": seed, "reason": "duplicate layout fingerprint"})
            continue
        seen.add(fingerprint)
        selected["held-out"].append({"seed": seed, "path": path.name, "fingerprint": fingerprint})
        if len(selected["held-out"]) == held_out["count"]:
            break
    write_json(output / "layouts.json", {"selected": selected, "rejected": rejected})


def task_list():
    scripted = [(red, PROTOCOL["training_seeds"], None) for red in PROTOCOL["scripted_reds"]]
    learned = [
        (f"learned-{condition}", [seed], condition) for seed in PROTOCOL["training_seeds"] for condition in CONDITIONS
    ]
    return [*scripted, *learned]


def checkpoint(root, condition, seed):
    spec = PROTOCOL["checkpoints"][condition][str(seed)]
    directory = root / spec["relative_directory"]
    step = PROTOCOL["checkpoint_step"]
    bundle = directory / f"checkpoint_{step}.safetensors"
    recipe = yaml.safe_load((directory / f"recipe_checkpoint_{step}.yaml").read_text())
    run = recipe["run"]
    if digest(bundle) != spec["sha256"] or (run["seed"], run["total_steps"], run["git_commit"]) != (
        seed,
        step,
        PROTOCOL["source_revision"],
    ):
        raise ValueError(f"Checkpoint provenance differs for {condition}, training seed {seed}")
    return bundle, recipe


@lru_cache(maxsize=1)
def scripted_environment(variant, *, training_mode, topology_path):
    return make_jax_env(variant, training_mode=training_mode, topology_path=topology_path)


def evaluate(args):
    panel = json.loads((args.output / "layouts.json").read_text())["selected"][args.panel]
    paths = tuple(args.output / "layouts" / case["path"] for case in panel)
    seed_by_fingerprint = {case["fingerprint"]: case["seed"] for case in panel}
    for case, path in zip(panel, paths, strict=True):
        if canonical_topology_fingerprint(path) != case["fingerprint"]:
            raise ValueError(f"Layout fingerprint differs for seed {case['seed']}")
    opponent, training_seeds, red_condition = task_list()[args.task]
    run_name = f"{args.panel}-{opponent}-blue-{','.join(map(str, training_seeds))}"
    task_directory = args.output / f"{args.panel}-task-{args.task}"
    task_directory.mkdir(exist_ok=False)
    seeds = PROTOCOL["episode_seeds"]
    expected_cases = {(case["seed"], seed) for case in panel for seed in seeds}
    context = MatchupEvaluationContext()
    experiment = mlflow.set_experiment(args.experiment)
    if not experiment.artifact_location.startswith("mlflow-artifacts:/"):
        raise ValueError("Evaluation requires the shared Jaxborg artifact service")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "source_revision": PROTOCOL["source_revision"],
                "panel": args.panel,
                "opponent": opponent,
                "checkpoint_step": PROTOCOL["checkpoint_step"],
            }
        )
        write_json(
            task_directory / "tracking.json", {"experiment_id": experiment.experiment_id, "run_id": run.info.run_id}
        )
        for path in (PROTOCOL_PATH, Path(__file__), args.output / "layouts.json"):
            mlflow.log_artifact(str(path))
        for training_seed in training_seeds:
            red_seed = (
                None
                if red_condition is None
                else next(seed for seed in PROTOCOL["training_seeds"] if seed != training_seed)
            )
            for condition in CONDITIONS:
                bundle, recipe = checkpoint(args.checkpoint_root, condition, training_seed)
                print(f"{args.panel} {opponent}, Blue {condition} seed {training_seed}", flush=True)
                variant = eval_variant(recipe)
                target = opponent[-1] if opponent.startswith("cia_") else None
                if red_seed is None:
                    (row,) = evaluate_jax_scripted_reds(
                        bundle,
                        base_variant=variant,
                        topology_paths=paths,
                        reds=(opponent,),
                        seeds=seeds,
                        recipe=recipe,
                        env_factory=scripted_environment,
                    )
                    episode_seeds = row["per_episode_seeds"]
                    fingerprints = row["per_episode_topology_fingerprints"]
                    role_maps = row["episode_role_map_ids"]
                    rewards = row["per_episode"]
                    cia = [episode[target] if target else None for episode in row["per_episode_cia"]]
                else:
                    red, _ = checkpoint(args.checkpoint_root, red_condition, red_seed)
                    result = evaluate_matchup(
                        bundle,
                        red,
                        backend="jax",
                        variant=variant,
                        seeds=seeds,
                        topology_path=paths,
                        cia=recipe["eval"]["cia"],
                        progress=False,
                        context=context,
                    )
                    episode_seeds, fingerprints = result.episode_seeds, result.episode_topology_fingerprints
                    role_maps = result.episode_role_map_ids
                    rewards, cia = result.blue_returns, [None] * len(result.blue_returns)
                episodes = [
                    {
                        "layout_seed": seed_by_fingerprint[fingerprint],
                        "episode_seed": seed,
                        "role_map_id": role_map,
                        "reward": reward,
                        "targeted_cia": score,
                    }
                    for seed, fingerprint, role_map, reward, score in zip(
                        episode_seeds, fingerprints, role_maps, rewards, cia, strict=True
                    )
                ]
                actual_cases = {(episode["layout_seed"], episode["episode_seed"]) for episode in episodes}
                if len(episodes) != len(expected_cases) or actual_cases != expected_cases:
                    raise ValueError("Evaluation did not produce exactly the protocol's cases")
                output = task_directory / f"{condition}-{training_seed}.json"
                write_json(
                    output,
                    {
                        "panel": args.panel,
                        "condition": condition,
                        "opponent": opponent,
                        "training_seed": training_seed,
                        "red_training_seed": red_seed,
                        "episodes": episodes,
                    },
                )
                mlflow.log_artifact(str(output), artifact_path="results")
                mlflow.log_metric(f"blue_reward.{condition}.{training_seed}", sum(rewards) / len(rewards))
                print(f"Saved {output.name} with {len(episodes)} episodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--panel", choices=("familiar", "held-out"), default="held-out")
    parser.add_argument("--task", type=int, choices=range(len(task_list())))
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--experiment")
    args = parser.parse_args()
    if jax.default_backend() != "cpu" or jax.__version__ != PROTOCOL["jax_version"]:
        parser.error("Use the pinned CPU JAX environment")
    if args.prepare:
        prepare_layouts(args.output)
    elif args.task is None or args.checkpoint_root is None or not args.experiment:
        parser.error("Evaluation needs --task, --checkpoint-root, and --experiment")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
