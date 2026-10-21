"""Compare two cumulative search budgets along one fresh frozen-Blue response run."""

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import response_oracle as oracle

CHECKPOINT_UPDATES = (104, 208, 312, 416)
BUDGET_UPDATES = (208, 416)
STRIDE = 48000


def select_budget_candidates(scores, checkpoint_updates=CHECKPOINT_UPDATES, budget_updates=BUDGET_UPDATES):
    if tuple(checkpoint_updates) != CHECKPOINT_UPDATES or tuple(budget_updates) != BUDGET_UPDATES:
        raise ValueError("unexpected prescribed checkpoint or budget updates")
    order = ["original", *[f"update-{update}" for update in checkpoint_updates]]
    if set(scores) != set(order) or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("selection requires finite scores for every prescribed checkpoint and original Red")
    return {
        str(budget): min(
            ["original", *[f"update-{update}" for update in checkpoint_updates if update <= budget]],
            key=lambda name: (scores[name], order.index(name)),
        )
        for budget in budget_updates
    }


def load_protocol(path):
    manifest = oracle.load_protocol(path)
    reference = f"runs:/{manifest['protocol_owner_run_id']}/protocol/manifest.json"
    if oracle.file_hash(oracle.resolve_artifact(reference)) != oracle.file_hash(path):
        raise ValueError("manifest differs from the immutable prepared protocol")
    selection = manifest["selection"]
    if (
        selection.get("checkpoint") != "budget_milestones"
        or selection.get("checkpoint_updates") != list(CHECKPOINT_UPDATES)
        or selection.get("budget_updates") != list(BUDGET_UPDATES)
        or manifest["trainable_team"] != "red"
        or len(manifest["training_seeds"]) != 1
        or manifest["oracle_budget_per_attempt"]
        != {"requested_steps": 20000000, "completed_steps": 19968000, "updates": 416, "steps_per_update": STRIDE}
    ):
        raise ValueError("manifest is not the prescribed one-seed Red budget diagnostic")
    return manifest


def checkpoint_candidates(manifest, result):
    """Resolve only completed milestones from the one continuous training run."""
    seed = manifest["training_seeds"][0]
    owner, _ = oracle.read_manifest(result["run_id"])
    expected_recipe = oracle.load_recipe(manifest["recipes"][str(seed)])
    if (
        owner["status"] != "FINISHED"
        or owner["kind"] != "training"
        or owner["seed"] != seed
        or owner["source"]["git_commit"] != manifest["source_revision"]
        or owner["recipe_hash"] != oracle.digest(expected_recipe)
        or owner["actual_steps"] != 416 * STRIDE
        or result["actual_steps"] != 416 * STRIDE
        or owner["completed_updates"] != 416
        or owner["requested_steps"] != 20000000
        or owner["inputs"][0]["sha256"] != manifest["source"]["sha256"]
        or owner["inputs"][1]["sha256"] != manifest["game"]["topology_sha256"]
        or owner["policy_sources"]["red"] != {"kind": "fresh", "seed": seed}
        or owner["parameter_checks"]["blue"]["changed"]
        or not owner["parameter_checks"]["red"]["changed"]
    ):
        raise ValueError("training identity, recipe, frozen opponent or completed budget differs")
    for team in ("blue", "red"):
        expected_config = oracle.project_jax(expected_recipe, team=team)
        config = owner["effective_config"][team]
        if any(config.get(key) != oracle.serializable(value) for key, value in expected_config.items()):
            raise ValueError(f"{team} effective training configuration differs")
    candidates = {
        "original": {
            "reference": manifest["source"]["checkpoint"],
            "sha256": manifest["source"]["sha256"],
            "update": 0,
            "step": 0,
        }
    }
    for update in CHECKPOINT_UPDATES:
        step = update * STRIDE
        reference = (
            result["final_checkpoint"]
            if update == 416
            else f"runs:/{result['run_id']}/checkpoints/checkpoint_{step}.safetensors"
        )
        if not reference.startswith(f"runs:/{result['run_id']}/checkpoints/"):
            raise ValueError("checkpoint does not belong to the prescribed training run")
        path = oracle.resolve_artifact(reference)
        record = owner["outputs"][reference.split("/", 2)[2]]
        sidecar = oracle.read_sidecar(path)
        with oracle.jax.default_device(oracle.jax.devices("cpu")[0]):
            bundle = oracle.load_jax_bundle(path)
        if (
            record["step"] != step
            or bundle.provenance["total_steps"] != step
            or bundle.provenance["seed"] != seed
            or bundle.provenance["train_run_id"] != result["run_id"]
            or sidecar["run"]["total_steps"] != step
            or sidecar["run"]["seed"] != seed
            or oracle.parameter_hash(bundle.policies["blue"].weights)
            != manifest["source"]["policies"]["blue"]["parameter_sha256"]
        ):
            raise ValueError("milestone step, training identity or frozen Blue differs")
        candidates[f"update-{update}"] = {
            "reference": reference,
            "sha256": oracle.file_hash(path),
            "sidecar_reference": f"runs:/{result['run_id']}/{record['sidecar']}",
            "sidecar_sha256": owner["outputs"][record["sidecar"]]["sha256"],
            "red_parameter_sha256": oracle.parameter_hash(bundle.policies["red"].weights),
            "update": update,
            "step": step,
        }
    return candidates


def checked_matchup(manifest, candidate, reference, seeds):
    row = oracle.checked_evaluation(reference, seeds)
    if len(row["per_episode_blue_returns"]) != len(seeds) or len(row["per_episode_red_returns"]) != len(seeds):
        raise ValueError("evaluation returns do not match the episode cohort")
    owner, _ = oracle.read_manifest(row["eval_id"])
    if (
        owner["inputs"][0]["sha256"] != manifest["source"]["sha256"]
        or owner["inputs"][1]["sha256"] != candidate["sha256"]
        or owner["inputs"][2]["sha256"] != manifest["game"]["topology_sha256"]
        or owner["source"]["git_commit"] != manifest["source_revision"]
    ):
        raise ValueError("evaluation policy, topology or source identity differs")
    return row


def comparison_statistics(rows, selections, seeds, confidence):
    longer = rows[selections["416"]]

    def comparison(baseline, challenger=longer):
        return oracle.paired_gap(
            baseline["per_episode_blue_returns"],
            challenger["per_episode_blue_returns"],
            baseline["per_episode_seeds"],
            challenger["per_episode_seeds"],
            samples=confidence["resamples"],
            seed=confidence["seed"],
            trainable_team="red",
        )

    if any(row["per_episode_seeds"] != list(seeds) for row in rows.values()):
        raise ValueError("confirmation cohorts differ")
    return {
        "longer_vs_original": comparison(rows["original"]),
        "longer_vs_shorter": comparison(rows[selections["208"]]),
        "shorter_vs_original": comparison(rows["original"], rows[selections["208"]]),
    }


def aggregate(manifest, state, report):
    candidates = checkpoint_candidates(manifest, state["training"])
    if candidates != state["checkpoints"]:
        raise ValueError("saved checkpoint identities changed")
    validation_seeds = manifest["randomness"]["validation_episode_roots"]
    test_seeds = manifest["randomness"]["final_test_episode_roots"]
    scores = {
        name: checked_matchup(manifest, candidate, state["validation"][name], validation_seeds)["blue_mean_return"]
        for name, candidate in candidates.items()
    }
    selections = select_budget_candidates(scores)
    if selections != state["selection"]["candidates"] or scores != state["selection"]["validation_scores"]:
        raise ValueError("locked validation selection changed")
    names = list(dict.fromkeys(["original", *selections.values()]))
    if set(state["test"]) != set(names):
        raise ValueError("confirmation must contain only the unique locked candidates and original Red")
    rows = {name: checked_matchup(manifest, candidates[name], state["test"][name], test_seeds) for name in names}
    statistics = comparison_statistics(rows, selections, test_seeds, manifest["confidence_interval"])
    metrics_ref = f"runs:/{state['training']['run_id']}/logs/metrics.jsonl"
    metrics = [json.loads(line) for line in oracle.resolve_artifact(metrics_ref).read_text().splitlines()]
    if [row["update_idx"] for row in metrics] != list(range(1, 417)) or metrics[-1]["env_steps"] != 416 * STRIDE:
        raise ValueError("training curve does not cover the complete continuous budget")
    if any(row["team.red.lr"] != 0.0003 for row in metrics):
        raise ValueError("training curve learning rate differs from the fixed protocol")
    report.update(
        inputs=[
            *report.manifest["inputs"],
            oracle.input_artifact(metrics_ref, role="continuous training curve"),
            *[oracle.input_artifact(candidate["reference"], role=name) for name, candidate in candidates.items()],
            *[
                oracle.input_artifact(reference, role=f"validation {name}")
                for name, reference in state["validation"].items()
            ],
            *[
                oracle.input_artifact(reference, role=f"confirmation {name}")
                for name, reference in state["test"].items()
            ],
        ]
    )
    summary = {
        **statistics,
        "source_steps": manifest["source"]["original_training_steps"],
        "training": state["training"],
        "checkpoints": candidates,
        "selection": state["selection"],
        "validation": state["validation"],
        "confirmation": state["test"],
        "confirmation_unique_matchups": len(names),
        "training_time_s": metrics[-1]["wall_time_s"],
        "interval_scope": (
            "pointwise 95% paired episode intervals, conditional on selected policies; "
            "not simultaneous intervals or training-seed uncertainty"
        ),
        "interpretation": (
            "One training seed and one topology. An added-budget gain whose held-out interval "
            "excludes zero supports search insufficiency for this case; a positive point estimate alone, "
            "a null or a plateau does not establish convergence or equilibrium."
        ),
    }
    report_dir = Path(manifest["report_dir"])
    report.write_json("results/summary.json", summary)
    report.export("results/summary.json", report_dir / "summary.json")
    episode_rows = [
        {
            "episode_seed": episode_seed,
            "original_blue_return": rows["original"]["per_episode_blue_returns"][index],
            "shorter_blue_return": rows[selections["208"]]["per_episode_blue_returns"][index],
            "longer_blue_return": rows[selections["416"]]["per_episode_blue_returns"][index],
        }
        for index, episode_seed in enumerate(test_seeds)
    ]
    oracle.csv_file(report.path("results/confirmation-episodes.csv"), episode_rows)
    report.publish(report.path("results/confirmation-episodes.csv"), "results/confirmation-episodes.csv")
    report.export("results/confirmation-episodes.csv", report_dir / "confirmation-episodes.csv")
    curve_rows = [
        {
            key: row[key]
            for key in ("update_idx", "env_steps", "wall_time_s", "team.red.return", "team.blue.return", "team.red.lr")
        }
        for row in metrics
    ]
    oracle.csv_file(report.path("results/training-curves.csv"), curve_rows)
    report.publish(report.path("results/training-curves.csv"), "results/training-curves.csv")
    report.export("results/training-curves.csv", report_dir / "training-curves.csv")
    validation_rows = [
        {
            "candidate": name,
            "response_steps": candidate["step"],
            "blue_mean_return": scores[name],
            "checkpoint_sha256": candidate["sha256"],
        }
        for name, candidate in candidates.items()
    ]
    oracle.csv_file(report.path("results/validation-curve.csv"), validation_rows)
    report.publish(report.path("results/validation-curve.csv"), "results/validation-curve.csv")
    report.export("results/validation-curve.csv", report_dir / "validation-curve.csv")
    original = statistics["longer_vs_original"]
    added = statistics["longer_vs_shorter"]
    shorter = statistics["shorter_vs_original"]
    message = (
        "# Red response budget diagnostic\n\n"
        "One fresh Red IPPO response trained continuously against the original frozen late IPPO Blue. "
        "The requested 20M budget completed 19,968,000 steps. Separate validation selections used original Red "
        "and prescribed checkpoints available by 9,984,000 or 19,968,000 steps. "
        "Both selections were frozen before the 600 paired confirmation episodes.\n\n"
        f"Shorter-budget selection: `{selections['208']}`. Longer-budget selection: `{selections['416']}`.\n\n"
        f"Longer versus original Red gain: {original['red_improvement']:.6f}, "
        f"95% paired interval {original['ci95']}.\n\n"
        f"Longer versus shorter-budget gain: {added['red_improvement']:.6f}, 95% paired interval {added['ci95']}.\n\n"
        f"Shorter versus original Red gain: {shorter['red_improvement']:.6f}, "
        f"95% paired interval {shorter['ci95']}.\n\n"
        "These are pointwise 95% intervals describing episode uncertainty conditional on the selected policies, "
        "not simultaneous intervals. An added-budget gain with its interval wholly above zero supports the "
        "conclusion that the shorter search was insufficient in this case; a null result or training plateau "
        "does not establish equilibrium. The study covers one training seed and one topology.\n\n"
        "[Summary and checkpoint identities](summary.json), [confirmation episodes](confirmation-episodes.csv), "
        "[training curves](training-curves.csv), [validation curve](validation-curve.csv).\n"
    )
    report.path("results/README.md").write_text(message)
    report.publish(report.path("results/README.md"), "results/README.md")
    report.export("results/README.md", report_dir / "README.md")
    return summary


@oracle.tracked_entrypoint
def execute(args, *, aggregate_only=False):
    manifest = load_protocol(args.manifest)
    if not aggregate_only:
        oracle.require_smoke(manifest, args.manifest)
        oracle.assigned_devices()
    lock = (Path(manifest["study_dir"]) / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = oracle.state_file(manifest)
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "campaign": manifest["campaign"],
            "manifest_sha256": oracle.file_hash(args.manifest),
            "validation": {},
            "test": {},
            "status": "prepared",
            "slurm_job_ids": [],
        }
    )
    if state["campaign"] != manifest["campaign"] or state["manifest_sha256"] != oracle.file_hash(args.manifest):
        raise ValueError("state belongs to another immutable protocol")
    if state["test"] and "selection" not in state:
        raise ValueError("confirmation exists without a locked validation selection")
    report = oracle.Run(
        {"meta": {"name": manifest["campaign"] + "-budget"}},
        backend="cpu",
        kind="comparison",
        inputs=[oracle.input_artifact(args.manifest, role="immutable budget protocol")],
        config={"protocol": manifest},
    )

    def save():
        oracle.write_json(path, state)
        report.publish(path, "logs/state.json", mutable=True)

    try:
        if not aggregate_only:
            state["slurm_job_ids"].append(os.environ["SLURM_JOB_ID"])
            seed = manifest["training_seeds"][0]
            if "training" not in state:
                result_path = Path(manifest["study_dir"]) / f"training-seed-{seed}.json"
                if state.get("training_started"):
                    if not result_path.exists():
                        raise RuntimeError(
                            "training was already attempted; no automatic restart or optimizer resume is allowed"
                        )
                    state["training"] = json.loads(result_path.read_text())
                else:
                    state["training_started"] = time.time()
                    state["status"] = "training"
                    save()
                    state["training"] = oracle.train(manifest, seed, manifest["recipes"][str(seed)])
                save()
            candidates = checkpoint_candidates(manifest, state["training"])
            if "checkpoints" in state and state["checkpoints"] != candidates:
                raise ValueError("persisted checkpoint identities changed")
            state["checkpoints"] = candidates
            state["status"] = "validation"
            save()
            scores = {}
            for name, candidate in candidates.items():
                seeds = manifest["randomness"]["validation_episode_roots"]
                if name not in state["validation"]:
                    _, state["validation"][name] = oracle.evaluate(
                        manifest, (name, candidate["reference"]), "validation", seeds, reuse=True
                    )
                    save()
                scores[name] = checked_matchup(manifest, candidate, state["validation"][name], seeds)[
                    "blue_mean_return"
                ]
            selections = select_budget_candidates(scores)
            identities = {key: candidates[name]["sha256"] for key, name in selections.items()}
            if "selection" in state:
                previous = state["selection"]
                if (
                    previous["candidates"] != selections
                    or previous["checkpoint_sha256"] != identities
                    or previous["validation_scores"] != scores
                ):
                    raise ValueError("locked validation selections changed")
            else:
                state["selection"] = {
                    "candidates": selections,
                    "checkpoint_sha256": identities,
                    "validation_scores": scores,
                    "frozen_before_test_epoch": time.time(),
                }
            state["status"] = "confirmation"
            save()
            report.write_json("selection/locked.json", state["selection"])
            for name in dict.fromkeys(["original", *selections.values()]):
                if name not in state["test"]:
                    _, state["test"][name] = oracle.evaluate(
                        manifest,
                        (name, candidates[name]["reference"]),
                        "confirmation",
                        manifest["randomness"]["final_test_episode_roots"],
                        reuse=True,
                    )
                    save()
            state["status"] = "evaluations_finished"
            save()
        summary = aggregate(manifest, state, report)
        state["status"] = "complete"
        state["summary_reference"] = f"runs:/{report.run_id}/results/summary.json"
        save()
        return summary
    except BaseException as exc:
        state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "epoch": time.time(),
            "stage": state["status"],
        }
        state["status"] = "failed"
        save()
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign", required=True)
    prepare.add_argument("--defender", required=True)
    for action in ("dry-run", "smoke", "run", "pilot", "aggregate"):
        sub.add_parser(action).add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        config, _, _ = oracle.load_campaign(args.campaign, args.defender)
        if config["selection"]["checkpoint"] != "budget_milestones":
            raise ValueError("this runner requires budget_milestones selection")
        oracle.prepare(args)
    elif args.action == "dry-run":
        print(json.dumps(load_protocol(args.manifest), indent=2))
    elif args.action == "smoke":
        load_protocol(args.manifest)
        oracle.smoke(args)
    elif args.action == "pilot":
        load_protocol(args.manifest)
        oracle.smoke(args)
        execute(args)
    else:
        execute(args, aggregate_only=args.action == "aggregate")


if __name__ == "__main__":
    main()
