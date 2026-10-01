"""Verify and publish the bounded Blue credit study, including a failed export."""

import argparse
import json
import shutil
from pathlib import Path

import jax
import numpy as np

from jaxborg.checkpoint import load_jax_bundle
from jaxborg.research_tracking import parameter_hash
from jaxborg.tracking import Run, experiment_root, file_hash
from scripts.experiments.analyze_blue_learning_mechanism import analyze, analyze_credit, paired, write_csv
from scripts.experiments.blue_learning_mechanism import restore_tree, write_json


def verification(root):
    control, variant, credit = (root / p for p in ("attempt-002", "lambda1-001", "signal-forks-001"))
    for directory in (control, variant):
        assert json.loads((directory / "manifest.json").read_text())["status"] == "FINISHED"
    episodes = 0
    for directory in (control, variant):
        for filename in ("validation-episodes.json", "confirmation-episodes.json"):
            for label, rows in json.loads((directory / filename).read_text()).items():
                for row in rows:
                    assert row["blue_return"] == sum(
                        row[k] for k in ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
                    )
                    assert row["illegal_actions"] == 0
                    if label.endswith("-no-block"):
                        assert row["block"] == row["blocked_pair_ticks"] == row["reward_asf"] == 0
                    episodes += 1
    starts = [restore_tree(p / "captures/calibrated-state") for p in (control, variant)]
    assert jax.tree.structure(starts[0]) == jax.tree.structure(starts[1])
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(starts[0]), jax.tree.leaves(starts[1])))
    minis = [restore_tree(p / "captures/update-1-minibatch") for p in (control, variant)]
    for index in (0, 1, 4, 5):
        assert jax.tree.structure(minis[0][index]) == jax.tree.structure(minis[1][index])
        assert all(
            np.array_equal(a, b) for a, b in zip(jax.tree.leaves(minis[0][index]), jax.tree.leaves(minis[1][index]))
        )
    gradients = [json.loads((p / "gradient-components.json").read_text()) for p in (control, variant)]
    assert all(point["reference_max_parameter_error"] == 0 for rows in gradients for point in rows)
    futures = json.loads((credit / "results.json").read_text())
    assert len(futures) == 24
    assert {row["cohort"] for row in futures} == {"discovery", "confirmation"}
    maximum = 0.0
    for row in futures:
        assert len(row["seeds"]) == 32
        for condition in row["results"].values():
            assert all(condition["first_action_legal"])
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(condition["normalized_mc_advantage"])
                            - np.asarray(condition["gae_lambda1_advantage"])
                        )
                    )
                ),
            )
        for comparison, measures in row["summaries"].items():
            alternative = comparison.removesuffix("-minus-natural")
            for metric, saved in measures.items():
                delta = np.asarray(row["results"][alternative][metric], dtype=np.float32) - np.asarray(
                    row["results"]["natural"][metric], dtype=np.float32
                )
                if delta.ndim > 1:
                    delta = delta.sum(-1)
                delta = delta.astype(float)
                assert float(delta.mean()) == saved["mean"]
                rng = np.random.default_rng(6500001)
                interval = np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [0.025, 0.975])
                np.testing.assert_allclose(interval, saved["ci95"], atol=1e-10)
    assert maximum < 1e-4
    hashes = []
    frozen = parameter_hash(starts[0][0]["red"]["params"])
    for path in sorted(root.rglob("*.safetensors")):
        model = load_jax_bundle(path)
        assert parameter_hash(model.policies["red"].weights) == frozen
        hashes.append(
            {
                "checkpoint": path.relative_to(root).as_posix(),
                "sha256": file_hash(path),
                "blue_parameter_hash": parameter_hash(model.policies["blue"].weights),
                "frozen_red_parameter_hash": frozen,
            }
        )
    write_csv(root / "checkpoint-hashes.csv", hashes)
    return {
        "checked_exported_episode_rows_including_cached_baselines": episodes,
        "all_reward_component_sums_exact": True,
        "all_evaluation_actions_legal": True,
        "matched_calibrated_parameters_adam_environment_rng_normalizers_exact": True,
        "first_minibatch_trajectory_indices_and_prior_normalizers_exact": True,
        "six_original_execution_updater_parameter_comparisons_max_error": 0.0,
        "lambda1_monte_carlo_identity_max_error": maximum,
        "all_conditional_credit_estimates_reproduced_from_raw_paired_futures": True,
        "all_saved_model_red_parameters_unchanged": True,
        "model_files_checked": len(hashes),
    }


def summarize(root):
    control, variant, credit = (root / p for p in ("attempt-002", "lambda1-001", "signal-forks-001"))
    analyze(control)
    analyze(variant)
    analyze_credit(credit)
    checks = verification(root)
    write_json(root / "verification.json", checks)
    episodes = json.loads((variant / "confirmation-episodes.json").read_text())
    comparisons = {}
    for suffix in ("", "-no-block"):
        for label, a, b in (
            ("lambda1-minus-initial", "warm-0", "warm-final"),
            ("lambda095-minus-initial", "warm-0", "lambda095-final"),
            ("lambda1-minus-lambda095", "lambda095-final", "warm-final"),
        ):
            comparisons[label + suffix] = paired(episodes[a + suffix], episodes[b + suffix])
    canonical = json.loads((variant / "confirmation-summary.json").read_text())
    for suffix in ("", "-no-block"):
        independent = comparisons["lambda1-minus-lambda095" + suffix]
        saved = canonical["lambda1-minus-lambda095" + suffix]
        assert independent["mean"] == saved["mean"]
        np.testing.assert_allclose([independent["low95"], independent["high95"]], saved["ci95"], atol=1e-10)
    means = {
        label: {k: float(np.mean([row[k] for row in rows])) for k in rows[0] if k != "seed"}
        for label, rows in episodes.items()
    }
    adjusted = []
    for row in json.loads((credit / "results.json").read_text()):
        if row["cohort"] != "confirmation":
            continue
        for metric in ("normalized_mc_advantage", "gae_advantage"):
            d = np.asarray(row["results"]["opposite"][metric]) - np.asarray(row["results"]["natural"][metric])
            rng = np.random.default_rng(6500001)
            boot = d[rng.integers(0, 32, (10000, 32))].mean(1)
            interval = np.quantile(boot, [0.05 / 48, 1 - 0.05 / 48])
            adjusted.append(
                {
                    "fork": row["fork"],
                    "metric": metric,
                    "mean": float(d.mean()),
                    "bonferroni24_low": float(interval[0]),
                    "bonferroni24_high": float(interval[1]),
                }
            )
    write_csv(root / "credit-multiplicity.csv", adjusted)
    summary = {
        "source_steps": 9600000,
        "training_seed": 11001,
        "historical_warm_steps": 1920000,
        "additional_steps_per_arm": 960000,
        "comparisons": comparisons,
        "confirmation_means": means,
        "resource_allocation_gpu_hours": 6114 / 3600,
        "scope": "one source, training seed and topology; weights-only historical warm start",
    }
    write_json(root / "summary.json", summary)
    write_json(
        root / "seed-ledger.json",
        {
            "archived_reproduction_reuse": list(range(4200000, 4200008)),
            "validation_reused_for_both_arms": [6200000, 6200031],
            "control_confirmation": [6400000, 6400127],
            "ablation_confirmation": [7200000, 7200127],
            "warm_rollout_root": 6100001,
            "initial_fork_futures": [[6300000 + i * 1000, 6300031 + i * 1000] for i in range(12)],
            "credit_discovery_futures": [[6700000 + i * 1000, 6700031 + i * 1000] for i in range(12)],
            "credit_confirmation_futures": [[6800000 + i * 1000, 6800031 + i * 1000] for i in range(12)],
            "bootstrap_root": 6500001,
            "avoid_other_computer_confirmation": [5200000, 5200127],
            "audit_limit": "newer investigation unavailable here; domains avoid its reported used cohort",
            "episode_accounting": {
                "new_reset_policy_episodes": 1792,
                "archived_reproduction_replays": 24,
                "conditional_arm_rows": 4224,
                "conditional_sanity_replays": 64,
                "saved_intervention_states": 12,
                "source_rollout_environments": 4,
            },
        },
    )
    plot(root, comparisons, means)
    report(root, summary)
    return summary


def plot(root, comparisons, means):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for directory, label in (("attempt-002", "lambda 0.95"), ("lambda1-001", "lambda 1")):
        rows = json.loads((root / directory / "training-metrics.json").read_text())
        axes[0].plot(
            [r["warm_steps"] / 1e6 for r in rows], [r["blue"]["raw_rollout_return"] for r in rows], label=label
        )
    axes[0].legend()
    axes[0].set(
        title="Same calibrated start and rollout RNG",
        xlabel="Additional training, M steps",
        ylabel="Raw training return",
    )
    for i, (key, label) in enumerate(
        (("lambda095-minus-initial", "lambda 0.95"), ("lambda1-minus-initial", "lambda 1"))
    ):
        r = comparisons[key]
        axes[1].errorbar(i, r["mean"], yerr=[[r["mean"] - r["low95"]], [r["high95"] - r["mean"]]], fmt="o", capsize=5)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["lambda 0.95", "lambda 1"],
        title="128 new paired confirmation episodes",
        ylabel="Gain over the 1.92M initial policy",
    )
    keys = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
    axes[2].bar(
        ["Attack impact", "Local work", "Blocked access", "Restore cost"],
        [means["warm-final"][k] - means["lambda095-final"][k] for k in keys],
    )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set(title="lambda 1 minus lambda 0.95", ylabel="Blue reward-component gain")
    axes[2].tick_params(axis="x", labelrotation=20)
    fig.suptitle("Bounded Blue credit ablation: IPPO source 9.6M / training seed 11001")
    fig.savefig(root / "matched-ablation.png", dpi=180)
    fig.savefig(root / "matched-ablation.pdf")
    plt.close(fig)


def report(root, summary):
    c = summary["comparisons"]

    def estimate(key):
        return f"{c[key]['mean']:,.2f} [{c[key]['low95']:,.2f}, {c[key]['high95']:,.2f}]"

    (root / "README.md").write_text(f"""# Why fresh Blue learns harmful traffic control

The bounded experiment identifies a credit-estimation failure: PPO's actor follows GAE signals that prefer
harmful Blocks under the learned critic. Using GAE lambda 1 to reduce intermediate bootstrap bias
substantially improves learning in a matched warm-start test. Training still bootstraps at rollout boundaries.
This supports an experiment-specific configuration change; it does not establish a universal training fix.

Both arms start from the same fresh-policy checkpoint at 1.92M historical steps, with identical calibrated
weights, fresh Adam, normalizers, environment and RNG. They train another 960,000 steps against the same
frozen 9.6M Red, stock CC4, topology seed 0, enhanced-v2 observations, stochastic actions and 500-step
episodes. Only GAE lambda changes from 0.95 to 1. Historical Adam and normalizers were unavailable, so these
are warm starts, not exact historical continuations.

## Learning mechanism

1. The lambda-0.95 control reproduces the decline. Its first reserved 128 episodes show a loss of 566.98
points [−673.14, −464.01]; excluding Block at evaluation leaves an inconclusive −66.28 [−159.26, 24.99]. A
second independent cohort confirms deterioration.
2. Same-minibatch replay attributes increased new mission-route Block probability mainly to actor gradients at
updates 1, 5 and 20. Direct critic gradients push against it; entropy is small after the first capture. Update
10 temporarily pushes against Blocks. Action-stratum gradients show negative Remove/Restore advantages
redistribute probability toward Blocks through the shared actor. At update 20, new mission-route Blocks also
receive positive advantages.
3. Saved-state interventions hold other simultaneous actions and intervention randomness fixed, and continue
frozen policies under common future seeds. At four open mission-route states, legal Allow improves total
return but lambda-0.95 GAE prefers Block. This also holds against a freshly sampled policy alternative. In the
reserved futures, two states have opposite-sign credit intervals even after Bonferroni correction over 12
states × 2 credit measures; the other two have pointwise opposite-sign intervals. Both discounted raw return
and normalized Monte Carlo credit favor Allow, so this is not explained solely by discounting or reward
scaling.
4. Lambda-one GAE matches measured normalized Monte Carlo advantage within 5.1e-6. The matched training
ablation below validates that reducing bootstrap bias improves policy performance. Lambda changes actor
advantages and critic targets together; this test does not fully separate their contributions to the eventual
gain.

Critic-input audits establish aliasing: own pending-action time and within-phase clock changes are invisible
in all 20 examined traffic decisions; reverse-direction blocking is invisible in 19. Green routing fails when
either direction is blocked. These limits plausibly contribute to inaccurate continuation estimates, but this
study does not isolate which missing input causes the bias. The improved lambda-one arm retains the same
observations.

## New held-out comparison

Higher Blue return is better. Each comparison uses 128 common reset seeds, 7200000–7200127, and 10,000 paired
bootstrap resamples. Intervals are pointwise 95%.

| Comparison | Paired gain [95% interval] |
| --- | ---: |
| Lambda 0.95 final minus initial | {estimate("lambda095-minus-initial")} |
| Lambda 1 final minus initial | {estimate("lambda1-minus-initial")} |
| Lambda 1 minus lambda 0.95 | {estimate("lambda1-minus-lambda095")} |
| Lambda 1 minus lambda 0.95, both Block-excluded | {estimate("lambda1-minus-lambda095-no-block")} |

Lambda 1 produces fewer Blocks and more Restores. Its gains include reduced blocked-access penalties, fewer
local-work failures and less attack impact, while it pays more Restore costs. The benefit extends beyond
traffic blocking. Original co-trained Blue appears in validation, but was not evaluated on this final cohort;
do not claim a confirmed improvement over original Blue from these tests.

![Matched credit ablation](matched-ablation.png)

## Evidence, code and controls

- [summary.json](summary.json), [verification.json](verification.json),
[checkpoint-hashes.csv](checkpoint-hashes.csv), and [seed-ledger.json](seed-ledger.json).
- `attempt-002/`: control recipes, intermediate checkpoints, numeric optimizer/environment states, captured
minibatches, raw metrics and paired episode results.
- `signal-forks-001/`: all discovery/confirmation conditional outcomes, measured credit and route persistence.
[Credit comparisons](signal-forks-001/analysis/credit-comparisons.csv), [credit
plot](signal-forks-001/analysis/credit-ranking.png), and [multiplicity checks](credit-multiplicity.csv).
- `lambda1-001/`: matched-start proof, explicit lambda-one recipe, full training logs, models and 768 final
policy-episode rows. [Confirmation summary](lambda1-001/confirmation-summary.json).
- `attempt-002/analysis/`: route/phase probabilities and choices from captured shuffled minibatches, component
gradients, action-stratum attribution and observation audits. These samples do not constitute every training
decision.
- `input-data/`: the three target input models/recipes, topology and archived evaluation data, with preserved
relative layout. Other checkpoint-index rows are not included. `provenance/` retains source archives,
dependencies and Slurm logs.

Eight archived returns reproduce exactly on the CUDA12/JAX 0.10.2 backend. All six captured full PPO updates
match the original execution updater's parameters exactly. Raw reward components sum to the score and
evaluated actions remain legal; frozen Red weights are unchanged. No PPO arithmetic defect is established. The
observed failure concerns estimated credit under the learned critic.

Developmental instrumentation defects are preserved separately: job 3874 stopped at a read-only fork selector;
job 3876 completed every calculation but failed on duplicate immutable config publication. Both were
corrected. The latter's source run remains FAILED; recovered raw results are published by the final analysis
owner. The earlier control wrapper also left MLflow `actual_steps` at zero; its saved 20-update metrics
establish 960,000 logical steps. Later execution records this correctly. None of these defects explains the
original Blue training behavior.

The four one-GPU community jobs used **1.70 allocated GPU hours**, including failed attempts and compilation.
Unique additional training is 960,000 steps per arm; one control update was physically replayed during
recovery. Conditional futures are not independent topology or training-seed replications and must not be
pooled with reset episodes.

## Use and next bounded test

Code is on
[stage-c/blue-learning-mechanism](https://github.com/ITM-Kitware/jaxborg/tree/stage-c/blue-learning-mechanism).
Control execution: `eead2dc99c0df816c59f5880b186bc72d3528fbd`; credit execution:
`5e52ab5f5bbf6511565d7761504f20abf30c7da4`; lambda-one execution: `eafbe4a615e8527f717189b171060289b662e947`.
The original fresh-Blue execution remains `0d577faab9ac3e0a06376e99e9592aadf128c130`.

Use `campaigns/response-oracles/blue-learning-credit-lambda1.yaml` for this tested condition. Training
defaults and stock environment rules are unchanged. Replicate the matched 20-update comparison at one other
source or training seed with fresh confirmation roots before extending the change. Fresh-from-random training,
other opponents, other topologies and full actor-versus-critic mediation remain unresolved. The original
campaign result remains **“the configured Blue search failed to find an improvement”**; the new diagnostic
does not retroactively change its selection or imply equilibrium.

For transfer, copy this directory or its dedicated archive. Statistical reanalysis needs the exported JSON/CSV
and NumPy; plots additionally need Matplotlib. Captured-gradient analysis needs the compatible repository/JAX
environment. Resolve source paths from the portable input layout and remap topology paths in copied recipes,
preserving the archived originals. The shared MLflow database is unnecessary for reanalysis.

```bash
git fetch origin stage-c/blue-learning-mechanism
git worktree add ../blue-learning origin/stage-c/blue-learning-mechanism
JAX_PLATFORMS=cpu PYTHONPATH=src:. python -m scripts.experiments.summarize_blue_learning_study \\
  /path/to/blue-learning-mechanism
```
""")


def publish(root):
    tracking = experiment_root()
    assert (tracking / "mlflow.db").exists(), "publish into the existing tracking database"
    runs = [
        json.loads((root / directory / "runtime.json").read_text()).get("owner")
        or json.loads((root / directory / "runtime.json").read_text())["run_id"]
        for directory in ("attempt-001", "attempt-002", "signal-forks-001", "lambda1-001")
    ]
    for run in runs:
        source = tracking / "artifacts" / run / "artifacts"
        for relative in (
            "manifest.json",
            "recipe.yaml",
            "source/source.tar.gz",
            "environment/installed.json",
            "environment/uv.lock",
            "logs/console.log",
        ):
            if (source / relative).exists():
                target = root / "provenance" / run / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
    records = [
        {
            "role": directory,
            "path": str(root / directory / filename),
            "sha256": file_hash(root / directory / filename),
            "source_run_id": run,
        }
        for directory, filename, run in zip(
            ("attempt-002", "signal-forks-001", "lambda1-001"),
            ("confirmation-episodes.json", "results.json", "confirmation-episodes.json"),
            runs[1:],
        )
    ]
    with Run(
        {"meta": {"name": "blue-learning-mechanism-final-analysis"}},
        backend="jax",
        kind="analysis",
        inputs=records,
        config={"recovered_failed_export": runs[2]},
    ) as owner:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.relative_to(root).parts[0] != "previews":
                owner.publish(path, "blue-learning/" + path.relative_to(root).as_posix())
    write_json(
        root / "analysis-provenance.json",
        {
            "owner": owner.run_id,
            "manifest": owner.manifest,
            "report": f"runs:/{owner.run_id}/blue-learning/README.md",
            "failed_computation_export_owner": runs[2],
            "recovered_raw_credit": f"runs:/{owner.run_id}/blue-learning/signal-forks-001/results.json",
        },
    )
    source = tracking / "artifacts" / owner.run_id / "artifacts"
    for relative in (
        "manifest.json",
        "recipe.yaml",
        "source/source.tar.gz",
        "environment/installed.json",
        "environment/uv.lock",
        "logs/console.log",
    ):
        if (source / relative).exists():
            target = root / "provenance" / owner.run_id / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    summary = summarize(a.directory.resolve())
    if a.publish:
        publish(a.directory.resolve())
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
