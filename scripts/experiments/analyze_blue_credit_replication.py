"""Audit a matched Blue lambda pair and optional raw conditional credit evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.experiments.analyze_blue_learning_mechanism import paired, write_csv


def read(path):
    return json.loads(path.read_text())


def same_numeric_state(control, variant):
    """Compare every archived calibrated leaf without importing JAX or unpickling."""
    base = "captures/calibrated-state"
    if read(control / (base + ".leaves.json")) != read(variant / (base + ".leaves.json")):
        raise ValueError("initial-state leaf layouts differ")
    with np.load(control / (base + ".npz")) as a, np.load(variant / (base + ".npz")) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("calibrated starting state differs")
    return True


def audit_pair(control, variant):
    for directory in (control, variant):
        if read(directory / "manifest.json")["status"] != "FINISHED":
            raise ValueError("both training arms must be completed")
    protocols = [yaml.safe_load((p / "config.yaml").read_text()) for p in (control, variant)]
    for field in ("source_steps", "training_seed", "warm_updates", "validation_episodes"):
        if protocols[0][field] != protocols[1][field]:
            raise ValueError(f"unmatched protocol: {field}")
    configs = [read(p / "effective-config.json") for p in (control, variant)]
    for team in configs[0]:
        expected = {**configs[0][team], "GAE_LAMBDA": 1.0}
        if configs[0][team]["GAE_LAMBDA"] != 0.95 or configs[1][team] != expected:
            raise ValueError("effective configs differ beyond GAE lambda")
    for directory, protocol, config in zip((control, variant), protocols, configs):
        metrics = read(directory / "training-metrics.json")
        if [r["update"] for r in metrics] != list(range(1, protocol["warm_updates"] + 1)):
            raise ValueError("incomplete or duplicate training updates")
        steps = config["blue"]["NUM_ENVS"] * config["blue"]["NUM_STEPS"]
        if any(r["warm_steps"] != r["update"] * steps for r in metrics):
            raise ValueError("training-step budget mismatch")
        rows = read(directory / "confirmation-episodes.json")
        expected = list(
            range(
                protocol["seeds"]["confirmation_start"],
                protocol["seeds"]["confirmation_start"] + protocol["confirmation_episodes"],
            )
        )
        if any([r["seed"] for r in cohort] != expected for cohort in rows.values()):
            raise ValueError("confirmation data differs from the registered cohort")
    same_numeric_state(control, variant)
    episodes = read(variant / "confirmation-episodes.json")
    old = read(control / "confirmation-episodes.json")
    common_cohort = [r["seed"] for r in old["warm-0"]] == [r["seed"] for r in episodes["warm-0"]]
    if common_cohort:
        for suffix in ("", "-no-block"):
            if old["warm-0" + suffix] != episodes["warm-0" + suffix]:
                raise ValueError("same-cohort initial policy evaluations differ")
            if old["warm-final" + suffix] != episodes["lambda095-final" + suffix]:
                raise ValueError("same-cohort control policy evaluations differ")
            if "original" + suffix in old and old["original" + suffix] != episodes.get("original" + suffix):
                raise ValueError("same-cohort original policy evaluations differ")
    if protocols[1].get("include_original_confirmation") and "original" not in episodes:
        raise ValueError("registered original defender evaluation is missing")
    for label, rows in episodes.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("duplicated episode seeds")
        for row in rows:
            if row["blue_return"] != sum(row[k] for k in ("reward_ria", "reward_lwf", "reward_asf", "action_cost")):
                raise ValueError("reward components differ from total score")
            if row["illegal_actions"] or (label.endswith("-no-block") and (row["block"] or row["reward_asf"])):
                raise ValueError("evaluation legality or Block-exclusion control failed")
    bootstrap_seed = protocols[1]["seeds"].get("bootstrap", 6500001)
    comparisons = {}
    for suffix in ("", "-no-block"):
        for name, a, b in (
            ("lambda1-minus-initial", "warm-0", "warm-final"),
            ("lambda095-minus-initial", "warm-0", "lambda095-final"),
            ("lambda1-minus-lambda095", "lambda095-final", "warm-final"),
        ):
            comparisons[name + suffix] = paired(episodes[a + suffix], episodes[b + suffix], seed=bootstrap_seed)
        if "original" + suffix in episodes:
            for name, label in (("initial", "warm-0"), ("lambda095", "lambda095-final"), ("lambda1", "warm-final")):
                comparisons[name + "-minus-original" + suffix] = paired(
                    episodes["original" + suffix], episodes[label + suffix], seed=bootstrap_seed
                )
    means = {
        label: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
        for label, rows in episodes.items()
    }
    return {
        "status": "VERIFIED_COMPLETED_PAIR",
        "training_seed": protocols[1]["training_seed"],
        "source_steps": protocols[1]["source_steps"],
        "independent_training_seeds_in_this_pair": 1,
        "calibrated_starting_numeric_leaves_exact": True,
        "only_effective_lambda_differs": True,
        "both_arms_confirmed_on_same_cohort": common_cohort,
        "original_defender_on_variant_confirmation_cohort": "original" in episodes,
        "confirmation_seeds": [r["seed"] for r in episodes["warm-0"]],
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
        "means": means,
        "supports_lambda_within_seed": all(
            comparisons[k]["low95"] > 0 for k in ("lambda1-minus-initial", "lambda1-minus-lambda095")
        ),
        "scope": "one matched training seed; episode intervals are not training-replication uncertainty",
    }


def audit_credit(directory):
    records = read(directory / "results.json")
    family_size = 2 * sum(r["cohort"] == "confirmation" for r in records)
    rows, corrected = [], []
    maximum_identity_error = 0.0
    for record in records:
        for condition in record["results"].values():
            if not all(condition["first_action_legal"]):
                raise ValueError("illegal intervention")
            maximum_identity_error = max(
                maximum_identity_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(condition["normalized_mc_advantage"])
                            - np.asarray(condition["gae_lambda1_advantage"])
                        )
                    )
                ),
            )
        for alternative in ("opposite", "policy", "sleep"):
            for metric in record["summaries"][alternative + "-minus-natural"]:
                # Saved device calculations subtract and sum in float32;
                # the original bootstrap then explicitly converts to float64.
                delta = np.asarray(record["results"][alternative][metric], dtype=np.float32) - np.asarray(
                    record["results"]["natural"][metric], dtype=np.float32
                )
                if delta.ndim > 1:
                    delta = delta.sum(-1)
                delta = delta.astype(float)
                if len(delta) != len(record["seeds"]):
                    raise ValueError("conditional seeds and futures differ in length")
                rng = np.random.default_rng(6500001)
                resamples = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
                interval = np.quantile(resamples, [0.025, 0.975])
                saved = record["summaries"][alternative + "-minus-natural"][metric]
                np.testing.assert_allclose(
                    [delta.mean(), *interval], [saved["mean"], *saved["ci95"]], rtol=0, atol=1e-10
                )
                row = {
                    "cohort": record["cohort"],
                    "fork": record["fork"],
                    "alternative": alternative,
                    "metric": metric,
                    "mean": float(delta.mean()),
                    "low95": float(interval[0]),
                    "high95": float(interval[1]),
                    "n": len(delta),
                }
                rows.append(row)
                if (
                    record["cohort"] == "confirmation"
                    and alternative == "opposite"
                    and metric in ("normalized_mc_advantage", "gae_advantage")
                ):
                    interval = np.quantile(resamples, [0.05 / (2 * family_size), 1 - 0.05 / (2 * family_size)])
                    corrected.append(
                        {
                            **row,
                            "family_size": family_size,
                            "familywise_low": float(interval[0]),
                            "familywise_high": float(interval[1]),
                        }
                    )
    if maximum_identity_error > 1e-4:
        raise ValueError("lambda-one terminal-credit identity failed")
    return rows, corrected, maximum_identity_error


def plot(result, control, variant, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for directory, label in ((control, "lambda 0.95"), (variant, "lambda 1")):
        rows = read(directory / "training-metrics.json")
        axes[0].plot(
            [r["warm_steps"] / 1e6 for r in rows], [r["blue"]["raw_rollout_return"] for r in rows], label=label
        )
    axes[0].set(xlabel="Additional training, M steps", ylabel="Raw training return")
    axes[0].legend()
    suffixes = ("", "-no-block")
    keys = ["lambda1-minus-lambda095"]
    if result["original_defender_on_variant_confirmation_cohort"]:
        keys.extend(["lambda095-minus-original", "lambda1-minus-original"])
    labels = []
    for i, (key, suffix) in enumerate((k, s) for k in keys for s in suffixes):
        row = result["comparisons"][key + suffix]
        axes[1].errorbar(
            i, row["mean"], yerr=[[row["mean"] - row["low95"]], [row["high95"] - row["mean"]]], fmt="o", capsize=4
        )
        labels.append(key.replace("-minus-", " − ") + ("\nno Block" if suffix else ""))
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, ylabel="Paired Blue score gain, 95% interval")
    axes[1].tick_params(axis="x", labelrotation=40, labelsize=8)
    components = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
    axes[2].bar(
        ["Impact", "Local work", "Blocked access", "Action cost"],
        [result["means"]["warm-final"][k] - result["means"]["lambda095-final"][k] for k in components],
    )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set(title="lambda 1 minus lambda 0.95", ylabel="Reward-component gain")
    axes[2].tick_params(axis="x", labelrotation=20)
    fig.suptitle(f"Source {result['source_steps']:,} / seed {result['training_seed']}: one matched training pair")
    fig.savefig(output / "matched-pair.png", dpi=180)
    fig.savefig(output / "matched-pair.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Path)
    parser.add_argument("--credit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_pair(args.control, args.variant)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.credit:
        rows, corrected, error = audit_credit(args.credit)
        write_csv(args.output / "raw-credit-reproduction.csv", rows)
        write_csv(args.output / "credit-multiplicity.csv", corrected)
        result["conditional_estimates_reproduced"] = len(rows)
        result["lambda1_credit_identity_max_error"] = error
    write_csv(
        args.output / "paired-comparisons.csv", [{"comparison": k, **v} for k, v in result["comparisons"].items()]
    )
    write_csv(args.output / "policy-means.csv", [{"policy": k, **v} for k, v in result["means"].items()])
    result["inputs"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in (args.control, args.variant)
        for p in (d / "confirmation-episodes.json", d / "config.yaml", d / "manifest.json")
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.control, args.variant, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "training_seed": result["training_seed"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
