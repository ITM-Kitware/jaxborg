"""CPU-only paired analysis of a bounded actor-GAE x critic-target experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.experiments.analyze_blue_credit_replication import same_numeric_state
from scripts.experiments.analyze_blue_learning_mechanism import paired, write_csv

ARMS = {"a095-c095": "control", "a100-c100": "lambda1", "a095-c100": "a095-c100", "a100-c095": "a100-c095"}
LAMBDAS = {"a095-c095": (0.95, 0.95), "a100-c100": (1.0, 1.0), "a095-c100": (0.95, 1.0), "a100-c095": (1.0, 0.95)}


def read(path):
    return json.loads(path.read_text())


def audit_arms(root):
    baseline = root / "control"
    base_config = read(baseline / "effective-config.json")
    base_protocol = yaml.safe_load((baseline / "config.yaml").read_text())
    remove = {"GAE_LAMBDA", "ACTOR_GAE_LAMBDA", "CRITIC_TARGET_LAMBDA"}
    facts = {}
    for label, relative in ARMS.items():
        p = root / relative
        manifest = read(p / "manifest.json")
        if manifest["status"] != "FINISHED":
            raise ValueError(f"arm is incomplete: {label}")
        config = read(p / "effective-config.json")
        protocol = yaml.safe_load((p / "config.yaml").read_text())
        for field in ("source_steps", "training_seed", "warm_updates", "normalization_calibration_updates"):
            if protocol[field] != base_protocol[field]:
                raise ValueError(f"unmatched {field}: {label}")
        if protocol["seeds"]["warm_rollout"] != base_protocol["seeds"]["warm_rollout"]:
            raise ValueError("unmatched training randomness")
        for team in base_config:
            if {k: v for k, v in config[team].items() if k not in remove} != {
                k: v for k, v in base_config[team].items() if k not in remove
            }:
                raise ValueError(f"non-credit setting differs: {label}/{team}")
            actual = (
                config[team].get("ACTOR_GAE_LAMBDA", config[team]["GAE_LAMBDA"]),
                config[team].get("CRITIC_TARGET_LAMBDA", config[team]["GAE_LAMBDA"]),
            )
            if actual != LAMBDAS[label]:
                raise ValueError(f"wrong actor/critic lambdas: {label}")
        same_numeric_state(baseline, p)
        metrics = read(p / "training-metrics.json")
        steps = config["blue"]["NUM_ENVS"] * config["blue"]["NUM_STEPS"]
        if [r["update"] for r in metrics] != list(range(1, base_protocol["warm_updates"] + 1)) or any(
            r["warm_steps"] != r["update"] * steps for r in metrics
        ):
            raise ValueError(f"wrong training budget: {label}")
        if manifest["actual_steps"] != base_protocol["warm_updates"] * steps:
            raise ValueError(f"manifest training budget differs: {label}")
        gradients = read(p / "gradient-components.json")
        if any(r["reference_max_parameter_error"] > 2e-6 for r in gradients):
            raise ValueError("reference PPO composition mismatch")
        facts[label] = {
            "run_id": manifest["run_id"],
            "source_revision": manifest["source"]["git_commit"],
            "actual_steps": manifest["actual_steps"],
            "max_reference_parameter_error": max(r["reference_max_parameter_error"] for r in gradients),
        }
    return facts


def contrast(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "low95": float(lo), "high95": float(hi), "n": len(values)}


def analyze(root, evaluation, output):
    facts = audit_arms(root)
    protocol = yaml.safe_load((evaluation / "config.yaml").read_text())
    episodes = read(evaluation / "per-episode.json")
    seeds = list(
        range(
            protocol["seeds"]["confirmation_start"],
            protocol["seeds"]["confirmation_start"] + protocol["confirmation_episodes"],
        )
    )
    labels = [*ARMS, "initial", "original"]
    if set(episodes) != {label + suffix for label in labels for suffix in ("", "-no-block")}:
        raise ValueError("missing factorial evaluation policy or control")
    for label, rows in episodes.items():
        if [r["seed"] for r in rows] != seeds:
            raise ValueError("factorial confirmation cohort differs from registration")
        for row in rows:
            if (
                row["blue_return"] != sum(row[k] for k in ("reward_ria", "reward_lwf", "reward_asf", "action_cost"))
                or row["illegal_actions"]
            ):
                raise ValueError("canonical score or action contract failed")
            if label.endswith("-no-block") and (row["block"] or row["reward_asf"]):
                raise ValueError("Block-exclusion control failed")
    estimates, factorial = {}, {}
    seed = protocol["seeds"]["bootstrap"]
    for suffix in ("", "-no-block"):
        for label in ARMS:
            for baseline in ("initial", "original", "a095-c095"):
                estimates[label + "-minus-" + baseline + suffix] = paired(
                    episodes[baseline + suffix], episodes[label + suffix], seed=seed
                )
        arrays = {label: np.array([r["blue_return"] for r in episodes[label + suffix]]) for label in ARMS}
        factorial["actor-main-effect" + suffix] = contrast(
            (arrays["a100-c095"] + arrays["a100-c100"] - arrays["a095-c095"] - arrays["a095-c100"]) / 2, seed
        )
        factorial["critic-main-effect" + suffix] = contrast(
            (arrays["a095-c100"] + arrays["a100-c100"] - arrays["a095-c095"] - arrays["a100-c095"]) / 2, seed
        )
        factorial["interaction" + suffix] = contrast(
            arrays["a100-c100"] - arrays["a095-c100"] - arrays["a100-c095"] + arrays["a095-c095"], seed
        )
        factorial["actor-effect-at-critic095" + suffix] = paired(
            episodes["a095-c095" + suffix], episodes["a100-c095" + suffix], seed=seed
        )
        factorial["actor-effect-at-critic100" + suffix] = paired(
            episodes["a095-c100" + suffix], episodes["a100-c100" + suffix], seed=seed
        )
        factorial["critic-effect-at-actor095" + suffix] = paired(
            episodes["a095-c095" + suffix], episodes["a095-c100" + suffix], seed=seed
        )
        factorial["critic-effect-at-actor100" + suffix] = paired(
            episodes["a100-c095" + suffix], episodes["a100-c100" + suffix], seed=seed
        )
    means = {
        label: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
        for label, rows in episodes.items()
    }
    result = {
        "status": "VERIFIED_FACTORIAL",
        "training_seed": protocol["training_seed"],
        "independent_training_seeds_in_factorial": 1,
        "scope": (
            "conditional differences between four trained policies; "
            "episode intervals are not uncertainty over training replications"
        ),
        "confirmation_seeds": seeds,
        "calibrated_state_exact": True,
        "arms": facts,
        "comparisons": estimates,
        "factorial_effects": factorial,
        "means": means,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output / "paired-comparisons.csv", [{"comparison": k, **v} for k, v in estimates.items()])
    write_csv(output / "factorial-effects.csv", [{"contrast": k, **v} for k, v in factorial.items()])
    write_csv(output / "policy-means.csv", [{"policy": k, **v} for k, v in means.items()])
    write_csv(
        output / "learning-signals.csv",
        [
            {"arm": label, **row}
            for label, directory in ARMS.items()
            for row in read(root / directory / "learning-signals.json")
        ],
    )
    plot(root, result, output)
    return result


def plot(root, result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for label, directory in ARMS.items():
        rows = read(root / directory / "training-metrics.json")
        axes[0].plot(
            [r["warm_steps"] / 1e6 for r in rows], [r["blue"]["raw_rollout_return"] for r in rows], label=label
        )
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel="Additional training, M steps", ylabel="Raw training return")
    for i, label in enumerate(ARMS):
        row = result["comparisons"][label + "-minus-original"]
        axes[1].errorbar(
            i, row["mean"], yerr=[[row["mean"] - row["low95"]], [row["high95"] - row["mean"]]], fmt="o", capsize=3
        )
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(xticks=range(4), xticklabels=list(ARMS), ylabel="Gain over original Blue, 95% interval")
    for i, key in enumerate(("actor-main-effect", "critic-main-effect", "interaction")):
        row = result["factorial_effects"][key]
        axes[2].errorbar(
            i, row["mean"], yerr=[[row["mean"] - row["low95"]], [row["high95"] - row["mean"]]], fmt="o", capsize=3
        )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set(
        xticks=range(3),
        xticklabels=["Actor", "Critic", "Interaction"],
        ylabel="Conditional factorial contrast, 95% interval",
    )
    for ax in axes[1:]:
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Blue credit factorial: one warm-start training seed, gamma .99")
    fig.savefig(output / "factorial.png", dpi=180)
    fig.savefig(output / "factorial.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.study_root, args.evaluation, args.output)["factorial_effects"], indent=2))


if __name__ == "__main__":
    main()
