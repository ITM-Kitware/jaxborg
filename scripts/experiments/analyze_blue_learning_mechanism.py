"""Reproduce summaries and plots from portable traffic-learning exports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired(a, b, seed=6500001):
    if [x["seed"] for x in a] != [x["seed"] for x in b]:
        raise ValueError("unpaired episodes")
    d = np.array([y["blue_return"] - x["blue_return"] for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    ci = np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [0.025, 0.975])
    return {"mean": float(d.mean()), "low95": float(ci[0]), "high95": float(ci[1]), "n": len(d)}


def analyze_credit(root, out=None):
    out = out or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    results = read(root / "results.json")
    rows = []
    reversals = []
    for result in results:
        for comparison, measures in result["summaries"].items():
            for metric, estimate in measures.items():
                rows.append(
                    {
                        "cohort": result["cohort"],
                        "fork": result["fork"],
                        "group": result["group"],
                        "tick": result["tick"],
                        "comparison": comparison,
                        "metric": metric,
                        "mean": estimate["mean"],
                        "low95": estimate["ci95"][0],
                        "high95": estimate["ci95"][1],
                        "n": estimate["n"],
                    }
                )
        measures = result["summaries"]["opposite-minus-natural"]
        mc, gae = measures["normalized_mc_advantage"], measures["gae_advantage"]
        if mc["mean"] * gae["mean"] < 0:
            reversals.append(
                {
                    "cohort": result["cohort"],
                    "fork": result["fork"],
                    "group": result["group"],
                    "monte_carlo": mc,
                    "gae": gae,
                    "both_intervals_exclude_zero": (
                        mc["ci95"][0] * mc["ci95"][1] > 0 and gae["ci95"][0] * gae["ci95"][1] > 0
                    ),
                }
            )
    write_csv(out / "credit-comparisons.csv", rows)
    (out / "credit-ranking-reversals.json").write_text(json.dumps(reversals, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, cohort in zip(axes, ("discovery", "confirmation")):
        for result in (r for r in results if r["cohort"] == cohort):
            measures = result["summaries"]["opposite-minus-natural"]
            mc, gae = measures["normalized_mc_advantage"], measures["gae_advantage"]
            color = "tab:red" if result["group"] == "harmful_new_block" else "tab:blue"
            ax.errorbar(
                mc["mean"],
                gae["mean"],
                xerr=[[mc["mean"] - mc["ci95"][0]], [mc["ci95"][1] - mc["mean"]]],
                yerr=[[gae["mean"] - gae["ci95"][0]], [gae["ci95"][1] - gae["mean"]]],
                fmt="o",
                color=color,
                alpha=0.8,
            )
            ax.annotate(
                str(result["fork"]), (mc["mean"], gae["mean"]), xytext=(4, 4), textcoords="offset points", fontsize=8
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set(
            title=cohort.title() + ": 32 new common futures per state",
            xlabel="Alternative − natural normalized Monte Carlo credit",
            ylabel="Alternative − natural GAE credit (lambda 0.95)",
        )
    fig.suptitle("Conditional action ranking: red sampled Blocks, blue sampled Allows")
    fig.savefig(out / "credit-ranking.png", dpi=180)
    fig.savefig(out / "credit-ranking.pdf")
    plt.close(fig)
    print(json.dumps({"states_per_cohort": len(results) // 2, "ranking_reversals": reversals}, indent=2))


def analyze(root, out=None):
    out = out or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    metrics = read(root / "training-metrics.json")
    write_csv(
        out / "training.csv", [{"update": r["update"], "warm_steps": r["warm_steps"], **r["blue"]} for r in metrics]
    )
    gradients = read(root / "gradient-components.json")
    gradrows = []
    for r in gradients:
        controls = {c["component"]: c for c in r["components"]}
        for c in r["components"]:
            row = {
                "update": r["update"],
                "component": c["component"],
                "gradient_norm": c["gradient_norm"],
                "sgd_harmful_direction": c["sgd_harmful_direction"],
                "reference_max_parameter_error": r["reference_max_parameter_error"],
                "gradient_additivity_error": r["gradient_additivity_error"],
            }
            for k, v in c["change"].items():
                row[k + "/delta"] = v
                row[k + "/delta_minus_zero"] = v - controls["zero"]["change"][k]
            gradrows.append(row)
    write_csv(out / "gradient-attribution.csv", gradrows)
    result["canonical_update_max_error"] = max(r["reference_max_parameter_error"] for r in gradients)
    result["gradient_additivity_max_error"] = max(r["gradient_additivity_error"] for r in gradients)
    result["gradient_components"] = gradrows
    full_updates = []
    for r in gradients:
        after = read(root / f"captures/update-{r['update']}-post-full-probabilities.json")
        before = r["components"][0]["before"]
        full_updates.append(
            {
                "update": r["update"],
                **{k + "/before": v for k, v in before.items()},
                **{k + "/after": v for k, v in after.items()},
            }
        )
    write_csv(out / "full-update-probabilities.csv", full_updates)
    result["full_update_probabilities"] = full_updates
    signals = read(root / "learning-signals.json")
    write_csv(out / "learning-signals.csv", signals)
    result["phase2_signals"] = [r for r in signals if r["phase"] == 2]
    evaluations = read(root / "validation-episodes.json")
    evalrows = []
    for label, episodes in evaluations.items():
        base = {"label": label, "n": len(episodes)}
        for k in episodes[0]:
            if k != "seed":
                base[k] = float(np.mean([x[k] for x in episodes]))
        evalrows.append(base)
    write_csv(out / "validation.csv", evalrows)
    result["validation"] = evalrows
    if (root / "confirmation-episodes.json").exists():
        confirmation = read(root / "confirmation-episodes.json")
        result["confirmation"] = {}
        for s in ("", "-no-block"):
            result["confirmation"]["warm-final-minus-initial" + s] = paired(
                confirmation["warm-0" + s], confirmation["warm-final" + s]
            )
    if (root / "fork-results.json").exists():
        forks = read(root / "fork-results.json")
        forkrows = []
        for i, r in enumerate(forks):
            for measure in ("discounted", "undiscounted"):
                for comparator in ("opposite", "sleep"):
                    diffs = np.asarray(r["results"][comparator][measure]) - np.asarray(r["results"]["natural"][measure])
                    rng = np.random.default_rng(6500001)
                    means = diffs.sum(-1)[rng.integers(0, len(diffs), (10000, len(diffs)))].mean(1)
                    ci = np.quantile(means, [0.025, 0.975])
                    forkrows.append(
                        {
                            "fork": i,
                            "group": r["group"],
                            "tick": r["tick"],
                            "env": r["env"],
                            "agent": r["agent"],
                            "ppo_advantage": r["ppo_advantage"],
                            "raw_gae": r["raw_gae"],
                            "measure": measure,
                            "comparator": comparator,
                            "mean": float(diffs.sum(-1).mean()),
                            "low95": float(ci[0]),
                            "high95": float(ci[1]),
                            **{
                                k: float(diffs[:, j].mean())
                                for j, k in enumerate(("reward_ria", "reward_lwf", "reward_asf", "action_cost"))
                            },
                        }
                    )
        write_csv(out / "fork-comparisons.csv", forkrows)
        result["fork_comparisons"] = forkrows
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(root, out, metrics, gradrows, evalrows, signals, result.get("fork_comparisons", []))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("gradient_components", "fork_comparisons", "phase2_signals")},
            indent=2,
        )
    )


def plot(root, out, metrics, gradrows, evalrows, signals, forkrows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = [r["warm_steps"] / 1e6 for r in metrics]
    for name in ("raw_rollout_return", "reward_asf", "reward_lwf", "action_cost"):
        axs[0, 0].plot(x, [r["blue"][name] for r in metrics], label=name)
    axs[0, 0].legend()
    axs[0, 0].set(
        xlabel="Additional warm-start training, M steps",
        ylabel="Raw training score",
        title="Training and score components",
    )
    for suffix in ("", "-no-block"):
        rows = [
            r
            for r in evalrows
            if r["label"].startswith("warm-")
            and r["label"].endswith(suffix)
            and (suffix or not r["label"].endswith("-no-block"))
        ]
        rows.sort(key=lambda r: int(r["label"].split("-")[1]))
        axs[0, 1].plot(
            [int(r["label"].split("-")[1]) * 0.048 for r in rows],
            [r["blue_return"] for r in rows],
            marker="o",
            label="Block excluded" if suffix else "Unmodified",
        )
    axs[0, 1].set(
        xlabel="Additional warm-start training, M steps", ylabel="Blue return", title="32 paired validation episodes"
    )
    axs[0, 1].legend()
    for component in ("actor", "critic", "entropy", "full"):
        rows = [r for r in gradrows if r["component"] == component]
        axs[1, 0].plot(
            [r["update"] * 0.048 for r in rows],
            [r["phase2/harmful_new_block/delta_minus_zero"] for r in rows],
            marker="o",
            label=component,
        )
    axs[1, 0].axhline(0, color="black", linewidth=0.5)
    axs[1, 0].set(
        xlabel="Additional warm-start training, M steps",
        ylabel="Change in harmful Block probability",
        title="Same minibatch and Adam state; zero-gradient control subtracted",
    )
    axs[1, 0].legend()
    rows = [r for r in forkrows if r["comparator"] == "opposite" and r["measure"] == "discounted"]
    if rows:
        for group, color, label in (
            ("harmful_new_block", "tab:red", "Sampled Block"),
            ("useful_allow", "tab:blue", "Sampled Allow"),
        ):
            selected = [r for r in rows if r["group"] == group]
            axs[1, 1].scatter(
                [r["ppo_advantage"] for r in selected], [r["mean"] for r in selected], color=color, label=label
            )
        axs[1, 1].legend()
        axs[1, 1].axhline(0, color="black", linewidth=0.5)
        axs[1, 1].axvline(0, color="black", linewidth=0.5)
    axs[1, 1].set(
        xlabel="PPO normalized advantage of sampled traffic action",
        ylabel="Discounted alternative minus natural return",
        title="32 common-future forks per captured state",
    )
    fig.suptitle("Blue traffic learning: source 9.6M / seed 11001 / reset optimizer warm start")
    fig.savefig(out / "mechanism.png", dpi=180)
    fig.savefig(out / "mechanism.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--credit", action="store_true", help="summarize a captured-state credit replay")
    a = p.parse_args()
    (analyze_credit if a.credit else analyze)(a.directory, a.output)
