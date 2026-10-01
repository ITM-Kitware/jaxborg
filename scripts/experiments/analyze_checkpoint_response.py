"""Reproduce checkpoint diagnostics from portable exports without MLflow or JAX."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def paired(early, baseline, bootstrap):
    if early["per_episode_seeds"] != baseline["per_episode_seeds"]:
        raise ValueError("paired episode seeds differ")
    x = np.asarray(early["per_episode_blue_returns"], dtype=float)
    y = np.asarray(baseline["per_episode_blue_returns"], dtype=float)
    if x.shape != y.shape or len(x) < 2 or not np.all(np.isfinite(x - y)):
        raise ValueError("paired return arrays differ or are invalid")
    difference = x - y
    rng = np.random.default_rng(bootstrap["seed"])
    # Independently resample paired differences; do not call the campaign helper.
    draws = rng.choice(difference, size=(bootstrap["samples"], len(difference)), replace=True).mean(axis=1)
    ci = np.quantile(draws, [0.025, 0.975])
    simultaneous = np.quantile(draws, [0.05 / 12, 1 - 0.05 / 12])
    return {
        "early_mean": float(x.mean()),
        "baseline_mean": float(y.mean()),
        "early_minus_baseline": float(difference.mean()),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "bonferroni_six_low": float(simultaneous[0]),
        "bonferroni_six_high": float(simultaneous[1]),
        "episodes": len(difference),
    }


def derive(report):
    manifest = read_json(report / "manifest.json")
    config = manifest["config"]
    pairs, original_pairs, behavior_changes, points, all_ids = [], [], [], {}, set()
    trace_count = config["behavior_trace_episodes"]
    episode_count, traced_episodes = 0, 0
    for name, point in manifest["points"].items():
        folder = report / name
        results = read_json(folder / "evaluations.json")
        summary = read_json(folder / "summary.json")
        if read_json(folder / "completed.json")["status"] != "complete":
            raise ValueError("point did not complete")
        for split, values in results.items():
            spec = config["episodes"][split]
            expected = list(range(spec["seed_start"], spec["seed_start"] + spec["count"]))
            for value in values.values():
                blue = np.asarray(value["per_episode_blue_returns"])
                if value["per_episode_seeds"] != expected or len(blue) != spec["count"]:
                    raise ValueError("episode seed/count differs from protocol")
                if not np.array_equal(-blue, value["per_episode_red_returns"]):
                    raise ValueError("evaluation rewards are not zero sum")
                if not np.isclose(blue.mean(), value["blue_mean_return"]):
                    raise ValueError("saved evaluation mean differs")
                all_ids.add(value["eval_id"])
                episode_count += len(blue)
        validation = results["validation"]
        order = summary["selection"]["candidate_order"]
        if set(validation) != {c["name"] for c in point["candidates"]}:
            raise ValueError("validation candidate pool differs")
        selected = max(order, key=lambda c: validation[c]["blue_mean_return"])
        if selected != summary["selection"]["selected"]:
            raise ValueError("saved selection differs")
        confirmations = results["confirmation"]
        trace = read_json(folder / "behavior.json")
        if trace["trace_episodes"] != trace_count or any(
            not e["returns_match_canonical"] or e["max_return_difference"] != 0 for e in trace["evidence"]
        ):
            raise ValueError("behavior trace validation failed")
        traces = {c: [] for c in confirmations}
        for row in trace["episodes"]:
            traces[row["candidate"]].append(row)
        for candidate, rows in traces.items():
            reference = confirmations[candidate]
            if (
                len(rows) != trace_count
                or [r["episode_seed"] for r in rows] != reference["per_episode_seeds"][:trace_count]
            ):
                raise ValueError("trace episode seeds differ")
            if [r["blue_return"] for r in rows] != reference["per_episode_blue_returns"][:trace_count]:
                raise ValueError("trace returns differ")
            for row in rows:
                if row["illegal_actions"] or not np.isclose(
                    row["blue_return"], sum(row[k] for k in ("reward_ria", "reward_lwf", "reward_asf", "action_cost"))
                ):
                    raise ValueError("trace legality or reward accounting differs")
            traced_episodes += len(rows)
        behavior = {row["candidate"]: row for row in trace["summaries"]}
        steps = point["protocol"]["source"]["original_training_steps"]
        for seed in point["protocol"]["training_seeds"]:
            early_name, final_name = f"seed-{seed}-step-1920000", f"seed-{seed}-step-final"
            early, final = confirmations[early_name], confirmations[final_name]
            gap = paired(early, final, config["bootstrap"])
            saved = summary["early_vs_final"][str(seed)]
            if not np.allclose(
                [gap["early_minus_baseline"], gap["ci95_low"], gap["ci95_high"]],
                [saved["blue_improvement"], *saved["ci95"]],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("independent bootstrap differs from saved report")
            prefix = {"source_steps": steps, "training_seed": seed}
            pairs.append({**prefix, **gap})
            original_pairs.append({**prefix, **paired(early, confirmations["original"], config["bootstrap"])})
            before, after = behavior[early_name], behavior[final_name]
            change = {**prefix, "trace_episodes": trace_count}
            for key in (
                "blue_return",
                "reward_ria",
                "reward_lwf",
                "reward_asf",
                "action_cost",
                "green_asf_count",
                "mean_blocked_pairs",
                "block_idle_fraction",
                "allow_idle_fraction",
                "restore_idle_fraction",
            ):
                change.update(
                    {f"early_{key}": before[key], f"final_{key}": after[key], f"change_{key}": after[key] - before[key]}
                )
            behavior_changes.append(change)
        best_fresh = max((c for c in order if c != "original"), key=lambda c: validation[c]["blue_mean_return"])
        points[name] = {
            "selected": selected,
            "validation_original_mean": validation["original"]["blue_mean_return"],
            "validation_best_fresh": best_fresh,
            "validation_best_fresh_mean": validation[best_fresh]["blue_mean_return"],
            "confirmation_original_mean": confirmations["original"]["blue_mean_return"],
            "report_owner": summary["canonical_owner"],
            "behavior_owner": trace["owner"],
        }
    return {
        "diagnosis-summary.json": {
            "points": points,
            "evaluation_runs": len(all_ids),
            "canonical_episodes": episode_count,
            "paired_comparisons": len(pairs),
            "trace_episodes": traced_episodes,
            "independently_reproduced": True,
            "bootstrap": config["bootstrap"],
            "interval_scope": (
                "pointwise 95%; also six-comparison Bonferroni percentile intervals; "
                "conditional on checkpoints and one source seed/topology"
            ),
        },
        "paired-comparisons.csv": pairs,
        "early-versus-original.csv": original_pairs,
        "behavior-changes.csv": behavior_changes,
    }


def training_context(results_root):
    rows = []
    for algorithm in ("ippo", "mappo"):
        with gzip.open(results_root / f"analysis-data/source-training/{algorithm}/metrics.jsonl.gz", "rt") as stream:
            metrics = [json.loads(line) for line in stream]
        for endpoint in (9600000, 49968000):
            window = [r for r in metrics if r["env_steps"] <= endpoint][-20:]
            row = {"source_algorithm": algorithm, "source_steps": endpoint, "window_updates": len(window)}
            for label, key in (
                ("blue_return", "team.blue.return"),
                ("reward_asf", "team.blue.reward_asf"),
                ("reward_ria", "team.blue.reward_ria"),
                ("restore_cost", "team.blue.action_cost"),
                ("green_asf_count", "backend.jax.game.green_asf_count"),
            ):
                row[label] = float(np.mean([r[key] for r in window]))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true", help="verify saved derivations without rewriting")
    args = parser.parse_args()
    outputs = derive(args.report_dir)
    outputs["source-training-context.csv"] = training_context(args.report_dir.parent.parent)
    for filename, value in outputs.items():
        path = args.report_dir / filename
        if args.check:
            if filename.endswith(".json"):
                if read_json(path) != value:
                    raise ValueError(f"saved derivation differs: {filename}")
            else:
                saved = list(csv.DictReader(path.open()))
                if len(saved) != len(value):
                    raise ValueError(f"saved table length differs: {filename}")
                for a, b in zip(saved, value, strict=True):
                    if a != {k: str(v) for k, v in b.items()}:
                        raise ValueError(f"saved table differs: {filename}")
        elif filename.endswith(".json"):
            path.write_text(json.dumps(value, indent=2) + "\n")
        else:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(value[0]))
                writer.writeheader()
                writer.writerows(value)
    print(json.dumps(outputs["diagnosis-summary.json"], indent=2))


if __name__ == "__main__":
    main()
