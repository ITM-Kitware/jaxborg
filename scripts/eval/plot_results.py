#!/usr/bin/env python3
"""Plot standardized evaluation JSON/JSONL rows.

With no input arguments, all result files directly under
``$JAXBORG_EXP_DIR/eval`` are included.  The plot compares mean reward and
uses the recorded sample standard deviation as the error bar. This folder mode
is for development. Research mode requires a saved explicit collation and verifies
its complete local artifact closure before producing an immutable report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _decode_json_objects(text: str, *, source: Path) -> list[dict[str, Any]]:
    """Read both one-object-per-line JSONL and pretty-printed JSON objects."""

    decoder = json.JSONDecoder()
    offset = 0
    objects: list[dict[str, Any]] = []
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            value, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse evaluation results in {source}: {exc}") from exc
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, Mapping):
                raise ValueError(f"evaluation result in {source} must be a JSON object")
            objects.append(dict(item))
    return objects


def result_files(inputs: Sequence[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.update(candidate for candidate in path.glob("*.jsonl") if candidate.is_file())
            files.update(candidate for candidate in path.glob("*.json") if candidate.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"evaluation result path not found: {path}")
    return sorted(files)


def load_result_rows(inputs: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in result_files(inputs):
        # Manifests describe execution, not evaluation scores.
        if path.parent.name == "manifests":
            continue
        for row in _decode_json_objects(path.read_text(), source=path):
            if "mean_reward" not in row:
                continue
            row["_source_path"] = str(path)
            rows.append(row)
    return rows


def _row_label(row: Mapping[str, Any]) -> str:
    if row.get("evaluation_id"):
        identity = str(row["evaluation_id"]).rsplit(":", 1)[-1]
        sampling = f"n={row['n_episodes']}"
        if row["std_reward"] is None:
            sampling += "; SD unavailable"
        return f"{row['condition_id']} / {row['replicate_id']}\nevaluation {identity[:12]}…; {sampling}"
    model_value = row.get("model")
    if not model_value and isinstance(row.get("policies"), Mapping):
        model_value = row["policies"].get("blue", {}).get("path")
    model = Path(str(model_value or "unknown-model")).stem
    evaluation = row.get("eval_name") or row.get("suite") or row.get("eval_env") or "evaluation"
    condition = row.get("eval_red") or row.get("variant")
    detail = f"{evaluation} / {condition}" if condition and condition != evaluation else str(evaluation)
    return f"{model}\n{detail}"


def _numeric_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        try:
            mean_reward = float(row["mean_reward"])
            value = row.get("std_reward")
            std_reward = None if value is None or row.get("n_episodes") == 1 else max(0.0, float(value))
        except (KeyError, TypeError, ValueError):
            continue
        selected.append({**row, "mean_reward": mean_reward, "std_reward": std_reward})
    return selected


def plot_results(rows: Sequence[Mapping[str, Any]], output: str | Path, *, show: bool = False) -> Path:
    selected = _numeric_rows(rows)
    if not selected:
        raise ValueError("no plottable evaluation rows with mean_reward were found")

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [_row_label(row) for row in selected]
    means = [row["mean_reward"] for row in selected]
    positions = list(range(len(selected)))

    figure, axis = plt.subplots(figsize=(11, max(4.0, 0.65 * len(selected) + 1.5)))
    axis.barh(positions, means, alpha=0.85)
    estimated = [index for index, row in enumerate(selected) if row["std_reward"] is not None]
    if estimated:
        axis.errorbar(
            [means[index] for index in estimated],
            estimated,
            xerr=[selected[index]["std_reward"] for index in estimated],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Mean reward (error bars: sample SD where available)")
    axis.set_title("Evaluation results")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


REPORT_CONFIGURATION = {
    "estimator": "arithmetic mean per selected evaluation",
    "uncertainty": "sample episode standard deviation for n >= 2; unavailable for n = 1; not a confidence interval",
    "sampling_unit": "episode conditional on fixed policies",
    "weighting": "equal episodes within each evaluation; no pooling across evaluations",
}


def analysis_sources(extra=None):
    """Identify the analyzer, dependency specification and lock, and study checks."""
    import jaxborg.experiment_records
    import jaxborg.experiment_records.core

    project = Path(__file__).resolve().parents[2]
    files = {"scripts/eval/plot_results.py": Path(__file__).resolve()}
    files.update({name: project / name for name in ("pyproject.toml", "uv.lock")})
    for module in (jaxborg, jaxborg.experiment_records, jaxborg.experiment_records.core):
        suffix = "/__init__.py" if Path(module.__file__).name == "__init__.py" else ".py"
        files["src/" + module.__name__.replace(".", "/") + suffix] = Path(module.__file__).resolve()
    for name, path in (extra or {}).items():
        if name in files:
            raise ValueError(f"Duplicate analysis source: {name}")
        files[name] = Path(path).resolve()
    return files


def analysis_identity(extra=None):
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in analysis_sources(extra).items()}
    return "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _relative_file(root, name):
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or ":" in name:
        raise ValueError(f"Evidence requires a safe relative artifact path: {name}")
    target = root / path
    for component in (target, *target.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"Evidence must not follow symbolic links: {name}")
    return target


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _file_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = os.fstat(stream.fileno()).st_size
    return {"sha256": digest, "size_bytes": size}


def _copy_evidence(records, source, destination):
    copied = {}
    for record in records:
        if record["kind"] != "artifact":
            continue
        name = record["uri"]
        expected = {key: record[key] for key in ("sha256", "size_bytes")}
        if name in copied:
            if copied[name] != expected:
                raise ValueError("Conflicting artifact locators")
            continue
        path = _relative_file(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _relative_file(source, name).open("rb") as reader, path.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        if _file_identity(path) != expected:
            raise ValueError(f"Artifact bytes differ: {record['id']}")
        copied[name] = expected


def write_collation_report(collation_path, evidence_root, output, *, configuration=None, source_files=None, check=None):
    """Plot an immutable selection from a temporary verified evidence snapshot.

    Optional study checks receive the validated collation and detached episode
    rows before any output is exposed. They may add scientific columns to rows.
    """
    from jaxborg.experiment_records import validate_collation

    collation_path, evidence_root, output = (
        Path(p).expanduser().resolve() for p in (collation_path, evidence_root, output)
    )
    root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(root) or output.is_relative_to(evidence_root) or evidence_root.is_relative_to(output):
        raise ValueError("Report output must be outside the checkout and evidence input")
    if output.exists():
        raise FileExistsError(output)
    collation = validate_collation(json.loads(collation_path.read_text()))
    configuration = REPORT_CONFIGURATION if configuration is None else configuration
    expected_analysis = {"source": analysis_identity(source_files), "configuration": configuration}
    if collation["analysis"] != expected_analysis:
        raise ValueError("Collation analysis source or resolved configuration differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".report-", dir=output.parent) as temporary:
        destination = Path(temporary) / "report"
        destination.mkdir()
        evidence = Path(temporary) / "evidence"
        _copy_evidence(collation["records"], evidence_root, evidence)
        indexed = {record["id"]: record for record in collation["records"]}
        for identity in collation["included"]:
            record = indexed[identity]
            for fact_name in ("resolved_config", "source_files"):
                fact = record["facts"].get(fact_name)
                if fact is None:
                    continue
                backing = [indexed[key] for key in record["inputs"] if indexed[key]["uri"] == fact["source"]]
                if (
                    len(backing) != 1
                    or json.loads(_relative_file(evidence, backing[0]["uri"]).read_text()) != fact["value"]
                ):
                    raise ValueError(f"Selected {fact_name} lacks matching retained input bytes")
                if fact_name == "source_files":
                    source_digest = hashlib.sha256(
                        json.dumps(fact["value"], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    if "source_identity" in record["facts"] and record["facts"]["source_identity"]["value"] not in {
                        source_digest,
                        "sha256:" + source_digest,
                    }:
                        raise ValueError("Selected source identity differs from the retained source manifest")
                    for name, digest in fact["value"].items():
                        matches = [
                            indexed[key]
                            for key in record["inputs"]
                            if indexed[key]["uri"].endswith("/source/" + name) and indexed[key]["sha256"] == digest
                        ]
                        if len(matches) != 1:
                            raise ValueError("Selected source manifest lacks matching retained source bytes")
        rows = []
        for selected in collation["rows"]:
            if selected["evaluation_id"] is None:
                continue
            evaluation = indexed[selected["evaluation_id"]]
            observations = []
            for identity in evaluation["raw_artifacts"]:
                artifact = indexed[identity]
                if artifact["role"] == "evaluation episode observation":
                    if artifact["producer_id"] != evaluation["id"]:
                        raise ValueError("Episode observation belongs to another evaluation")
                    observations.append(json.loads(_relative_file(evidence, artifact["uri"]).read_text()))
            observations.sort(key=lambda row: row["episode_index"])
            budgets = [budget for budget in evaluation["budgets"] if budget["unit"] == "episodes"]
            if (
                len(budgets) != 1
                or budgets[0]["requested"] != budgets[0]["completed"]
                or budgets[0]["completed"] != len(observations)
            ):
                raise ValueError("Selected evaluation lacks its completed episode budget")
            if not observations or [row["episode_index"] for row in observations] != list(range(len(observations))):
                raise ValueError("Episode observations must cover consecutive unique indices")
            rewards = [row["reward"] for row in observations]
            if any(
                isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                for value in rewards
            ):
                raise ValueError("Episode rewards must be finite numbers")
            mean = statistics.mean(rewards)
            metrics = [metric for metric in evaluation["metrics"] if metric["name"] == "blue_return"]
            if len(metrics) != 1 or metrics[0]["unit"] != "score" or metrics[0]["value"] != mean:
                raise ValueError("Blue return metric differs from episode observations")
            definitions = [metric for metric in collation["metric_definitions"] if metric["name"] == "blue_return"]
            if (
                len(definitions) != 1
                or definitions[0]["aggregation"] not in {"mean", "arithmetic mean"}
                or definitions[0]["unit"] != "score"
            ):
                raise ValueError("Protocol must define arithmetic mean Blue return in score units")
            rows.append(
                {
                    **selected,
                    "mean_reward": mean,
                    "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else None,
                    "n_episodes": len(rewards),
                    "episodes": observations,
                    "correction": evaluation.get("extensions", {}).get("evaluation_correction"),
                }
            )
        if not rows:
            raise ValueError("Collation has no selected episode evaluations")
        if check is not None:
            check(collation, rows)
        _json(destination / "collation.json", collation)
        _json(destination / "joined.json", rows)
        _json(
            destination / "selection-status.json", {"excluded": collation["excluded"], "missing": collation["missing"]}
        )
        table = [
            {key: value for key, value in row.items() if key not in {"episodes", "metrics", "correction"}}
            for row in rows
        ]
        with (destination / "joined.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(table[0]))
            writer.writeheader()
            writer.writerows(table)
        for name, source in analysis_sources(source_files).items():
            path = _relative_file(destination / "analysis-source", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
        plot_results(rows, destination / "evaluation.png")
        files = {
            path.relative_to(destination).as_posix(): _file_identity(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        _json(
            destination / "manifest.json",
            {
                "schema_version": 2,
                "collation_id": collation["id"],
                "analysis": expected_analysis,
                "files": files,
                "corrections": {row["evaluation_id"]: row["correction"] for row in rows if row["correction"]},
            },
        )
        if analysis_identity(source_files) != expected_analysis["source"]:
            raise ValueError("Analysis source changed during report generation")
        if output.exists():
            raise FileExistsError(output)
        os.rename(destination, output)
    return output


def _print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    print(f"{'evaluation':<32} {'n':>7} {'mean':>12} {'std':>12}  source")
    for row in _numeric_rows(rows):
        name = str(row.get("eval_name") or row.get("suite") or row.get("eval_env") or "evaluation")
        condition = row.get("eval_red") or row.get("variant")
        if condition:
            name = f"{name}/{condition}"
        deviation = "unavailable" if row["std_reward"] is None else f"{row['std_reward']:.2f}"
        print(
            f"{name:<32} {int(row.get('n_episodes', 0)):>7} "
            f"{row['mean_reward']:>12.2f} {deviation:>12}  {row.get('_source_path', '')}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).expanduser().resolve()
    parser = argparse.ArgumentParser(description="Plot jaxborg evaluation JSON/JSONL results")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Result file or directory; repeat to combine inputs (default: $JAXBORG_EXP_DIR/eval)",
    )
    parser.add_argument("--collation", type=Path, help="Research mode: immutable explicit collation")
    parser.add_argument("--evidence-root", type=Path, help="Root for the collation artifact paths")
    parser.add_argument("--recipe", help="Only include rows with this recipe_name")
    parser.add_argument("--name", help="Only include rows with this eval_name")
    parser.add_argument("--output", help="PNG path (default: $JAXBORG_EXP_DIR/eval/plots/<timestamp>.png)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive matplotlib window")
    args = parser.parse_args(argv)

    if args.collation is not None:
        if args.evidence_root is None or args.output is None or args.input or args.recipe or args.name or args.show:
            parser.error(
                "--collation requires --evidence-root and a new --output directory; "
                "input/filter/show options are development-only"
            )
        print(f"Wrote report: {write_collation_report(args.collation, args.evidence_root, args.output)}")
        return
    if args.evidence_root is not None:
        parser.error("--evidence-root requires --collation")
    inputs = args.input or [str(exp_dir / "eval")]
    rows = load_result_rows(inputs)
    if args.recipe:
        rows = [row for row in rows if row.get("recipe_name") == args.recipe]
    if args.name:
        rows = [row for row in rows if row.get("eval_name") == args.name]
    if not rows:
        raise SystemExit("No matching evaluation result rows found.")

    _print_rows(rows)
    output = args.output or exp_dir / "eval" / "plots" / f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.png"
    output_path = plot_results(rows, output, show=args.show)
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
