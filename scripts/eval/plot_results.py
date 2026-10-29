#!/usr/bin/env python3
"""Plot standardized evaluation JSON/JSONL rows.

With no input arguments, all result files directly under
``$JAXBORG_EXP_DIR/eval`` are included.  The plot compares mean reward and
uses the recorded sample standard deviation as the error bar.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
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
            std_reward = float(row.get("std_reward", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        selected.append({**row, "mean_reward": mean_reward, "std_reward": max(0.0, std_reward)})
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
    standard_deviations = [row["std_reward"] for row in selected]
    positions = list(range(len(selected)))

    figure, axis = plt.subplots(figsize=(11, max(4.0, 0.65 * len(selected) + 1.5)))
    axis.barh(positions, means, xerr=standard_deviations, alpha=0.85, capsize=3)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Mean reward (error bar: sample standard deviation)")
    axis.set_title("Evaluation results")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def _print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    print(f"{'evaluation':<32} {'n':>7} {'mean':>12} {'std':>12}  source")
    for row in _numeric_rows(rows):
        name = str(row.get("eval_name") or row.get("suite") or row.get("eval_env") or "evaluation")
        condition = row.get("eval_red") or row.get("variant")
        if condition:
            name = f"{name}/{condition}"
        print(
            f"{name:<32} {int(row.get('n_episodes', 0)):>7} "
            f"{row['mean_reward']:>12.2f} {row['std_reward']:>12.2f}  {row.get('_source_path', '')}"
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
    parser.add_argument("--recipe", help="Only include rows with this recipe_name")
    parser.add_argument("--name", help="Only include rows with this eval_name")
    parser.add_argument("--output", help="PNG path (default: $JAXBORG_EXP_DIR/eval/plots/<timestamp>.png)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive matplotlib window")
    args = parser.parse_args(argv)

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
