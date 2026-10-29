from __future__ import annotations

import json

from scripts.eval.plot_results import load_result_rows, plot_results


def test_loads_pretty_json_and_jsonl_rows_while_ignoring_non_score_json(tmp_path):
    pretty = tmp_path / "pretty.jsonl"
    pretty.write_text(json.dumps({"eval_name": "one", "mean_reward": 3.0, "std_reward": 0.5}, indent=2))
    jsonl = tmp_path / "suite.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"eval_name": "two", "eval_red": "fsm", "mean_reward": 4.0}),
                json.dumps({"eval_name": "two", "eval_red": "cia_c", "mean_reward": 5.0}),
            ]
        )
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"evaluations": []}))

    rows = load_result_rows([tmp_path])

    assert [row["mean_reward"] for row in rows] == [3.0, 4.0, 5.0]
    assert all("_source_path" in row for row in rows)


def test_plot_results_writes_png(tmp_path):
    output = plot_results(
        [
            {
                "model": "/models/model_a.safetensors",
                "eval_name": "stochastic",
                "eval_red": "fsm",
                "mean_reward": -12.0,
                "std_reward": 2.0,
            },
            {
                "model": "/models/model_a.safetensors",
                "eval_name": "deterministic",
                "eval_red": "fsm",
                "mean_reward": -8.0,
                "std_reward": 1.0,
            },
        ],
        tmp_path / "plot.png",
    )

    assert output.is_file()
    assert output.stat().st_size > 0
