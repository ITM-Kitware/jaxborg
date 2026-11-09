"""Freeze compact numeric evidence from the maintained behavior analyzer output."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan_path = args.check / "plan.json"
    summary_path = args.check / "analysis/summary.json"
    plan = json.loads(plan_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["plan_sha256"] != digest(plan_path) or summary["missing_cells"]:
        raise ValueError("Complete frozen panel required")
    episodes = [
        {
            key: e[key]
            for key in (
                "cell",
                "checkpoint_steps",
                "opponent",
                "case",
                "trace_artifact",
                "trace_sha256",
                "totals",
                "chosen_counts",
                "completed_counts",
                "idle_decisions_with_available_type",
                "host_coverage_of_idle_opportunities",
            )
        }
        for e in summary["episodes"]
    ]
    cells = []
    for cell in plan["cells"]:
        receipt_path = args.check / "cells" / cell["id"] / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if not receipt["canonical_parity"]:
            raise ValueError("Canonical parity failed")
        selected = [e for e in episodes if e["cell"] == cell["id"]]
        totals, chosen = Counter(), Counter()
        for e in selected:
            totals.update(e["totals"])
            chosen.update(e["chosen_counts"])
        cells.append(
            {
                "cell": cell["id"],
                "evaluation_id": receipt["evaluation_id"],
                "receipt_sha256": digest(receipt_path),
                "episodes": len(selected),
                "checkpoint": cell["checkpoint"],
                "canonical_parity": True,
                "pooled_totals": dict(totals),
                "mean_totals_per_episode": {k: v / len(selected) for k, v in totals.items()},
                "pooled_chosen_counts": dict(chosen),
                "chosen_share_of_genuine_idle_decisions": {k: v / totals["idle_decisions"] for k, v in chosen.items()},
            }
        )
    output = {
        "schema_version": 1,
        "evaluation_source_revision": plan["source_revision"],
        "training_source_revision": plan["protocol"]["training_source_revision"],
        "check_root": str(args.check.resolve()),
        "plan_sha256": digest(plan_path),
        "maintained_summary_sha256": digest(summary_path),
        "builder_sha256": digest(__file__),
        "analysis_source_sha256": summary["analyzer_sha256"],
        "protocol": plan["protocol"],
        "topologies": plan["topologies"],
        "red": plan["red"],
        "cells": cells,
        "episodes": episodes,
        "paired_cases": summary["paired_cases"],
        "limits": summary["limits"]
        + [
            "On-policy states diverge; these are not same-state action preference tests.",
            "The four-root21000 cases are exploratory and do not replace the64-case checkpoint score panels.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": digest(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
