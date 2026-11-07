"""Execute one declared frozen-policy diagnostic through retained evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from prepare import protocol_for

from jaxborg.evaluation.joint_capture import run_matchup
from jaxborg.evaluation.scripted_capture import run_frozen_scripted
from jaxborg.scenarios.cc4.game_variant import GameVariant


def verified(path, digest):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Input differs from prepared manifest: {path}")
    return path


def run(manifest_path, root, trial_id, expected_revision):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_revision"] != expected_revision:
        raise ValueError("Manifest source revision differs from requested execution")
    protocol = json.loads((manifest_path.parent / "protocol.json").read_text())
    if protocol != protocol_for(manifest):
        raise ValueError("Prepared manifest differs from its immutable protocol")
    cells = [cell for cell in manifest["cells"] if cell["trial_id"] == trial_id]
    if len(cells) != 1:
        raise ValueError("Trial must identify exactly one prepared cell")
    cell = cells[0]
    if not cell.get("executable", True):
        raise ValueError(cell["blocked_reason"])
    blue = next(policy for policy in manifest["policies"]["blue"] if policy["id"] == cell["blue_id"])
    blue_path = verified(blue["model"], blue["sha256"])
    sidecar = verified(blue["sidecar"], blue["sidecar_sha256"])
    bank = manifest["topology_banks"][cell["bank_id"]]
    paths = [verified(item["path"], item["sha256"]) for item in bank["snapshots"]]
    recipe_path = verified(cell["recipe_path"], cell["recipe_sha256"])
    recipe = yaml.safe_load(recipe_path.read_text())
    settings = manifest["screening"]
    scripted = cell["kind"] == "scripted"
    variant = GameVariant(**cell["variant"])

    def condition_check(configuration):
        actual = configuration["base_variant" if scripted else "variant"]
        if actual["num_steps"] != settings["horizon"] or not actual["cage4_enhanced_obs"]:
            raise ValueError("Resolved game differs from declared horizon/observations")
        if actual["blue_block_policy"] != "cc4" or actual["red_reward"] != "zero_sum":
            raise ValueError("Resolved payoff/action contract differs from stock")
        if configuration["topology"]["sha256"] != [item["sha256"] for item in bank["snapshots"]]:
            raise ValueError("Resolved topology bank differs from declaration")
        if configuration["deterministic"] or configuration["episodes_per_seed"] != 1:
            raise ValueError("Diagnostic requires stochastic policy sampling and exact episode roots")

    common = dict(
        root=Path(root),
        protocol=protocol,
        trial_id=trial_id,
        expected_revision=expected_revision,
        seeds=settings["seeds"],
        episodes_per_seed=1,
        deterministic=False,
        condition_check=condition_check,
        entrypoint=Path(__file__),
        source_roots=(Path(__file__).parent,),
    )
    if scripted:
        capture, result = run_frozen_scripted(
            blue_path,
            source_recipe=sidecar,
            effective_recipe=recipe_path,
            base_variant=variant,
            topology_paths=paths,
            reds=cell["reds"],
            progress=False,
            **common,
        )
        summary = [
            {"red": row["eval_red"], "mean_reward": row["mean_reward"], "n_episodes": row["n_episodes"]}
            for row in result
        ]
    else:
        red = next(policy for policy in manifest["policies"]["red"] if policy["id"] == cell["red_id"])
        red_path = verified(red["model"], red["sha256"])
        verified(red["sidecar"], red["sidecar_sha256"])
        capture, result = run_matchup(
            blue_path,
            red_path,
            variant=variant,
            topology_path=paths,
            recipe=recipe,
            supplied_recipe=recipe,
            overrides={"study_cell": cell, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
            **common,
        )
        summary = {
            "mean_reward": sum(result.blue_returns) / len(result.blue_returns),
            "n_episodes": len(result.blue_returns),
        }
    receipt = {
        "trial_id": trial_id,
        "attempt_id": capture.attempt_id,
        "evaluation_id": capture.evaluation_id,
        "directory": str(capture.directory),
        "summary": summary,
    }
    destination = Path(root) / "receipts"
    destination.mkdir(exist_ok=True)
    with (destination / f"{capture.attempt_id.rsplit(':', 1)[-1]}.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    run(args.manifest, args.root, args.trial, args.expected_revision)
