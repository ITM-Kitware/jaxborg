"""Register completion of the three-by-three saved-checkpoint payoff matrix."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "jaxborg-exp/shared/studies/checkpoint-audit-150m-20261101"
STEPS = [99840000, 124800000, 150000000]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def main():
    assert not (ROOT / "manifest.json").exists()
    old = json.loads((SOURCE / "manifest.json").read_text())
    report = json.loads((SOURCE / "analysis/audit-report.json").read_text())
    assert report["status"] == "complete_four_contrast_checkpoint_audit"
    assert report["study_manifest_sha256"] == sha(SOURCE / "manifest.json")
    assert not subprocess.check_output(
        ["git", "-C", old["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", old["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == old["source_revision"]
    )
    for folder in ["eval", "analysis", "inputs", "logs", "provenance"]:
        (ROOT / folder).mkdir(exist_ok=True)
    for item in old["inputs"].values():
        assert sha(item["path"]) == item["sha256"] and sha(item["sidecar_path"]) == item["sidecar_sha256"]
    for key in ["topology", "recipe"]:
        assert sha(old[key]["path"]) == old[key]["sha256"]
    retained = {}
    for cell in report["cells"]:
        name = cell["name"]
        path = SOURCE / "eval" / (name + ".json")
        assert sha(path) == cell["result_sha256"]
        d = json.loads(path.read_text())
        assert d["seeds"] == d["per_episode_seeds"] == list(range(51000000, 51004096))
        assert d["git_commit"] == old["source_revision"] and d["n_episodes"] == 4096
        dest = ROOT / "eval" / path.name
        assert not dest.exists()
        shutil.copy2(path, dest)
        retained[name] = {
            "path": str(dest),
            "source_path": str(path),
            "sha256": sha(dest),
            "eval_id": d["eval_id"],
            "canonical_reference": f"runs:/{d['eval_id']}/evaluations/result.json",
        }
    cells = [
        {"name": f"b{b}-r{r}", "blue_steps": b, "red_steps": r, "reused": f"b{b}-r{r}" in retained}
        for b in STEPS
        for r in STEPS
    ]
    new = [c for c in cells if not c["reused"]]
    assert len(retained) == 5 and len(new) == 4
    now = datetime.now(timezone.utc).isoformat()
    save(
        ROOT / "seed-ledger.json",
        {
            "registered_utc": now,
            "evaluation_seed_range": [51000000, 51004095],
            "reuse": "Intentionally complete the prior audit matrix on its identical fixed4096-root GPU cohort",
            "prior_study": str(SOURCE),
            "reused_cells": 5,
            "new_cells": 4,
            "independent_confirmation": False,
        },
    )
    m = {
        "study": ROOT.name,
        "registered_utc": now,
        "checkout": old["checkout"],
        "python": old["python"],
        "store": old["store"],
        "source_revision": old["source_revision"],
        "source_training_run_id": old["source_training_run_id"],
        "inputs": old["inputs"],
        "topology": old["topology"],
        "recipe": old["recipe"],
        "steps": STEPS,
        "cells": cells,
        "reused_cells": retained,
        "episode_seed_range": [51000000, 51004095],
        "source_audit": {
            "manifest_path": str(SOURCE / "manifest.json"),
            "manifest_sha256": sha(SOURCE / "manifest.json"),
            "report_path": str(SOURCE / "analysis/audit-report.json"),
            "report_sha256": sha(SOURCE / "analysis/audit-report.json"),
        },
        "protocol": {
            "episodes_per_cell": 4096,
            "batch_size": 64,
            "episode_length": 500,
            "variant": "cc4_stock",
            "stochastic": True,
            "topology_sampling": "exhaustive",
            "family_size": 24,
            "alpha": 0.05,
            "practical_margin": 50,
            "comparison_direction": "Earlier checkpoint own return minus later checkpoint own return",
            "contrasts": (
                "NineRed pairwise gains at fixedBlue plus nineBlue pairwise gains at fixedRed; "
                "six equal-opponent-panel gains"
            ),
            "panel_variance_unit": (
                "Average the three opponent-specific differences within episode root before computing uncertainty"
            ),
            "intervals": "Two-sided Bonferroni paired normal approximation across24 contrasts",
            "interpretation": (
                "Exploratory follow-up motivated by the already inspected150M audit; not independent confirmation"
            ),
            "stopping": "Complete the fixed four missing cells; no extra checkpoints, roots or adaptive extensions",
        },
        "analysis_rules": {
            "dominance": (
                "An earlier checkpoint beats a later one across this pool only if all3 "
                "opponent-specific corrected lower bounds exceed0"
            ),
            "ranking_reversal": (
                "Same checkpoint pair has a positive corrected interval at one opponent and a "
                "negative corrected interval at another"
            ),
            "better_response_cycle": (
                "A directed loop of statistically supported unilateral payoff improvements among "
                "the9 saved policy pairs"
            ),
            "training_cycle": (
                "A payoff loop or rank reversal alone does not establish a repeating temporal training cycle"
            ),
        },
        "resources": {"partition": "community", "gpus": 1, "cpus": 8, "memory_gib": 32, "walltime": "02:30:00"},
        "internal_cutoff": "2026-11-02T07:55:00-05:00",
        "user_cutoff": "2026-11-02T08:00:00-05:00",
        "limits": [
            "One source training seed and one training topology",
            "Three saved ages cannot by themselves establish temporal recurrence",
            "No Nash certificate or upper bound on untested challengers",
            "Intervals are approximate; existing outcomes motivated this exploratory question",
        ],
        "launcher_hashes": {p.name: sha(p) for p in ROOT.iterdir() if p.suffix in [".py", ".sh"]},
    }
    save(ROOT / "manifest.json", m)
    print(
        json.dumps(
            {
                "status": "registered",
                "manifest_sha256": sha(ROOT / "manifest.json"),
                "reused_cells": 5,
                "new_cells": new,
                "family_size": 24,
            }
        )
    )


if __name__ == "__main__":
    main()
