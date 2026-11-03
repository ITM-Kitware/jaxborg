"""Run fixed Red searches and independent pooled evaluation before the cutoff."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def state_save(state):
    temp = ROOT / "state.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(ROOT / "state.json")


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "community"
    assert os.environ.get("JAX_PLATFORMS") == "cuda"
    child_env = {**os.environ, "JAX_PLATFORMS": "cuda"}
    child_env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    m = json.loads((ROOT / "manifest.json").read_text())
    cutoff = datetime.fromisoformat(m["internal_cutoff"]).timestamp()
    job_end = time.time() + 13.9 * 3600
    cutoff = min(cutoff, job_end)
    assert not subprocess.check_output(
        ["git", "-C", m["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", m["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == m["source_revision"]
    )
    for name, expected in m["launcher_hashes"].items():
        assert sha(ROOT / name) == expected, name
    for item in m["inputs"].values():
        assert sha(item["path"]) == item["sha256"] and sha(item["sidecar_path"]) == item["sidecar_sha256"]
    assert sha(m["topology"]["path"]) == m["topology"]["sha256"]
    assert sha(m["evaluation_recipe"]["path"]) == m["evaluation_recipe"]["sha256"]
    for records in m["recipes"].values():
        for item in records.values():
            assert sha(item["path"]) == item["sha256"]
    os.chdir(m["checkout"])
    sys.path[:0] = [m["checkout"] + "/src", m["checkout"]]
    from jaxborg.checkpoint import load_jax_bundle
    from jaxborg.recipe import load
    from jaxborg.research_tracking import parameter_hash
    from jaxborg.tracking import read_manifest, resolve_artifact

    assert not (ROOT / "state.json").exists(), "Inspect retained work before a new attempt"
    state = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "manifest_sha256": sha(ROOT / "manifest.json"),
        "smoke": {},
        "training": {},
        "validation": {},
        "confirmation": {},
    }
    state_save(state)

    def execute(command):
        assert cutoff - time.time() > 60, "Cutoff reached"
        subprocess.run(command, env=child_env, check=True, timeout=max(1, cutoff - time.time() - 30))

    def train(target, seed, recipe, dest, expected_steps, label):
        assert not dest.exists()
        before = time.monotonic()
        execute(
            [
                m["python"],
                "scripts/train/algorithms/ippo_jax.py",
                "--recipe",
                recipe,
                "--seed",
                str(seed),
                "--tag",
                m["study"] + "-" + label,
                "--run-result",
                str(dest),
            ]
        )
        result = json.loads(dest.read_text())
        assert result["actual_steps"] == expected_steps
        owner = read_manifest(result["run_id"])[0]
        assert owner["status"] == "FINISHED"
        assert owner["trainable_teams"] == ["red"] and owner["frozen_teams"] == ["blue"]
        assert owner["policy_sources"]["red"] == {"kind": "fresh", "seed": seed}
        assert owner["policy_sources"]["blue"]["kind"] == "checkpoint"
        assert sha(owner["policy_sources"]["blue"]["path"]) == m["inputs"][str(target)]["sha256"]
        saved = load_jax_bundle(resolve_artifact(result["final_checkpoint"]))
        assert parameter_hash(saved.policies["blue"].weights) == m["inputs"][str(target)]["parameter_hashes"]["blue"]
        checks = owner["parameter_checks"]
        assert not checks["blue"]["changed"] and checks["red"]["changed"]
        assert parameter_hash(saved.policies["red"].weights) == checks["red"]["current_sha256"]
        assert saved.provenance["seed"] == seed and saved.provenance["total_steps"] == expected_steps
        return {
            **result,
            "target_steps": target,
            "training_seed": seed,
            "wall_seconds": time.monotonic() - before,
            "model_sha256": sha(resolve_artifact(result["final_checkpoint"])),
            "frozen_blue_verified": True,
        }

    for target in m["targets"]:
        recipe = m["smoke_recipes"][str(target)]
        assert sha(recipe["path"]) == recipe["sha256"]
        state["smoke"][str(target)] = train(
            target,
            recipe["training_seed"],
            recipe["path"],
            ROOT / "smoke" / f"b{target}.json",
            recipe["actual_steps"],
            f"smoke-b{target}",
        )
        state_save(state)
    order = [(target, seed) for seed in m["training_seeds"] for target in m["targets"]]
    durations = []
    for index, (target, seed) in enumerate(order):
        upper = max(m["runtime_budget"]["conservative_training_seconds_per_attempt"], max(durations, default=0) * 1.35)
        needed = (
            (len(order) - index) * upper
            + m["runtime_budget"]["reserved_evaluation_seconds"]
            + m["runtime_budget"]["margin_seconds"]
        )
        if cutoff - time.time() < needed:
            state.update(
                status="incomplete_time_budget", reason="Conservative remaining planned work does not fit cutoff"
            )
            state_save(state)
            raise SystemExit("Insufficient remaining time for the complete registered study")
        key = f"b{target}-seed{seed}"
        print("TRAINING_ATTEMPT", index + 1, len(order), key, flush=True)
        result = train(
            target,
            seed,
            m["recipes"][str(target)][str(seed)]["path"],
            ROOT / "training" / (key + ".json"),
            9984000,
            key,
        )
        state["training"][key] = result
        durations.append(result["wall_seconds"])
        state_save(state)
        print("TRAINING_COMPLETE", key, result["final_checkpoint"], flush=True)
    candidates = [
        {
            "name": "saved-r150000000",
            "checkpoint": m["inputs"]["150000000"]["canonical_reference"],
            "sha256": m["inputs"]["150000000"]["sha256"],
            "source_steps": 150000000,
            "seed": 43,
            "run_id": m["source_training_run_id"],
        }
    ]
    for step in [99840000, 124800000]:
        item = m["inputs"][str(step)]
        candidates.append(
            {
                "name": f"saved-r{step}",
                "checkpoint": item["canonical_reference"],
                "sha256": item["sha256"],
                "source_steps": step,
                "seed": 43,
                "run_id": m["source_training_run_id"],
            }
        )
    for target, seed in order:
        key = f"b{target}-seed{seed}"
        result = state["training"][key]
        candidates.append(
            {
                "name": key,
                "checkpoint": result["final_checkpoint"],
                "sha256": result["model_sha256"],
                "source_steps": 9984000,
                "seed": seed,
                "run_id": result["run_id"],
            }
        )
    assert len(candidates) == 9
    save(ROOT / "candidate-pool.json", {"candidates": candidates})

    def evaluate(target, candidate, split):
        name = f"b{target}-{candidate['name']}"
        output = ROOT / split / (name + ".json")
        assert not output.exists()
        bounds = m["evaluation_seed_ranges"][split]
        print("EVALUATE", split, name, flush=True)
        execute(
            [
                m["python"],
                "scripts/eval/eval_matchup.py",
                "--recipe",
                m["evaluation_recipe"]["path"],
                "--policy-backend",
                "jax",
                "--blue-path",
                m["inputs"][str(target)]["canonical_reference"],
                "--red-path",
                candidate["checkpoint"],
                "--episodes-per-seed",
                "1",
                "--seeds",
                f"{bounds[0]}-{bounds[1]}",
                "--topology-path",
                m["topology"]["path"],
                "--topology-sampling",
                "exhaustive",
                "--name",
                m["study"] + "-" + split + "-" + name,
                "--output",
                str(output),
            ]
        )
        d = json.loads(output.read_text())
        seeds = list(range(bounds[0], bounds[1] + 1))
        assert d["per_episode_seeds"] == d["seeds"] == seeds and d["n_episodes"] == len(seeds)
        assert all(
            math.isfinite(b) and b == -r for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"])
        )
        assert sha(d["policies"]["blue"]["path"]) == m["inputs"][str(target)]["sha256"]
        assert sha(d["policies"]["red"]["path"]) == candidate["sha256"]
        return {"path": str(output), "sha256": sha(output), "eval_id": d["eval_id"], "blue_mean": d["blue_mean_return"]}

    for target in m["targets"]:
        for candidate in candidates:
            key = f"b{target}-{candidate['name']}"
            state["validation"][key] = evaluate(target, candidate, "validation")
            state_save(state)
    selected = {}
    for target in m["targets"]:
        winner = min(
            candidates, key=lambda c: (state["validation"][f"b{target}-{c['name']}"]["blue_mean"], candidates.index(c))
        )
        selected[str(target)] = winner
    selection = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "validation": state["validation"],
        "tie_order": [c["name"] for c in candidates],
        "candidate_pool_sha256": sha(ROOT / "candidate-pool.json"),
    }
    save(ROOT / "selection.json", selection)
    state["selection_sha256"] = sha(ROOT / "selection.json")
    state_save(state)
    confirm = []
    for candidate in [candidates[0], selected["124800000"], selected["150000000"]]:
        if candidate["sha256"] not in [c["sha256"] for c in confirm]:
            confirm.append(candidate)
    for target in m["targets"]:
        for candidate in confirm:
            key = f"b{target}-{candidate['name']}"
            state["confirmation"][key] = evaluate(target, candidate, "confirmation")
            state_save(state)
    from analyze import analyze

    report = analyze(ROOT)
    state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    state_save(state)
    from jaxborg.tracking import Run, input_artifact

    records = [
        input_artifact(ROOT / "manifest.json", role="registered matched response study"),
        input_artifact(ROOT / "selection.json", role="validation-frozen selections"),
    ]
    records += [input_artifact(v["path"], role=k) for k, v in state["confirmation"].items()]
    with Run(
        load(m["evaluation_recipe"]["path"]),
        backend="cpu",
        kind="analysis",
        name=m["study"] + "-analysis",
        config=m["confirmation"],
        inputs=records,
    ) as run:
        for name in ["response-report.json", "response-contrasts.csv"]:
            run.publish(ROOT / "analysis" / name, "analysis/" + name)
        for name in m["launcher_hashes"]:
            run.publish(ROOT / name, "protocol/" + name)
        reference = f"runs:/{run.run_id}/analysis/response-report.json"
    save(
        ROOT / "provenance/completion.json",
        {
            "complete": True,
            "completed_utc": state["completed_utc"],
            "job_id": os.environ["SLURM_JOB_ID"],
            "analysis_reference": reference,
        },
    )
    print(
        json.dumps({"status": "complete", "contrasts": report["contrasts"], "analysis_reference": reference}),
        flush=True,
    )


if __name__ == "__main__":
    main()
