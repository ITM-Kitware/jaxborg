"""Observable study tables, selection failures and portable reconstruction."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[1] / "studies/red-update-schedules/analyze.py"
SPEC = importlib.util.spec_from_file_location("red_update_analysis", MODULE_PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def inputs():
    selected = []
    for condition in study.RECIPES:
        for seed in (42, 100):
            evaluations = []
            for opponent in study.OPPONENTS:
                row = {
                    "eval_id": f"{condition}-{seed}-{opponent}",
                    "eval_red": opponent,
                    "eval_env": "jax_fsm",
                    "policy_backend": "jax",
                    "base_variant": "cia_resilience",
                    "cia_metric": "resilience",
                    "cia_config": {"role_assignment": "fixed_per_topology"},
                    "stochastic": True,
                    "blue_busy_action_masking": True,
                    "topology_sampling": "exhaustive",
                    "episode_horizon": 500,
                    "wall_time_s": 2.0,
                    "n_episodes": 4,
                    "per_episode": [1.0, 3.0, 5.0, 7.0] if seed == 42 else [3.0, 5.0, 7.0, 9.0],
                    "per_episode_cia": [{"c": 1.0, "i": 2.0, "a": 3.0}] * 4,
                    "per_episode_seeds": [1000, 1001, 1000, 1001],
                    "per_episode_topology_fingerprints": ["topology-one"] * 2 + ["topology-two"] * 2,
                    "episode_role_map_ids": ["roles-one"] * 2 + ["roles-two"] * 2,
                    "topology_role_maps": [
                        {"topology_fingerprint": f"topology-{name}", "role_map_id": f"roles-{name}"}
                        for name in ("one", "two")
                    ],
                }
                evaluations.append({"row": row})
            selected.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "attempt": {
                        "id": f"train-{condition}-{seed}",
                        "facts": {"training_cost": {"value": {"loop_elapsed_seconds": 10}}},
                        "budgets": [
                            {"unit": "environment_steps", "completed": 96000},
                            {"unit": "red_policy_updates", "completed": 2},
                        ],
                    },
                    "evaluations": evaluations,
                    "updates": [{"update_idx": 1, "team.red.updated": False}],
                }
            )
    return {
        "protocol": {
            "extensions": {
                "red_update": {
                    "values": {
                        "mode": "substantive",
                        "training_seeds": [42, 100],
                        "topology_count": 2,
                        "eval_seeds": [1000, 1001],
                        "episode_seeds": [1000, 1001],
                        "episodes_per_seed": 1,
                        "horizon": 500,
                        "weights": "equal episodes, topologies and training seeds",
                    }
                }
            }
        },
        "selection": {"selected": [], "exploratory": False, "revision_of": None, "rationale": "Initial analysis"},
        "selected": selected,
    }


def test_comparison_keeps_seed_variability_and_actual_cost_separate():
    result = study.tables(inputs())
    assert len(result["episodes"]) == 96
    assert len(result["cases"]) == 48
    assert len(result["training-seeds"]) == 24
    assert not result["missing"]
    row = result["comparison"][0]
    assert row["reward_mean"] == 5.0
    assert row["reward_training_seed_sd"] == pytest.approx(2**0.5)
    assert row["n_training_seeds"] == 2
    assert result["costs"][0]["evaluation_wall_seconds"] == 8.0
    assert result["costs"][0]["red_policy_updates"] == 2
    assert len(result["costs"]) == 6


def test_missing_cells_never_become_a_complete_policy_mean():
    data = inputs()
    data["selected"][0]["evaluations"].pop(0)
    result = study.tables(data)
    assert result["missing"] == [
        {"condition": "baseline", "training_seed": 42, "opponent": "fsm", "reason": "no explicit complete selection"}
    ]
    assert result["comparison"][0]["reward_mean"] is None
    assert result["comparison"][0]["n_training_seeds"] == 1
    assert result["comparison"][0]["reward_min"] == 6.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["episode_role_map_ids"].__setitem__(0, "different-role"),
        lambda row: row["per_episode_topology_fingerprints"].__setitem__(0, "different-topology"),
        lambda row: row["per_episode_seeds"].__setitem__(0, 999),
        lambda row: row["per_episode_cia"].pop(),
        lambda row: row.__setitem__("episode_horizon", 8),
        lambda row: row.__setitem__("stochastic", False),
        lambda row: row["per_episode"].__setitem__(0, float("nan")),
    ],
)
def test_incompatible_or_partial_evaluation_is_rejected(mutate):
    data = inputs()
    mutate(data["selected"][-1]["evaluations"][-1]["row"])
    with pytest.raises(ValueError):
        study.tables(data)


def test_report_is_portable_and_tampering_fails(tmp_path):
    original = tmp_path / "original"
    study.render(inputs(), original)
    rebuilt = tmp_path / "rebuilt"
    study.rebuild(original, rebuilt)
    assert (original / "comparison.csv").read_bytes() == (rebuilt / "comparison.csv").read_bytes()
    assert (original / "manifest.json").read_bytes() == (rebuilt / "manifest.json").read_bytes()
    assert "rankings remain exploratory" in (rebuilt / "report.md").read_text()
    (original / "analysis-inputs.json").write_text("{}")
    with pytest.raises(ValueError, match="byte identity"):
        study.rebuild(original, tmp_path / "tampered")


def test_new_analyzer_cannot_replace_a_saved_result(tmp_path, monkeypatch):
    original = tmp_path / "original"
    study.render(inputs(), original)
    changed = tmp_path / "changed.py"
    changed.write_bytes(MODULE_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(study, "__file__", str(changed))
    with pytest.raises(ValueError, match="exact retained analyzer"):
        study.rebuild(original, tmp_path / "replacement")


def test_prepare_uses_existing_recipes_and_never_launches(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "prepared",
        mode="smoke",
        expected_revision="a" * 40,
        seeds=[42],
        eval_seeds=[1000],
        steps=None,
        num_envs=None,
        topologies=None,
        horizon=None,
        episodes_per_seed=1,
        resources=None,
        wall_time=None,
    )
    study.prepare(args)
    import yaml

    protocol = json.loads((args.output / "protocol.json").read_text())
    assert len(protocol["trials"]) == 3
    assert not (args.output / "evidence").exists()
    adaptive = yaml.safe_load((args.output / "adaptive.yaml").read_text())
    assert adaptive["core"]["anneal_lr"] is False
    assert adaptive["train"]["team_overrides"]["blue"]["core"]["lr"] == 0.0001
    assert adaptive["train"]["adaptive_updates"]["phase_switch_blue_updates"] == 1
    assert adaptive["train"]["total_timesteps"] == 32
    assert adaptive["eval"]["after_training"][0]["args"][-2:] == ["--horizon", "8"]
    assert len((args.output / "commands.sh").read_text().splitlines()) == 3


def test_substantive_prepare_requires_a_declared_budget(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "prepared",
        mode="substantive",
        expected_revision="a" * 40,
        seeds=[42, 100],
        eval_seeds=[1000],
        steps=None,
        num_envs=None,
        topologies=None,
        horizon=None,
        episodes_per_seed=1,
        resources=None,
        wall_time=None,
    )
    with pytest.raises(ValueError, match="Substantive preparation requires"):
        study.prepare(args)
    assert not args.output.exists()


def record_fixture(tmp_path):
    """Minimal retained graph to exercise exact selection and byte ownership."""
    data = inputs()
    spec = data["protocol"]["extensions"]["red_update"]["values"]
    spec.update(
        source_revision="a" * 40,
        recipes={
            "baseline": {
                "train": {"variant": "cc4_stock"},
                "core": {"lr": 0.0003},
                "eval": {
                    "variant": "cia_resilience",
                    "topology_generation": {
                        "generator": "jax",
                        "seed_start": 100,
                        "count": 2,
                        "cache_dir": str(tmp_path / "cache"),
                    },
                },
            }
        },
        completed_steps=96000,
    )
    protocol_id = f"red-update:{study.digest(study.encoded(spec))}"
    protocol = {"id": protocol_id, "kind": "protocol", **data["protocol"]}
    attempt = {
        "id": "training",
        "kind": "attempt",
        "execution_kind": "training",
        "protocol_id": protocol_id,
        "trial_id": "baseline:42",
        "status": "succeeded",
        "retry_of": None,
        "budgets": [],
        "facts": {
            name: {"value": value}
            for name, value in {
                "resolved_config": {
                    "seed_roles": {"environment_reset_root": 42},
                    "recipe": spec["recipes"]["baseline"],
                },
                "source_identity": "source",
                "source_revision": "a" * 40,
                "source_files": {"shared.py": "bytes"},
                "policy_contract": {"blue": "architecture"},
                "game_contract": {"rules": "stock"},
                "topology": {"sha256": ["topology"]},
                "checkpoint_96000": {"model": "model"},
            }.items()
        },
    }
    model = {
        "kind": "artifact",
        "id": "model",
        "producer_id": "training",
        "sha256": "model-digest",
        "role": "joint policy checkpoint",
    }
    evaluation = {
        "kind": "evaluation",
        "id": "evaluation",
        "status": "succeeded",
        "inputs": ["model"],
        "facts": {name: attempt["facts"][name] for name in ("source_revision", "source_files")},
    }
    from dataclasses import replace

    from jaxborg.evaluation.capture_store import json_configuration
    from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint
    from jaxborg.recipe import eval_variant
    from jaxborg.scenarios.cc4.topology import load_topology, save_topology

    topology_artifacts, fingerprints = [], []
    for i in range(2):
        const = load_topology(Path(__file__).parents[1] / f"scripts/dev/topology_bank/shape_{i:02}.snapshot.npz")
        path = tmp_path / f"topology-{i}.npz"
        save_topology(const, path, metadata={"source": "generated", "source_seed": 100 + i})
        raw = path.read_bytes()
        topology_artifacts.append(
            {
                "kind": "artifact",
                "id": f"topology-{i}",
                "producer_id": "evaluation",
                "uri": path.name,
                "size_bytes": len(raw),
                "sha256": study.digest(raw),
                "role": f"topology {i}",
            }
        )
        fingerprints.append(canonical_topology_fingerprint(const))
    evaluation["inputs"].extend(artifact["id"] for artifact in topology_artifacts)
    evaluation["facts"]["resolved_config"] = {
        "value": {
            "recipe": spec["recipes"]["baseline"],
            "base_variant": json_configuration(replace(eval_variant(spec["recipes"]["baseline"]), num_steps=500)),
            "seeds": [1000, 1001],
            "episodes_per_seed": 1,
            "deterministic": False,
            "reds": list(study.OPPONENTS),
        }
    }
    evaluation["facts"]["input_artifacts"] = {
        "value": {f"topology {i}": artifact["id"] for i, artifact in enumerate(topology_artifacts)}
    }
    row = data["selected"][0]["evaluations"][0]["row"]
    row["per_episode_topology_fingerprints"] = [fingerprint for fingerprint in fingerprints for _ in range(2)]
    for role, fingerprint in zip(row["topology_role_maps"], fingerprints):
        role["topology_fingerprint"] = fingerprint
    row.update(
        evaluation_id="evaluation",
        training_attempt_id="training",
        training_checkpoint_artifact_id="model",
        model_sha256="model-digest",
    )
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "evaluation.jsonl").write_bytes(raw)
    artifact = {
        "kind": "artifact",
        "id": "rows",
        "producer_id": "evaluation",
        "uri": "evaluation.jsonl",
        "size_bytes": len(raw),
        "sha256": study.digest(raw),
        "role": "scripted opponent episode results",
    }
    selection = {
        "protocol_id": protocol_id,
        "selected": [
            {
                "condition": "baseline",
                "seed": 42,
                "attempt_id": "training",
                "checkpoint_id": "model",
                "evaluations": [{"artifact_id": "rows", "eval_id": row["eval_id"]}],
            }
        ],
    }
    return [protocol, attempt, model, evaluation, artifact, *topology_artifacts], selection


def test_explicit_ids_follow_checkpoint_ownership_and_verify_bytes(tmp_path, monkeypatch):
    from jaxborg.evaluation import capture_store

    records, selection = record_fixture(tmp_path)
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    result = study.collect(tmp_path, selection)
    assert result["selected"][0]["evaluations"][0]["row"]["eval_red"] == "fsm"
    assert result["attempt_inventory"][0]["id"] == "training"
    (tmp_path / "evaluation.jsonl").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Artifact bytes differ"):
        study.collect(tmp_path, selection)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda records, selection: selection["selected"].append(selection["selected"][0]),
        lambda records, selection: records[2].__setitem__("producer_id", "another-training"),
        lambda records, selection: records[3].__setitem__("inputs", []),
        lambda records, selection: records[1].__setitem__("status", "failed"),
        lambda records, selection: selection["selected"][0]["evaluations"][0].__setitem__("eval_id", "absent"),
    ],
)
def test_selection_rejects_duplicate_retry_partial_and_wrong_input(tmp_path, monkeypatch, mutate):
    from jaxborg.evaluation import capture_store

    records, selection = record_fixture(tmp_path)
    mutate(records, selection)
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    with pytest.raises(ValueError):
        study.collect(tmp_path, selection)


def test_consistently_wrong_source_revision_is_not_the_declared_study(tmp_path, monkeypatch):
    from jaxborg.evaluation import capture_store

    records, selection = record_fixture(tmp_path)
    second = copy.deepcopy(records[1])
    second.update(id="training100", trial_id="baseline:100")
    second["facts"]["resolved_config"]["value"]["seed_roles"]["environment_reset_root"] = 100
    records.append(second)
    selection["selected"].append(
        {**selection["selected"][0], "seed": 100, "attempt_id": "training100", "evaluations": []}
    )
    for record in (records[1], records[3], second):
        record["facts"]["source_revision"] = {"value": "b" * 40}
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    with pytest.raises(ValueError, match="declared source pin"):
        study.collect(tmp_path, selection)


def test_unrelated_later_attempt_does_not_change_saved_comparison(tmp_path, monkeypatch):
    from jaxborg.evaluation import capture_store

    records, selection = record_fixture(tmp_path)
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    first = study.collect(tmp_path, selection)
    study.render(first, tmp_path / "first")
    records.append(
        {
            "kind": "attempt",
            "id": "unrelated-later-attempt",
            "trial_id": "adaptive:100",
            "status": "failed",
            "retry_of": None,
            "budgets": [],
        }
    )
    second = study.collect(tmp_path, selection)
    study.render(second, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/manifest.json").read_bytes() == (tmp_path / "second/manifest.json").read_bytes()


def test_same_size_alternate_evaluation_bank_is_not_the_declared_suite(tmp_path, monkeypatch):
    from jaxborg.evaluation import capture_store
    from jaxborg.scenarios.cc4.topology import load_topology, save_topology

    records, selection = record_fixture(tmp_path)
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    study.collect(tmp_path, selection)
    artifact = records[5]
    path = tmp_path / artifact["uri"]
    const = load_topology(path)
    save_topology(const, path, metadata={"source": "generated", "source_seed": 200})
    raw = path.read_bytes()
    artifact.update(size_bytes=len(raw), sha256=study.digest(raw))
    with pytest.raises(ValueError, match="topology provenance differs"):
        study.collect(tmp_path, selection)
