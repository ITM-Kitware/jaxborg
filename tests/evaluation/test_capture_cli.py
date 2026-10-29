import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.capture_store import load_records
from jaxborg.experiment_records import collate, validate_collation, validate_records
from jaxborg.policies import make_torch_policy


@pytest.fixture(autouse=True)
def isolated_tracking_destination(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("JAXBORG_EXP_DIR", raising=False)


@pytest.fixture
def protocol():
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": "capture-smoke:protocol:1",
        "study_id": "capture-smoke",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config", "source_files"], "evaluation": ["resolved_config"]},
        "trials": [
            {"id": name, "condition_id": "tiny-policy", "replicate_id": name, "matched_unit_id": None}
            for name in ("trial:one", "trial:unlaunched")
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policy",
                "uncertainty": "none",
            }
        ],
    }


@pytest.fixture
def tiny_model(tmp_path):
    model = tmp_path / "model_tiny.pt"
    policy = make_torch_policy("shared", obs_dim=BLUE_OBS_SIZE, action_dim=242, hidden_dim=8, hidden_layers=1)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    torch.save(policy.state_dict(), model)
    model.with_name("recipe_tiny.yaml").write_text(
        yaml.safe_dump(
            {"arch": {"name": "shared", "hidden_dim": 8, "hidden_layers": 1}, "train": {"variant": "cc4_stock"}}
        )
    )
    return model


def _cli(*arguments):
    checkout = Path(__file__).resolve().parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(checkout / "src"), str(checkout)]),
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "jaxborg.evaluation.capture", *map(str, arguments)],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _result(completed):
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def _observations(root, records, evaluation_id):
    return sorted(
        [
            json.loads((root / record["uri"]).read_text())
            for record in records
            if record["kind"] == "artifact"
            and record["producer_id"] == evaluation_id
            and record["role"] == "evaluation episode observation"
        ],
        key=lambda observation: observation["episode_index"],
    )


def _run(root, protocol, model, **overrides):
    from jaxborg.evaluation.capture import run_evaluation

    return run_evaluation(
        **(
            {
                "root": root,
                "protocol": protocol,
                "trial_id": "trial:one",
                "model": model,
                "seeds": [17, 18],
                "horizon": 2,
                "deterministic": True,
                "policy_seed": 100,
                "allow_dirty": True,
                "entrypoint": Path(__file__),
            }
            | overrides
        )
    )


@pytest.mark.parametrize("sidecar_suffix", ["yaml", "yml"])
def test_cli_declares_runs_real_cyborg_and_exports_verified_records(tmp_path, protocol, tiny_model, sidecar_suffix):
    sidecar = tiny_model.with_name(f"recipe_tiny.{sidecar_suffix}")
    original = tiny_model.with_name("recipe_tiny.yaml")
    if sidecar != original:
        original.rename(sidecar)
    sidecar_bytes = sidecar.read_bytes()
    protocol = protocol | {
        "trials": [
            *protocol["trials"],
            {
                "id": "trial:stochastic",
                "condition_id": "tiny-policy-stochastic",
                "replicate_id": "trial:stochastic",
                "matched_unit_id": None,
            },
        ]
    }
    root = tmp_path / "store"
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    declared = _result(_cli("declare", "--root", root, "--protocol", declaration))
    assert declared == {"protocol_id": protocol["id"], "status": "declared"}
    assert load_records(root) == [protocol]

    completed = _result(
        _recipe_cli(
            "--model",
            tiny_model,
            "--evidence-root",
            root,
            "--protocol",
            declaration,
            "--trial",
            "trial:one",
            "--episode-seeds",
            18,
            17,
            "--horizon",
            2,
            "--policy-seed",
            100,
            "--deterministic",
            "--allow-dirty",
        )
    )
    assert completed["status"] == "succeeded"
    stochastic = _run(root, protocol, tiny_model, trial_id="trial:stochastic", seeds=[17], deterministic=False)
    output = tmp_path / "records.json"
    exported = _result(_cli("export", "--root", root, "--output", output))
    records = json.loads(output.read_text())
    assert exported == {"output": str(output), "records": len(records)}
    assert records == load_records(root)
    index = validate_records(records)
    attempt, evaluation = [index[completed[key]] for key in ("attempt_id", "evaluation_id")]
    assert attempt["trial_id"] == "trial:one"
    assert attempt["status"] == evaluation["status"] == "succeeded"
    assert attempt["retry_of"] is None
    assert evaluation["attempt_id"] == attempt["id"]
    assert evaluation["budgets"][0] == {"unit": "episodes", "requested": 2, "completed": 2}
    observations = _observations(root, records, evaluation["id"])
    assert next(budget for budget in evaluation["budgets"] if budget["unit"] == "completed_episode_environment_steps")[
        "completed"
    ] == sum(row["steps"] for row in observations)
    assert [row["seed"] for row in observations] == [18, 17]
    assert all(0 < row["steps"] <= 2 for row in observations)
    assert [row["policy_seed"] for row in observations] == [100, 101]
    assert all(math.isfinite(row["reward"]) for row in observations)
    assert evaluation["metrics"] == [
        {"name": "blue_return", "unit": "score", "value": sum(row["reward"] for row in observations) / 2}
    ]
    artifacts = [record for record in records if record["kind"] == "artifact"]
    recipe_artifacts = [record for record in artifacts if record["role"] == "policy recipe"]
    assert len(recipe_artifacts) == 2
    for artifact in recipe_artifacts:
        retained = root / artifact["uri"]
        assert retained.name == sidecar.name
        assert retained.read_bytes() == sidecar_bytes
    evaluation_artifacts = [record for record in artifacts if record["producer_id"] == evaluation["id"]]
    assert len([record for record in evaluation_artifacts if record["role"] == "evaluation reset topology"]) == 2
    assert len([record for record in evaluation_artifacts if record["role"] == "evaluation reset metadata"]) == 2
    for artifact in evaluation_artifacts:
        if artifact["role"] == "evaluation reset metadata":
            reset = json.loads((root / artifact["uri"]).read_text())
            assert reset["seed"] == observations[reset["episode_index"]]["seed"]
            assert index[reset["topology_id"]]["role"] == "evaluation reset topology"
    assert any(record["role"] == "evaluation source" for record in artifacts)
    assert all(not Path(record["uri"]).is_absolute() and (root / record["uri"]).is_file() for record in artifacts)
    assert {
        "src/jaxborg/evaluation/capture.py",
        "src/jaxborg/evaluation/capture_store.py",
        "src/jaxborg/evaluation/cyborg_runner.py",
        "src/jaxborg/evaluation/provenance.py",
        "pyproject.toml",
        "uv.lock",
    } <= attempt["facts"]["source_files"]["value"].keys()
    dependencies = {item["name"].lower(): item for item in attempt["facts"]["dependencies"]["value"]}
    for name in ("cyborg", "jax", "torch"):
        assert dependencies[name]["version"] == importlib.metadata.version(name)
    assert (
        attempt["facts"]["resolved_config"]["value"]["execution"]["dependencies"]
        == attempt["facts"]["dependencies"]["value"]
    )
    config = attempt["facts"]["resolved_config"]["value"]
    assert config["horizon"] == config["variant"]["num_steps"] == 2
    assert config["deterministic"] is True
    stochastic_attempt = index[stochastic.attempt_id]
    assert stochastic_attempt["trial_id"] == "trial:stochastic"
    assert stochastic_attempt["status"] == "succeeded"
    assert stochastic_attempt["facts"]["resolved_config"]["value"]["deterministic"] is False
    assert len(_observations(root, records, stochastic.evaluation_id)) == 1
    result = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={"trial:one": attempt["id"], "trial:stochastic": stochastic.attempt_id},
        selected_evaluations=[evaluation["id"], stochastic.evaluation_id],
        analysis={"source": "test_capture_cli.py", "configuration": {"selection": "two parameter settings"}},
    )
    assert result["included"] == sorted(
        [attempt["id"], evaluation["id"], stochastic.attempt_id, stochastic.evaluation_id]
    )
    assert [(row["trial_id"], row["reason"]) for row in result["missing"]] == [("trial:unlaunched", "no_attempt")]
    assert validate_collation(result) == result
    relocated = tmp_path / "relocated-evidence"
    shutil.move(root, relocated)
    tiny_model.unlink()
    sidecar.unlink()
    assert load_records(relocated) == records


@pytest.mark.parametrize("error_type,status", [(RuntimeError, "failed"), (KeyboardInterrupt, "cancelled")])
def test_partial_failure_and_explicit_retry_preserve_one_scientific_trial(
    tmp_path, protocol, tiny_model, monkeypatch, error_type, status
):
    from jaxborg.evaluation import cyborg_runner

    root = tmp_path / "store"
    original = cyborg_runner.rollout_episode

    def interrupted(env, variant, ep_seed, agent, **kwargs):
        if ep_seed == 18:
            raise error_type("stop after first completed episode")
        return original(env, variant, ep_seed, agent, **kwargs)

    monkeypatch.setattr(cyborg_runner, "rollout_episode", interrupted)
    torch.manual_seed(312)
    with pytest.raises(error_type, match="stop after first completed episode"):
        _run(root, protocol, tiny_model)
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(312)))
    records = load_records(root)
    failed = next(record for record in records if record["kind"] == "attempt")
    evaluation = next(record for record in records if record["kind"] == "evaluation")
    assert failed["status"] == evaluation["status"] == status
    assert evaluation["budgets"][0] == {"unit": "episodes", "requested": 2, "completed": 1}
    observations = _observations(root, records, evaluation["id"])
    assert next(budget for budget in evaluation["budgets"] if budget["unit"] == "completed_episode_environment_steps")[
        "completed"
    ] == sum(row["steps"] for row in observations)
    assert [(row["episode_index"], row["seed"]) for row in observations] == [(0, 17)]
    assert 0 < observations[0]["steps"] <= 2
    assert evaluation["metrics"] == []
    assert evaluation["facts"]["error"]["value"] == {
        "type": error_type.__name__,
        "message": "stop after first completed episode",
    }

    monkeypatch.setattr(cyborg_runner, "rollout_episode", original)
    retry = _run(root, protocol, tiny_model, retry_of=failed["id"])
    records = load_records(root)
    assert _observations(root, records, retry.evaluation_id)[0] == observations[0]
    attempts = [record for record in records if record["kind"] == "attempt"]
    assert len(attempts) == 2
    assert {record["trial_id"] for record in attempts} == {"trial:one"}
    assert next(record for record in attempts if record["id"] == retry.attempt_id)["retry_of"] == failed["id"]
    result = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={"trial:one": retry.attempt_id},
        selected_evaluations=[retry.evaluation_id],
        analysis={"source": "test_capture_cli.py", "configuration": {"selection": "explicit retry"}},
    )
    assert result["included"] == sorted([retry.attempt_id, retry.evaluation_id])
    assert {row["id"] for row in result["excluded"]} == {failed["id"], evaluation["id"]}
    assert result["missing"] == [
        {
            "id": "trial:unlaunched",
            "trial_id": "trial:unlaunched",
            "condition_id": "tiny-policy",
            "replicate_id": "trial:unlaunched",
            "matched_unit_id": None,
            "reason": "no_attempt",
        }
    ]
    assert validate_collation(result) == result


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_invalid_checkpoint_is_retained_as_failed_execution(tmp_path, protocol, tiny_model, damage):
    root = tmp_path / "store"
    if damage == "missing":
        tiny_model.unlink()
    else:
        tiny_model.write_bytes(b"invalid checkpoint")
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    completed = _recipe_cli(
        "--model",
        tiny_model,
        "--evidence-root",
        root,
        "--protocol",
        declaration,
        "--trial",
        "trial:one",
        "--episode-seeds",
        17,
        18,
        "--horizon",
        2,
        "--deterministic",
        "--allow-dirty",
    )
    assert completed.returncode != 0
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["status"] == "failed"
    records = load_records(root)
    executions = [record for record in records if record["kind"] in {"attempt", "evaluation"}]
    assert len(executions) == 2
    assert all(record["status"] == "failed" for record in executions)
    assert all(record["budgets"][0] == {"unit": "episodes", "requested": 2, "completed": 0} for record in executions)
    failure = next(
        record for record in records if record["kind"] == "artifact" and record["role"] == "evaluation failure"
    )
    assert json.loads((root / failure["uri"]).read_text())["type"]
    evaluation = next(record for record in executions if record["kind"] == "evaluation")
    assert result["evaluation_id"] == evaluation["id"]
    assert result["attempt_id"] == next(record["id"] for record in executions if record["kind"] == "attempt")
    assert _observations(root, records, evaluation["id"]) == []


@pytest.mark.parametrize("role", ["policy checkpoint", "evaluation source"])
def test_export_rejects_tampered_retained_inputs(tmp_path, protocol, tiny_model, role):
    root = tmp_path / "store"
    _run(root, protocol, tiny_model, seeds=[17])
    artifact = next(record for record in load_records(root) if record["kind"] == "artifact" and record["role"] == role)
    (root / artifact["uri"]).write_bytes(b"changed retained bytes")
    output = tmp_path / "records.json"
    completed = _cli("export", "--root", root, "--output", output)
    assert completed.returncode != 0
    assert "recorded bytes" in completed.stderr
    assert not output.exists()


def test_source_verification_failure_prevents_successful_terminal_record(tmp_path, protocol, tiny_model, monkeypatch):
    from jaxborg.evaluation import provenance

    original = provenance.collect_provenance

    def changed_sources(**kwargs):
        collected = original(**kwargs)

        def verify_sources():
            raise ValueError("Evaluation source changed during execution")

        return SimpleNamespace(
            configuration=collected.configuration,
            facts=collected.facts,
            source_files=collected.source_files,
            hashes=collected.hashes,
            verify_sources=verify_sources,
        )

    monkeypatch.setattr(provenance, "collect_provenance", changed_sources)
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="source changed during execution"):
        _run(root, protocol, tiny_model, seeds=[17])
    records = load_records(root)
    executions = [record for record in records if record["kind"] in {"attempt", "evaluation"}]
    assert all(record["status"] == "failed" for record in executions)
    evaluation = next(record for record in executions if record["kind"] == "evaluation")
    assert evaluation["budgets"][0] == {"unit": "episodes", "requested": 1, "completed": 1}
    assert len(_observations(root, records, evaluation["id"])) == 1
    assert evaluation["metrics"] == []
    assert evaluation["facts"]["error"]["value"]["message"] == "Evaluation source changed during execution"


def test_source_copy_window_mismatch_fails_before_any_episode(tmp_path, protocol, tiny_model, monkeypatch):
    from jaxborg.evaluation import provenance

    original = provenance.collect_provenance

    def substituted_source_copy(**kwargs):
        collected = original(**kwargs)
        source_name = "src/jaxborg/evaluation/cyborg_runner.py"
        replacement = tmp_path / "copied_source.py"
        replacement.write_bytes(collected.source_files[source_name].read_bytes() + b"\nCOPY_WINDOW_VARIANT = True\n")
        return SimpleNamespace(
            configuration=collected.configuration,
            facts=collected.facts,
            source_files=collected.source_files | {source_name: replacement},
            hashes=collected.hashes,
            verify_sources=collected.verify_sources,
        )

    monkeypatch.setattr(provenance, "collect_provenance", substituted_source_copy)
    root = tmp_path / "store"
    message = "Retained evaluation source differs from its captured identity"
    with pytest.raises(ValueError, match=message):
        _run(root, protocol, tiny_model, seeds=[17])
    records = load_records(root)
    executions = [record for record in records if record["kind"] in {"attempt", "evaluation"}]
    assert len(executions) == 2
    assert all(record["status"] == "failed" for record in executions)
    assert all(record["budgets"][0] == {"unit": "episodes", "requested": 1, "completed": 0} for record in executions)
    evaluation = next(record for record in executions if record["kind"] == "evaluation")
    assert _observations(root, records, evaluation["id"]) == []
    assert evaluation["metrics"] == []
    assert evaluation["facts"]["error"]["value"] == {"type": "ValueError", "message": message}


@pytest.mark.parametrize(
    "payload,error_type,message",
    [
        (None, "FileNotFoundError", "No recipe sidecar found"),
        ("arch: [", "ParserError", "while parsing"),
        ("[]", "ValueError", "must contain an arch mapping"),
        ("{}", "ValueError", "must contain an arch mapping"),
        ("arch: {}", "ValueError", "must declare arch.name"),
        ("arch: {name: 7}", "ValueError", "must declare arch.name"),
    ],
)
def test_invalid_recipe_is_rejected_before_capture_and_cannot_use_loader_fallback(
    tmp_path, protocol, tiny_model, payload, error_type, message
):
    sidecar = tiny_model.with_name("recipe_tiny.yaml")
    if payload is None:
        sidecar.unlink()
    else:
        sidecar.write_text(payload)
    root = tmp_path / "store"
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    completed = _recipe_cli(
        "--model",
        tiny_model,
        "--evidence-root",
        root,
        "--protocol",
        declaration,
        "--trial",
        "trial:one",
        "--episode-seeds",
        17,
        "--horizon",
        2,
        "--allow-dirty",
    )
    assert completed.returncode != 0
    failure = json.loads(completed.stdout.splitlines()[-1])
    assert failure["status"] == "failed"
    assert failure["error"]["type"] == error_type
    assert message in failure["error"]["message"]
    assert "attempt_id" not in failure and "evaluation_id" not in failure
    assert not root.exists()


@pytest.mark.parametrize("deterministic", [True, False])
def test_explicit_native_evaluation_matches_retained_capture_settings_and_observations(
    tmp_path, protocol, tiny_model, deterministic
):
    from jaxborg.checkpoint import read_sidecar
    from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg, resolve_torch_evaluation_settings

    recipe_bytes = tiny_model.with_name("recipe_tiny.yaml").read_bytes()
    model_bytes = tiny_model.read_bytes()
    settings = resolve_torch_evaluation_settings(read_sidecar(tiny_model), deterministic=deterministic, horizon=2)
    torch.manual_seed(312)
    rewards, seeds = evaluate_on_cyborg(
        tiny_model,
        variant=settings.variant,
        seeds=[17, 18],
        episodes_per_seed=1,
        deterministic=deterministic,
        policy_seed=100,
        workers=1,
        progress=False,
    )
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(312)))
    torch.manual_seed(999)
    root = tmp_path / "store"
    capture = _run(root, protocol, tiny_model, deterministic=deterministic)
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(999)))
    records = load_records(root)
    observations = _observations(root, records, capture.evaluation_id)
    assert [row["reward"] for row in observations] == rewards
    assert [row["seed"] for row in observations] == seeds == [17, 18]
    assert [row["policy_seed"] for row in observations] == [100, 101]
    attempt = next(r for r in records if r["id"] == capture.attempt_id)
    config = attempt["facts"]["resolved_config"]["value"]
    assert config["variant"] == asdict(settings.variant)
    assert config["policy"] == settings.policy and config["deterministic"] == settings.deterministic
    evaluation = next(r for r in records if r["id"] == capture.evaluation_id)
    assert evaluation["metrics"] == [{"name": "blue_return", "unit": "score", "value": sum(rewards) / len(rewards)}]
    assert tiny_model.read_bytes() == model_bytes
    assert tiny_model.with_name("recipe_tiny.yaml").read_bytes() == recipe_bytes
    for artifact in records:
        if artifact["kind"] == "artifact":
            payload = (root / artifact["uri"]).read_bytes()
            assert len(payload) == artifact["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]


def test_capture_still_rejects_duplicate_explicit_seeds_before_creating_attempt(tmp_path, protocol, tiny_model):
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="unique"):
        _run(root, protocol, tiny_model, seeds=[17, 17])
    assert not root.exists()


def test_capture_execution_uses_retained_inputs_after_originals_disappear(tmp_path, protocol, tiny_model, monkeypatch):
    from jaxborg.evaluation import capture

    model_hash = hashlib.sha256(tiny_model.read_bytes()).hexdigest()
    recipe_path = tiny_model.with_name("recipe_tiny.yaml")
    recipe_hash = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    original = capture.start_evaluation

    def retain_then_remove_originals(*args, **kwargs):
        retained = original(*args, **kwargs)
        tiny_model.unlink()
        recipe_path.unlink()
        return retained

    monkeypatch.setattr(capture, "start_evaluation", retain_then_remove_originals)
    root = tmp_path / "store"
    retained = _run(root, protocol, tiny_model, seeds=[17])
    records = load_records(root)
    assert next(r for r in records if r["id"] == retained.attempt_id)["status"] == "succeeded"
    assert len(_observations(root, records, retained.evaluation_id)) == 1
    for role, expected in (("policy checkpoint", model_hash), ("policy recipe", recipe_hash)):
        artifact = next(r for r in records if r.get("role") == role)
        assert artifact["sha256"] == expected
        assert hashlib.sha256((root / artifact["uri"]).read_bytes()).hexdigest() == expected


def _recipe_cli(*arguments):
    checkout = Path(__file__).resolve().parents[2]
    workspace = Path(arguments[arguments.index("--model") + 1]).parent
    return subprocess.run(
        [sys.executable, str(checkout / "scripts/eval/eval_recipe.py"), *map(str, arguments)],
        cwd=workspace,
        env=os.environ
        | {
            "PYTHONPATH": os.pathsep.join([str(checkout / "src"), str(checkout)]),
            "JAXBORG_EXP_DIR": str(workspace / "legacy-results"),
            "MLFLOW_EXPERIMENT_NAME": "captured-recipe-cli",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )


def _assert_recipe_tracking(tmp_path, receipt, native_status):
    from mlflow import MlflowClient

    destination = tmp_path / "legacy-results"
    assert not (destination / "eval").exists()
    client = MlflowClient(tracking_uri=f"sqlite:///{destination / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("captured-recipe-cli")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == native_status
    for name, value in {
        "role": "execution",
        "state": "unverified",
        "attempt_id": receipt["attempt_id"],
        "evaluation_id": receipt["evaluation_id"],
        "scientific_status": receipt["status"],
        "delivery": "current",
    }.items():
        assert run.data.tags[f"jaxborg.discovery.{name}"] == value


def test_recipe_research_cli_retains_expanded_seeds_and_resolved_override(tmp_path, protocol, tiny_model):
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    root = tmp_path / "evidence"
    completed = _recipe_cli(
        "--model",
        tiny_model,
        "--evidence-root",
        root,
        "--protocol",
        declaration,
        "--trial",
        "trial:one",
        "--seeds",
        "17,18",
        "--episodes-per-seed",
        "2",
        "--policy-seed",
        "100",
        "--horizon",
        "1",
        "--eval-red",
        "sleep",
        "--allow-dirty",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(completed.stdout)
    records = load_records(root)
    evaluation = next(record for record in records if record["id"] == receipt["evaluation_id"])
    assert receipt["status"] == evaluation["status"] == "succeeded"
    config = evaluation["facts"]["resolved_config"]["value"]
    assert config["seeds"] == [34, 35, 36, 37]
    assert config["policy_seeds"] == [100, 101, 102, 103]
    assert config["horizon"] == 1
    assert config["execution"]["expected_revision"] == config["execution"]["source_revision"]
    assert "scripts/eval/eval_recipe.py" in evaluation["facts"]["source_files"]["value"]
    assert config["overrides"] == {"eval_red": "sleep"}
    assert "eval" not in config["recipe"]
    assert config["variant"]["red_agent"] == "sleep"
    _assert_recipe_tracking(tmp_path, receipt, "FINISHED")


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "requires --evidence-root"),
        (["--evidence-root", "unused", "--protocol", "unused", "--trial", "one", "--workers", "2"], "serial"),
        (["--development-no-capture", "--trial", "one"], "cannot be combined"),
        (["--development-no-capture", "--episode-seeds", "17"], "requires research capture"),
        (["--episode-seeds", "17", "--episodes-per-seed", "2"], "cannot be combined"),
        (
            ["--evidence-root", "unused", "--protocol", "unused", "--trial", "one", "--output", "unused"],
            "immutable store",
        ),
    ],
)
def test_recipe_research_cli_rejects_ambiguous_or_unsupported_requests(tmp_path, tiny_model, extra, message):
    completed = _recipe_cli("--model", tiny_model, *extra)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert not (tmp_path / "legacy-results").exists()


def test_recipe_research_cli_failure_returns_retained_identity(tmp_path, protocol, tiny_model):
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    tiny_model.write_bytes(b"invalid checkpoint")
    root = tmp_path / "evidence"
    completed = _recipe_cli(
        "--model",
        tiny_model,
        "--evidence-root",
        root,
        "--protocol",
        declaration,
        "--trial",
        "trial:one",
        "--seeds",
        "17",
        "--episodes-per-seed",
        "1",
        "--horizon",
        "1",
        "--allow-dirty",
    )
    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "failed"
    records = load_records(root)
    assert next(record for record in records if record["id"] == receipt["attempt_id"])["status"] == "failed"
    assert next(record for record in records if record["id"] == receipt["evaluation_id"])["status"] == "failed"
    _assert_recipe_tracking(tmp_path, receipt, "FAILED")


def test_recipe_research_cli_rejects_wrong_source_revision_before_execution(tmp_path, protocol, tiny_model):
    declaration = tmp_path / "protocol.json"
    declaration.write_text(json.dumps(protocol))
    root = tmp_path / "evidence"
    completed = _recipe_cli(
        "--model",
        tiny_model,
        "--evidence-root",
        root,
        "--protocol",
        declaration,
        "--trial",
        "trial:one",
        "--source-revision",
        "0" * 40,
        "--allow-dirty",
    )
    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert "revision" in receipt["error"]["message"].lower()
    assert "attempt_id" not in receipt
    assert not root.exists()


@pytest.mark.parametrize("interrupt_delivery", [False, True])
def test_recorded_native_evaluation_indexes_running_and_preserves_terminal_result(
    tmp_path, protocol, tiny_model, monkeypatch, interrupt_delivery
):
    import mlflow

    from jaxborg.evaluation import cyborg_runner

    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "native-evaluation")
    client = mlflow.MlflowClient(tracking_uri=uri)
    execute = cyborg_runner.run_torch_episodes
    seen = []

    def observe_running(*args, **kwargs):
        experiment = client.get_experiment_by_name("native-evaluation")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].info.status == "RUNNING"
        assert runs[0].data.tags["jaxborg.discovery.scientific_status"] == "running"
        seen.append(runs[0].info.run_id)
        return execute(*args, **kwargs)

    monkeypatch.setattr(cyborg_runner, "run_torch_episodes", observe_running)
    if interrupt_delivery:
        deliver = mlflow.MlflowClient.log_dict

        def interrupt_terminal(self, run_id, dictionary, artifact_file):
            if artifact_file == "execution.json" and dictionary["records"][0]["status"] == "succeeded":
                raise KeyboardInterrupt("terminal delivery interrupted")
            return deliver(self, run_id, dictionary, artifact_file)

        monkeypatch.setattr(mlflow.MlflowClient, "log_dict", interrupt_terminal)
        with pytest.raises(KeyboardInterrupt, match="terminal delivery interrupted"):
            _run(tmp_path / "evidence", protocol, tiny_model, seeds=[17])
    else:
        _run(tmp_path / "evidence", protocol, tiny_model, seeds=[17])
    records = load_records(tmp_path / "evidence")
    attempt = next(record for record in records if record["kind"] == "attempt")
    assert attempt["status"] == "succeeded"
    assert len(seen) == 1
    run = client.get_run(seen[0])
    assert run.data.tags["jaxborg.discovery.attempt_id"] == attempt["id"]
    assert run.data.tags["jaxborg.discovery.scientific_status"] == "succeeded"
    assert run.data.tags["jaxborg.discovery.delivery"] == ("failed" if interrupt_delivery else "current")
    assert run.info.status == ("FAILED" if interrupt_delivery else "FINISHED")
