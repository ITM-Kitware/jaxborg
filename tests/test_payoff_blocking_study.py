"""Four-condition evidence is checked before a scientific report is exposed."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from jaxborg.experiment_records import collate
from jaxborg.recipe import eval_variant, train_variant
from scripts.eval.plot_results import analysis_identity

_spec = importlib.util.spec_from_file_location(
    "payoff_blocking_study", Path(__file__).parents[1] / "experiments/reward-offset/study.py"
)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def configuration(condition):
    recipe, builder, check = study.bind_condition(condition)
    config = {
        "variant": asdict(replace(eval_variant(recipe), num_steps=2)),
        "execution": {
            "dependencies": [{"name": "jax", "version": "0.10.2"}],
            "runtime": {"python": "3.11.0", "system": "Linux"},
        },
        "recipe": recipe,
        "team_configs": {
            team: {"LR": recipe["core"]["lr"], "TRAIN_VARIANT": asdict(train_variant(recipe))}
            for team in ("blue", "red")
        },
        "initialization": {"kind": "fresh", "seed_roots_retained": True},
        "deterministic": False,
        "horizon": 2,
        "episodes_per_seed": 1,
        "builder": {
            "module": "reward_offset_rules",
            "function": "build_env",
            "arguments": builder.keywords,
            "source_hashes": {
                "experiments/reward-offset/rules.py": hashlib.sha256((study.HERE / "rules.py").read_bytes()).hexdigest()
            },
        },
    }
    check(config)
    assert train_variant(recipe).blue_block_policy == config["variant"]["blue_block_policy"]
    return config


def write_fixture(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    protocol = study.protocol()
    records = [protocol]
    selected_attempts, selected_evaluations = {}, []

    def artifact(name, payload, producer, role):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        identity = f"artifact:{name}"
        records.append(
            {
                "schema_version": 1,
                "kind": "artifact",
                "id": identity,
                "producer_id": producer,
                "uri": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "role": role,
            }
        )
        return identity

    def fact(value):
        return {"value": value, "source": "fixture.json"}

    for condition in study.CONDITIONS:
        config = configuration(condition)
        training_id = f"attempt:{condition}:train"
        evaluation_attempt = f"attempt:{condition}:evaluate"
        evaluation_id = f"evaluation:{condition}"
        for procedure, identity in (("train", training_id), ("evaluate", evaluation_attempt)):
            trial = f"{condition}:{procedure}:1"
            selected_attempts[trial] = identity
            records.append(
                {
                    "schema_version": 1,
                    "kind": "attempt",
                    "id": identity,
                    "protocol_id": protocol["id"],
                    "trial_id": trial,
                    "status": "succeeded",
                    "execution_kind": "training" if procedure == "train" else "evaluation",
                    "retry_of": None,
                    "facts": {
                        "resolved_config": fact(config),
                        "source_identity": fact("fixture-source"),
                        "seed_roles": fact({"initialization": 42}),
                    },
                    "budgets": [
                        {
                            "unit": "environment_steps" if procedure == "train" else "episodes",
                            "requested": 2,
                            "completed": 2,
                        }
                    ],
                    "inputs": [],
                }
            )
        policies = {
            team: artifact(f"{condition}/{team}.weights", team.encode(), training_id, f"{team} checkpoint")
            for team in ("blue", "red")
        }
        offset = study.CONDITIONS[condition]["offset"]
        observations = [
            artifact(
                f"{condition}/{i}.json",
                json.dumps({"episode_index": i, "seed": 17 + i, "reward": reward + offset * 2, "steps": 2}).encode(),
                evaluation_id,
                "evaluation episode observation",
            )
            for i, reward in enumerate((2.0, 4.0))
        ]
        records.append(
            {
                "schema_version": 1,
                "kind": "evaluation",
                "id": evaluation_id,
                "attempt_id": evaluation_attempt,
                "inputs": list(policies.values()),
                "status": "succeeded",
                "facts": {
                    "resolved_config": fact(config),
                    "source_identity": fact("fixture-source"),
                    "policies": fact(policies),
                    "backend": fact("jax"),
                    "eval_env": fact("jax"),
                    "policy_contract": fact("canonical joint CC4"),
                    "topology": fact({"mode": "generative"}),
                    "seed_roles": fact({"environment": [17, 18]}),
                },
                "budgets": [{"unit": "episodes", "requested": 2, "completed": 2}],
                "metrics": [{"name": "blue_return", "unit": "score", "value": 3.0 + offset * 2}],
                "raw_artifacts": observations,
            }
        )
        for identity in (training_id, evaluation_attempt, evaluation_id):
            record = next(record for record in records if record["id"] == identity)
            artifact_id = artifact(
                f"{identity.replace(':', '-')}/configuration.json",
                json.dumps(config).encode(),
                identity,
                "resolved evaluation configuration",
            )
            record["inputs"].append(artifact_id)
            record["facts"]["resolved_config"]["source"] = f"{identity.replace(':', '-')}/configuration.json"
            source_name = "experiments/reward-offset/rules.py"
            source_bytes = (study.HERE / "rules.py").read_bytes()
            source_id = artifact(
                f"{identity.replace(':', '-')}/source/{source_name}", source_bytes, None, "evaluation source"
            )
            manifest = {source_name: hashlib.sha256(source_bytes).hexdigest()}
            manifest_name = f"{identity.replace(':', '-')}/source-files.json"
            manifest_id = artifact(manifest_name, json.dumps(manifest).encode(), identity, "evaluation source manifest")
            record["inputs"].extend((source_id, manifest_id))
            topology_payload = (
                b"training snapshot fixture" if identity == training_id else b"evaluation snapshot fixture"
            )
            topology_id = artifact(
                f"{identity.replace(':', '-')}/topology.npz", topology_payload, None, "topology input"
            )
            record["inputs"].append(topology_id)
            record["facts"]["topology"] = fact(
                {"sampling": "exhaustive", "sha256": [hashlib.sha256(topology_payload).hexdigest()]}
            )
            record["facts"].update(
                source_files={"value": manifest, "source": manifest_name},
                source_identity=fact(
                    hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                ),
                source_revision=fact("fixture-revision"),
                dependencies=fact([{"name": "jax", "version": "0.10.2"}]),
                runtime=fact({"python": "3.11.0", "system": "Linux"}),
            )
        selected_evaluations.append(evaluation_id)
    analysis = {"source": analysis_identity(study.sources()), "configuration": study.report_configuration()}
    selection = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts=selected_attempts,
        selected_evaluations=selected_evaluations,
        analysis=analysis,
    )
    path = tmp_path / "collation.json"
    path.write_text(json.dumps(selection))
    return root, path


def test_four_arm_report_separates_payoff_and_original_score(tmp_path):
    root, selection = write_fixture(tmp_path)
    output = tmp_path / "report"
    study.main(["analyze", "--collation", str(selection), "--evidence-root", str(root), "--output", str(output)])
    rows = json.loads((output / "joined.json").read_text())
    by_condition = {row["condition_id"]: row for row in rows}
    assert {name: row["mean_reward"] for name, row in by_condition.items()} == {
        "baseline": 3.0,
        "offset": 5.0,
        "blocking": 3.0,
        "combined": 5.0,
    }
    assert all(row["original_score_mean"] == 3.0 for row in rows)
    assert all(row["std_reward"] == pytest.approx(2**0.5) for row in rows)
    assert (
        json.loads((output / "selection-status.json").read_text())["missing"][0]["trial_id"]
        == "baseline:evaluate:unlaunched"
    )


def test_report_replays_from_only_archived_analysis_sources(tmp_path):
    root, selection = write_fixture(tmp_path)
    output = tmp_path / "report"
    study.main(["analyze", "--collation", str(selection), "--evidence-root", str(root), "--output", str(output)])
    archive = output / "analysis-source"
    original_files = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    rebuilt = tmp_path / "rebuilt"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(archive / "src"), str(archive))),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(archive / "experiments/reward-offset/study.py"),
            "analyze",
            "--collation",
            str(output / "collation.json"),
            "--evidence-root",
            str(root),
            "--output",
            str(rebuilt),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rebuilt / "joined.json").read_bytes() == (output / "joined.json").read_bytes()
    assert (rebuilt / "joined.csv").read_bytes() == (output / "joined.csv").read_bytes()
    assert {
        str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original_files


@pytest.mark.parametrize(
    "damage",
    [
        "condition",
        "source",
        "seed",
        "budget",
        "checkpoint",
        "missing-cell",
        "topology",
        "rule-source",
        "optimizer",
        "initialization",
        "deterministic",
        "builder-module",
        "training-dependency",
        "evaluation-dependency",
        "training-runtime",
        "evaluation-runtime",
    ],
)
def test_scientific_mismatches_are_rejected_without_report(tmp_path, damage):
    root, selection_path = write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    index = {record["id"]: record for record in selection["records"]}
    evaluation = index["evaluation:combined"]
    if damage == "condition":
        evaluation["facts"]["resolved_config"]["value"]["builder"]["arguments"]["offset"] = 0
    elif damage == "source":
        evaluation["facts"]["source_identity"]["value"] = "other"
    elif damage == "seed":
        evaluation["facts"]["seed_roles"]["value"]["environment"] = [99, 100]
    elif damage == "budget":
        index["attempt:combined:train"]["budgets"][0]["requested"] = 4
    elif damage == "checkpoint":
        identity = index["evaluation:baseline"]["facts"]["policies"]["value"]["blue"]
        evaluation["facts"]["policies"]["value"]["blue"] = identity
        evaluation["inputs"].append(identity)
    elif damage == "missing-cell":
        index[selection["protocol_id"]]["trials"].pop()
    elif damage == "topology":
        evaluation["facts"]["topology"]["value"] = {"mode": "fixed"}
    elif damage == "rule-source":
        evaluation["facts"]["resolved_config"]["value"]["builder"]["source_hashes"] = {}
    elif damage == "optimizer":
        index["attempt:combined:train"]["facts"]["resolved_config"]["value"]["team_configs"]["blue"]["LR"] = 0.9
    elif damage == "initialization":
        index["attempt:combined:train"]["facts"]["resolved_config"]["value"]["initialization"]["kind"] = "warm-start"
    elif damage == "deterministic":
        evaluation["facts"]["resolved_config"]["value"]["deterministic"] = True
    elif damage.endswith("dependency"):
        target = index["attempt:combined:train"] if damage.startswith("training") else evaluation
        target["facts"]["dependencies"]["value"][0]["version"] = "different"
        target["facts"]["resolved_config"]["value"]["execution"]["dependencies"][0]["version"] = "different"
    elif damage.endswith("runtime"):
        target = index["attempt:combined:train"] if damage.startswith("training") else evaluation
        target["facts"]["runtime"]["value"]["python"] = "different"
        target["facts"]["resolved_config"]["value"]["execution"]["runtime"]["python"] = "different"
    else:
        evaluation["facts"]["resolved_config"]["value"]["builder"]["module"] = "another_implementation"
    for record in selection["records"]:
        if record["kind"] not in {"attempt", "evaluation"}:
            continue
        config = record["facts"]["resolved_config"]
        artifact = next(index[identity] for identity in record["inputs"] if index[identity]["uri"] == config["source"])
        payload = json.dumps(config["value"]).encode()
        (root / artifact["uri"]).write_bytes(payload)
        artifact.update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    rebuilt = collate(
        selection["records"],
        **{key: selection[key] for key in ("protocol_id", "selected_attempts", "selected_evaluations", "analysis")},
    )
    selection_path.write_text(json.dumps(rebuilt))
    with pytest.raises(ValueError):
        study.main(
            [
                "analyze",
                "--collation",
                str(selection_path),
                "--evidence-root",
                str(root),
                "--output",
                str(tmp_path / "bad"),
            ]
        )
    assert not (tmp_path / "bad").exists()


def test_topology_recipe_declarations_keep_training_and_evaluation_disjoint(tmp_path):
    from jaxborg.topology_banks import parse_topology_generation, validate_topology_split

    declarations = study.topology_declarations(tmp_path)
    validate_topology_split(declarations, repo_root=study.HERE.parents[1])
    train = parse_topology_generation(declarations["train"], scope="train", repo_root=study.HERE.parents[1])
    evaluation = parse_topology_generation(declarations["eval"], scope="eval", repo_root=study.HERE.parents[1])
    assert train.seeds == (0,) and evaluation.seeds == (1,)
    assert set(train.paths).isdisjoint(evaluation.paths)
    assert not (tmp_path / "topologies").exists()


@pytest.fixture
def condition_inputs(tmp_path, monkeypatch):
    recipe, builder, check = study.bind_condition("baseline")
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_bytes((study.HERE / "recipe.yaml").read_bytes())
    monkeypatch.setattr(study, "HERE", tmp_path)
    monkeypatch.setattr(study, "bind_condition", lambda condition: (recipe, builder, check))
    banks = {}
    for name in ("train", "eval"):
        path = tmp_path / f"{name}.npz"
        path.write_bytes(name.encode())
        banks[name] = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "expected_revision": "pinned",
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "train_topology": banks["train"],
        "eval_topology": banks["eval"],
    }


@pytest.mark.parametrize("target", ["recipe.yaml", "train.npz", "eval.npz"])
def test_condition_rejects_changed_inputs_before_execution(tmp_path, condition_inputs, target):
    (tmp_path / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        study.run_condition(tmp_path, "baseline", **condition_inputs)
    assert not (tmp_path / "baseline.result.json").exists()
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("target", ["recipe.yaml", "train.npz", "eval.npz"])
@pytest.mark.parametrize("phase", ["binding", "training", "evaluation"])
def test_changed_inputs_stop_following_execution_and_result_receipt(
    tmp_path, monkeypatch, condition_inputs, target, phase
):
    from types import SimpleNamespace

    import jaxborg.evaluation.capture_store as store
    import jaxborg.evaluation.joint_capture as joint

    records, completed = [], []
    monkeypatch.setattr(store, "load_records", lambda root: records)
    bind = study.bind_condition

    def bind_condition(condition):
        result = bind(condition)
        if phase == "binding":
            (tmp_path / target).write_bytes(b"changed")
        return result

    def train(args, recipe, *arguments, **kwargs):
        completed.append("training")
        records.extend(
            [
                {
                    "id": "training",
                    "kind": "attempt",
                    "facts": {
                        f"checkpoint_{recipe['train']['total_timesteps']}": {
                            "value": {"model": "model", "sidecar": "sidecar"}
                        }
                    },
                },
                {"id": "model", "uri": "model.safetensors"},
            ]
        )
        if phase == "training":
            (tmp_path / target).write_bytes(b"changed")
        return SimpleNamespace(attempt_id="training")

    def evaluate(*args, **kwargs):
        completed.append("evaluation")
        if phase == "evaluation":
            (tmp_path / target).write_bytes(b"changed")
        return SimpleNamespace(attempt_id="evaluated", evaluation_id="evaluation"), None

    monkeypatch.setattr(study, "bind_condition", bind_condition)
    monkeypatch.setitem(sys.modules, "scripts.train.algorithms.ippo_jax", SimpleNamespace(_run_joint_training=train))
    monkeypatch.setattr(joint, "run_matchup", evaluate)
    with pytest.raises(ValueError, match="bytes differ"):
        study.run_condition(tmp_path, "baseline", **condition_inputs)
    assert completed == {"binding": [], "training": ["training"], "evaluation": ["training", "evaluation"]}[phase]
    assert not (tmp_path / "baseline.result.json").exists()


def test_execution_receipts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "result.json"
    study._write_receipt(path, {"selected_attempt": "original"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study._write_receipt(path, {"selected_attempt": "replacement"})
    assert path.read_bytes() == original


def test_condition_processes_are_fresh_and_failure_propagates(tmp_path):
    pids = []
    for condition in study.CONDITIONS:
        receipt = tmp_path / f"{condition}.json"
        study._wait_condition(
            [
                sys.executable,
                "-c",
                "import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))",
                str(receipt),
            ]
        )
        pids.append(int(receipt.read_text()))
    assert len(set(pids)) == 4 and os.getpid() not in pids
    with pytest.raises(subprocess.CalledProcessError) as error:
        study._wait_condition([sys.executable, "-c", "raise SystemExit(7)"])
    assert error.value.returncode == 7


def test_interrupt_is_forwarded_only_to_owned_child_and_reaped(monkeypatch):
    import signal

    calls = []

    class Child:
        waits = 0

        def wait(self):
            self.waits += 1
            calls.append("wait")
            if self.waits == 1:
                raise KeyboardInterrupt
            assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
            return 130

        def poll(self):
            return None

        def send_signal(self, value):
            calls.append(value)

    def launch(command, **kwargs):
        assert command == ["owned-worker"]
        assert kwargs == {"start_new_session": True}
        return Child()

    monkeypatch.setattr(subprocess, "Popen", launch)
    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        study._wait_condition(["owned-worker"])
    assert calls == ["wait", signal.SIGINT, "wait"]
    assert signal.getsignal(signal.SIGINT) == previous_handler


@pytest.mark.parametrize("failure", [subprocess.CalledProcessError(7, "worker"), KeyboardInterrupt()])
def test_failed_or_interrupted_child_stops_study_without_collation(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    import jaxborg.evaluation.provenance as provenance
    import jaxborg.topology_banks as topology

    monkeypatch.setattr(
        provenance,
        "collect_provenance",
        lambda **kwargs: SimpleNamespace(
            hashes={},
            verify_sources=lambda: None,
        ),
    )

    def bank(section, *, scope, repo_root):
        path = tmp_path / f"{scope}.npz"
        path.write_bytes(scope.encode())
        return [path]

    monkeypatch.setattr(topology, "materialize_topology_bank", bank)
    calls = []

    def child(command):
        calls.append(command)
        raise failure

    monkeypatch.setattr(study, "_wait_condition", child)
    output = tmp_path / "study"
    with pytest.raises(type(failure)):
        study.run_study(output, expected_revision="pinned")
    assert len(calls) == 1
    assert calls[0][-1] == "baseline"
    assert (output / "evidence" / "protocol.json").exists()
    assert not (output / "collation.json").exists()
    assert not (output / "report").exists()


def test_coordinator_uses_four_fresh_children_and_exact_receipts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jaxborg.evaluation.capture_store as store
    import jaxborg.evaluation.provenance as provenance
    import jaxborg.topology_banks as topology
    import scripts.eval.plot_results as plots

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, fixture_selection = write_fixture(fixture)
    records = json.loads(fixture_selection.read_text())["records"]

    def source(**kwargs):
        assert kwargs["source_roots"] == (study.HERE, study.HERE.parents[1] / "scripts/train/algorithms")
        return SimpleNamespace(hashes={}, verify_sources=lambda: None)

    monkeypatch.setattr(provenance, "collect_provenance", source)

    def bank(section, *, scope, repo_root):
        path = tmp_path / f"{scope}.npz"
        path.write_bytes(scope.encode())
        return [path]

    monkeypatch.setattr(topology, "materialize_topology_bank", bank)
    monkeypatch.setattr(store, "load_records", lambda root: records)
    wait = study._wait_condition
    pids = []

    def child(command):
        condition = command[-1]
        output = Path(command[command.index("--output") + 1])
        assert (
            command[command.index("--recipe-sha256") + 1]
            == hashlib.sha256((study.HERE / "recipe.yaml").read_bytes()).hexdigest()
        )
        for scope in ("train", "eval"):
            offset = command.index(f"--{scope}-topology")
            path, digest = command[offset + 1 : offset + 3]
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        result = {
            "selected_attempts": {
                f"{condition}:train:1": f"attempt:{condition}:train",
                f"{condition}:evaluate:1": f"attempt:{condition}:evaluate",
            },
            "selected_evaluations": [f"evaluation:{condition}"],
        }
        receipt = output / f"{condition}.result.json"
        wait(
            [
                sys.executable,
                "-c",
                "import json,os,sys; value=json.loads(sys.argv[2]); value['pid']=os.getpid(); "
                "stream=open(sys.argv[1], 'x'); json.dump(value,stream); stream.close()",
                str(receipt),
                json.dumps(result),
            ]
        )
        pids.append(json.loads(receipt.read_text())["pid"])

    monkeypatch.setattr(study, "_wait_condition", child)
    reports = []
    monkeypatch.setattr(plots, "write_collation_report", lambda *args, **kwargs: reports.append(args))
    output = study.run_study(tmp_path / "run", expected_revision="pinned")
    selection = json.loads((output / "collation.json").read_text())
    assert len(set(pids)) == 4 and os.getpid() not in pids
    assert selection["selected_evaluations"] == sorted(f"evaluation:{condition}" for condition in study.CONDITIONS)
    assert len(selection["selected_attempts"]) == 8
    assert len(reports) == 1
