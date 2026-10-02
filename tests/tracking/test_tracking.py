"""Native MLflow integration checks in disposable repositories/experiment roots."""

import copy
import json
import shutil
import subprocess
import tarfile

import mlflow
import pytest
from mlflow import MlflowClient

from jaxborg import tracking as t
from jaxborg.checkpoint import read_sidecar, write_sidecar


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "checkout"
    (path / "src").mkdir(parents=True)
    (path / "src/main.py").write_text("print('source')\n")
    (path / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", "https://example.invalid/jaxborg"], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(t, "_REPO", path)
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "experiments"))
    monkeypatch.delenv("JAXBORG_ALLOW_DIRTY", raising=False)
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    yield path
    if t._current:
        t._current.finish("FAILED")
    mlflow.end_run()


def new_run(**kwargs):
    return t.Run(
        {"meta": {"name": "same"}, "algorithm": "ippo", "train": {"total_timesteps": 2}},
        backend="cpu",
        seed=42,
        config={"seed": 42},
        **kwargs,
    )


def test_unique_runs_survive_checkout_removal(repo):
    ids, roots = [], []
    for _ in range(2):
        with new_run() as r:
            ids.append(r.run_id)
            roots.append(r.root)
            r.write_json("checkpoints/policy.json", {"weights": [1, 2]})
    assert ids[0] != ids[1] and roots[0] != roots[1]
    shutil.rmtree(repo)
    for run_id in ids:
        manifest, root = t.read_manifest(run_id)
        assert manifest["status"] == "FINISHED"
        assert t.resolve_artifact(f"runs:/{run_id}/checkpoints/policy.json").exists()
        with tarfile.open(root / "source/source.tar.gz") as archive:
            assert archive.extractfile("src/main.py").read() == b"print('source')\n"


@pytest.mark.parametrize("value", [None, "relative", "checkout/results"])
def test_invalid_roots_fail_before_outputs(repo, monkeypatch, value):
    root = repo / "results"
    if value is None:
        monkeypatch.delenv("JAXBORG_EXP_DIR")
    else:
        monkeypatch.setenv("JAXBORG_EXP_DIR", str(root) if value.startswith("checkout") else value)
    with pytest.raises(ValueError, match="absolute|checkout"):
        new_run()
    assert not root.exists()
    assert mlflow.active_run() is None


def test_incompatible_experiment_fails(repo):
    t.configure()
    client = MlflowClient()
    client.create_experiment("ippo-cc4", artifact_location=(repo / "mlruns").as_uri())
    with pytest.raises(ValueError, match="incompatible"):
        new_run()
    assert not (repo / "mlruns").exists()


def test_experiment_root_ignores_empty_read_only_git_marker(repo):
    root = repo.parent / "experiments"
    marker = root / ".git"
    marker.mkdir(parents=True)
    marker.chmod(0o555)
    try:
        assert t.experiment_root() == root.resolve()
        assert not (root / "mlflow.db").exists()
    finally:
        marker.chmod(0o755)


def test_experiment_root_rejects_linked_worktree(repo, monkeypatch):
    worktree = repo.parent / "other-checkout"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), "HEAD"], check=True)
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(worktree / "results"))
    with pytest.raises(ValueError, match="inside a Git checkout"):
        t.experiment_root()
    assert not (worktree / "results").exists()


def test_dirty_and_wrong_sha_rejected_and_override_archived(repo, monkeypatch):
    sha = t.git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="Wrong launch SHA"):
        t.verify_launch("0" * 40)
    (repo / "src/main.py").write_text("print('edited')\n")
    (repo / "src/new.py").write_text("NEW = True\n")
    (repo / ".env").write_text("SECRET=value\n")
    with pytest.raises(ValueError, match="Dirty"):
        new_run()
    assert not (repo.parent / "experiments").exists()
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    with new_run() as r:
        assert r.manifest["source"]["git_commit"] == sha
        assert not r.manifest["source"]["canonical"]
        with tarfile.open(r.root / "source/source.tar.gz") as archive:
            assert archive.extractfile("src/main.py").read() == b"print('edited')\n"
            assert "src/new.py" in archive.getnames()
            assert ".env" not in archive.getnames()
        assert (r.root / "source/working-tree.patch").stat().st_size > 0


@pytest.mark.parametrize("error,status", [(RuntimeError("bug"), "FAILED"), (KeyboardInterrupt(), "KILLED")])
def test_partial_artifacts_survive_failure(repo, error, status):
    ids = []

    @t.tracked_entrypoint
    def interrupted():
        r = new_run()
        ids.append(r.run_id)
        r.write_json("checkpoints/policy.json", {"weights": [1]})
        print("completed checkpoint retained")
        raise error

    with pytest.raises(type(error)):
        interrupted()
    m, root = t.read_manifest(ids[0])
    assert m["status"] == status == MlflowClient().get_run(ids[0]).info.status
    assert t.resolve_artifact(f"runs:/{ids[0]}/checkpoints/policy.json").is_file()
    assert "completed checkpoint retained" in (root / "logs/console.log").read_text()


def test_atomic_and_immutable_publication(repo, monkeypatch):
    with new_run() as r:
        p = r.path("checkpoints/policy.bin")
        p.write_bytes(b"complete")
        original = MlflowClient.log_artifact

        def failed(self, run_id, path, artifact_path=None):
            original(self, run_id, path, artifact_path)
            raise OSError("upload interrupted")

        monkeypatch.setattr(MlflowClient, "log_artifact", failed)
        with pytest.raises(OSError):
            r.publish(p, "checkpoints/policy.bin")
        assert "checkpoints/policy.bin" not in r.manifest["outputs"]
        monkeypatch.setattr(MlflowClient, "log_artifact", original)
        ref = r.publish(p, "checkpoints/policy.bin")
        assert t.resolve_artifact(ref).read_bytes() == b"complete"
        with pytest.raises(ValueError, match="immutable"):
            r.publish(p, "checkpoints/policy.bin")


def test_legacy_inputs_retained_and_exports_verified(repo, tmp_path):
    p = tmp_path / "legacy/model_old.pt"
    p.parent.mkdir()
    p.write_bytes(b"legacy bytes")
    recipe = {"meta": {"name": "legacy"}, "arch": {"name": "shared"}}
    write_sidecar(p.with_name("recipe_old.yaml"), recipe, seed=1, total_steps=12, backend="cyborg")
    original_recipe = copy.deepcopy(recipe)
    with new_run(inputs=[t.input_artifact(str(p), role="Blue")]) as r:
        assert r.input_path(0).read_bytes() == b"legacy bytes"
        assert read_sidecar(r.input_path(0))["run"]["total_steps"] == 12
        assert t.resolve_artifact(str(p)) == p
        ref = r.write_json("evaluations/result.json", {"mean": 1})
        out = tmp_path / "exports/output.json"
        r.export("evaluations/result.json", out)
        assert t.file_hash(out) == t.file_hash(t.resolve_artifact(ref))
        assert r.manifest["exports"][0]["reference"] == ref
        with pytest.raises(ValueError, match="canonical"):
            r.export("evaluations/result.json", r.root / "other.json")
        p.unlink()
        assert r.input_path(0).is_file()
    assert recipe == original_recipe


def test_reuse_fingerprint_and_output_validation(repo, tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"one")
    recipe = {"eval": {"variant": "stock"}}
    config = {"seeds": [1], "episodes": 2, "topologies": ["abc"], "reward": "stock"}

    def fingerprint(*, evaluator="v1", cfg=None, deps=None):
        return t.evaluation_fingerprint(
            [t.input_artifact(model)],
            recipe,
            cfg or config,
            evaluator_identity=evaluator,
            dependencies=deps or {"jax": "1"},
        )

    fp = fingerprint()
    with new_run(kind="evaluation", fingerprint=fp) as r:
        r.write_json("evaluations/result.json", {"mean": 2})
        run_id = r.run_id
    assert t.find_reusable(fp, ["evaluations/result.json"]) == run_id
    model.write_bytes(b"two")
    assert fingerprint() != fp
    model.write_bytes(b"one")
    assert fingerprint(evaluator="v2") != fp
    assert fingerprint(deps={"jax": "2"}) != fp
    for key in ("seeds", "episodes", "topologies", "reward"):
        changed = copy.deepcopy(config)
        changed[key] = "changed"
        assert fingerprint(cfg=changed) != fp
    result = t.resolve_artifact(f"runs:/{run_id}/evaluations/result.json")
    result.write_text("modified")
    assert t.find_reusable(fp, ["evaluations/result.json"]) is None
    result.unlink()
    assert t.find_reusable(fp, ["evaluations/result.json"]) is None


def test_corrected_evaluation_and_multiple_inputs_preserve_original(repo):
    refs = []
    for team in ("Blue", "Red"):
        with new_run() as r:
            refs.append(r.write_json(f"checkpoints/{team}.json", {"weights": [1]}))
    inputs = [t.input_artifact(ref, role=team) for ref, team in zip(refs, ("Blue", "Red"))]
    with new_run(kind="evaluation", inputs=inputs) as original:
        original.write_json("evaluations/result.json", {"mean": 1})
    original_manifest = (original.root / "manifest.json").read_bytes()
    with new_run(kind="evaluation", inputs=inputs, supersedes=original.run_id, bug_reference="example-25") as corrected:
        corrected.write_json("evaluations/result.json", {"mean": 2})
    assert original.run_id != corrected.run_id
    assert corrected.manifest["supersedes_eval_run_id"] == original.run_id
    assert len({x["source_run_id"] for x in corrected.manifest["inputs"]}) == 2
    assert (original.root / "manifest.json").read_bytes() == original_manifest
    with new_run(
        kind="comparison", inputs=[t.input_artifact(f"runs:/{corrected.run_id}/evaluations/result.json")]
    ) as report:
        report.write_json("evaluations/comparison.json", {"evaluation": corrected.run_id})
    assert report.manifest["inputs"][0]["source_run_id"] == corrected.run_id


def test_launch_pin_checks_environment_and_source(repo, monkeypatch):
    from jaxborg import launch

    monkeypatch.chdir(repo)
    record = launch.pin()
    assert launch.verify(record) == "cpu"
    m = json.loads(record.read_text())
    assert t.git("rev-parse", f"refs/jaxborg/launches/{m['sha']}") == m["sha"]
    monkeypatch.setattr(launch, "dependency_snapshot", lambda: {"changed": "1"})
    with pytest.raises(ValueError, match="dependencies"):
        launch.verify(record)


def test_gpu_discovery_refuses_unallocated_host(repo, monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        t.assigned_devices()
