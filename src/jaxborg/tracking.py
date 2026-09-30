"""JAXborg provenance and publication checks over native MLflow tracking.

Local shared-filesystem deployment only. MLflow owns identity and locations;
this module owns the manifest, complete artifact hashes and evaluation contract.
"""

from __future__ import annotations

import argparse
import copy
import functools
import getpass
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse, urlunparse

import mlflow
import yaml
from mlflow import MlflowClient

SCHEMA_VERSION = 1
_REPO = Path(__file__).resolve().parents[2]
_current = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items() if not str(k).startswith("__")}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def digest(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, repo=None):
    return subprocess.check_output(["git", "-C", str(repo or _REPO), *args], stderr=subprocess.DEVNULL).decode().strip()


def experiment_root():
    raw = os.environ.get("JAXBORG_EXP_DIR")
    if not raw or not Path(raw).is_absolute():
        raise ValueError("Set JAXBORG_EXP_DIR to an absolute experiment root outside every Git checkout")
    root = Path(raw).resolve()
    # Catch this worktree, other worktrees, and roots inside unrelated checkouts.
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            raise ValueError(f"JAXBORG_EXP_DIR is inside a Git checkout: {parent}")
    try:
        worktrees = git("worktree", "list", "--porcelain").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        worktrees = []  # Read-only access still works after the producing checkout is retired.
    for line in worktrees:
        if line.startswith("worktree ") and root.is_relative_to(Path(line[9:]).resolve()):
            raise ValueError("JAXBORG_EXP_DIR must be outside every checkout")
    return root


def local_artifact_path(uri):
    u = urlparse(uri)
    if u.scheme not in ("", "file") or u.netloc not in ("", "localhost"):
        raise ValueError(f"Only local MLflow artifacts are supported: {uri}")
    return Path(unquote(u.path)).resolve()


def configure(experiment=None):
    root = experiment_root()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    client = MlflowClient()
    if experiment:
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            client.create_experiment(experiment, artifact_location=(root / "artifacts").as_uri())
            exp = client.get_experiment_by_name(experiment)
        location = local_artifact_path(exp.artifact_location)
        if not location.is_relative_to(root / "artifacts") or exp.lifecycle_stage != "active":
            raise ValueError(
                f"Experiment {experiment!r} has incompatible artifact storage {exp.artifact_location}; "
                "choose a new JAXBORG_MLFLOW_EXPERIMENT or a new root"
            )
        mlflow.set_experiment(experiment)
    return root / "mlflow.db"


def verify_launch(expected_sha=None, *, repo=None, allow_dirty=False):
    repo = Path(repo or _REPO)
    sha = git("rev-parse", "HEAD", repo=repo)
    if expected_sha and sha != expected_sha:
        raise ValueError(f"Wrong launch SHA: expected {expected_sha}, found {sha}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all", repo=repo))
    if dirty and not allow_dirty:
        raise ValueError(
            "Dirty source checkout; commit changes and use a clean pinned checkout. "
            "JAXBORG_ALLOW_DIRTY=1 permits archived, noncanonical development runs only"
        )
    return sha, dirty


def dependency_snapshot():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        record = {"version": distribution.version}
        direct = distribution.read_text("direct_url.json")
        if direct:
            origin = json.loads(direct)
            parsed = urlparse(origin.get("url", ""))
            if origin.get("dir_info", {}).get("editable") and name.lower() == "jaxborg":
                origin["url"] = "editable-project: source identity recorded separately"
            else:
                # Dependency URLs must never archive authentication or query secrets.
                origin["url"] = urlunparse(parsed._replace(netloc=parsed.hostname or "", query="", fragment=""))
            record["origin"] = origin
        packages[name] = record
    return dict(sorted(packages.items()))


def assigned_devices():
    """Enumerate GPUs only under Slurm; outside Slurm explicitly select CPU."""
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("JAX execution requires a Slurm allocation or explicit JAX_PLATFORMS=cpu")
    import jax

    devices = jax.devices()
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("JAXBORG_ALLOW_CPU") != "1":
        if not any(d.platform == "gpu" for d in devices):
            raise ValueError("Allocated job has no JAX GPU backend; prepare an isolated --extra cuda environment")
    return [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in devices]


def _source_files(repo):
    # Archive version-controlled project material only, plus narrowly scoped
    # untracked source during a deliberate development run. Never follow links.
    tracked = git("ls-files", repo=repo).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", repo=repo).splitlines()
    for name in tracked + untracked:
        p = Path(name)
        if any(part.startswith(".") and part not in (".github",) for part in p.parts):
            continue
        if any(
            part.lower() in ("credentials", "secrets", "data", "datasets", "mlruns", "__pycache__") for part in p.parts
        ):
            continue
        if p.suffix.lower() in (".pem", ".key", ".pt", ".safetensors", ".sqlite", ".db"):
            continue
        if name in untracked and (
            p.parts[0] not in ("src", "scripts", "recipes", "tests", "docs")
            or p.suffix not in (".py", ".sh", ".yaml", ".yml", ".md", ".toml")
        ):
            continue
        full = repo / p
        if full.is_file() and not full.is_symlink():
            yield name, full


def source_identity(*, repo=None):
    repo = Path(repo or _REPO)
    sha, dirty = verify_launch(
        os.environ.get("JAXBORG_EXPECTED_SHA"), repo=repo, allow_dirty=os.environ.get("JAXBORG_ALLOW_DIRTY") == "1"
    )
    files = {name: file_hash(p) for name, p in _source_files(repo)}
    return {
        "git_commit": sha,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "repository": git("config", "--get", "remote.origin.url", repo=repo),
        "dirty": dirty,
        "canonical": not dirty,
        "files": files,
        "content_hash": digest(files),
    }


def game_contract(config):
    # Preserve the caller's construction and key handling; identify the code
    # implementing observation/reward/action/topology as well as full variants.
    names = (
        "src/jaxborg/constants.py",
        "src/jaxborg/env.py",
        "src/jaxborg/observations.py",
        "src/jaxborg/blue_observation_contract.py",
        "src/jaxborg/blue_ioc.py",
        "src/jaxborg/learned_red.py",
        "src/jaxborg/joint_env.py",
        "src/jaxborg/scenarios/cc4/game_variant.py",
        "src/jaxborg/evaluation/episode_seeds.py",
        "src/jaxborg/actions/encoding.py",
        "src/jaxborg/scenarios/cc4/topology.py",
    )
    return {
        "resolved": serializable(config),
        "implementation": {n: file_hash(_REPO / n) for n in names if (_REPO / n).exists()},
    }


def _relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or str(p) in (".", ""):
        raise ValueError(f"Invalid artifact path: {name}")
    return p


def read_manifest(run_id):
    configure()
    client = MlflowClient()
    root = local_artifact_path(client.get_run(run_id).info.artifact_uri)
    if not root.is_relative_to(experiment_root() / "artifacts"):
        raise ValueError(f"Run {run_id} is outside configured artifact storage; load old checkpoints by path")
    return json.loads((root / "manifest.json").read_text()), root


def resolve_artifact(reference):
    """Resolve runs:/<id>/<artifact> or a legacy file; verify canonical hashes."""
    if not str(reference).startswith("runs:/"):
        path = Path(reference).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    run_id, name = str(reference)[6:].split("/", 1)
    _relative(name)
    manifest, root = read_manifest(run_id)
    record = manifest["outputs"].get(name)
    path = root / name
    if not record or not record.get("complete") or not path.is_file() or file_hash(path) != record["sha256"]:
        raise ValueError(f"Missing, incomplete or changed artifact: {reference}")
    # Policy loading requires the matching sidecar to be intact as well.
    if record.get("sidecar"):
        resolve_artifact(f"runs:/{run_id}/{record['sidecar']}")
    return path


def resolve_directory(reference):
    if not str(reference).startswith("runs:/"):
        return Path(reference).resolve()
    run_id, prefix = str(reference)[6:].split("/", 1)
    _relative(prefix)
    manifest, root = read_manifest(run_id)
    names = [n for n in manifest["outputs"] if n.startswith(prefix.rstrip("/") + "/")]
    if not names:
        raise ValueError(f"No completed artifacts under {reference}")
    for name in names:
        resolve_artifact(f"runs:/{run_id}/{name}")
    return root / prefix


def input_artifact(reference, *, role="input"):
    path = resolve_artifact(reference)
    run_id = str(reference)[6:].split("/", 1)[0] if str(reference).startswith("runs:/") else None
    if run_id is None:
        # A canonical local artifact path still retains its run lineage.
        for parent in path.parents:
            if (parent / "manifest.json").is_file():
                m = json.loads((parent / "manifest.json").read_text())
                name = str(path.relative_to(parent))
                if name in m.get("outputs", {}):
                    resolve_artifact(f"runs:/{m['run_id']}/{name}")
                    run_id = m["run_id"]
                break
    metadata = {
        "role": role,
        "reference": str(reference),
        "path": str(path),
        "sha256": file_hash(path),
        "source_run_id": run_id,
        "source_step": None,
        "train_code_identity": None,
        "canonical_reference": str(reference) if str(reference).startswith("runs:/") else None,
    }
    if path.suffix in (".pt", ".safetensors"):
        from jaxborg.checkpoint import read_sidecar

        try:
            recipe = read_sidecar(path)
        except FileNotFoundError:
            recipe = None
        if recipe:
            metadata["sidecar_hash"] = digest(recipe)
            metadata["source_step"] = recipe.get("run", {}).get("total_steps")
            metadata["train_code_identity"] = recipe.get("run", {}).get("git_commit")
            metadata["source_run_id"] = run_id or recipe.get("run", {}).get("train_run_id")
            stem = path.stem.removeprefix("model_")
            sidecar = next(
                p for p in (path.with_name(f"recipe_{stem}.yaml"), path.with_name(f"recipe_{stem}.yml")) if p.exists()
            )
            metadata["sidecar_path"] = str(sidecar)
            metadata["sidecar_sha256"] = file_hash(sidecar)
    if run_id and not metadata["canonical_reference"]:
        for parent in path.parents:
            if (parent / "manifest.json").is_file():
                m = json.loads((parent / "manifest.json").read_text())
                if m.get("run_id") == run_id and str(path.relative_to(parent)) in m.get("outputs", {}):
                    metadata["canonical_reference"] = f"runs:/{run_id}/{path.relative_to(parent)}"
                break
    return metadata


def evaluation_fingerprint(inputs, recipe, config, *, evaluator_identity=None, dependencies=None):
    return digest(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": [
                {
                    k: v
                    for k, v in x.items()
                    if k not in ("path", "reference", "source_run_id", "canonical_reference", "sidecar_path")
                }
                for x in inputs
            ],
            "recipe": recipe,
            "contract": game_contract(config),
            "evaluator": evaluator_identity
            or {k: v for k, v in source_identity().items() if k in ("git_commit", "content_hash")},
            "dependencies": dependency_snapshot() if dependencies is None else dependencies,
        }
    )


def find_reusable(fingerprint, expected_outputs):
    configure()
    client = MlflowClient()
    exps = client.search_experiments()
    for run in client.search_runs(
        [x.experiment_id for x in exps],
        filter_string=f"tags.`evaluation.fingerprint` = '{fingerprint}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    ):
        try:
            manifest, _ = read_manifest(run.info.run_id)
            if (
                manifest["status"] != "FINISHED"
                or manifest["kind"] != "evaluation"
                or manifest.get("evaluation_fingerprint") != fingerprint
            ):
                continue
            if not expected_outputs or not manifest.get("outputs"):
                continue
            # Validate every advertised scientific output, not just one filename.
            for name in set(expected_outputs) | set(manifest["outputs"]):
                resolve_artifact(f"runs:/{run.info.run_id}/{name}")
            return run.info.run_id
        except (OSError, KeyError, ValueError):
            continue
    return None


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.stream.write(value)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, key):
        return getattr(self.stream, key)


class Run:
    """One attempt. Stage output, publish complete files, then optionally export."""

    def __init__(
        self,
        recipe,
        *,
        backend,
        seed=None,
        kind="training",
        name=None,
        config=None,
        inputs=(),
        tags=None,
        fingerprint=None,
        supersedes=None,
        bug_reference=None,
    ):
        global _current
        if _current is not None or mlflow.active_run():
            raise ValueError("An MLflow run is already active")
        experiment_root()  # Fail before making any experiment outputs.
        source = source_identity()
        if supersedes and read_manifest(supersedes)[0]["kind"] != "evaluation":
            raise ValueError("supersedes_eval_run_id must identify an evaluation run")
        experiment = os.environ.get("JAXBORG_MLFLOW_EXPERIMENT", f"{recipe.get('algorithm', 'analysis')}-cc4")
        configure(experiment)
        human_name = name or (
            f"{recipe.get('algorithm', 'analysis')}-{backend}-"
            f"{recipe.get('meta', {}).get('name', 'unnamed')}-seed{seed}"
        )
        self.active = mlflow.start_run(run_name=human_name)
        self.info = self.active.info
        self.run_id = self.info.run_id
        self.root = local_artifact_path(self.info.artifact_uri)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix="jaxborg-stage-", dir=experiment_root()))
        self._cleanups = []
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "name": human_name,
            "kind": kind,
            "status": "RUNNING",
            "started_utc": utc_now(),
            "owner": getpass.getuser(),
            "campaign": os.environ.get("JAXBORG_CAMPAIGN"),
            "source": source,
            "train_code_identity": source["content_hash"] if kind == "training" else None,
            "eval_code_identity": source["content_hash"] if kind == "evaluation" else None,
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "recipe": serializable(recipe),
            "recipe_hash": digest(recipe),
            "effective_config": serializable(config or {}),
            "game_contract": game_contract(config or {}),
            "seed": seed,
            "inputs": copy.deepcopy(list(inputs)),
            "outputs": {},
            "exports": [],
            "actual_steps": 0 if kind == "training" else None,
            "supersedes_eval_run_id": supersedes,
            "evaluation_fingerprint": fingerprint,
            "bug_reference": bug_reference,
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "dependencies": dependency_snapshot(),
                "lockfile_hash": file_hash(_REPO / "uv.lock"),
                "node": platform.node(),
                "runtime": {
                    k: os.environ[k]
                    for k in (
                        "SLURM_JOB_ID",
                        "SLURM_JOB_GPUS",
                        "SLURM_JOB_NODELIST",
                        "CUDA_VISIBLE_DEVICES",
                        "JAX_PLATFORMS",
                        "XLA_FLAGS",
                        "OMP_NUM_THREADS",
                        "JAX_ENABLE_X64",
                        "JAXBORG_EXPECTED_SHA",
                        "JAXBORG_ALLOW_CPU",
                    )
                    if k in os.environ
                },
            },
        }
        index = {
            "run.kind": kind,
            "recipe.name": recipe.get("meta", {}).get("name", "unnamed"),
            "algorithm": recipe.get("algorithm", "analysis"),
            "backend": backend,
            "seed": str(seed),
            "arch.name": recipe.get("arch", {}).get("name", "unknown"),
            "git.commit": source["git_commit"],
            "git.branch": source["branch"],
            "source.canonical": str(source["canonical"]).lower(),
            "recipe.hash": digest(recipe),
            "config.hash": digest(config or {}),
            "campaign": self.manifest["campaign"] or "",
            **(tags or {}),
        }
        for phase, key in (("train", "TRAIN_VARIANT"), ("eval", "EVAL_VARIANT")):
            variant = serializable((config or {}).get(key) or (config or {}).get("variant"))
            if variant:
                index[f"game.{phase}.hash"] = digest(variant)
                index[f"game.{phase}.variant"] = variant.get("name", "unknown")
        for i, item in enumerate(inputs):
            index[f"input.{i}.sha256"] = item["sha256"]
            index[f"input.{i}.run_id"] = item.get("source_run_id") or "unknown"
        if fingerprint:
            index["evaluation.fingerprint"] = fingerprint
        if supersedes:
            index["supersedes_eval_run_id"] = supersedes
        if bug_reference:
            index["bug.reference"] = bug_reference
        mlflow.set_tags({k: str(v) for k, v in index.items()})
        self._write_manifest()
        try:
            for i, item in enumerate(self.manifest["inputs"]):
                path = Path(item["path"])
                if file_hash(path) != item["sha256"]:
                    raise ValueError("Input changed before launch")
                if not item.get("canonical_reference"):
                    name = f"inputs/{i}/{path.name}"
                    item["retained_reference"] = self.publish(path, name)
                    if item.get("sidecar_path"):
                        sidecar = Path(item["sidecar_path"])
                        if file_hash(sidecar) != item["sidecar_sha256"]:
                            raise ValueError("Input sidecar changed before launch")
                        self.publish(sidecar, f"inputs/{i}/{sidecar.name}")
            self._write_manifest()
            self.write_json("environment/installed.json", self.manifest["environment"])
            self.publish(_REPO / "uv.lock", "environment/uv.lock")
            archive = self.path("source/source.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for filename in source["files"]:
                    tar.add(_REPO / filename, arcname=filename, recursive=False)
            source["archive_reference"] = self.publish(archive, "source/source.tar.gz")
            source["archive_sha256"] = self.manifest["outputs"]["source/source.tar.gz"]["sha256"]
            if source["dirty"]:
                patch = self.path("source/working-tree.patch")
                patch.write_text(git("diff", "HEAD", "--", *source["files"].keys()) + "\n")
                self.publish(patch, "source/working-tree.patch")
            rp = self.path("recipe.yaml")
            rp.write_text(yaml.safe_dump(serializable(recipe), sort_keys=False))
            self.publish(rp, "recipe.yaml")
        except BaseException:
            self.finish("FAILED")
            raise
        self._stdout, self._stderr = sys.stdout, sys.stderr
        (self.root / "logs").mkdir(exist_ok=True)
        self._log = (self.root / "logs/console.log").open("a", buffering=1)
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._log), _Tee(sys.stderr, self._log)
        _current = self
        print(f"MLflow run: {self.run_id}\nArtifacts: {self.info.artifact_uri}", flush=True)

    def _write_manifest(self):
        temp = self.root / "manifest.json.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temp.replace(self.root / "manifest.json")

    def update(self, **fields):
        self.manifest.update(serializable(fields))
        if "effective_config" in fields:
            self.manifest["game_contract"] = game_contract(fields["effective_config"])
            MlflowClient().set_tag(self.run_id, "config.hash", digest(fields["effective_config"]))
        self._write_manifest()

    def on_close(self, callback):
        self._cleanups.append(callback)

    def path(self, name):
        p = self.staging / _relative(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def input_path(self, index):
        item = self.manifest["inputs"][index]
        return resolve_artifact(item.get("canonical_reference") or item["retained_reference"])

    def publish(self, source, name, *, sidecar=None, step=None, mutable=False):
        name = str(_relative(name))
        if name in self.manifest["outputs"] and not mutable:
            raise ValueError(f"Completed artifact is immutable: {name}")
        if mutable and not name.startswith("logs/"):
            raise ValueError("Only live logs may be republished")
        if sidecar:
            resolve_artifact(f"runs:/{self.run_id}/{sidecar}")
        source = Path(source)
        expected = file_hash(source)
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Upload via the native client, then atomically rename within its
        # resolved local artifact location. A failed upload cannot advertise
        # a half-written checkpoint as complete.
        with tempfile.TemporaryDirectory(dir=self.staging) as tmp:
            pending = Path(tmp) / ("." + destination.name + ".pending")
            shutil.copyfile(source, pending)
            parent = str(PurePosixPath(name).parent)
            MlflowClient().log_artifact(self.run_id, str(pending), None if parent == "." else parent)
            uploaded = destination.parent / pending.name
            if file_hash(uploaded) != expected:
                raise ValueError("Artifact upload hash mismatch")
            uploaded.replace(destination)
        if file_hash(destination) != expected:
            raise ValueError("Published artifact hash mismatch")
        self.manifest["outputs"][name] = {
            "sha256": expected,
            "size": destination.stat().st_size,
            "complete": True,
            "completed_utc": utc_now(),
            "step": step,
            "sidecar": sidecar,
        }
        self._write_manifest()
        return f"runs:/{self.run_id}/{name}"

    def write_json(self, name, value):
        p = self.path(name)
        p.write_text(json.dumps(serializable(value), indent=2) + "\n")
        return self.publish(p, name)

    def export(self, name, destination):
        reference = f"runs:/{self.run_id}/{name}"
        export_artifact(reference, destination, record=False)
        self.manifest["exports"].append(
            {
                "reference": reference,
                "destination": str(Path(destination).resolve()),
                "sha256": self.manifest["outputs"][name]["sha256"],
            }
        )
        self._write_manifest()
        print(f"Export: {destination}\nCanonical: {reference}")

    def finish(self, status="FINISHED", error=None):
        global _current
        for callback in reversed(self._cleanups):
            try:
                callback()
            except Exception as exc:
                print(f"Resource cleanup failed: {exc}", file=sys.stderr)
                if status == "FINISHED":
                    status, error = "FAILED", repr(exc)
        if hasattr(self, "_log"):
            sys.stdout, sys.stderr = self._stdout, self._stderr
            self._log.close()
            self.publish(self.root / "logs/console.log", "logs/console.log", mutable=True)
        self.update(status=status, ended_utc=utc_now(), error=error)
        mlflow.end_run(status=status)
        shutil.rmtree(self.staging)
        if _current is self:
            _current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            traceback.print_exception(exc_type, exc, tb)
        self.finish(
            "KILLED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED" if exc else "FINISHED",
            repr(exc) if exc else None,
        )


def current_run():
    if _current is None:
        raise RuntimeError("Start a run before writing outputs")
    return _current


def export_artifact(reference, destination, *, record=True):
    source = resolve_artifact(reference)
    destination = Path(destination).resolve()
    if destination == source:
        return
    if destination.is_relative_to(experiment_root() / "artifacts"):
        raise ValueError("Exports cannot overwrite canonical run artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        shutil.copyfile(source, tmp)
        if file_hash(tmp) != file_hash(source):
            raise ValueError("Export hash mismatch")
        tmp.replace(destination)
        if record and str(reference).startswith("runs:/"):
            run_id, _ = str(reference)[6:].split("/", 1)
            manifest, root = read_manifest(run_id)
            manifest["exports"].append(
                {
                    "reference": reference,
                    "destination": str(destination),
                    "sha256": file_hash(source),
                    "exported_utc": utc_now(),
                }
            )
            pending = root / "manifest.json.tmp"
            pending.write_text(json.dumps(manifest, indent=2) + "\n")
            pending.replace(root / "manifest.json")
    finally:
        tmp.unlink(missing_ok=True)


def tracked_entrypoint(fn):
    """Finish a manually started caller run on normal return, error or SIGTERM."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        previous = signal.getsignal(signal.SIGTERM)

        def canceled(_sig, _frame):
            raise KeyboardInterrupt("SIGTERM")

        signal.signal(signal.SIGTERM, canceled)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if _current is not None:
                traceback.print_exc()
                _current.finish("KILLED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED", repr(exc))
            raise
        else:
            if _current is not None:
                _current.finish()
            return result
        finally:
            signal.signal(signal.SIGTERM, previous)

    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Inspect MLflow provenance and resolve immutable artifacts")
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("run_id")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("reference")
    search = sub.add_parser("search")
    search.add_argument("--filter", default="")
    annotate = sub.add_parser("annotate")
    annotate.add_argument("run_id")
    annotate.add_argument("--bug", required=True)
    annotate.add_argument("--note", required=True)
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("run_id")
    reconcile.add_argument("--status", choices=["KILLED", "FAILED"], required=True)
    reconcile.add_argument("--note", required=True)
    args = parser.parse_args()
    configure()
    client = MlflowClient()
    if args.action == "inspect":
        print(json.dumps(read_manifest(args.run_id)[0], indent=2))
    elif args.action == "resolve":
        print(resolve_artifact(args.reference))
    elif args.action == "search":
        for r in client.search_runs([e.experiment_id for e in client.search_experiments()], args.filter):
            print(r.info.run_id, r.info.status, r.data.tags.get("mlflow.runName"))
    elif args.action == "annotate":
        client.set_tag(args.run_id, "bug.reference", args.bug)
        client.set_tag(args.run_id, "bug.impact_note", args.note)
    else:
        r = client.get_run(args.run_id)
        if r.info.status != "RUNNING":
            raise ValueError("Only an interrupted RUNNING run can be reconciled")
        m, root = read_manifest(args.run_id)
        m.update(status=args.status, reconciliation_note=args.note, reconciled_utc=utc_now())
        temp = root / "manifest.json.tmp"
        temp.write_text(json.dumps(m, indent=2) + "\n")
        temp.replace(root / "manifest.json")
        client.set_terminated(args.run_id, args.status)


if __name__ == "__main__":
    main()
