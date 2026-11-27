"""Source and installed dependency identity for retained evaluation evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dependencies():
    dependencies = []
    for distribution in importlib.metadata.distributions():
        item = {"name": distribution.metadata["Name"], "version": distribution.version}
        direct = distribution.read_text("direct_url.json")
        if direct:
            vcs = json.loads(direct).get("vcs_info")
            if vcs:
                item["vcs"] = vcs["vcs"]
                item["commit_id"] = vcs["commit_id"]
        dependencies.append(item)
    return sorted(dependencies, key=lambda item: (item["name"].lower(), item["version"]))


@dataclass(frozen=True)
class Provenance:
    facts: dict
    source_files: dict[str, Path]
    configuration: dict
    hashes: dict[str, str]
    source_roots: tuple[Path, ...] = field(default_factory=tuple)

    def verify_sources(self):
        """Reject a source snapshot that changed during execution."""
        for name, path in self.source_files.items():
            if not path.is_file() or _digest(path) != self.hashes[name]:
                raise ValueError(f"Evaluation source changed during execution: {name}")
        package = Path(__file__).resolve().parents[1]
        root = package.parent.parent
        for directory in (package, *self.source_roots):
            source_names = {path.relative_to(root).as_posix() for path in directory.rglob("*.py")}
            recorded_names = {name for name, path in self.source_files.items() if path.is_relative_to(directory)}
            if source_names != recorded_names:
                raise ValueError("Evaluation source file set changed during execution")
        if _git(root, "rev-parse", "HEAD") != self.configuration["source_revision"]:
            raise ValueError("Executing source revision changed during execution")
        _check_local_imports(root, self.source_files)


def writer_identity():
    """Identify the process namespace in which a writer PID can be checked."""
    identity = {"hostname": socket.gethostname(), "pid": os.getpid()}
    try:
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        identity["pid_namespace"] = os.readlink("/proc/self/ns/pid")
    except OSError:
        pass
    return identity


def collect_provenance(*, allow_dirty=False, expected_revision=None, entrypoint=None, source_roots=()):
    """Capture the executing checkout, without credentials or unrelated files.

    This entrypoint requires a source checkout. Installed package versions describe
    the execution environment; the retained lock describes the intended resolution.
    Neither is substituted for the other.
    """
    package = Path(__file__).resolve().parents[1]
    root = package.parent.parent
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Evaluation capture requires a source checkout")
    revision = _git(root, "rev-parse", "HEAD")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Executing source revision differs from the pinned request")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=all"))
    if dirty and not allow_dirty:
        raise ValueError("Evaluation source checkout is dirty; use --allow-dirty for a retained development execution")
    for name, module in tuple(sys.modules.items()):
        if name == "jaxborg" or name.startswith("jaxborg."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(package):
                raise ValueError(f"Evaluation imported source from another checkout: {name}")
    files = {path.relative_to(root).as_posix(): path for path in sorted(package.rglob("*.py"))}
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if not path.is_file():
            raise ValueError(f"Evaluation source checkout lacks {name}")
        files[name] = path
    declared_roots = tuple(Path(path).resolve() for path in source_roots)
    for directory in declared_roots:
        if not directory.is_dir() or not directory.is_relative_to(root):
            raise ValueError("Local helper source roots must be directories in the executing checkout")
        for path in directory.rglob("*.py"):
            if path.is_symlink():
                raise ValueError("Local source closure must not follow symbolic links")
            files[path.relative_to(root).as_posix()] = path
    if entrypoint is not None:
        path = Path(entrypoint).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("Entrypoint must be a retained file in the executing checkout")
        files[path.relative_to(root).as_posix()] = path
    _check_local_imports(root, files)
    hashes = {name: _digest(path) for name, path in files.items()}
    manifest_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    configuration = {
        "source_revision": revision,
        "working_tree_dirty": dirty,
        "development_override": bool(allow_dirty),
        "source_manifest_sha256": manifest_hash,
        "expected_revision": expected_revision,
        "working_tree_patch": _git(root, "diff", "HEAD", "--", *files) if dirty else "",
        "command": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "writer": writer_identity(),
        "dependencies": _dependencies(),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    facts = {
        "source_revision": {"value": revision, "source": "git HEAD of executing source checkout"},
        "source_identity": {"value": manifest_hash, "source": "SHA256 of canonical retained source-file hashes"},
        "development_execution": {
            "value": dirty or bool(allow_dirty),
            "source": "checkout status and supplied override",
        },
        "dependencies": {"value": configuration["dependencies"], "source": "installed distribution metadata"},
        "runtime": {"value": configuration["runtime"], "source": "Python platform observations"},
    }
    return Provenance(facts, files, configuration, hashes, declared_roots)


def _dependency_roots():
    roots = {
        Path(path).resolve()
        for key, path in sysconfig.get_paths().items()
        if key in {"stdlib", "platstdlib", "purelib", "platlib"}
    }
    for distribution in importlib.metadata.distributions():
        direct = distribution.read_text("direct_url.json")
        if direct:
            value = json.loads(direct)
            location = urlparse(value.get("url", ""))
            if location.scheme == "file" and distribution.metadata["Name"].lower() != "jaxborg":
                roots.add(Path(unquote(location.path)).resolve())
    return roots


def _imported_module(value):
    if inspect.ismodule(value):
        return value
    return inspect.getmodule(value) if inspect.isfunction(value) or inspect.isclass(value) else None


def _check_import_values(values, retained, dependencies):
    for value in values:
        imported = _imported_module(value)
        imported_path = getattr(imported, "__file__", None)
        if imported_path:
            candidate = Path(imported_path).resolve()
            if candidate not in retained and not any(candidate.is_relative_to(root) for root in dependencies):
                raise ValueError(f"Imported local helper is not in declared source closure: {candidate}")


def _check_local_imports(root, files):
    del root
    retained = {path.resolve() for path in files.values()}
    dependencies = _dependency_roots()
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve() in retained:
            _check_import_values(tuple(vars(module).values()), retained, dependencies)
