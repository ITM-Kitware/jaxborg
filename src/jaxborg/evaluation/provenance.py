"""Source and installed dependency identity for retained evaluation evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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

    def verify_sources(self):
        """Reject a source snapshot that changed during execution."""
        for name, path in self.source_files.items():
            if not path.is_file() or _digest(path) != self.hashes[name]:
                raise ValueError(f"Evaluation source changed during execution: {name}")
        package = Path(__file__).resolve().parents[1]
        source_names = {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
        recorded_names = {
            name.removeprefix("src/jaxborg/") for name in self.source_files if name.startswith("src/jaxborg/")
        }
        if source_names != recorded_names:
            raise ValueError("Evaluation source file set changed during execution")


def collect_provenance(*, allow_dirty=False):
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
    hashes = {name: _digest(path) for name, path in files.items()}
    manifest_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    configuration = {
        "source_revision": revision,
        "working_tree_dirty": dirty,
        "development_override": bool(allow_dirty),
        "source_manifest_sha256": manifest_hash,
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
    return Provenance(facts, files, configuration, hashes)
