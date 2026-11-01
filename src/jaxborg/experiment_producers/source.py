"""Identify the executable local sources consumed by a saved frozen analysis."""

import hashlib
import importlib
from pathlib import Path

from jaxborg.experiment_records import json_sha256

MODULES = (
    "jaxborg",
    "jaxborg.frozen_bundle",
    "jaxborg.experiment_records",
    "jaxborg.experiment_records.core",
    "jaxborg.experiment_producers",
    "jaxborg.experiment_producers.local",
    "jaxborg.experiment_producers.selection",
    "jaxborg.experiment_producers.source",
    "jaxborg.evidence_archive",
)


def analysis_source_files(analyzer):
    """Return canonical repository paths mapped to actual imported module files."""
    files = {"studies/frozen-defender-response/analyze.py": Path(analyzer).resolve()}
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        suffix = "/__init__.py" if path.name == "__init__.py" else ".py"
        files["src/" + name.replace(".", "/") + suffix] = path
    return files


def analysis_source_identity(analyzer):
    """Hash canonical path-to-SHA256 JSON for the analyzer and imported helpers."""
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in analysis_source_files(analyzer).items()
    }
    return "sha256:" + json_sha256(hashes)
