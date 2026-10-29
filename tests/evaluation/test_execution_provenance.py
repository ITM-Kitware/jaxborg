"""Exercise pinned source requests in isolated source checkouts."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from jaxborg.evaluation import provenance


@pytest.fixture
def checkout(tmp_path):
    package = tmp_path / "src/jaxborg/evaluation"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    shutil.copyfile(provenance.__file__, package / "provenance.py")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.1"\n')
    (tmp_path / "uv.lock").write_text("version = 1\n")
    study = tmp_path / "study"
    study.mkdir()
    (study / "helper.py").write_text("OFFSET = 1\n")
    (study / "rules.py").write_text("from study import helper\ndef build(variant, *, offset=1):\n    return variant\n")
    (study / "run.py").write_text("# Maintained study entrypoint.\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Initial fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    return tmp_path, revision


def execute(checkout, code):
    root, _ = checkout
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src") + os.pathsep + str(root), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_pinned_entrypoint_and_bound_builder_helpers_are_retained(checkout):
    _, revision = checkout
    result = execute(
        checkout,
        f"""
import json
from pathlib import Path
from functools import partial
from study.rules import build
from jaxborg.evaluation.provenance import collect_provenance, builder_identity
p = collect_provenance(expected_revision={revision!r}, entrypoint=Path('study/run.py'), source_roots=[Path('study')])
p.verify_sources()
print(json.dumps({{"builder": builder_identity(partial(build, offset=3), p), "files": list(p.source_files)}}))
""",
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["builder"]["arguments"] == {"offset": 3}
    assert {"study/helper.py", "study/rules.py"} == value["builder"]["source_hashes"].keys()
    assert "study/run.py" in value["files"]


@pytest.mark.parametrize("damage", ["revision", "missing_helper", "helper_class", "external_helper", "mutation"])
def test_pinned_execution_rejects_incomplete_or_changed_sources(checkout, damage):
    root, revision = checkout
    if damage == "missing_helper":
        (root / "src/jaxborg/helper_user.py").write_text("from study import helper\n")
    if damage == "helper_class":
        (root / "study/helper.py").write_text("class Rule:\n    pass\n")
        (root / "src/jaxborg/helper_user.py").write_text("from study.helper import Rule\n")
    if damage == "external_helper":
        external = root.parent / f"{root.name}_external"
        external.mkdir()
        (external / "outside.py").write_text("class Rule:\n    pass\n")
        (root / "src/jaxborg/helper_user.py").write_text(
            f"import sys\nsys.path.insert(0, {str(external)!r})\nfrom outside import Rule\n"
        )
    mutation = 'Path("src/jaxborg/__init__.py").write_text("VALUE = 2")' if damage == "mutation" else ""
    result = execute(
        checkout,
        f"""
from pathlib import Path
from jaxborg.evaluation.provenance import collect_provenance
{"import jaxborg.helper_user" if damage in {"missing_helper", "helper_class", "external_helper"} else ""}
p = collect_provenance(expected_revision={("0" * 40 if damage == "revision" else revision)!r}, allow_dirty=True)
{mutation}
p.verify_sources()
""",
    )
    assert result.returncode != 0
    assert {
        "revision": "pinned request",
        "missing_helper": "not in declared source closure",
        "helper_class": "not in declared source closure",
        "external_helper": "not in declared source closure",
        "mutation": "source changed",
    }[damage] in result.stderr


def test_unregistered_builder_module_cannot_hide_external_helper_classes(checkout):
    root, revision = checkout
    external = root.parent / f"{root.name}_unregistered"
    external.mkdir()
    (external / "outside.py").write_text("class Rule:\n    pass\n")
    (root / "study/rules.py").write_text(
        f"import sys\nsys.path.insert(0, {str(external)!r})\nfrom outside import Rule\n"
        "def build(variant, *, offset=1):\n    return Rule()\n"
    )
    result = execute(
        checkout,
        f"""
from pathlib import Path
import importlib.util
from jaxborg.evaluation.provenance import collect_provenance, builder_identity
spec = importlib.util.spec_from_file_location('unregistered_rules', 'study/rules.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
p = collect_provenance(expected_revision={revision!r}, allow_dirty=True, source_roots=[Path('study')])
builder_identity(module.build, p)
""",
    )
    assert result.returncode != 0
    assert "not in declared source closure" in result.stderr
