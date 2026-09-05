from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jaxborg.recipe import project_eval, project_jax
from jaxborg.topology_bank_cli import main as materialize_main
from jaxborg.topology_banks import (
    MAX_TOPOLOGY_BANK_SIZE,
    expand_topology_bank,
    parse_topology_generation,
    validate_eval_topology_override,
    validate_topology_split,
)


def _recipe(*, train_topology=None, eval_topology=None):
    train = {
        "episode_length": 500,
        "total_timesteps": 1_000,
        "variant": "cc4_stock",
    }
    evaluation = {"variant": "cc4_stock"}
    if train_topology:
        train.update(train_topology)
    if eval_topology:
        evaluation.update(eval_topology)
    return {
        "meta": {"name": "topology-generation-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": train,
        "eval": evaluation,
        "jax": {"num_envs": 1},
    }


def _generation(cache_dir: Path, **overrides):
    config = {
        "generator": "jax",
        "seed_start": 7,
        "count": 3,
        "cache_dir": str(cache_dir),
    }
    config.update(overrides)
    return {"topology_generation": config}


def test_count_and_inclusive_end_expand_to_deterministic_seed_paths(tmp_path):
    count_section = _generation(tmp_path / "count")
    end_section = _generation(tmp_path / "end", count=None, seed_end=9)
    del end_section["topology_generation"]["count"]

    count_paths = expand_topology_bank(count_section, scope="train", repo_root=tmp_path)
    end_paths = expand_topology_bank(end_section, scope="eval", repo_root=tmp_path)

    assert [path.name for path in count_paths] == [
        "jax_seed_0000000007.snapshot.npz",
        "jax_seed_0000000008.snapshot.npz",
        "jax_seed_0000000009.snapshot.npz",
    ]
    assert [path.name for path in end_paths] == [path.name for path in count_paths]
    assert not (tmp_path / "count").exists()
    assert not (tmp_path / "end").exists()


def test_materialization_is_lazy_and_cached_in_projection(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    calls = []

    def fake_export(generator, seed, path):
        calls.append((generator, seed, path))
        path.write_bytes(b"snapshot")

    monkeypatch.setattr(topology_banks, "_export_snapshot", fake_export)
    monkeypatch.setattr(topology_banks, "_validate_cached_snapshot", lambda *_args, **_kwargs: None)
    recipe = _recipe(train_topology=_generation(tmp_path / "cache"))

    first = project_jax(recipe)["TOPOLOGY_BANK"]
    second = project_jax(recipe)["TOPOLOGY_BANK"]

    assert first == second
    assert len(first) == 3
    assert [(generator, seed) for generator, seed, _ in calls] == [("jax", 7), ("jax", 8), ("jax", 9)]
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in first)


def test_project_eval_materializes_only_eval_declaration(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    calls = []

    def fake_export(generator, seed, path):
        calls.append((generator, seed))
        path.write_bytes(b"snapshot")

    monkeypatch.setattr(topology_banks, "_export_snapshot", fake_export)
    monkeypatch.setattr(topology_banks, "_validate_cached_snapshot", lambda *_args, **_kwargs: None)
    recipe = _recipe(eval_topology=_generation(tmp_path / "eval", generator="cyborg", count=2))

    inspected = project_eval(recipe)
    assert calls == []
    assert len(inspected["TOPOLOGY_BANK"]) == 2

    projected = project_eval(recipe, materialize_topologies=True)

    assert calls == [("cyborg", 7), ("cyborg", 8)]
    assert [path.name for path in projected["TOPOLOGY_BANK"]] == [
        "cyborg_seed_0000000007.snapshot.npz",
        "cyborg_seed_0000000008.snapshot.npz",
    ]
    assert projected["TOPOLOGY_SAMPLING"] == "exhaustive"


def test_project_eval_accepts_random_bank_sampling(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    monkeypatch.setattr(topology_banks, "_export_snapshot", lambda generator, seed, path: path.write_bytes(b"x"))
    monkeypatch.setattr(topology_banks, "_validate_cached_snapshot", lambda *_args, **_kwargs: None)
    recipe = _recipe(eval_topology=_generation(tmp_path / "eval", count=1))
    recipe["eval"]["topology_sampling"] = "random"

    assert project_eval(recipe, materialize_topologies=True)["TOPOLOGY_SAMPLING"] == "random"


def test_materialize_cli_dry_run_prints_both_disjoint_ranges_without_writing(tmp_path, capsys):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    recipe = _recipe(
        train_topology=_generation(train_dir, seed_start=0, count=2),
        eval_topology=_generation(eval_dir, seed_start=10, count=1),
    )
    recipe_path = tmp_path / "split.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    materialize_main(["--recipe", str(recipe_path), "--dry-run"])

    output = capsys.readouterr().out
    assert "train: 2 topology snapshot(s)" in output
    assert "eval: 1 topology snapshot(s)" in output
    assert "jax_seed_0000000000.snapshot.npz" in output
    assert "jax_seed_0000000010.snapshot.npz" in output
    assert not train_dir.exists()
    assert not eval_dir.exists()


def test_legacy_explicit_bank_still_projects(tmp_path):
    path = tmp_path / "existing.snapshot.npz"
    recipe = _recipe(train_topology={"topology_bank": [str(path)]})

    assert project_jax(recipe)["TOPOLOGY_BANK"] == (path.resolve(),)


def test_duplicate_explicit_paths_are_rejected(tmp_path):
    path = tmp_path / "duplicate.snapshot.npz"
    recipe = _recipe(train_topology={"topology_bank": [str(path), str(path)]})

    with pytest.raises(ValueError, match="duplicate paths"):
        project_jax(recipe)


@pytest.mark.parametrize("bank", [None, False, 0, {}, [], ""])
def test_falsey_or_empty_explicit_bank_is_rejected(tmp_path, bank):
    recipe = _recipe(train_topology={"topology_bank": bank})

    with pytest.raises(ValueError, match="topology_bank"):
        project_jax(recipe)


@pytest.mark.parametrize("entry", [None, False, 0, ""])
def test_invalid_explicit_bank_entry_is_rejected(tmp_path, entry):
    recipe = _recipe(train_topology={"topology_bank": [entry]})

    with pytest.raises(ValueError, match=r"topology_bank\[0\] must be a non-empty path"):
        project_jax(recipe)


def test_empty_bank_cannot_silently_coexist_with_generation(tmp_path):
    section = _generation(tmp_path)
    section["topology_bank"] = []

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_topology_generation(section, scope="train", repo_root=tmp_path)


def test_single_sided_explicit_bank_does_not_read_snapshot_metadata(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    path = tmp_path / "not-created-yet.snapshot.npz"
    recipe = _recipe(train_topology={"topology_bank": [str(path)]})
    monkeypatch.setattr(
        topology_banks,
        "_path_provenance",
        lambda _path: pytest.fail("one-sided split validation must not inspect metadata"),
    )

    validate_topology_split(recipe, repo_root=tmp_path)


def test_cache_reuse_requires_strict_provenance_and_a_usable_snapshot(tmp_path, monkeypatch):
    from jaxborg import topology_banks
    from jaxborg.scenarios.cc4 import topology

    path = tmp_path / "cached.snapshot.npz"
    path.touch()
    metadata = {
        "format": topology.TOPOLOGY_SNAPSHOT_FORMAT,
        "format_version": topology.TOPOLOGY_SNAPSHOT_VERSION,
        "source": "generated",
        "source_seed": 7,
        "jaxborg_git_sha": "recorded-for-audit",
    }
    loaded = []
    monkeypatch.setattr(topology, "load_topology_metadata", lambda _path: metadata)
    monkeypatch.setattr(topology, "load_topology", lambda loaded_path: loaded.append(loaded_path))

    topology_banks._validate_cached_snapshot(path, source="generated", seed=7)

    assert loaded == [path]

    metadata["source_seed"] = True
    with pytest.raises(ValueError, match="provenance"):
        topology_banks._validate_cached_snapshot(path, source="generated", seed=1)

    metadata["source_seed"] = 7
    metadata["format_version"] = -1
    with pytest.raises(ValueError, match="snapshot format"):
        topology_banks._validate_cached_snapshot(path, source="generated", seed=7)


def test_generation_range_has_a_safe_bank_size_limit(tmp_path):
    section = _generation(tmp_path, count=MAX_TOPOLOGY_BANK_SIZE + 1)

    with pytest.raises(ValueError, match="maximum supported bank size"):
        parse_topology_generation(section, scope="train", repo_root=tmp_path)


def test_generated_source_seed_overlap_fails_before_materialization(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    monkeypatch.setattr(
        topology_banks,
        "_export_snapshot",
        lambda *_args, **_kwargs: pytest.fail("overlap must fail before generation"),
    )
    recipe = _recipe(
        train_topology=_generation(tmp_path / "train", seed_start=0, count=5),
        eval_topology=_generation(tmp_path / "eval", seed_start=4, count=2),
    )

    with pytest.raises(ValueError, match="source-seed overlap: generated:4"):
        project_jax(recipe)


def test_explicit_path_overlap_is_rejected(tmp_path):
    path = tmp_path / "same.snapshot.npz"
    recipe = _recipe(
        train_topology={"topology_bank": [str(path)]},
        eval_topology={"topology_bank": [str(path)]},
    )

    with pytest.raises(ValueError, match="topology path overlap"):
        project_eval(recipe)


def test_cli_eval_override_cannot_select_a_generated_training_snapshot(tmp_path):
    train_topology = _generation(tmp_path / "train", seed_start=20, count=2)
    recipe = _recipe(train_topology=train_topology)
    training_paths = expand_topology_bank(train_topology, scope="train", repo_root=tmp_path)

    with pytest.raises(ValueError, match="topology path overlap"):
        validate_eval_topology_override(recipe, [training_paths[0]], repo_root=tmp_path)


def test_distinct_explicit_paths_with_same_source_seed_are_rejected(tmp_path, monkeypatch):
    from jaxborg import topology_banks

    train_path = tmp_path / "train.snapshot.npz"
    eval_path = tmp_path / "eval.snapshot.npz"
    train_path.touch()
    eval_path.touch()
    monkeypatch.setattr(topology_banks, "_path_provenance", lambda _path: ("cyborg", 12))
    recipe = _recipe(
        train_topology={"topology_bank": [str(train_path)]},
        eval_topology={"topology_bank": [str(eval_path)]},
    )

    with pytest.raises(ValueError, match="source-seed overlap: cyborg:12"):
        project_eval(recipe)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"generator": "unknown"}, "generator must be one of"),
        ({"count": 0}, "count must be positive"),
        ({"seed_start": -1}, "seeds must be in"),
        ({"seed_start": True}, "seed_start must be an integer"),
        ({"seed_start": 1.5}, "seed_start must be an integer"),
        ({"count": True}, "count must be an integer"),
        ({"count": 2.5}, "count must be an integer"),
        ({"seed_end": 9}, "exactly one of seed_end or count"),
        ({"surprise": True}, "unknown keys"),
    ],
)
def test_invalid_generation_schema_is_rejected(tmp_path, updates, message):
    section = _generation(tmp_path)
    section["topology_generation"].update(updates)

    with pytest.raises(ValueError, match=message):
        parse_topology_generation(section, scope="train", repo_root=tmp_path)


def test_explicit_bank_and_generation_are_mutually_exclusive(tmp_path):
    section = _generation(tmp_path)
    section["topology_bank"] = [tmp_path / "explicit.snapshot.npz"]

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_topology_generation(section, scope="train", repo_root=tmp_path)
