import sqlite3

import pytest

from jaxborg.experiment_producers import smoke_store
from jaxborg.experiment_records import validate_records


def test_duplicate_smoke_names_preserve_unknown_links(tmp_path):
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE runs(run_uuid TEXT,name TEXT,status TEXT); "
            "CREATE TABLE params(key TEXT,value TEXT,run_uuid TEXT); "
            "CREATE TABLE tags(key TEXT,value TEXT,run_uuid TEXT); "
            "CREATE TABLE metrics(key TEXT,value REAL,run_uuid TEXT,step INT,timestamp INT);"
        )
        connection.executemany(
            "INSERT INTO runs VALUES (?, ?, ?)", [("first", "same", "FINISHED"), ("second", "same", "FINISHED")]
        )
        connection.execute("INSERT INTO metrics VALUES ('env_steps',2000,'second',2000,1)")
    (tmp_path / "model.safetensors").write_bytes(b"unlinked")
    records = smoke_store(tmp_path)
    validate_records(records)
    attempts = [r for r in records if r["kind"] == "attempt"]
    assert len(attempts) == 2
    assert all(row["budgets"][0]["completed"]["value"] is None for row in attempts)
    observed = next(row for row in attempts if row["id"].endswith(":second"))
    assert observed["facts"]["observed_env_steps_max"]["value"] == 2000
    assert all(r["trial_id"]["value"] is None and r["protocol_id"]["value"] is None for r in attempts)
    assert all(r["retry_of"]["value"] is None and r["inputs"] == [] for r in attempts)
    assert all(r["facts"]["checkpoint_link"]["value"] is None for r in attempts)
    assert all(r["producer_id"] is None for r in records if r["kind"] == "artifact")
    assert not any(r["kind"] == "protocol" for r in records)


def test_unknown_status_rejected(tmp_path):
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE runs(run_uuid TEXT,name TEXT,status TEXT); "
            "CREATE TABLE params(key TEXT,value TEXT,run_uuid TEXT); "
            "CREATE TABLE tags(key TEXT,value TEXT,run_uuid TEXT); "
            "CREATE TABLE metrics(key TEXT,value REAL,run_uuid TEXT,step INT,timestamp INT);"
        )
        connection.execute("INSERT INTO runs VALUES ('unknown','same','UNRECOGNIZED')")
    with pytest.raises(ValueError, match="Unrecognized retained execution status"):
        smoke_store(tmp_path)


def test_active_database_wal_rejected(tmp_path):
    (tmp_path / "mlflow.db-wal").write_bytes(b"uncheckpointed")
    with pytest.raises(ValueError, match="checkpointed"):
        smoke_store(tmp_path)


def test_expected_cells_survive_missing_attempts():
    from jaxborg.experiment_producers.local import frozen_cells

    protocol = {
        "training_seeds": [1, 2],
        "selection": {"candidates": ["original", "seed-1", "seed-2"]},
        "validation_episodes": 100,
        "final_test_episodes": 600,
    }
    trials, mapped, budgets = frozen_cells(protocol, {}, "protocol")
    assert len(trials) == 7
    assert mapped == budgets == {}
    assert {cell["replicate_id"] for cell in trials if cell["condition_id"] == "fresh-red-frozen-blue"} == {"1", "2"}


def test_one_execution_cannot_fill_multiple_intended_cells():
    from jaxborg.experiment_producers.local import frozen_cells

    protocol = {
        "training_seeds": [1, 2],
        "selection": {"candidates": []},
        "validation_episodes": 100,
        "final_test_episodes": 600,
    }
    state = {"training": {"seed-1": {"run_id": "same"}, "seed-2": {"run_id": "same"}}}
    with pytest.raises(ValueError, match="multiple intended"):
        frozen_cells(protocol, state, "protocol")


def test_source_identity_changes_with_an_imported_helper(tmp_path, monkeypatch):
    from jaxborg.experiment_producers import source

    analyzer = tmp_path / "analyze.py"
    helper = tmp_path / "helper.py"
    analyzer.write_text("print('analyze')\n")
    helper.write_text("CONTRACT = 1\n")
    monkeypatch.setattr(
        source,
        "analysis_source_files",
        lambda _: {
            "experiments/frozen-defender-response/analyze.py": analyzer,
            "src/jaxborg/experiment_records/core.py": helper,
        },
    )
    before = source.analysis_source_identity(analyzer)
    helper.write_text("CONTRACT = 2\n")
    assert source.analysis_source_identity(analyzer) != before
