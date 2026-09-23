"""Latest completed budgets and paired training seeds drive diversity plots."""

import sqlite3

import pandas as pd

from plots.cotraining.data import completed_runs, paired_rows


def test_completed_runs_use_saved_budget_and_latest_run_not_largest_budget(tmp_path):
    db = tmp_path / "mlflow.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE runs(run_uuid TEXT,name TEXT,status TEXT,lifecycle_stage TEXT,start_time INT); "
            "CREATE TABLE metrics(run_uuid TEXT,key TEXT,step INT); "
            "CREATE TABLE params(run_uuid TEXT,key TEXT,value TEXT);"
        )
        for uid, start, steps, budget in [
            ("old", 1, 69984000, 70000000),
            ("new", 2, 49968000, 50000000),
            ("partial", 3, 10000000, 50000000),
        ]:
            conn.execute(
                "INSERT INTO runs VALUES(?,?,'FINISHED','active',?)", (uid, "ippo-jax-cotraining_lstm-seed42", start)
            )
            conn.execute("INSERT INTO metrics VALUES(?,'team.blue.return',?)", (uid, steps))
            conn.execute("INSERT INTO params VALUES(?,'recipe.train.total_timesteps',?)", (uid, str(budget)))
    result = completed_runs(db, families=["lstm"])
    assert result.ids == frozenset({"new"})


def test_pairing_excludes_missing_training_seed_separately_for_each_opponent():
    rows = [
        {"family": "lstm", "condition": c, "seed": s, "red": red}
        for red in ["fsm", "cia_c"]
        for c in ["single", "diverse"]
        for s in [42, 100, 200]
        if not (c == "diverse" and s == 200 and red == "cia_c")
    ]
    actual = paired_rows(pd.DataFrame(rows), ["family", "seed", "red"])
    assert len(actual) == 10
    assert set(actual[actual.red == "fsm"].seed) == {42, 100, 200}
    assert set(actual[actual.red == "cia_c"].seed) == {42, 100}
