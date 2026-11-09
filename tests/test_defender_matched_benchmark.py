"""Evidence-contract tests for the fixed-policy matched benchmark (no simulator)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1] / "studies" / "defender-transfer"
spec = importlib.util.spec_from_file_location("matched_benchmark", STUDY / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MatchedBenchmarkContracts(unittest.TestCase):
    def test_full_and_smoke_ledgers_are_explicit_and_disjoint(self):
        definition = json.loads((STUDY / "benchmark-protocol.json").read_text())
        policies = [{"id": name} for name in definition["full_policy_ids"]]
        cases = [
            {
                "case_id": f"stock-{i:04}",
                "env_seed": 23000000 + i,
                "policy_seed": 21000000 + i,
                "dynamics_seed": 22000000 + i,
            }
            for i in range(100)
        ]
        full = benchmark.ledger_for(policies, cases, definition, "full")
        smoke = benchmark.ledger_for(policies, cases, definition, "smoke")
        assert len(full) == 96
        assert len(smoke) == 3
        assert {row["case_index"] for row in full} == set(range(16))
        assert {row["case_index"] for row in smoke} == {99}
        assert len({row["episode_id"] for row in full + smoke}) == 99
        with self.assertRaisesRegex(ValueError, "exactly one"):
            benchmark.ledger_for(policies[:-1], cases, definition, "full")

    def test_retention_rejects_tampering_and_path_escape(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        source = tmp_path / "source"
        source.write_bytes(b"original checkpoint")
        inputs = tmp_path / "inputs"
        item = benchmark.retain(source, inputs, "model.safetensors")
        assert benchmark.checked_path(inputs, item).read_bytes() == b"original checkpoint"
        (inputs / item["path"]).write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "differs"):
            benchmark.checked_path(inputs, item)
        with self.assertRaisesRegex(ValueError, "differs"):
            benchmark.checked_path(inputs, {"path": "../source", "sha256": benchmark.digest(source)})

    def test_episode_contract_rejects_wrong_seed_or_incomplete_trace(self):
        declared = {"policy_seed": 2, "dynamics_seed": 3}
        result = {
            "reward": -3.0,
            "steps": 2,
            "snapshot_sha256": "abc",
            **declared,
            "per_step_returns": [-1.0, -2.0],
            "per_step_components": [{}, {}],
            "per_step_actions": [[0] * 5, [0] * 5],
            "per_step_done": [False, True],
            "per_step_phases": [0, 0],
        }
        benchmark.validate_episode(result, declared, {"sha256": "abc"}, 2)
        with self.assertRaisesRegex(ValueError, "seed"):
            benchmark.validate_episode({**result, "policy_seed": 9}, declared, {"sha256": "abc"}, 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            benchmark.validate_episode({**result, "per_step_actions": [[0] * 5]}, declared, {"sha256": "abc"}, 2)
        with self.assertRaisesRegex(ValueError, "return differs"):
            benchmark.validate_episode({**result, "reward": -10}, declared, {"sha256": "abc"}, 2)

    def test_pairing_uses_case_identity_and_rejects_changed_scenarios(self):
        shared = {"snapshot_sha256": "same", "policy_seed": 2, "dynamics_seed": 3}
        rows = [
            {"policy_id": "a", "case_id": "x", "reward": -1, **shared},
            {"policy_id": "a", "case_id": "y", "reward": -10, **shared},
            {"policy_id": "b", "case_id": "y", "reward": -4, **shared},
            {"policy_id": "b", "case_id": "x", "reward": -3, **shared},
        ]
        means, gaps = benchmark.summarize(rows)
        assert means[0]["mean_return"] == -5.5
        assert [(gap["case_id"], gap["return_gap"]) for gap in gaps] == [("x", 2), ("y", -6)]
        with self.assertRaisesRegex(ValueError, "identical scenario coverage"):
            benchmark.summarize(rows[:-1])
        with self.assertRaisesRegex(ValueError, "seed contracts"):
            benchmark.summarize([*rows[:-1], {**rows[-1], "dynamics_seed": 999}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            benchmark.summarize([*rows, rows[0]])


if __name__ == "__main__":
    unittest.main()
