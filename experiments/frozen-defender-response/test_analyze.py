"""Exercise the report CLI using synthetic retained records only."""

import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import runpy
import statistics
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCRIPT = HERE / "analyze.py"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


class SyntheticBundle:
    def __init__(self, root):
        self.root = root
        self.payloads = {}
        self.owners = {}
        self.revision = "a" * 40
        self.seeds = [11, 22, 33]
        self.validation_seeds = [100, 101]
        self.test_seeds = [200, 201, 202]
        self.source_policy = "runs:/source/source/model.safetensors"
        self.source_hash = sha256(b"synthetic original checkpoint identity")
        self.topology = "runs:/source/topologies/topology.npz"
        self.topology_hash = sha256(b"synthetic topology identity")
        self.topology_path = "/original/location/topology.npz"
        self.policies = {
            "blue": {"obs_dim": 450, "action_dim": 242, "arch": {"name": "mappo"}},
            "red": {"obs_dim": 706, "action_dim": 1106, "arch": {"name": "mappo"}},
        }
        self.challenger_policies = copy.deepcopy(self.policies)
        self.challenger_policies["red"]["arch"] = {"name": "shared"}
        self.variant = {
            "name": "cc4_stock",
            "num_steps": 500,
            "red_reward": "zero_sum",
            "blue_observation_version": 2,
            "cage4_enhanced_obs": True,
        }
        source_code = b"EPISODE_LENGTH = 500\n"
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as stream:
            member = tarfile.TarInfo("src/contract.py")
            member.size = len(source_code)
            stream.addfile(member, io.BytesIO(source_code))
        source_archive = archive.getvalue()
        lock = b"version = 1\n"
        self.source = {
            "git_commit": self.revision,
            "dirty": False,
            "files": {"src/contract.py": sha256(source_code)},
            "archive_reference": "runs:/source/source/archive.tar",
            "archive_sha256": sha256(source_archive),
        }
        self.lock_hash = sha256(lock)
        self.owner("source", "analysis", {})
        self.add("source", "source/archive.tar", source_archive)
        self.add("source", "environment/uv.lock", lock)
        sidecar = self.add("source", "source/recipe.yaml", b"name: synthetic-source\n")
        self.protocol = {
            "campaign": "synthetic-response",
            "source_revision": self.revision,
            "protocol_owner_run_id": "source",
            "source": {
                "checkpoint": self.source_policy,
                "sha256": self.source_hash,
                "sidecar_reference": sidecar,
                "sidecar_sha256": sha256(self.payloads[sidecar]),
                "policies": {
                    team: {
                        "obs_dim": policy["obs_dim"],
                        "action_dim": policy["action_dim"],
                        "architecture": policy["arch"],
                        "parameter_sha256": sha256(team.encode()),
                    }
                    for team, policy in self.policies.items()
                },
            },
            "challenger": {"algorithm": "ippo", "architecture": {"name": "shared"}},
            "game": {
                "rules": "cc4_stock",
                "episode_length": 500,
                "reward": "zero_sum",
                "blue_observation_version": 2,
                "enhanced_observations": True,
                "topology": self.topology,
                "topology_sha256": self.topology_hash,
                "layout_count": 1,
                "topology_reconstruction": "Synthetic topology identity only",
            },
            "randomness": {
                "validation_episode_roots": self.validation_seeds,
                "final_test_episode_roots": self.test_seeds,
                "training_environment_roots": self.seeds,
                "training_policy_rollout_roots": [seed + 1 for seed in self.seeds],
                "episode_seed_scheme": "base_times_count_plus_replica_v1",
            },
            "oracle_budget_per_attempt": {
                "requested_steps": 21,
                "completed_steps": 20,
                "updates": 2,
                "steps_per_update": 10,
            },
            "training_seeds": self.seeds,
            "validation_episodes": len(self.validation_seeds),
            "final_test_episodes": len(self.test_seeds),
            "selection": {
                "rule": "lowest validation mean Blue episode return; original Red is fallback",
                "candidates": ["original", *[f"seed-{seed}" for seed in self.seeds]],
                "tie_break": ["original", *[f"seed-{seed}" for seed in self.seeds]],
            },
            "confidence_interval": {"resamples": 10000, "seed": 7},
            "dependencies": {"lockfile_sha256": self.lock_hash},
        }
        self.state = {
            "campaign": self.protocol["campaign"],
            "training": {},
            "validation": {},
            "test": {},
            "started_epoch": 0,
            "finished_epoch": 100,
        }
        self.candidates = {"original": (self.source_policy, self.source_hash, self.policies)}
        for index, seed in enumerate(self.seeds, 1):
            name = f"seed-{seed}"
            run = f"train-{seed}"
            config = {team: {"TRAIN_VARIANT": self.variant, "EVAL_VARIANT": self.variant} for team in ("blue", "red")}
            owner = self.owner(run, "training", config)
            owner.update(
                {
                    "seed": seed,
                    "trainable_teams": ["red"],
                    "frozen_teams": ["blue"],
                    "requested_steps": 21,
                    "actual_steps": 20,
                    "completed_updates": 2,
                    "inputs": [
                        self.input("frozen blue policy", self.source_policy, self.source_hash),
                        self.input("training topology", self.topology, self.topology_hash),
                    ],
                    "parameter_checks": {
                        "blue": {
                            "initial_sha256": sha256(b"blue"),
                            "current_sha256": sha256(b"blue"),
                            "changed": False,
                            "trainable": False,
                        },
                        "red": {
                            "initial_sha256": sha256(b"red-initial"),
                            "current_sha256": sha256(b"red-trained"),
                            "changed": True,
                            "trainable": True,
                        },
                    },
                    "policy_sources": {"red": {"kind": "fresh", "seed": seed}},
                    "seed_contract": {"environment_reset_root": seed, "policy_and_rollout_root": seed + 1},
                    "policy_contract": copy.deepcopy(self.challenger_policies),
                }
            )
            checkpoint = f"runs:/{run}/checkpoints/model.safetensors"
            checkpoint_hash = sha256(checkpoint.encode())
            owner["outputs"]["checkpoints/model.safetensors"] = {
                "complete": True,
                "step": 20,
                "sha256": checkpoint_hash,
                "size": 123,
                "sidecar": "checkpoints/recipe.yaml",
            }
            self.add(run, "checkpoints/recipe.yaml", b"name: synthetic-challenger\n")
            attempt = {"run_id": run, "final_checkpoint": checkpoint, "actual_steps": 20, "requested_steps": 21}
            self.state["training"][name] = attempt
            self.add_json(run, "training/result.json", attempt)
            rows = [
                {
                    "update_idx": update,
                    "env_steps": update * 10,
                    "wall_time_s": index * update * 5,
                    "throughput_sps": 2.0,
                    "team.red.return": index * 10 + update,
                    "team.blue.return": -(index * 10 + update),
                }
                for update in (1, 2)
            ]
            self.add(run, "logs/metrics.jsonl", b"".join(json.dumps(row).encode() + b"\n" for row in rows))
            self.candidates[name] = (checkpoint, checkpoint_hash, self.challenger_policies)
        scores = dict(zip(self.candidates, [-10.0, -11.0, -14.0, -12.0]))
        for name, score in scores.items():
            self.state["validation"][name] = self.evaluation(
                f"validate-{name}", self.validation_seeds, name, [score - 1, score + 1]
            )
        self.state["test"]["original"] = self.evaluation("test-original", self.test_seeds, "original", [-10, -20, -30])
        self.state["test"]["selected"] = self.evaluation("test-selected", self.test_seeds, "seed-22", [-11, -22, -33])
        chosen = self.candidates["seed-22"]
        self.state["selection"] = {
            "candidate": "seed-22",
            "checkpoint": chosen[0],
            "checkpoint_sha256": chosen[1],
            "validation_scores": scores,
            "frozen_before_test_epoch": 10,
        }
        self.owner("report", "analysis", {})
        self.protocol_ref = self.add_json("source", "protocol/manifest.json", self.protocol)
        self.state_ref = self.add_json("report", "state.json", self.state)
        self.summary_ref = self.add_json(
            "report",
            "summary.json",
            {
                "baseline_blue_mean": -20.0,
                "selected_blue_mean": -22.0,
                "red_improvement": 2.0,
                "episodes": 3,
                "ci95": [1.0, 3.0],
                "bootstrap_samples": 10000,
                "bootstrap_seed": 7,
            },
        )
        self.config = {"tail_updates": 1}
        self.refresh()

    def owner(self, run, kind, config):
        owner = {
            "run_id": run,
            "kind": kind,
            "status": "FINISHED",
            "started_utc": "1970-01-01T00:00:20+00:00",
            "ended_utc": "1970-01-01T00:00:30+00:00",
            "source": copy.deepcopy(self.source),
            "environment": {"lockfile_hash": self.lock_hash},
            "effective_config": copy.deepcopy(config),
            "game_contract": {"implementation": self.source["files"], "resolved": copy.deepcopy(config)},
            "inputs": [],
            "outputs": {},
        }
        self.owners[run] = owner
        return owner

    def input(self, role, reference, digest):
        return {
            "role": role,
            "canonical_reference": reference,
            "reference": reference,
            "sha256": digest,
            "path": self.topology_path if "topology" in role else "/original/model.safetensors",
        }

    def add(self, run, name, data):
        reference = f"runs:/{run}/{name}"
        self.payloads[reference] = data
        return reference

    def add_json(self, run, name, value):
        return self.add(run, name, encoded(value))

    def evaluation(self, run, seeds, candidate, blue):
        policy_reference, policy_hash, policies = self.candidates[candidate]
        config = {
            "variant": self.variant,
            "episodes_per_seed": 1,
            "deterministic": False,
            "policy_backend": "jax",
            "cia": {"enabled": False},
            "episode_seed_scheme": "base_times_count_plus_replica_v1",
            "policy_rng_contract": "PRNGKey(episode_seed); split for reset, per-step policies and environment",
            "seeds": seeds,
            "episode_seeds": seeds,
            "topology_sampling": "exhaustive",
            "topology_paths": [self.topology_path],
        }
        owner = self.owner(run, "evaluation", config)
        owner["inputs"] = [
            self.input("blue policy", self.source_policy, self.source_hash),
            self.input("red policy", policy_reference, policy_hash),
            self.input("evaluation topology", self.topology, self.topology_hash),
        ]
        return self.add_json(
            run,
            "evaluations/result.json",
            {
                "eval_id": run,
                "git_commit": self.revision,
                "episodes_per_seed": 1,
                "stochastic": True,
                "policy_backend": "jax",
                "seeds": seeds,
                "per_episode_seeds": seeds,
                "n_episodes": len(seeds),
                "total_episodes": len(seeds),
                "wall_time_s": 4.0,
                "topology_sampling": "exhaustive",
                "topology_paths": [self.topology_path],
                "per_episode_topology_paths": [self.topology_path] * len(seeds),
                "policies": {
                    team: {
                        "observation_dim": policy["obs_dim"],
                        "action_dim": policy["action_dim"],
                        "arch": policy["arch"],
                    }
                    for team, policy in policies.items()
                },
                "per_episode_blue_returns": blue,
                "per_episode_red_returns": [-value for value in blue],
                "blue_mean_return": sum(blue) / len(blue),
                "red_mean_return": -sum(blue) / len(blue),
                "blue_std_return": statistics.stdev(blue),
            },
        )

    def modify_json(self, reference, change):
        value = json.loads(self.payloads[reference])
        change(value)
        self.payloads[reference] = encoded(value)

    def path(self, reference):
        return self.root / reference.removeprefix("runs:/")

    def refresh(self):
        self.payloads[self.protocol_ref] = encoded(self.protocol)
        self.payloads[self.state_ref] = encoded(self.state)
        for reference, data in self.payloads.items():
            run, name = reference.removeprefix("runs:/").split("/", 1)
            self.owners[run]["outputs"][name] = {"complete": True, "size": len(data), "sha256": sha256(data)}
        files = {}
        for reference, data in {
            **self.payloads,
            **{f"runs:/{run}/manifest.json": encoded(owner) for run, owner in self.owners.items()},
        }.items():
            path = self.path(reference)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[str(path.relative_to(self.root))] = {
                "reference": reference,
                "size": len(data),
                "sha256": sha256(data),
            }
        config = encoded(self.config)
        (self.root / "config.json").write_bytes(config)
        files["config.json"] = {"size": len(config), "sha256": sha256(config)}
        index = {
            "format": "frozen-defender-response-v1",
            "source_store": "file:///original/shared-store",
            "protocol": self.protocol_ref,
            "state": self.state_ref,
            "reference_summary": self.summary_ref,
            "configuration": "config.json",
            "files": files,
        }
        (self.root / "index.json").write_bytes(encoded(index))


class ReportCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = SyntheticBundle(self.root / "downloaded" / "inputs")
        self.output = self.root / "unrelated" / "report"

    def run_cli(self, *arguments, script=SCRIPT, output=None, environment=None):
        environment = {**os.environ, "MPLCONFIGDIR": str(self.root / "matplotlib"), **(environment or {})}
        return subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                str(self.bundle.root),
                "--output",
                str(output or self.output),
                *arguments,
            ],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_rejected(self, message):
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.output.exists())

    def csv_rows(self, name):
        with (self.output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_offline_report_outputs_and_provenance(self):
        before = {str(path): sha256(path.read_bytes()) for path in self.bundle.root.rglob("*") if path.is_file()}
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = read_json(self.output / "summary.json")
        self.assertEqual(
            {
                key: summary[key]
                for key in ("baseline_blue_mean", "selected_blue_mean", "red_improvement", "ci95", "episodes")
            },
            {
                "baseline_blue_mean": -20.0,
                "selected_blue_mean": -22.0,
                "red_improvement": 2.0,
                "ci95": [1.0, 3.0],
                "episodes": 3,
            },
        )
        self.assertEqual(read_json(self.output / "selection.json")["candidate"], "seed-22")
        self.assertEqual(
            read_json(self.output / "duration.json"),
            {"training_seconds": 60.0, "evaluation_seconds": 24.0, "pipeline_seconds": 100},
        )
        episodes = self.csv_rows("per-episode.csv")
        self.assertEqual([int(row["episode_seed"]) for row in episodes], [200, 201, 202])
        self.assertEqual([float(row["red_improvement"]) for row in episodes], [1.0, 2.0, 3.0])
        validation = self.csv_rows("validation.csv")
        self.assertEqual(len(validation), 4)
        self.assertEqual([row["candidate"] for row in validation if row["selected"] == "True"], ["seed-22"])
        self.assertEqual({row["evaluation"] for row in validation}, set(self.bundle.state["validation"].values()))
        self.assertEqual(len(self.csv_rows("summary.csv")), 2)
        self.assertEqual(len(self.csv_rows("learning-curves.csv")), 6)
        tails = self.csv_rows("learning-tail.csv")
        self.assertEqual([float(row["change"]) for row in tails], [1.0, 1.0, 1.0])
        self.assertEqual([float(row["final_window_mean_red_return"]) for row in tails], [12.0, 22.0, 32.0])
        for figure in ("comparison", "learning-curves"):
            self.assertTrue((self.output / f"{figure}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("<svg", (self.output / f"{figure}.svg").read_text())
        manifest = read_json(self.output / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["producer_revision"], self.bundle.revision)
        self.assertEqual(manifest["source_store"], "file:///original/shared-store")
        self.assertEqual(manifest["configuration_source"]["values"], {"tail_updates": 1})
        self.assertEqual(manifest["resolved_configuration"], read_json(self.output / "resolved-config.json"))
        self.assertEqual(manifest["resolved_configuration"]["bootstrap"], {"seed": 7, "resamples": 10000})
        self.assertEqual(manifest["input_index_sha256"], sha256((self.bundle.root / "index.json").read_bytes()))
        self.assertEqual(manifest["inputs"], read_json(self.bundle.root / "index.json")["files"])
        self.assertEqual(read_json(self.output / "input-index.json"), read_json(self.bundle.root / "index.json"))
        self.assertEqual(len(manifest["analysis_source"]["git_commit"]), 40)
        self.assertEqual(manifest["command"][1], str(SCRIPT))
        self.assertEqual(manifest["working_directory"], str(self.root))
        self.assertTrue(manifest["reference_interval_verified"])
        self.assertTrue(manifest["limitations"])
        for name, metadata in manifest["outputs"].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(metadata, {"size": len(data), "sha256": sha256(data)})
        source = manifest["analysis_source"]
        expected_files = {"pyproject.toml", "uv.lock", ".python-version", "README.md", "experiments/README.md"} | {
            f"experiments/frozen-defender-response/{name}"
            for name in ("analyze.py", "index_bundle.py", "test_analyze.py", "README.md")
        }
        self.assertEqual(set(source["files"]), expected_files)
        for name in expected_files:
            data = (PROJECT / name).read_bytes()
            self.assertEqual((self.output / "analysis-source" / name).read_bytes(), data)
            self.assertEqual(source["files"][name], sha256(data))
        for dependency in ("numpy", "matplotlib"):
            self.assertEqual(source["dependencies"][dependency], importlib.metadata.version(dependency))
        self.assertIn("[input index](input-index.json)", (self.output / "README.md").read_text())
        after = {str(path): sha256(path.read_bytes()) for path in self.bundle.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_shared_project_source_snapshot_runs_without_git(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        snapshot = self.output / "analysis-source"
        replay = self.root / "replayed-report"
        result = self.run_cli(script=snapshot / SCRIPT.relative_to(PROJECT), output=replay, environment={"PATH": ""})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        original = read_json(self.output / "manifest.json")["analysis_source"]
        copied = read_json(replay / "manifest.json")["analysis_source"]
        self.assertIsNone(copied["git_commit"])
        self.assertIsNone(copied["dirty"])
        self.assertEqual(copied["files"], original["files"])
        for name in ("summary.json", "resolved-config.json", "per-episode.csv"):
            self.assertEqual((replay / name).read_bytes(), (self.output / name).read_bytes())
        for name in ("pyproject.toml", "uv.lock", ".python-version"):
            self.assertEqual((replay / "analysis-source" / name).read_bytes(), (PROJECT / name).read_bytes())

    def test_source_snapshot_does_not_claim_enclosing_repository_revision(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for arguments in (
            ["init", "--quiet", "--template="],
            [
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "Create unrelated fixture",
            ],
        ):
            subprocess.run(["git", "-C", str(self.root), *arguments], check=True, capture_output=True)
        replay = self.root / "replayed-report"
        result = self.run_cli(script=self.output / "analysis-source" / SCRIPT.relative_to(PROJECT), output=replay)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        copied = read_json(replay / "manifest.json")["analysis_source"]
        self.assertIsNone(copied["git_commit"])
        self.assertIsNone(copied["dirty"])

    def test_parameter_overrides_are_recorded_without_editing_source(self):
        source_hash = sha256(SCRIPT.read_bytes())
        override = self.root / "trial-settings.json"
        override.write_bytes(encoded({"tail_updates": 1, "figure": {"dpi": 72}, "bootstrap": {"seed": 17}}))
        result = self.run_cli("--config", str(override), "--bootstrap-seed", "23", "--bootstrap-resamples", "256")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = read_json(self.output / "manifest.json")
        self.assertEqual(manifest["resolved_configuration"]["bootstrap"], {"seed": 23, "resamples": 256})
        self.assertEqual(manifest["resolved_configuration"]["figure"]["dpi"], 72)
        self.assertEqual(manifest["configuration_source"]["values"]["bootstrap"]["seed"], 17)
        self.assertEqual(manifest["configuration_source"]["sha256"], sha256(override.read_bytes()))
        self.assertEqual(manifest["overrides"], {"bootstrap": {"seed": 23, "resamples": 256}})
        self.assertFalse(manifest["reference_interval_verified"])
        self.assertEqual(read_json(self.output / "summary.json")["red_improvement"], 2.0)
        self.assertEqual(source_hash, sha256(SCRIPT.read_bytes()))

    def test_raw_byte_corruption_is_rejected(self):
        path = self.bundle.path(self.bundle.state["test"]["selected"])
        path.write_bytes(path.read_bytes() + b" ")
        self.assert_rejected("Hash mismatch")

    def test_missing_or_duplicate_final_seed_is_rejected(self):
        reference = self.bundle.state["test"]["selected"]
        original = self.bundle.payloads[reference]
        for seeds in ([200, 201], [200, 200, 202]):
            with self.subTest(seeds=seeds):
                self.bundle.payloads[reference] = original
                self.bundle.modify_json(reference, lambda row: row.update(per_episode_seeds=seeds))
                self.bundle.refresh()
                self.assert_rejected("evaluation cases")

    def test_incompatible_validation_contract_is_rejected(self):
        owner = self.bundle.owners["validate-seed-11"]
        owner["effective_config"]["variant"]["num_steps"] = 499
        owner["game_contract"]["resolved"] = copy.deepcopy(owner["effective_config"])
        self.bundle.refresh()
        self.assert_rejected("Horizon, reward, or observation contract differs")

    def test_overlapping_validation_and_test_cohorts_are_rejected(self):
        self.bundle.protocol["randomness"]["validation_episode_roots"] = [100, 200]
        self.bundle.refresh()
        self.assert_rejected("Validation/test cohorts overlap")

    def test_wrong_selection_with_valid_hashes_is_rejected(self):
        selection = self.bundle.state["selection"]
        selection["candidate"] = "seed-11"
        selection["checkpoint"], selection["checkpoint_sha256"] = self.bundle.candidates["seed-11"][:2]
        self.bundle.refresh()
        self.assert_rejected("Recorded selection differs from raw validation scores")

    def test_wrong_validation_checkpoint_is_rejected(self):
        owner = self.bundle.owners["validate-seed-11"]
        red = next(value for value in owner["inputs"] if value["role"] == "red policy")
        red["canonical_reference"], red["sha256"] = self.bundle.candidates["seed-33"][:2]
        self.bundle.refresh()
        self.assert_rejected("Wrong red policy identity")

    def test_challenger_contract_must_match_protocol(self):
        wrong_architecture = {"name": "different-network"}
        self.bundle.owners["train-11"]["policy_contract"]["red"]["arch"] = wrong_architecture
        reference = self.bundle.state["validation"]["seed-11"]
        self.bundle.modify_json(
            reference,
            lambda row: row["policies"]["red"].update(arch=wrong_architecture),
        )
        self.bundle.refresh()
        self.assert_rejected("Challenger policy contract differs")

    def test_missing_training_update_is_rejected(self):
        reference = "runs:/train-22/logs/metrics.jsonl"
        self.bundle.payloads[reference] = self.bundle.payloads[reference].splitlines(keepends=True)[0]
        self.bundle.refresh()
        self.assert_rejected("Missing or duplicate training updates")

    def test_changed_frozen_policy_is_rejected(self):
        self.bundle.owners["train-11"]["parameter_checks"]["blue"]["changed"] = True
        self.bundle.refresh()
        self.assert_rejected("Recorded Blue freeze evidence differs")

    def test_mismatched_producer_source_is_rejected(self):
        self.bundle.owners["test-selected"]["source"]["git_commit"] = "b" * 40
        self.bundle.refresh()
        self.assert_rejected("Producer source revision differs")

    def saved_selection(self, source=None):
        from jaxborg.experiment_producers import frozen_bundle
        from jaxborg.experiment_producers.source import analysis_source_identity
        from jaxborg.experiment_records import collate

        functions = runpy.run_path(str(SCRIPT))
        configuration = functions["resolve_config"](self.bundle.protocol, self.bundle.config, {})
        records = frozen_bundle(self.bundle.root)
        protocol = next(row for row in records if row["kind"] == "protocol")
        selected_attempts = {
            row["trial_id"]: row["id"]
            for row in records
            if row["kind"] == "attempt" and isinstance(row["trial_id"], str)
        }
        selection = collate(
            records,
            protocol_id=protocol["id"],
            selected_attempts=selected_attempts,
            selected_evaluations=[row["id"] for row in records if row["kind"] == "evaluation"],
            analysis={"source": source or analysis_source_identity(SCRIPT), "configuration": configuration},
        )
        path = self.root / "collation.json"
        path.write_bytes(encoded(selection))
        return path

    def test_saved_selection_cli_archives_all_executable_helpers(self):
        from jaxborg.experiment_producers.source import analysis_source_files

        selection = self.saved_selection()
        result = self.run_cli("--selection", str(selection))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(read_json(self.output / "collation.json"), read_json(selection))
        source = read_json(self.output / "manifest.json")["analysis_source"]
        for name, path in analysis_source_files(SCRIPT).items():
            self.assertEqual(source["files"][name], sha256(path.read_bytes()))
            self.assertEqual((self.output / "analysis-source" / name).read_bytes(), path.read_bytes())

    def test_saved_selection_rejects_wrong_executable_source(self):
        selection = self.saved_selection(source="sha256:" + "0" * 64)
        result = self.run_cli("--selection", str(selection))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Analyzer source differs", result.stderr)
        self.assertFalse(self.output.exists())

    def test_saved_selection_rejects_changed_configuration(self):
        selection = self.saved_selection()
        result = self.run_cli("--selection", str(selection), "--bootstrap-seed", "123")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("configuration differs", result.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir(parents=True)
        marker = self.output / "existing.txt"
        marker.write_text("keep this result")
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Output directory already exists", result.stderr)
        self.assertEqual(marker.read_text(), "keep this result")


if __name__ == "__main__":
    unittest.main()
