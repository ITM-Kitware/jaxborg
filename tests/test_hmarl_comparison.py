"""Final-opponent selection, stateful learned matchups, and launcher barriers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import struct

from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy, _run_jax_matchup_episodes_batched
from jaxborg.evaluation.stateful_blue import StatefulBluePolicy
from jaxborg.policies.base import RecurrentPolicy
from jaxborg.policies.categorical import Categorical
from jaxborg.pretrained.hmarl_eval import final_red_metadata
from jaxborg.scenarios.cc4.game_variant import GameVariant

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/temp/run_lstm_hmarl.sh"


class CounterBlue(StatefulBluePolicy):
    def initialize(self, state):
        return state, jnp.int32(0)

    def select_actions(self, weights, state, key, carry, *, deterministic):
        return jnp.array([carry + 1]), carry + 1


class ConstantBlue(nn.Module):
    def __call__(self, obs, mask):
        return Categorical(logits=jnp.zeros((1, 6))), jnp.zeros(1)


class CounterRed(RecurrentPolicy):
    def initialize_carry(self, batch_size):
        return jnp.zeros(batch_size, dtype=jnp.int32)

    def __call__(self, carry, obs, mask, resets):
        carry = jnp.where(resets[0], 0, carry) + 1
        logits = jax.nn.one_hot(carry, 6)[None] * 100
        return carry, Categorical(logits=logits), jnp.zeros(obs.shape[:2])


@struct.dataclass
class TinyState:
    time: object
    red_agent_active: object
    ot_service_stopped: object
    host_service_reliability: object
    host_decoy_reliability: object


@struct.dataclass
class TinyEnvState:
    state: TinyState


class TinyJointEnv:
    blue_agents = ("blue_0",)
    red_agents = ("red_0",)

    def observations(self):
        return {name: jnp.zeros(1) for name in self.blue_agents + self.red_agents}

    def reset(self, key):
        return self.observations(), TinyEnvState(
            TinyState(
                jnp.int32(0), jnp.array([False]), jnp.array([False]), jnp.full((1, 1), 100), jnp.full((1, 1), 100)
            )
        )

    def reset_at_topology(self, key, index):
        return self.reset(key)

    def get_avail_actions(self, state):
        return {name: jnp.ones(6, dtype=bool) for name in self.blue_agents + self.red_agents}

    def step_env(self, key, state, actions):
        new = state.state.replace(time=state.state.time + 1, red_agent_active=jnp.array([True]))
        reward = jnp.float32(10 * actions["blue_0"] + actions["red_0"])
        return self.observations(), state.replace(state=new), {"blue_0": reward}, {"__all__": new.time >= 4}, {}


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("stateful", [False, True])
def test_joint_scan_preserves_blue_memory_red_reset_and_terminal_state(batch_size, stateful):
    blue = CounterBlue() if stateful else ConstantBlue()
    policies = {
        "blue": LoadedMatchupPolicy("blue", "jax", blue, {}, {}),
        "red": LoadedMatchupPolicy("red", "jax", CounterRed(), {}, {}),
    }
    kwargs = dict(
        variant=GameVariant(name="tiny", num_steps=7),
        env=TinyJointEnv(),
        episode_seeds=[1, 2, 3],
        topology_indices=[0, 0, 0],
        role_arrays=[jnp.array([1])] * 3,
        deterministic=True,
        batch_size=batch_size,
    )
    rewards, cia = _run_jax_matchup_episodes_batched(policies, **kwargs)
    # Blue's four decisions sum to 10; Red restarts after dormancy: 1,1,2,3.
    np.testing.assert_array_equal(rewards, [107 if stateful else 7] * 3)
    np.testing.assert_array_equal(cia, np.zeros((3, 3)))
    np.testing.assert_array_equal(_run_jax_matchup_episodes_batched(policies, **kwargs)[0], rewards)


def make_final_model(tmp_path, *, teams=("blue", "red"), recorded_name="model_run.safetensors"):
    path = tmp_path / "model_run.safetensors"
    path.touch()
    (tmp_path / "recipe_run.yaml").write_text(
        yaml.safe_dump(
            {
                "meta": {"name": "cotraining_lstm"},
                "run": {"model": recorded_name, "trainable_teams": list(teams), "seed": 42, "total_steps": 49968000},
            }
        )
    )
    return path


def test_final_red_metadata_rejects_historical_and_non_cotrained_models(tmp_path):
    path = make_final_model(tmp_path)
    assert final_red_metadata(path) == {"recipe_name": "cotraining_lstm", "seed": 42, "total_steps": 49968000}
    historical = tmp_path / "checkpoint_48000000.safetensors"
    historical.touch()
    with pytest.raises(ValueError, match="final model"):
        final_red_metadata(historical)
    make_final_model(tmp_path, teams=("blue",))
    with pytest.raises(ValueError, match="cotrained"):
        final_red_metadata(path)
    make_final_model(tmp_path, recorded_name="model_another.safetensors")
    with pytest.raises(ValueError, match="matching sidecar"):
        final_red_metadata(path)


def launcher(tmp_path, *args, **overrides):
    if not LAUNCHER.is_file():
        pytest.skip("Local launcher is in the ignored scripts/temp directory")
    env = {
        **os.environ,
        "JAXBORG_REPO_DIR": str(ROOT),
        "JAXBORG_COMPARISON_DIR": str(tmp_path / "comparison"),
        "JAXBORG_SEEDS": "42 100 200",
        "CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second",
        **overrides,
    }
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, text=True, capture_output=True, timeout=60)


def test_launcher_dry_run_two_conditions_final_opponents_and_gpu_mapping(tmp_path):
    result = launcher(tmp_path, "--dry-run", "--run-id", "check")
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
    trains = [cmd for cmd in commands if "scripts/train/algorithms/ippo_jax.py" in cmd]
    hmarl = [cmd for cmd in commands if "scripts/eval/eval_hmarl.py" in cmd]
    assert len(trains) == 6 and len(hmarl) == 2 and len(commands) == 14
    assert all("JAXBORG_SKIP_POST_TRAINING_EVAL=1" in cmd for cmd in trains)
    expected = {
        f"{recipe}_seed{seed}_check"
        for recipe in ("cotraining_lstm", "cotraining_lstm_env_diversity")
        for seed in (42, 100, 200)
    }
    assert {cmd[cmd.index("--tag") + 1] for cmd in trains} == expected
    for i, cmd in enumerate(hmarl):
        assert f"CUDA_VISIBLE_DEVICES=GPU-{'first' if i == 0 else 'second'}" in cmd
        reds = [cmd[j + 1] for j, arg in enumerate(cmd) if arg == "--red-model"]
        assert {Path(red).name for red in reds} == {f"model_{tag}.safetensors" for tag in expected}
        assert all("/checkpoints/" not in red for red in reds)
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "0,0"},
        {"JAXBORG_SEEDS": "42 42"},
        {"JAXBORG_SEEDS": "-1"},
    ],
)
def test_launcher_invalid_resources_fail_before_work(tmp_path, overrides):
    result = launcher(tmp_path, "--dry-run", **overrides)
    assert result.returncode != 0
    assert not (tmp_path / "comparison").exists()


@pytest.fixture
def fake_uv(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    executable = bindir / "uv"
    executable.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
train = "scripts/train/algorithms/ippo_jax.py" in args
scripted = "scripts/eval/eval_scripted_reds_jax.py" in args
hmarl = "scripts/eval/eval_hmarl.py" in args and "--prepare-only" not in args
if not (train or scripted or hmarl):
    sys.exit(0)
tag = args[args.index("--tag") + 1] if train else args[args.index("--output") + 1]
def record(event):
    with open(os.environ["FAKE_EVENTS"], "a") as file:
        file.write(json.dumps(dict(event=event, train=train, tag=tag, gpu=os.environ["CUDA_VISIBLE_DEVICES"])) + "\\n")
record("start")
time.sleep(0.05)
if train and os.environ.get("FAKE_FAIL") == tag:
    record("fail")
    sys.exit(3)
if train:
    path = Path(os.environ["JAXBORG_EXP_DIR"]) / "ippo_jax" / tag / f"model_{tag}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
record("finish")
""")
    executable.chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}", "FAKE_EVENTS": str(tmp_path / "events.jsonl")}


def test_launcher_waits_for_training_and_uses_at_most_two_workers(tmp_path, fake_uv):
    result = launcher(tmp_path, "--run-id", "check", **fake_uv)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in Path(fake_uv["FAKE_EVENTS"]).read_text().splitlines()]
    active = set()
    trained = 0
    for event in events:
        if event["event"] == "start":
            if not event["train"]:
                assert trained == 6
            assert event["gpu"] not in active
            active.add(event["gpu"])
            assert len(active) <= 2
        else:
            active.remove(event["gpu"])
            trained += int(event["train"])
    assert not active and trained == 6
    assert (tmp_path / "comparison/recipes/check/training.complete").exists()
    again = launcher(tmp_path, "--run-id", "check", **fake_uv)
    assert again.returncode != 0 and "already exists" in again.stderr
    rerun = launcher(tmp_path, "--eval-only", "--run-id", "check", **fake_uv)
    assert rerun.returncode == 0, rerun.stderr
    added = [json.loads(line) for line in Path(fake_uv["FAKE_EVENTS"]).read_text().splitlines()][len(events) :]
    assert len(added) == 16 and all(not event["train"] for event in added)
    wrong_seeds = launcher(tmp_path, "--eval-only", "--run-id", "check", JAXBORG_SEEDS="42", **fake_uv)
    assert wrong_seeds.returncode != 0 and "must match" in wrong_seeds.stderr


def test_launcher_training_failure_blocks_every_evaluation(tmp_path, fake_uv):
    result = launcher(tmp_path, "--run-id", "check", FAKE_FAIL="cotraining_lstm_seed42_check", **fake_uv)
    assert result.returncode != 0
    events = [json.loads(line) for line in Path(fake_uv["FAKE_EVENTS"]).read_text().splitlines()]
    assert all(event["train"] for event in events)
    assert not (tmp_path / "comparison/recipes/check/training.complete").exists()
    retry = launcher(tmp_path, "--eval-only", "--run-id", "check", **fake_uv)
    assert retry.returncode != 0 and "completed-training marker" in retry.stderr
