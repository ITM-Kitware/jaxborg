"""Held-out topology-bank wiring for JAX-native evaluation scripts."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.eval import baselines_jax, cec_phase6_eval_jax


def test_phase6_eval_env_forwards_held_out_topology_bank(monkeypatch, tmp_path):
    bank = [tmp_path / "held-out-10.snapshot.npz", tmp_path / "held-out-11.snapshot.npz"]
    sentinel_env = object()
    captured = {}

    monkeypatch.setattr(cec_phase6_eval_jax, "variant_for_red", lambda *args, **kwargs: CC4_STOCK)

    def fake_make_jax_env(variant, **kwargs):
        captured["variant"] = variant
        captured["kwargs"] = kwargs
        return sentinel_env

    monkeypatch.setattr(cec_phase6_eval_jax, "make_jax_env", fake_make_jax_env)

    variant, env = cec_phase6_eval_jax._build_eval_env(
        "fsm",
        resilience_roles=False,
        topology_path=bank,
    )

    assert variant is CC4_STOCK
    assert env is sentinel_env
    assert captured == {"variant": CC4_STOCK, "kwargs": {"topology_path": bank}}


def test_phase6_run_eval_resolves_and_records_topology_bank(monkeypatch, tmp_path):
    bank = [tmp_path / "nested" / "held-out-a.snapshot.npz", tmp_path / "held-out-b.snapshot.npz"]
    variant = replace(CC4_STOCK, num_steps=1)
    captured = {}

    class FakePolicy:
        def apply(self, params, obs, mask):
            return SimpleNamespace(logits=jnp.zeros((1,), dtype=jnp.float32)), None

    class FakeEnv:
        def reset(self, key):
            obs = {f"blue_{index}": jnp.zeros((1,), dtype=jnp.float32) for index in range(NUM_BLUE_AGENTS)}
            return obs, jnp.int32(0)

        def reset_at_topology(self, key, topology_index):
            return self.reset(key)

        def get_avail_actions(self, state):
            return {f"blue_{index}": jnp.ones((1,), dtype=bool) for index in range(NUM_BLUE_AGENTS)}

        def step(self, key, state, actions):
            obs = {f"blue_{index}": jnp.zeros((1,), dtype=jnp.float32) for index in range(NUM_BLUE_AGENTS)}
            rewards = {f"blue_{index}": jnp.float32(1.0) for index in range(NUM_BLUE_AGENTS)}
            return obs, state, rewards, {"__all__": jnp.bool_(True)}, {}

    fake_runner = types.ModuleType("jaxborg.evaluation.jax_runner")
    fake_runner.load_jax_checkpoint = lambda path: (FakePolicy(), {}, {"meta": {"name": "test"}, "train": {}})
    monkeypatch.setitem(sys.modules, "jaxborg.evaluation.jax_runner", fake_runner)
    monkeypatch.setattr(cec_phase6_eval_jax, "variant_for_red", lambda *args, **kwargs: variant)

    def fake_make_jax_env(selected_variant, **kwargs):
        captured["variant"] = selected_variant
        captured["kwargs"] = kwargs
        return FakeEnv()

    monkeypatch.setattr(cec_phase6_eval_jax, "make_jax_env", fake_make_jax_env)
    monkeypatch.setattr(cec_phase6_eval_jax, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(cec_phase6_eval_jax.jax, "jit", lambda function: function)

    row = cec_phase6_eval_jax.run_eval(
        model_path=tmp_path / "model.safetensors",
        eval_red="fsm",
        episodes=1,
        seed=7,
        topology_path=bank,
    )

    resolved = tuple(path.resolve() for path in bank)
    assert captured == {"variant": variant, "kwargs": {"topology_path": resolved}}
    assert row["topology_paths"] == [str(path) for path in resolved]
    assert row["topology_sampling"] == "exhaustive"
    assert row["episodes"] == 1
    assert row["total_episodes"] == 2
    assert row["n_episodes"] == 2
    assert row["per_episode_topology_paths"] == [str(path) for path in resolved]
    assert row["mean_reward"] == 1.0


def test_phase6_cli_accepts_multiple_and_repeated_topology_paths(monkeypatch, tmp_path):
    model = tmp_path / "model.safetensors"
    model.touch()
    output = tmp_path / "result.jsonl"
    bank = [tmp_path / f"held-out-{index}.snapshot.npz" for index in range(3)]
    captured = {}

    def fake_run_eval(**kwargs):
        captured.update(kwargs)
        return {
            "recipe_name": "test",
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "n_episodes": 1,
            "wall_time_s": 0.01,
            "topology_paths": [str(path.resolve()) for path in bank],
        }

    monkeypatch.setattr(cec_phase6_eval_jax, "run_eval", fake_run_eval)
    monkeypatch.setattr(cec_phase6_eval_jax.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(cec_phase6_eval_jax.jax, "devices", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cec_phase6_eval_jax.py",
            "--model",
            str(model),
            "--eval-red",
            "fsm",
            "--episodes",
            "1",
            "--topology-path",
            str(bank[0]),
            str(bank[1]),
            "--topology-path",
            str(bank[2]),
            "--output",
            str(output),
        ],
    )

    cec_phase6_eval_jax.main()

    assert captured["topology_path"] == [str(path) for path in bank]
    assert output.is_file()


def test_jax_baseline_evaluate_forwards_and_reports_resolved_topology_bank(monkeypatch, tmp_path, capsys):
    bank = [tmp_path / "held-out-a.snapshot.npz", tmp_path / "held-out-b.snapshot.npz"]
    sentinel_env = object()
    captured = {}

    monkeypatch.setattr(baselines_jax, "resolve_eval_variant", lambda **kwargs: CC4_STOCK)

    def fake_make_jax_env(variant, **kwargs):
        captured["variant"] = variant
        captured["kwargs"] = kwargs
        return sentinel_env

    monkeypatch.setattr(baselines_jax, "make_jax_env", fake_make_jax_env)
    calls = []

    def fake_sleep(env, key, topology_index=None):
        calls.append(topology_index)
        return 0.0

    monkeypatch.setattr(baselines_jax, "run_sleep_episode", fake_sleep)

    baselines_jax.evaluate("sleep", seed=10, max_eps=1, topology_path=bank)

    resolved = tuple(path.resolve() for path in bank)
    assert captured == {"variant": CC4_STOCK, "kwargs": {"topology_path": resolved}}
    assert calls == [0, 1]
    stdout = capsys.readouterr().out
    assert "2 snapshot(s), exhaustive" in stdout
    assert all(str(path) in stdout for path in resolved)


def test_jax_baseline_cli_forwards_topology_paths(monkeypatch, tmp_path):
    bank = [tmp_path / f"held-out-{index}.snapshot.npz" for index in range(3)]
    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(baselines_jax, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baselines_jax.py",
            "--policy",
            "random",
            "--topology-path",
            str(bank[0]),
            str(bank[1]),
            "--topology-path",
            str(bank[2]),
        ],
    )

    baselines_jax.main()

    assert captured["args"] == ("random", None, 10)
    assert captured["kwargs"]["topology_path"] == [str(path) for path in bank]


def test_empty_topology_bank_is_rejected():
    with pytest.raises(ValueError, match="at least one snapshot"):
        cec_phase6_eval_jax._resolve_topology_paths([])

    with pytest.raises(ValueError, match="at least one snapshot"):
        baselines_jax._resolve_topology_paths([])
