from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.evaluation import jax_scripted_red
from jaxborg.evaluation.cia.fixed_topology import EvaluationCase
from jaxborg.evaluation.jax_scripted_red import (
    JaxScriptedRedEpisode,
    attach_results_to_mlflow,
    evaluate_jax_scripted_reds,
    run_jax_scripted_red_episode,
)
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK, CIA_RESILIENCE
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_NONE, ROLE_WEB

_CIA_CONFIG = {
    "enabled": True,
    "metric": "resilience",
    "role_assignment": "fixed_per_topology",
}


@dataclass(frozen=True)
class _FakeState:
    state: object
    const: object
    extras: dict

    def replace(self, **changes):
        return replace(self, **changes)


class _OneStepEnv:
    agents = ("blue_0", "blue_1")

    def __init__(self, expected_roles):
        self.expected_roles = np.asarray(expected_roles)
        self.mask_role_maps = []
        self.step_role_maps = []

    def reset_at_topology(self, key, topology_index):
        del key
        assert topology_index == 4
        obs = {agent: jnp.zeros(2, dtype=jnp.float32) for agent in self.agents}
        state = _FakeState(
            state=object(),
            const=object(),
            extras={"host_resilience_role": jnp.zeros_like(jnp.asarray(self.expected_roles))},
        )
        return obs, state

    def get_avail_actions(self, state):
        roles = np.asarray(state.extras["host_resilience_role"])
        self.mask_role_maps.append(roles)
        return {agent: jnp.ones(3, dtype=jnp.bool_) for agent in self.agents}

    def step_env(self, key, state, actions):
        del key
        self.step_role_maps.append(np.asarray(state.extras["host_resilience_role"]))
        assert set(actions) == set(self.agents)
        obs = {agent: jnp.zeros(2, dtype=jnp.float32) for agent in self.agents}
        rewards = {agent: jnp.float32(2.5) for agent in self.agents}
        dones = {agent: True for agent in self.agents} | {"__all__": True}
        return obs, state, rewards, dones, {}

    def step(self, *args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("rollout must use non-resetting step_env")


@pytest.mark.parametrize("backend", ["jax", "cyborg"])
def test_episode_injects_fixed_roles_before_blue_and_red_act(monkeypatch, backend):
    roles = (0, 1, 2, 3)
    case = EvaluationCase(
        topology_index=4,
        topology_path=Path("fixed.snapshot.npz"),
        topology_fingerprint="fingerprint",
        base_seed=17,
        replicate_index=0,
        episode_seed=17,
        host_roles=roles,
        role_map_id="role-map",
    )
    env = _OneStepEnv(roles)
    selected_role_maps = []

    def fake_torch_actions(policy, obs, mask, lookups, *, seed, deterministic):
        del policy, obs, mask, lookups, seed, deterministic
        selected_role_maps.append(env.mask_role_maps[-1])
        return np.zeros(2, dtype=np.int32)

    monkeypatch.setattr(jax_scripted_red, "_torch_blue_actions", fake_torch_actions)
    monkeypatch.setattr(
        jax_scripted_red,
        "cyborg_blue_flat_to_jax_lookup",
        lambda const, agent_id: np.arange(3, dtype=np.int32),
    )
    monkeypatch.setattr(
        jax_scripted_red,
        "score_resilience_state",
        lambda state, role_map: jnp.asarray([-10.0, -20.0, -30.0]),
    )
    scan_roles = []

    def fake_scan(weights, key, topology_index, role_map, **kwargs):
        del weights, key, kwargs
        assert int(topology_index) == case.topology_index
        scan_roles.append(np.asarray(role_map))
        return jnp.float32(2.5), jnp.asarray([-10.0, -20.0, -30.0])

    monkeypatch.setattr(jax_scripted_red, "_run_jax_scripted_red_episode_scan", fake_scan)
    policy = LoadedMatchupPolicy("blue", backend, object(), object(), {})

    result = run_jax_scripted_red_episode(
        policy,
        env=env,
        variant=replace(CIA_RESILIENCE, num_steps=3),
        case=case,
        deterministic=True,
    )

    assert result == JaxScriptedRedEpisode(2.5, (-10.0, -20.0, -30.0))
    if backend == "jax":
        assert len(scan_roles) == 1
        np.testing.assert_array_equal(scan_roles[0], roles)
        assert not env.step_role_maps
        return
    assert len(env.step_role_maps) == 1
    for observed in selected_role_maps + env.step_role_maps:
        np.testing.assert_array_equal(observed, roles)


@pytest.mark.parametrize(("suffix", "backend"), [(".safetensors", "jax"), (".pt", "cyborg")])
def test_sweep_supports_both_blue_bundle_backends_and_reuses_cases(
    monkeypatch,
    tmp_path,
    suffix,
    backend,
):
    model = tmp_path / f"model{suffix}"
    model.touch()
    topologies = [tmp_path / "a.snapshot.npz", tmp_path / "b.snapshot.npz"]
    for topology in topologies:
        topology.touch()
    roles_a = (0, 1, 2, 3)
    roles_b = (3, 2, 1, 0)
    cases = (
        EvaluationCase(0, topologies[0], "fp-a", 10, 0, 10, roles_a, "map-a"),
        EvaluationCase(1, topologies[1], "fp-b", 10, 0, 10, roles_b, "map-b"),
    )
    monkeypatch.setattr(jax_scripted_red, "build_evaluation_cases", lambda *args: cases)
    loader_calls = []

    def fake_loader(path, *, team, backend):
        loader_calls.append((path, team, backend))
        return LoadedMatchupPolicy(team, backend, object(), object(), {"bundle_trainable": True})

    env_calls = []
    monkeypatch.setattr(jax_scripted_red, "_git_commit", lambda: "abc123")

    def fake_env(variant, **kwargs):
        env = object()
        env_calls.append((variant, kwargs, env))
        return env

    episode_calls = []

    def fake_episode(policy, *, env, variant, case, deterministic):
        episode_calls.append((policy.backend, env, variant.red_agent, case.role_map_id, deterministic))
        index = 1.0 if case.role_map_id == "map-a" else 2.0
        return JaxScriptedRedEpisode(index, (-10.0 * index, -20.0 * index, -30.0 * index))

    rows = evaluate_jax_scripted_reds(
        model,
        base_variant=CIA_RESILIENCE,
        topology_paths=topologies,
        reds=("fsm", "cia_c"),
        seeds=(10,),
        deterministic=True,
        recipe={
            "meta": {"name": "co-train"},
            "run": {"train_run_id": "run-1"},
            "eval": {"cia": _CIA_CONFIG},
        },
        policy_loader=fake_loader,
        env_factory=fake_env,
        episode_runner=fake_episode,
    )

    assert loader_calls == [(model.resolve(), "blue", backend)]
    assert [call[0].red_agent for call in env_calls] == ["finite_state", "c"]
    assert [call[0].name for call in env_calls] == ["cia_resilience_fsm", "cia_c"]
    assert all(call[1] == {"training_mode": False, "topology_path": tuple(topologies)} for call in env_calls)
    assert [call[3] for call in episode_calls] == ["map-a", "map-b", "map-a", "map-b"]
    assert all(call[-1] is True for call in episode_calls)
    for row in rows:
        assert row["eval_env"] == "jax_fsm"
        assert row["topology_sampling"] == "exhaustive"
        assert row["episode_role_map_ids"] == ["map-a", "map-b"]
        assert row["cia_summary"] == {
            "n": 2,
            "c": {"mean": -15.0, "std": pytest.approx(7.0710678118654755)},
            "i": {"mean": -30.0, "std": pytest.approx(14.142135623730951)},
            "a": {"mean": -45.0, "std": pytest.approx(21.213203435596427)},
        }
        assert [item["role_map_id"] for item in row["topology_role_maps"]] == ["map-a", "map-b"]


def test_sweep_rejects_roleless_variant_before_loading_model(tmp_path):
    topology = tmp_path / "topology.snapshot.npz"
    topology.touch()
    with pytest.raises(ValueError, match="resilience_roles=True"):
        evaluate_jax_scripted_reds(
            tmp_path / "missing.safetensors",
            base_variant=CC4_STOCK,
            topology_paths=[topology],
        )


def test_sweep_rejects_missing_topology_bank_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="non-empty topology bank"):
        evaluate_jax_scripted_reds(
            tmp_path / "missing.safetensors",
            base_variant=CIA_RESILIENCE,
            topology_paths=None,
        )


def test_public_sweep_rejects_disabled_cia_before_building_cases(tmp_path):
    topology = tmp_path / "topology.snapshot.npz"
    topology.touch()

    with pytest.raises(ValueError, match="eval.cia.enabled"):
        evaluate_jax_scripted_reds(
            tmp_path / "missing.safetensors",
            base_variant=CIA_RESILIENCE,
            topology_paths=[topology],
            recipe={"eval": {"cia_metric": "resilience"}},
        )


def test_mlflow_metrics_preserve_rewards_and_add_cia(monkeypatch):
    rows = [
        {
            "eval_name": "scripted-reds",
            "eval_red": "cia_a",
            "mean_reward": 4.0,
            "std_reward": 0.5,
            "n_episodes": 8,
            "train_run_id": "run-123",
            "cia_summary": {
                "n": 8,
                "c": {"mean": -1.0, "std": 1.5},
                "i": {"mean": -2.0, "std": 2.5},
                "a": {"mean": -3.0, "std": 3.5},
            },
        }
    ]
    captured = {}
    monkeypatch.setattr(
        "jaxborg.mlflow_setup.attach_eval_metrics",
        lambda run_id, metrics: captured.update(run_id=run_id, metrics=metrics),
    )

    attach_results_to_mlflow(rows)

    assert captured == {
        "run_id": "run-123",
        "metrics": {
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.mean_reward": 4.0,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.std_reward": 0.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.episodes": 8.0,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.c.mean": -1.0,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.c.std": 1.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.c.mean_minus_std": -2.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.c.mean_plus_std": 0.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.i.mean": -2.0,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.i.std": 2.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.i.mean_minus_std": -4.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.i.mean_plus_std": 0.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.a.mean": -3.0,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.a.std": 3.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.a.mean_minus_std": -6.5,
            "eval.after_training.scripted-reds.scripted_red.cia_a.blue.cia.a.mean_plus_std": 0.5,
        },
    }


def test_legacy_metric_alias_does_not_enable_jax_scripted_cia():
    with pytest.raises(ValueError, match="eval.cia.enabled"):
        jax_scripted_red._cia_recipe_configuration({"eval": {"cia_metric": "resilience"}})


class _BatchedEnv:
    """Env exposing the reset/step surface ``_supports_batched_eval`` requires."""

    agents = ("blue_0",)

    def reset(self, key):  # pragma: no cover - surface probe only
        raise AssertionError("batched sweep resets inside the vmapped scan")

    def reset_at_topology(self, key, topology_index):  # pragma: no cover
        raise AssertionError("batched sweep resets inside the vmapped scan")

    def step_env(self, key, state, actions):  # pragma: no cover
        raise AssertionError("batched sweep steps inside the vmapped scan")


# Each case carries a valid AUTH/DB/WEB map; only the topology index varies.
_BANK_ROLES = (ROLE_NONE, ROLE_AUTH, ROLE_DB, ROLE_WEB)


def _fake_batched_scan(weights, key, topology_index, role_map, **kwargs):
    """Derive reward/CIA from the traced case inputs so ``vmap`` can map it.

    The reward mixes the topology index with a draw from the episode key, so a
    chunk that mismatched either would not reproduce the sequential values.
    """
    del weights, role_map, kwargs
    index = jnp.asarray(topology_index, dtype=jnp.float32)
    reward = index * 10.0 + jax.random.uniform(key)
    return reward, jnp.stack([-index, -2.0 * index, -3.0 * index])


def _expected_reward(case: EvaluationCase) -> float:
    return 10.0 * case.topology_index + float(jax.random.uniform(jax.random.PRNGKey(case.episode_seed)))


def _bank_cases(count: int, seeds: tuple[int, ...]) -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase(
            topology_index=topology_index,
            topology_path=Path(f"t{topology_index}.snapshot.npz"),
            topology_fingerprint=f"fp-{topology_index}",
            base_seed=seed,
            replicate_index=0,
            episode_seed=seed,
            host_roles=_BANK_ROLES,
            role_map_id=f"map-{topology_index}",
        )
        for topology_index in range(count)
        for seed in seeds
    )


def test_jax_sweep_vmaps_cases_and_preserves_per_case_order(monkeypatch, tmp_path):
    """Batched chunks must reproduce the sequential per-case results in order."""
    model = tmp_path / "model.safetensors"
    model.touch()
    topology = tmp_path / "topology.snapshot.npz"
    topology.touch()
    cases = _bank_cases(5, (1000, 1001))

    monkeypatch.setenv("JAXBORG_EVAL_BATCH_SIZE", "4")  # 10 cases -> 4 + 4 + 2
    monkeypatch.setattr(jax_scripted_red, "build_evaluation_cases", lambda *args: cases)
    monkeypatch.setattr(jax_scripted_red, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(jax_scripted_red, "_run_jax_scripted_red_episode_scan", _fake_batched_scan)

    def unreachable(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("a JAX Blue policy must not use the per-case seam")

    monkeypatch.setattr(jax_scripted_red, "run_jax_scripted_red_episode", unreachable)

    rows = evaluate_jax_scripted_reds(
        model,
        base_variant=CIA_RESILIENCE,
        topology_paths=[topology],
        reds=("cia_a",),
        seeds=(1000, 1001),
        recipe={"meta": {"name": "co-train"}, "eval": {"cia": _CIA_CONFIG}},
        policy_loader=lambda path, *, team, backend: LoadedMatchupPolicy(
            team, backend, object(), object(), {"bundle_trainable": True}
        ),
        env_factory=lambda variant, **kwargs: _BatchedEnv(),
    )

    (row,) = rows
    expected = [_expected_reward(case) for case in cases]
    assert row["per_episode"] == pytest.approx(expected)
    assert row["n_episodes"] == len(cases)
    assert [record["c"] for record in row["per_episode_cia"]] == pytest.approx(
        [-float(case.topology_index) for case in cases]
    )
    assert row["episode_role_map_ids"] == [case.role_map_id for case in cases]


def test_jax_sweep_batches_report_progress_per_chunk(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model.safetensors"
    model.touch()
    topology = tmp_path / "topology.snapshot.npz"
    topology.touch()
    cases = _bank_cases(5, (1000,))

    monkeypatch.setenv("JAXBORG_EVAL_BATCH_SIZE", "2")
    monkeypatch.setattr(jax_scripted_red, "build_evaluation_cases", lambda *args: cases)
    monkeypatch.setattr(jax_scripted_red, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(jax_scripted_red, "_run_jax_scripted_red_episode_scan", _fake_batched_scan)

    evaluate_jax_scripted_reds(
        model,
        base_variant=CIA_RESILIENCE,
        topology_paths=[topology],
        reds=("cia_a",),
        seeds=(1000,),
        progress=True,
        recipe={"meta": {"name": "co-train"}, "eval": {"cia": _CIA_CONFIG}},
        policy_loader=lambda path, *, team, backend: LoadedMatchupPolicy(
            team, backend, object(), object(), {"bundle_trainable": True}
        ),
        env_factory=lambda variant, **kwargs: _BatchedEnv(),
    )

    printed = capsys.readouterr().out
    assert "  cia_a episodes 2/5" in printed
    assert "  cia_a episodes 5/5" in printed


def test_torch_blue_sweep_keeps_the_per_case_seam(monkeypatch, tmp_path):
    """Only a JAX policy vmaps; Torch Blue inference stays on the host."""
    model = tmp_path / "model.pt"
    model.touch()
    topology = tmp_path / "topology.snapshot.npz"
    topology.touch()
    cases = _bank_cases(2, (1000,))

    monkeypatch.setattr(jax_scripted_red, "build_evaluation_cases", lambda *args: cases)
    monkeypatch.setattr(jax_scripted_red, "_git_commit", lambda: "abc123")

    def unreachable(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("Torch Blue must not be vmapped")

    monkeypatch.setattr(jax_scripted_red, "_run_jax_scripted_red_episodes_batched", unreachable)
    seen = []

    def fake_episode(policy, *, env, variant, case, deterministic):
        seen.append(case.role_map_id)
        return JaxScriptedRedEpisode(1.0, (-1.0, -2.0, -3.0))

    monkeypatch.setattr(jax_scripted_red, "run_jax_scripted_red_episode", fake_episode)

    evaluate_jax_scripted_reds(
        model,
        base_variant=CIA_RESILIENCE,
        topology_paths=[topology],
        reds=("cia_a",),
        seeds=(1000,),
        recipe={"meta": {"name": "co-train"}, "eval": {"cia": _CIA_CONFIG}},
        policy_loader=lambda path, *, team, backend: LoadedMatchupPolicy(
            team, backend, object(), object(), {"bundle_trainable": True}
        ),
        env_factory=lambda variant, **kwargs: _BatchedEnv(),
    )

    assert seen == [case.role_map_id for case in cases]
