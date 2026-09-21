from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import yaml
from CybORG.Agents.Wrappers import BlueFlatWrapper

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.checkpoint import save_jax_bundle, save_torch_bundle
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation import matchup_runner
from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.evaluation.matchup_runner import (
    LoadedMatchupPolicy,
    cyborg_blue_flat_to_jax_lookup,
    load_matchup_policy,
)
from jaxborg.learned_red import RED_OBS_SIZE, RED_POLICY_ACTION_DIM
from jaxborg.parity.translate import build_mappings_from_cyborg, cyborg_blue_to_jax
from jaxborg.policies import make_torch_policy
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

ARCH = {
    "name": "shared",
    "hidden_dim": 4,
    "hidden_layers": 1,
    "activation": "tanh",
}


def _jax_entry(team: str, *, obs_dim: int | None = None, action_dim: int | None = None):
    expected_obs = BLUE_OBS_SIZE if team == "blue" else RED_OBS_SIZE
    expected_action = BLUE_ALLOW_TRAFFIC_END if team == "blue" else RED_POLICY_ACTION_DIM
    return {
        "weights": {"params": {"marker": jnp.asarray([1.0])}},
        "team": team,
        "obs_dim": expected_obs if obs_dim is None else obs_dim,
        "action_dim": expected_action if action_dim is None else action_dim,
        "arch": ARCH,
        "trainable": team == "blue",
        "source": {"experiment": f"{team}_source"},
    }


def _write_sidecar(model_path: Path, *, team: str, run_id: str, seed: int) -> None:
    tag = model_path.stem.removeprefix("model_")
    sidecar = {
        "meta": {"name": f"{team}_recipe"},
        "run": {"train_run_id": run_id, "seed": seed},
    }
    model_path.with_name(f"recipe_{tag}.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))


def test_load_jax_matchup_policies_keeps_independent_source_provenance(tmp_path):
    blue_path = tmp_path / "model_blue_a.safetensors"
    red_path = tmp_path / "model_red_b.safetensors"
    save_jax_bundle(
        blue_path,
        {"blue": _jax_entry("blue")},
        provenance={"training": "blue-provenance"},
    )
    save_jax_bundle(
        red_path,
        {"red": _jax_entry("red")},
        provenance={"training": "red-provenance"},
    )
    _write_sidecar(blue_path, team="blue", run_id="run-blue", seed=11)
    _write_sidecar(red_path, team="red", run_id="run-red", seed=22)

    blue = load_matchup_policy(blue_path, team="blue", backend="jax")
    red = load_matchup_policy(red_path, team="red", backend="jax")

    assert blue.team == "blue"
    assert red.team == "red"
    assert blue.source["path"] == str(blue_path.resolve())
    assert red.source["path"] == str(red_path.resolve())
    assert blue.source["train_run_id"] == "run-blue"
    assert red.source["train_run_id"] == "run-red"
    assert blue.source["train_seed"] == 11
    assert red.source["train_seed"] == 22
    assert blue.source["bundle_source"] == {"experiment": "blue_source"}
    assert red.source["bundle_source"] == {"experiment": "red_source"}
    assert blue.source["bundle_teams"] == ["blue"]
    assert red.source["bundle_teams"] == ["red"]
    assert blue.source["bundle_provenance"] == {"training": "blue-provenance"}
    assert red.source["bundle_provenance"] == {"training": "red-provenance"}


def test_load_torch_matchup_policy_recreates_requested_team(tmp_path):
    model_path = tmp_path / "model_red_torch.pt"
    network = make_torch_policy(
        "shared",
        obs_dim=RED_OBS_SIZE,
        action_dim=RED_POLICY_ACTION_DIM,
        hidden_dim=4,
        hidden_layers=1,
    )
    save_torch_bundle(
        model_path,
        {
            "red": {
                "weights": network.state_dict(),
                "team": "red",
                "obs_dim": RED_OBS_SIZE,
                "action_dim": RED_POLICY_ACTION_DIM,
                "arch": ARCH,
                "trainable": False,
                "source": {"path": "/training/red.pt"},
            }
        },
        provenance={"seed": 7},
    )

    loaded = load_matchup_policy(model_path, team="red", backend="torch")

    assert loaded.backend == "cyborg"
    assert loaded.team == "red"
    assert not loaded.module.training
    assert loaded.source["bundle_source"] == {"path": "/training/red.pt"}
    assert loaded.source["bundle_provenance"] == {"seed": 7}
    for key, value in network.state_dict().items():
        torch.testing.assert_close(loaded.weights[key], value)


def test_torch_blue_lookup_matches_every_valid_blueflatwrapper_action():
    env = make_cyborg_env(
        CC4_STOCK,
        11,
        wrapper_class=BlueFlatWrapper,
        wrapper_kwargs={"pad_spaces": True},
    )
    reset_cyborg_env(env, CC4_STOCK, ep_seed=12)
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    for agent_id, agent_name in enumerate(env.possible_agents):
        lookup = cyborg_blue_flat_to_jax_lookup(const, agent_id)
        assert lookup.shape == (BLUE_ALLOW_TRAFFIC_END,)
        for action_idx, (action, valid) in enumerate(
            zip(env.actions(agent_name), env.action_mask(agent_name), strict=True)
        ):
            if not valid:
                continue
            if type(action).__name__ == "Sleep":
                expected = BLUE_SLEEP
            elif type(action).__name__ == "DeployDecoy":
                expected = encode_blue_action(
                    "DeployDecoy",
                    mappings.hostname_to_idx[action.hostname],
                    agent_id,
                    const=const,
                )
            else:
                expected = cyborg_blue_to_jax(action, agent_name, mappings, const=const)
            assert int(lookup[action_idx]) == expected
        assert np.all(lookup[np.asarray(env.action_mask(agent_name), dtype=bool)] >= 0)


def test_matchup_loader_rejects_wrong_team_dimensions_and_format(tmp_path):
    blue_only = tmp_path / "model_blue.safetensors"
    wrong_red = tmp_path / "model_red.safetensors"
    save_jax_bundle(blue_only, {"blue": _jax_entry("blue")})
    save_jax_bundle(
        wrong_red,
        {"red": _jax_entry("red", obs_dim=RED_OBS_SIZE - 1)},
    )

    with pytest.raises(ValueError, match="has no red policy"):
        load_matchup_policy(blue_only, team="red", backend="jax")
    with pytest.raises(ValueError, match="observation dimension mismatch"):
        load_matchup_policy(wrong_red, team="red", backend="jax")
    with pytest.raises(ValueError, match="does not match cyborg backend"):
        load_matchup_policy(blue_only, team="blue", backend="cyborg")


def test_run_matchup_rejects_mixed_policy_backends_before_environment_setup():
    policies = {
        "blue": LoadedMatchupPolicy("blue", "jax", None, None, {}),
        "red": LoadedMatchupPolicy("red", "cyborg", None, None, {}),
    }

    with pytest.raises(ValueError, match="mixed JAX/Torch"):
        matchup_runner.run_matchup_episode(policies, variant=CC4_STOCK, seed=42)


def test_evaluate_matchup_expands_seeds_and_reports_zero_sum_returns(monkeypatch):
    load_calls = []

    def fake_load(path, *, team, backend):
        load_calls.append((Path(path), team, backend))
        return LoadedMatchupPolicy(
            team,
            "jax",
            None,
            None,
            {"path": str(path), "source_run": f"run-{team}"},
        )

    sentinel_env = object()
    episode_calls = []

    def fake_episode(
        policies,
        *,
        variant,
        seed,
        deterministic=False,
        topology_path=None,
        env=None,
        topology_index=None,
    ):
        episode_calls.append((set(policies), variant.name, seed, deterministic, env, topology_index))
        return float(seed)

    monkeypatch.setattr(matchup_runner, "load_matchup_policy", fake_load)
    monkeypatch.setattr(matchup_runner, "run_matchup_episode", fake_episode)
    monkeypatch.setattr(matchup_runner, "make_joint_jax_env", lambda *args, **kwargs: sentinel_env)

    result = matchup_runner.evaluate_matchup(
        "blue-a.safetensors",
        "red-b.safetensors",
        backend="jax",
        variant=CC4_STOCK,
        seeds=[10, 20],
        episodes_per_seed=2,
        deterministic=True,
        progress=False,
    )

    assert load_calls == [
        (Path("blue-a.safetensors"), "blue", "jax"),
        (Path("red-b.safetensors"), "red", "jax"),
    ]
    assert [call[2] for call in episode_calls] == [10, 11, 20, 21]
    assert all(call[0] == {"blue", "red"} and call[3] for call in episode_calls)
    assert all(call[4:] == (sentinel_env, None) for call in episode_calls)
    assert result.episode_seeds == [10, 11, 20, 21]
    assert result.blue_returns == [10.0, 11.0, 20.0, 21.0]
    assert result.red_returns == [-10.0, -11.0, -20.0, -21.0]
    assert result.policies["blue"]["source_run"] == "run-blue"
    assert result.policies["red"]["source_run"] == "run-red"
    assert result.topology_paths == []
    assert result.episode_topology_paths == [None, None, None, None]
    assert result.topology_sampling == "generative"


def test_evaluate_matchup_forwards_and_reports_held_out_topology_bank(tmp_path, monkeypatch):
    bank = [tmp_path / "held-out-a.npz", tmp_path / "held-out-b.npz"]

    def fake_load(path, *, team, backend):
        return LoadedMatchupPolicy(team, "jax", None, None, {"path": str(path)})

    sentinel_env = object()
    factory_calls = []
    episode_assignments = []

    def fake_episode(
        policies,
        *,
        variant,
        seed,
        deterministic=False,
        topology_path=None,
        env=None,
        topology_index=None,
    ):
        episode_assignments.append((env, topology_index))
        return 0.0

    monkeypatch.setattr(matchup_runner, "load_matchup_policy", fake_load)
    monkeypatch.setattr(matchup_runner, "run_matchup_episode", fake_episode)
    monkeypatch.setattr(
        matchup_runner,
        "make_joint_jax_env",
        lambda *args, **kwargs: factory_calls.append((args, kwargs)) or sentinel_env,
    )

    result = matchup_runner.evaluate_matchup(
        "blue.safetensors",
        "red.safetensors",
        backend="jax",
        variant=CC4_STOCK,
        seeds=[7, 8],
        progress=False,
        topology_path=bank,
    )

    resolved_bank = [path.resolve() for path in bank]
    assert len(factory_calls) == 1
    assert factory_calls[0][1] == {"training_mode": False, "topology_path": resolved_bank}
    assert episode_assignments == [
        (sentinel_env, 0),
        (sentinel_env, 0),
        (sentinel_env, 1),
        (sentinel_env, 1),
    ]
    assert result.topology_paths == [str(path) for path in resolved_bank]
    assert result.episode_topology_paths == [
        str(resolved_bank[0]),
        str(resolved_bank[0]),
        str(resolved_bank[1]),
        str(resolved_bank[1]),
    ]
    assert result.topology_sampling == "exhaustive"
    assert result.episode_seeds == [7, 8, 7, 8]


def test_evaluate_matchup_can_sample_held_out_bank_randomly(tmp_path, monkeypatch):
    bank = [tmp_path / "held-out-a.npz", tmp_path / "held-out-b.npz"]

    monkeypatch.setattr(
        matchup_runner,
        "load_matchup_policy",
        lambda path, *, team, backend: LoadedMatchupPolicy(team, "jax", None, None, {}),
    )
    sentinel_env = object()
    factory_calls = []
    episode_assignments = []

    def fake_episode(
        policies,
        *,
        variant,
        seed,
        deterministic=False,
        topology_path=None,
        env=None,
        topology_index=None,
    ):
        episode_assignments.append((env, topology_index))
        return 0.0

    monkeypatch.setattr(matchup_runner, "run_matchup_episode", fake_episode)
    monkeypatch.setattr(
        matchup_runner,
        "make_joint_jax_env",
        lambda *args, **kwargs: factory_calls.append((args, kwargs)) or sentinel_env,
    )
    result = matchup_runner.evaluate_matchup(
        "blue.safetensors",
        "red.safetensors",
        backend="jax",
        variant=CC4_STOCK,
        seeds=[7, 8],
        progress=False,
        topology_path=bank,
        topology_sampling="random",
    )

    resolved_bank = [path.resolve() for path in bank]
    assert len(factory_calls) == 1
    assert factory_calls[0][1] == {"training_mode": False, "topology_path": resolved_bank}
    assert episode_assignments == [(sentinel_env, None), (sentinel_env, None)]
    assert result.episode_topology_paths == [None, None]
    assert result.topology_sampling == "random"


def test_run_matchup_episode_forwards_topology_bank_to_joint_env(monkeypatch):
    policies = {
        "blue": LoadedMatchupPolicy("blue", "jax", None, None, {}),
        "red": LoadedMatchupPolicy("red", "jax", None, None, {}),
    }
    bank = [Path("held-out-a.npz"), Path("held-out-b.npz")]
    captured = {}

    def fake_env(variant, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("environment captured")

    monkeypatch.setattr(matchup_runner, "make_joint_jax_env", fake_env)

    with pytest.raises(RuntimeError, match="environment captured"):
        matchup_runner.run_matchup_episode(
            policies,
            variant=CC4_STOCK,
            seed=42,
            topology_path=bank,
        )

    assert captured == {"training_mode": False, "topology_path": bank}
