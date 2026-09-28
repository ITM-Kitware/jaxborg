"""Focused contracts for imported weights, policy observations, and eval recipes."""

from __future__ import annotations

import copy
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import struct

from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.decoy_telemetry import record_decoy_connection
from jaxborg.actions.encoding import BLUE_ANALYSE_START, BLUE_REMOVE_START, BLUE_SLEEP
from jaxborg.evaluation.matchup_runner import cyborg_blue_flat_to_jax_lookup
from jaxborg.pretrained.hmarl import (
    HMARLPolicy,
    _host_slots,
    action_layout,
    actor_logits,
    branch_mask,
    policy_observations,
    update_memory,
)
from jaxborg.pretrained.hmarl_import import actor_arrays, read_weight_prefix
from jaxborg.recipe import _validate, load, project_cleanrl, project_eval, project_jax
from jaxborg.state import create_initial_state


@struct.dataclass
class EnvState:
    state: object
    const: object


@pytest.fixture
def hmarl_state(jax_const):
    env = EnvState(create_initial_state(), jax_const.replace(cage4_enhanced_obs=True))
    return HMARLPolicy("expert").initialize(env)


def _torch_weights(agent, branch, seed=10):
    rng = np.random.default_rng(seed)
    n = 3 if agent == 4 else 1
    inp = 1 + n * {"investigate": 48, "recover": 16, "master": 75}[branch]
    out = 2 if branch == "master" else 2 + 80 * n
    weights = {}
    for layer, (i, o) in zip(("_hidden_layers.0", "_hidden_layers.1", "_logits"), ((inp, 256), (256, 256), (256, out))):
        weights[f"internal_model.{layer}._model.0.weight"] = rng.normal(0, 0.04, (o, i)).astype(np.float32)
        weights[f"internal_model.{layer}._model.0.bias"] = rng.normal(0, 0.04, o).astype(np.float32)
    return weights


@pytest.mark.parametrize("agent", [0, 4])
@pytest.mark.parametrize("branch", ["investigate", "recover", "master"])
def test_imported_actor_matches_torch_linear_tanh(agent, branch):
    weights = _torch_weights(agent, branch)
    converted = actor_arrays(
        read_weight_prefix(pickle.dumps({"weights": weights}, protocol=5)), agent=agent, branch=branch
    )
    obs_dim = weights["internal_model._hidden_layers.0._model.0.weight"].shape[1]
    obs = np.random.default_rng(123).integers(0, 4, (12, obs_dim)).astype(np.float32)
    x = torch.from_numpy(obs)
    for layer in ("_hidden_layers.0", "_hidden_layers.1", "_logits"):
        prefix = f"internal_model.{layer}._model.0"
        x = torch.nn.functional.linear(
            x, torch.from_numpy(weights[prefix + ".weight"]), torch.from_numpy(weights[prefix + ".bias"])
        )
        if layer != "_logits":
            x = torch.tanh(x)
    actual = jax.jit(partial(actor_logits, agent=agent, branch=branch))(converted, obs=obs)
    np.testing.assert_allclose(actual, x.numpy(), atol=2e-6, rtol=2e-5)


class _ExecutableTail:
    def __reduce__(self):
        return eval, ("1 / 0",)


def test_importer_does_not_unpickle_executable_tail_or_allow_executable_weights():
    weights = _torch_weights(0, "recover")
    raw = pickle.dumps({"weights": weights, "config": _ExecutableTail()}, protocol=5)
    assert set(read_weight_prefix(raw)) == set(weights)
    raw = pickle.dumps({"weights": {"bad": _ExecutableTail(), "other": 1}}, protocol=5)
    with pytest.raises(pickle.UnpicklingError, match="Unsupported checkpoint global"):
        read_weight_prefix(raw)


def test_importer_rejects_mismatched_checkpoint_shape():
    with pytest.raises(ValueError, match="shapes"):
        actor_arrays(_torch_weights(0, "recover"), agent=4, branch="recover")


@pytest.mark.parametrize("agent", range(5))
def test_action_order_and_masks_match_unpadded_upstream(agent, jax_const):
    n_subnets = 3 if agent == 4 else 1
    n = n_subnets * 16
    lookup = np.asarray(action_layout(n_subnets))
    reference = cyborg_blue_flat_to_jax_lookup(jax_const, agent)[: len(lookup)]
    # Traffic is never selected by either released sub-policy.
    select = np.ones(len(lookup), dtype=bool)
    select[3 * n + 2 : -n] = False
    np.testing.assert_array_equal(lookup[select], reference[select])
    valid = jnp.arange(n) % 3 != 0
    inv = np.asarray(branch_mask(n_subnets, "investigate", valid, False))
    rec = np.asarray(branch_mask(n_subnets, "recover", valid, False))
    assert inv.sum() == 2 * int(valid.sum())
    assert rec.sum() == 2 * n  # Includes invalid padded recovery actions upstream.
    assert np.all(rec[n + 1 : 3 * n + 1])
    for branch in ("investigate", "recover"):
        busy = np.asarray(branch_mask(n_subnets, branch, valid, True))
        assert busy.sum() == 1
        assert lookup[busy][0] == BLUE_SLEEP


def test_observed_ioc_memory_priorities_and_recovery(hmarl_state):
    env, memory = hmarl_state
    _, hosts, valid = _host_slots(env.const, 0)
    slot = int(np.flatnonzero(valid[0])[0])
    host = int(hosts[0, slot])
    # Latent compromise alone must never become an observation.
    state = env.state.replace(host_compromised=env.state.host_compromised.at[host].set(2))
    memory, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slot]) == 0
    state = state.replace(blue_file_evidence=state.blue_file_evidence.at[host].set(1))
    memory, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slot]) == 2  # User file.
    state = state.replace(blue_file_evidence=state.blue_file_evidence.at[host].set(2))
    memory, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slot]) == 1  # Root file.
    state = state.replace(blue_file_evidence=jnp.zeros_like(state.blue_file_evidence))
    memory, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slot]) == 1  # Empty Analyse preserves history.
    state = state.replace(blue_recovered_this_step=state.blue_recovered_this_step.at[host].set(True))
    _, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slot]) == 0


def test_decoy_remote_origin_is_delivered_one_tick_later(hmarl_state):
    env, memory = hmarl_state
    _, targets, valid = _host_slots(env.const, 0)
    target = int(targets[0, np.flatnonzero(valid[0])[0]])
    _, origins, valid = _host_slots(env.const, 1)
    slot = int(np.flatnonzero(valid[0])[0])
    origin = int(origins[0, slot])
    state = env.state.replace(blue_old_decoy_sources=env.state.blue_old_decoy_sources.at[target, 0].set(origin))
    memory, iocs = update_memory(state, env.const, memory)
    assert not np.asarray(iocs[1]).any()
    assert memory["delivered"][origin]
    memory, iocs = update_memory(state, env.const, memory)
    assert int(iocs[1][0, slot]) == 3
    assert not np.asarray(iocs[0]).any()  # IOC identifies the origin, not the decoy destination.


def test_only_first_decoy_ioc_per_subnet_and_file_overrides(hmarl_state):
    env, memory = hmarl_state
    _, hosts, valid = _host_slots(env.const, 0)
    slots = np.flatnonzero(valid[0])[:2]
    a, b = (int(hosts[0, i]) for i in slots)
    memory["decoys"] = memory["decoys"].at[a].set(True).at[b].set(True)
    memory, iocs = update_memory(env.state, env.const, memory)
    assert int(iocs[0][0, slots[0]]) == 3
    assert int(iocs[0][0, slots[1]]) == 0
    state = env.state.replace(blue_file_evidence=env.state.blue_file_evidence.at[a].set(2))
    _, iocs = update_memory(state, env.const, memory)
    assert int(iocs[0][0, slots[0]]) == 1


@pytest.mark.parametrize("agent", [0, 4])
def test_observation_subnet_order_and_busy_sentinel(agent, hmarl_state):
    env, memory = hmarl_state
    n = 3 if agent == 4 else 1
    iocs = jnp.arange(16 * n).reshape(n, 16) % 4
    master, inv, rec = policy_observations(env.state, env.const, agent, memory, iocs)
    assert (master.size, inv.size, rec.size) == (1 + 75 * n, 1 + 48 * n, 1 + 16 * n)
    np.testing.assert_array_equal(np.asarray(master[1:]).reshape(n, 75)[:, -16:], iocs)
    np.testing.assert_array_equal(np.asarray(inv[1:]).reshape(n, 48)[:, -16:], iocs)
    np.testing.assert_array_equal(rec[1:], iocs.reshape(-1))
    busy = env.state.replace(blue_pending_ticks=env.state.blue_pending_ticks.at[agent].set(1))
    for obs in policy_observations(busy, env.const, agent, memory, iocs):
        np.testing.assert_array_equal(obs, np.ones_like(obs))


def test_telemetry_is_optional_and_monitor_ages_it(hmarl_state, monkeypatch):
    from jaxborg.actions import red_common

    env, _ = hmarl_state
    _, hosts, valid = _host_slots(env.const, 0)
    target, origin = (int(hosts[0, i]) for i in np.flatnonzero(valid[0])[:2])
    monkeypatch.setattr(red_common, "select_scan_execution_source_host", lambda *a: jnp.int32(origin))
    plain = create_initial_state()
    assert record_decoy_connection(plain, env.const, 0, target, True) is plain
    state = record_decoy_connection(env.state, env.const, 0, target, False)
    assert int(state.blue_decoy_sources[target, 0]) == -1
    state = record_decoy_connection(state, env.const, 0, target, True)
    assert int(state.blue_decoy_sources[target, 0]) == origin
    state = apply_blue_monitor(state, env.const, 0)
    assert int(state.blue_old_decoy_sources[target, 0]) == origin
    assert int(state.blue_decoy_sources[target, 0]) == -1
    state = apply_blue_monitor(state, env.const, 0)
    assert int(state.blue_old_decoy_sources[target, 0]) == -1


@pytest.mark.parametrize("variant", ["expert", "meta"])
def test_jitted_hierarchy_uses_released_masks_and_reset_memory(variant, hmarl_state):
    env, memory = hmarl_state
    tensors = {}
    for agent in range(5):
        for branch in ("investigate", "recover", "master"):
            arrays = actor_arrays(_torch_weights(agent, branch), agent=agent, branch=branch)
            arrays = {k: np.zeros_like(v) for k, v in arrays.items()}
            # All ties pick first valid action; Meta picks Recover after reset.
            if branch == "master":
                arrays[f"agent{agent}.master.2.bias"][1] = 100
            tensors.update(arrays)
    model = HMARLPolicy(variant)
    select = jax.jit(partial(model.select_actions, deterministic=True))
    first, _ = select(tensors, env, jax.random.PRNGKey(3), memory)
    assert int(first[0]) >= BLUE_ANALYSE_START
    assert int(first[0]) < BLUE_ANALYSE_START + 16
    _, hosts, valid = _host_slots(env.const, 0)
    host = int(hosts[0, np.flatnonzero(valid[0])[0]])
    state = env.state.replace(time=jnp.int32(1), blue_file_evidence=env.state.blue_file_evidence.at[host].set(2))
    selected, updated = select(tensors, env.replace(state=state), jax.random.PRNGKey(3), memory)
    # Recover's first slot may be padded (and therefore maps to Sleep).
    expected = BLUE_REMOVE_START if valid[0, 0] else BLUE_SLEEP
    assert int(selected[0]) == expected
    _, fresh = model.initialize(env)
    assert not np.asarray(fresh["files"]).any()
    assert np.asarray(updated["files"]).any()
    state = state.replace(blue_pending_ticks=jnp.ones(5, dtype=jnp.int32))
    selected, _ = select(tensors, env.replace(state=state), jax.random.PRNGKey(3), updated)
    np.testing.assert_array_equal(selected, np.full(5, BLUE_SLEEP))


@pytest.mark.parametrize("variant", ["expert", "meta"])
def test_eval_only_recipe_matches_cotraining_cases_and_rejects_training(variant):
    recipe = load(f"hmarl_{variant}")
    reference = load("cotraining_lstm_env_diversity")
    ev = recipe["eval"]
    assert ev["topology_generation"] == reference["eval"]["topology_generation"]
    # Same seeds as the final-model FSM/CIA sweep (not the cheaper checkpoint curve).
    scripted = next(job for job in reference["eval"]["after_training"] if job["name"] == "scripted-reds")
    assert ev["seeds"] == scripted["args"][scripted["args"].index("--seeds") + 1]
    assert ev["episodes_per_seed"] == 6
    assert ev["episode_length"] == reference["train"]["episode_length"] == 500
    assert ev["reds"] == ["fsm", "cia_c", "cia_i", "cia_a", "aggressive", "stealthy", "impact"]
    assert ev["deterministic"] is False
    assert ev["cia"] == reference["eval"]["cia"]
    assert "train" not in recipe and "core" not in recipe
    assert project_eval(recipe)["EVAL_VARIANT"].num_steps == 500
    for project in (project_jax, project_cleanrl):
        with pytest.raises(ValueError, match="evaluation-only"):
            project(recipe)
    broken = copy.deepcopy(recipe)
    broken["pretrained"]["architecture"]["hidden_dim"] = 128
    with pytest.raises(ValueError, match="256x256"):
        _validate(broken, source="test")


def test_pretrained_cli_prepare_only_and_overrides(monkeypatch, tmp_path):
    from jaxborg.pretrained import hmarl_eval

    bundle = tmp_path / "actor.safetensors"
    bundle.touch()
    loaded = []
    monkeypatch.setattr(hmarl_eval, "load_policy", lambda path, **kwargs: loaded.append((path, kwargs)))
    hmarl_eval.main(["--recipe", "hmarl_meta", "--model", str(bundle), "--prepare-only"])
    assert loaded == [(bundle, {"variant": "meta"})]


@struct.dataclass
class _ScanState:
    time: object
    blue_pending_ticks: object
    ot_service_stopped: object
    host_service_reliability: object
    host_decoy_reliability: object


@struct.dataclass
class _ScanEnvState:
    state: object
    extras: dict


class _ScanEnv:
    agents = ("blue_0",)

    def reset_at_topology(self, key, topology_index):
        del key, topology_index
        state = _ScanState(jnp.int32(0), jnp.zeros(1), jnp.zeros(3, bool), jnp.full((3, 1), 100), jnp.full((3, 1), 100))
        return {"blue_0": jnp.zeros(1)}, _ScanEnvState(state, {"host_resilience_role": jnp.zeros(3, jnp.int32)})

    def step_env(self, key, env_state, actions):
        del key
        state = env_state.state.replace(
            time=env_state.state.time + 1, ot_service_stopped=jnp.array([True, False, False])
        )
        return (
            {"blue_0": jnp.zeros(1)},
            env_state.replace(state=state),
            {"blue_0": actions["blue_0"].astype(jnp.float32)},
            {"__all__": state.time >= 2},
            {},
        )


def test_stateful_adapter_scans_batches_and_resets_without_terminal_overcount():
    from jaxborg.evaluation.jax_scripted_red import _run_jax_scripted_red_episode_scan
    from jaxborg.evaluation.stateful_blue import StatefulBluePolicy

    class CounterPolicy(StatefulBluePolicy):
        def initialize(self, state):
            return state, jnp.int32(0)

        def select_actions(self, weights, state, key, carry, *, deterministic):
            del weights, state, key, deterministic
            return jnp.array([carry + 1]), carry + 1

    scan = partial(
        _run_jax_scripted_red_episode_scan,
        policy_module=CounterPolicy(),
        env=_ScanEnv(),
        num_steps=5,
        deterministic=True,
    )
    batch = jax.vmap(scan, in_axes=(None, 0, 0, 0))
    args = ({}, jax.random.split(jax.random.PRNGKey(0), 2), jnp.array([0, 1]), jnp.array([[1, 0, 0], [0, 1, 0]]))
    rewards, cia = batch(*args)
    np.testing.assert_array_equal(rewards, [3, 3])  # Decisions 1+2, then stop.
    np.testing.assert_array_equal(cia, [[-10, -10, -10], [0, 0, 0]])
    np.testing.assert_array_equal(batch(*args)[0], rewards)


def test_import_checksum_and_atomic_actor_bundle(monkeypatch, tmp_path):
    import hashlib
    import json

    from safetensors.numpy import load_file

    from jaxborg.pretrained import hmarl_import

    raw = pickle.dumps({"weights": _torch_weights(0, "recover"), "unused": 0}, protocol=5)
    source = tmp_path / "policy_state.pkl"
    source.write_bytes(raw)
    manifest = tmp_path / "manifest.json"
    entry = dict(path=source.name, agent=0, branch="recover", sha256=hashlib.sha256(raw).hexdigest())
    manifest.write_text(json.dumps({"checkpoints": [entry]}))
    monkeypatch.setattr(hmarl_import, "MANIFEST_PATH", manifest)
    output = tmp_path / "converted.safetensors"
    hmarl_import.import_checkpoint(output, upstream_dir=tmp_path)
    assert len(load_file(output)) == 6
    before = output.read_bytes()
    source.write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        hmarl_import.import_checkpoint(output, upstream_dir=tmp_path)
    assert output.read_bytes() == before


def test_scan_emits_only_successful_observable_decoy_connections(hmarl_state, monkeypatch):
    from jaxborg.actions import red_common, red_scan_unified

    env, _ = hmarl_state
    _, hosts, valid = _host_slots(env.const, 0)
    target, origin = (int(hosts[0, i]) for i in np.flatnonzero(valid[0])[:2])
    for module in (red_common, red_scan_unified):
        monkeypatch.setattr(module, "select_scan_execution_source_host", lambda *a: jnp.int32(origin))
    monkeypatch.setattr(red_scan_unified, "can_reach_subnet_from_source_host", lambda *a: jnp.bool_(True))
    state = env.state.replace(
        red_discovered_hosts=env.state.red_discovered_hosts.at[0, target].set(True),
        host_decoys=env.state.host_decoys.at[target, 0].set(True),
    )
    scan = partial(
        red_scan_unified.apply_scan_unified,
        const=env.const,
        agent_id=0,
        target_host=target,
        key=jax.random.PRNGKey(1),
        has_detection_roll=jnp.bool_(False),
        detection_rate=jnp.float32(0),
    )
    result = scan(state=state)
    assert int(result.blue_decoy_sources[target, 0]) == origin
    failed = state.replace(red_discovered_hosts=jnp.zeros_like(state.red_discovered_hosts))
    assert int(scan(state=failed).blue_decoy_sources[target, 0]) == -1
    no_decoy = state.replace(host_decoys=jnp.zeros_like(state.host_decoys))
    assert int(scan(state=no_decoy).blue_decoy_sources[target, 0]) == -1
