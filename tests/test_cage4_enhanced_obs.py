"""Enhanced evidence is observed, delayed, local, persistent, and portable."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.File import File
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.blue_analyse import apply_blue_analyse
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.duration import process_blue_with_duration
from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.blue_observation_contract import BLUE_HOST_SLOTS, blue_obs_size, enhanced_obs_enabled
from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle
from jaxborg.critic_observations import get_critic_obs
from jaxborg.evaluation.enhanced_blue_wrapper import EnhancedBlueFlatWrapper, EnhancedEnterpriseMAE
from jaxborg.evaluation.jax_runner import load_jax_checkpoint
from jaxborg.evaluation.matchup_runner import load_matchup_policy
from jaxborg.observations import get_blue_obs, update_blue_observation_memory
from jaxborg.parity.translate import build_mappings_from_cyborg
from jaxborg.policies import init_policy_params, initial_carry, policy_from_arch, policy_step
from jaxborg.recipe import eval_variant, load, team_recipe, train_variant
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from jaxborg.state import create_initial_state


def _native(wrapper=EnhancedBlueFlatWrapper, version=1):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=SleepAgent, red_agent_class=SleepAgent, steps=500
    )
    env = wrapper(CybORG(sg, "sim", seed=42), pad_spaces=True, blue_observation_version=version)
    obs, _ = env.reset(seed=42)
    return env, obs


def _extra(obs):
    return np.asarray(obs)[210:].reshape(4, BLUE_HOST_SLOTS)


@pytest.fixture
def scenario():
    env, obs = _native()
    const = build_const_from_cyborg(env.env).replace(cage4_enhanced_obs=True, blue_observation_version=1)
    state = create_initial_state().replace(host_services=const.initial_services)
    mappings = build_mappings_from_cyborg(env.env)
    host = next(h for h in env.hosts("blue_agent_0") if "router" not in h and h in mappings.hostname_to_idx)
    index = mappings.hostname_to_idx[host]
    slot = int(np.flatnonzero(np.asarray(const.obs_host_map[const.blue_obs_subnets[0, 0], :16]) == index)[0])
    return env, obs, const, state, host, index, slot


def test_reset_layout_and_legacy_prefix_match_native(scenario):
    env, obs, const, state, _, _, _ = scenario
    legacy_const = const.replace(cage4_enhanced_obs=False)
    for b in range(5):
        name = f"blue_agent_{b}"
        actual = get_blue_obs(state, const, b)
        assert actual.shape == (402,)
        np.testing.assert_array_equal(actual[:210], get_blue_obs(state, legacy_const, b))
        np.testing.assert_array_equal(actual, obs[name])
        assert env.observation_space(name).contains(obs[name])
    assert blue_obs_size() == 210


@pytest.mark.parametrize("level,name", [(1, "cmd.sh"), (2, "escalate.sh")])
def test_analyse_delayed_file_evidence_and_native_parity(scenario, level, name):
    env, obs, const, state, host, index, slot = scenario
    native_host = env.env.environment_controller.state.hosts[host]
    native_host.files.append(File(name, "/tmp", native_host.users[0], density=0.9, signed=False))
    state = state.replace(host_file_artifact=state.host_file_artifact.at[index].set(level))
    # Hidden changes alone cannot affect observations, including another Blue agent's vector.
    for b in range(5):
        np.testing.assert_array_equal(get_blue_obs(state, const, b), obs[f"blue_agent_{b}"])
    native_action = next(
        i for i, a in enumerate(env.actions("blue_agent_0")) if type(a).__name__ == "Analyse" and a.hostname == host
    )
    jax_action = encode_blue_action("Analyse", index, 0, const=const)
    with jax.disable_jit():
        state = process_blue_with_duration(state, const, 0, jax_action)
    obs, *_ = env.step({"blue_agent_0": native_action})
    assert _extra(obs["blue_agent_0"])[1, slot] == 0
    assert state.blue_file_evidence[index] == 0
    with jax.disable_jit():
        state = process_blue_with_duration(state, const, 0, BLUE_SLEEP)
    obs, *_ = env.step({})
    assert _extra(obs["blue_agent_0"])[1, slot] == level / 2
    np.testing.assert_array_equal(_extra(get_blue_obs(state, const, 0)), _extra(obs["blue_agent_0"]))
    # Removing the live implant file behind the wrapper cannot alter remembered evidence.
    native_host.files.clear()
    for _ in range(3):
        obs, *_ = env.step({})
    assert _extra(obs["blue_agent_0"])[1, slot] == level / 2
    # A new successful negative Analyse replaces stale evidence.
    env.step({"blue_agent_0": native_action})
    obs, *_ = env.step({})
    assert _extra(obs["blue_agent_0"])[1, slot] == 0
    obs, _ = env.reset(seed=42)
    assert not _extra(obs["blue_agent_0"])[1:].any()


def test_analyse_does_not_leak_other_hosts_or_change_legacy(scenario):
    _, _, const, state, _, index, _ = scenario
    state = state.replace(host_file_artifact=state.host_file_artifact.at[index].set(2))
    assert apply_blue_analyse(state, const.replace(cage4_enhanced_obs=False), 0, index) is state
    after = apply_blue_analyse(state, const, 1, index)
    np.testing.assert_array_equal(after.blue_file_evidence, state.blue_file_evidence)
    for invalid in (-1, const.host_active.size):
        after = apply_blue_analyse(state, const, 0, invalid)
        assert not np.asarray(after.blue_file_evidence).any()


@pytest.mark.parametrize("action", ["Remove", "Restore"])
def test_recovery_clears_memory_on_completion_but_remove_leaves_files(scenario, action):
    env, _, const, state, host, index, slot = scenario
    native_host = env.env.environment_controller.state.hosts[host]
    native_host.files.append(File("cmd.sh", "/tmp", native_host.users[0], density=0.9))
    state = state.replace(
        host_file_artifact=state.host_file_artifact.at[index].set(1),
        blue_file_evidence=state.blue_file_evidence.at[index].set(1),
        blue_process_memory=state.blue_process_memory.at[index].set(True),
        blue_network_memory=state.blue_network_memory.at[index].set(True),
    )
    env._evidence["blue_agent_0"][1:, slot] = [0.5, 1.0, 1.0]
    native_action = next(
        i for i, a in enumerate(env.actions("blue_agent_0")) if type(a).__name__ == action and a.hostname == host
    )
    ticks = 3 if action == "Remove" else 5
    for tick in range(ticks):
        obs, *_ = env.step({"blue_agent_0": native_action} if tick == 0 else {})
        if tick < ticks - 1:
            assert _extra(obs["blue_agent_0"])[1:, slot].any()
    with jax.disable_jit():
        fn = apply_blue_remove if action == "Remove" else apply_blue_restore
        state = fn(state, const, 0, index)
    state = update_blue_observation_memory(state, const)
    assert not _extra(obs["blue_agent_0"])[1:, slot].any()
    assert not _extra(get_blue_obs(state, const, 0))[1:, slot].any()
    assert int(state.host_file_artifact[index]) == (1 if action == "Remove" else 0)
    assert any(f.name == "cmd.sh" for f in native_host.files) == (action == "Remove")


def test_alert_history_persists_after_transient_flags_expire(scenario):
    _, _, const, state, _, index, slot = scenario
    state = state.replace(
        host_exploit_detected=state.host_exploit_detected.at[index].set(True),
        host_activity_detected=state.host_activity_detected.at[index].set(True),
    )
    state = update_blue_observation_memory(state, const)
    state = state.replace(
        host_exploit_detected=jnp.zeros_like(state.host_exploit_detected),
        host_activity_detected=jnp.zeros_like(state.host_activity_detected),
    )
    state = update_blue_observation_memory(state, const)
    assert _extra(get_blue_obs(state, const, 0))[2:, slot].all()
    assert not _extra(get_blue_obs(state, const, 1))[2:].any()


@pytest.mark.parametrize("name", ["shared", "recurrent", "mappo"])
def test_enhanced_checkpoint_roundtrip_and_inference(scenario, tmp_path, name):
    _, native_obs, const, state, _, _, _ = scenario
    arch = dict(name=name, hidden_dim=8, hidden_layers=1)
    if name == "recurrent":
        arch.update(cell="lstm", trunk="shared")
    if name == "mappo":
        arch.update(cage4_enhanced_obs=True, critic_input="joint_observations", blue_observation_version=1)
    policy = policy_from_arch(arch, action_dim=BLUE_ALLOW_TRAFFIC_END)
    params = init_policy_params(policy, jax.random.PRNGKey(0), obs_dim=402)
    path = tmp_path / "model_test.safetensors"
    save_jax_bundle(path, {"blue": PolicyBundleEntry(params, "blue", 402, BLUE_ALLOW_TRAFFIC_END, arch)})
    recipe = {"arch": arch, "core": {}, "cage4_enhanced_obs": True, "run": {"blue_observation_version": 1}}
    (tmp_path / "recipe_test.yaml").write_text(yaml.safe_dump(recipe))
    loaded, weights, _ = load_jax_checkpoint(path)
    matchup = load_matchup_policy(path, team="blue", backend="jax")
    assert matchup.source["observation_dim"] == 402
    obs = jnp.stack([jnp.asarray(native_obs[f"blue_agent_{i}"]) for i in range(5)])
    pi, value, _ = policy_step(
        loaded, weights, obs, jnp.ones((5, BLUE_ALLOW_TRAFFIC_END), dtype=bool), carry=initial_carry(loaded, 5)
    )
    assert np.isfinite(pi.logits).all()
    if name == "mappo":
        critic = get_critic_obs(state, const, "joint_observations")
        assert critic.shape == (5, loaded.critic_obs_dim)
        _, value = loaded.apply(weights, obs, critic_obs=critic)
        assert np.isfinite(value).all()
    recipe["cage4_enhanced_obs"] = False
    (tmp_path / "recipe_test.yaml").write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="observation dimension mismatch"):
        load_jax_checkpoint(path)


def test_flag_is_enabled_only_for_actual_cotraining_recipes():
    for path in Path("recipes").rglob("*.yaml"):
        recipe = load(str(path))
        if recipe.get("kind") == "pretrained_eval":
            assert enhanced_obs_enabled(recipe)
            continue
        expected = path.parent.name == "cotraining" and recipe.get("train", {}).get("teams") == "both"
        assert enhanced_obs_enabled(recipe) is expected, path
        assert train_variant(recipe).cage4_enhanced_obs is expected
        assert eval_variant(recipe).cage4_enhanced_obs is expected
        if expected:
            benchmark = next(e for e in recipe["eval"]["after_training"] if e["name"] == "cage4-benchmark")
            assert benchmark["script"] == "scripts/eval/eval_scripted_reds.py"
            assert "1000-1099" in benchmark["args"]
            for team in ("blue", "red"):
                arch = team_recipe(recipe, team)["arch"]
                if arch["name"] == "mappo":
                    assert arch["cage4_enhanced_obs"] is True
    with pytest.raises(ValueError, match="YAML boolean"):
        enhanced_obs_enabled({"cage4_enhanced_obs": "True"})


def test_enterprise_wrapper_enhanced_shape():
    env, obs = _native(EnhancedEnterpriseMAE)
    assert all(o.shape == (402,) for o in obs.values())
    obs, *_ = env.step({})
    assert all(o.shape == (402,) for o in obs.values())


def test_implant_artifacts_survive_withdraw_and_track_escalation(scenario):
    from jaxborg.actions.red_common import apply_exploit_success
    from jaxborg.actions.red_privesc import apply_privesc
    from jaxborg.actions.red_withdraw import apply_withdraw

    _, _, const, state, _, index, _ = scenario
    key = jax.random.PRNGKey(11)
    with jax.disable_jit():
        failed = apply_exploit_success(state, const, 0, index, jnp.bool_(False), key)
        assert failed.host_file_artifact[index] == 0
        state = apply_exploit_success(state, const, 0, index, jnp.bool_(True), key)
        assert state.host_file_artifact[index] == 1
        assert state.blue_file_evidence[index] == 0
        state = state.replace(
            red_primary_is_abstract=state.red_primary_is_abstract.at[0].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(index),
            red_primary_pid=state.red_primary_pid.at[0].set(state.red_session_pids[0, index, 0]),
        )
        state = apply_privesc(state, const, 0, index, key)
        assert state.host_file_artifact[index] == 2
        assert state.blue_file_evidence[index] == 0
        state = apply_withdraw(state, const, 0, index)
        assert not state.red_sessions[0, index]
        assert state.host_file_artifact[index] == 2
        state = apply_blue_analyse(state, const, 0, index)
        assert state.blue_file_evidence[index] == 2
        state = apply_blue_restore(state, const, 0, index)
        assert state.host_file_artifact[index] == 0


def test_native_and_jax_alert_memory_parity(scenario):
    env, _, const, state, host, index, _ = scenario
    events = env.env.environment_controller.state.hosts[host].events
    events.process_creation.append({})
    events.network_connections.append({})
    state = state.replace(
        host_exploit_detected=state.host_exploit_detected.at[index].set(True),
        host_activity_detected=state.host_activity_detected.at[index].set(True),
    )
    state = update_blue_observation_memory(state, const)
    obs = env.observation_change("blue_agent_0", {})
    np.testing.assert_array_equal(get_blue_obs(state, const, 0), obs)
    events.process_creation.clear()
    events.network_connections.clear()
    state = state.replace(
        host_exploit_detected=jnp.zeros_like(state.host_exploit_detected),
        host_activity_detected=jnp.zeros_like(state.host_activity_detected),
    )
    state = update_blue_observation_memory(state, const)
    obs = env.observation_change("blue_agent_0", {})
    np.testing.assert_array_equal(get_blue_obs(state, const, 0), obs)


def test_jax_snapshot_reset_and_batched_step_shapes(scenario, tmp_path):
    from dataclasses import replace

    from jaxborg.evaluation.jax_env_factory import make_jax_env, make_joint_jax_env
    from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
    from jaxborg.scenarios.cc4.topology import save_topology

    _, _, const, _, _, _, _ = scenario
    snapshot = tmp_path / "topology.npz"
    save_topology(const.replace(cage4_enhanced_obs=False), snapshot)
    variant = replace(CC4_STOCK, cage4_enhanced_obs=True, blue_observation_version=1)
    for factory in (make_jax_env, make_joint_jax_env):
        env = factory(variant, topology_path=snapshot)
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert state.const.cage4_enhanced_obs
        assert obs["blue_0"].shape == env.observation_space("blue_0").shape == (402,)
        batched_obs, batched_state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        assert batched_obs["blue_0"].shape == (2, 402)
        actions = {agent: jnp.int32(BLUE_SLEEP if agent.startswith("blue") else 0) for agent in env.agents}
        next_obs, next_state, *_ = jax.eval_shape(env.step, jax.random.PRNGKey(0), state, actions)
        assert next_obs["blue_0"].shape == (402,)
        assert next_state.const.cage4_enhanced_obs
        assert batched_state.const.cage4_enhanced_obs


def test_native_enhanced_lstm_rollout_ends_at_episode_horizon():
    from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env
    from jaxborg.evaluation.jax_runner import run_episode
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    variant = GameVariant(name="smoke", cage4_enhanced_obs=True, blue_observation_version=1, num_steps=3)
    env = make_cyborg_env(variant, 9, wrapper_class=BlueFlatWrapper)
    module = policy_from_arch(
        dict(name="recurrent", cell="lstm", hidden_dim=8, hidden_layers=1), action_dim=BLUE_ALLOW_TRAFFIC_END
    )
    params = init_policy_params(module, jax.random.PRNGKey(0), 402)
    score = run_episode(env, variant, 9, module, params, False, jax.random.PRNGKey(1))
    assert np.isfinite(score)
    assert env.env.environment_controller.determine_done()


def test_torch_enhanced_checkpoint_and_joint_buffer_contract(tmp_path):
    from jaxborg.checkpoint import save_torch_bundle
    from jaxborg.evaluation.cyborg_runner import _pad_obs_mask, load_torch_policy
    from jaxborg.recipe import project_cleanrl
    from scripts.train.algorithms.ippo_cyborg import _make_joint_runtimes

    recipe = load("cotraining")
    recipe["run"] = {"blue_observation_version": 1}
    recipe["arch"].update(hidden_dim=8, hidden_layers=1)
    cfg = project_cleanrl(recipe)
    runtimes = _make_joint_runtimes(recipe, cfg, seed=0, num_envs=1, num_steps=2)
    assert runtimes["blue"].rollout["obs"].shape == (2, 1, 5, 402)
    blue = runtimes["blue"]
    path = tmp_path / "model_torch.pt"
    save_torch_bundle(path, {"blue": PolicyBundleEntry(blue.agent.state_dict(), "blue", 402, 242, blue.arch)})
    (tmp_path / "recipe_torch.yaml").write_text(yaml.safe_dump(recipe))
    loaded, _ = load_torch_policy(path)
    env, obs = _native(EnhancedEnterpriseMAE)
    obs, _, _, _, info = env.step({})
    obs, mask = _pad_obs_mask(obs, info)
    assert obs.shape == (5, 402)
    import torch

    with torch.no_grad():
        actions = loaded.deterministic_action(torch.from_numpy(obs), torch.from_numpy(mask))
    assert actions.shape == (5,)
