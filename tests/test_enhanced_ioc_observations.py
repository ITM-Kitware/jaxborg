"""Flag-gated IOC evidence, one-tick bounded delivery, native parity and old-model loading."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.blue_ioc import _host_slots, initialize_blue_ioc
from jaxborg.blue_observation_contract import blue_obs_size, enhanced_obs_version, recipe_blue_obs_size
from jaxborg.checkpoint import read_sidecar, write_sidecar
from jaxborg.evaluation.enhanced_blue_wrapper import EnhancedBlueFlatWrapper, EnhancedEnterpriseMAE
from jaxborg.observations import get_blue_obs, update_blue_observation_memory
from jaxborg.recipe import load, train_variant
from jaxborg.state import create_initial_state


def host_slot(const, agent, n=0):
    _, hosts, valid = _host_slots(const, agent)
    slot = int(np.flatnonzero(np.asarray(valid).reshape(-1))[n])
    return int(np.asarray(hosts).reshape(-1)[slot]), slot


def extras(state, const, agent):
    return np.asarray(get_blue_obs(state, const, agent))[210:].reshape(5, 48)


@pytest.fixture
def observed(jax_const):
    return initialize_blue_ioc(create_initial_state()), jax_const.replace(cage4_enhanced_obs=True)


def test_existing_flag_selects_new_layout_and_sidecars_preserve_old_versions(tmp_path):
    recipe = load("cotraining_lstm")
    assert recipe_blue_obs_size(recipe) == 450
    assert train_variant(recipe).blue_observation_version == 2
    path = tmp_path / "recipe_new.yaml"
    write_sidecar(path, recipe, seed=0, total_steps=1, backend="jax")
    saved = read_sidecar(tmp_path / "model_new.safetensors")
    assert saved["run"]["blue_observation_version"] == 2
    assert recipe_blue_obs_size(saved) == 450
    del saved["run"]["blue_observation_version"]
    assert enhanced_obs_version(saved) == 1
    assert recipe_blue_obs_size(saved) == 402
    assert train_variant(saved).blue_observation_version == 1
    recipe["cage4_enhanced_obs"] = False
    assert recipe_blue_obs_size(recipe) == 210
    assert blue_obs_size(True) == 450
    assert blue_obs_size(True, 1) == 402
    # Inactive enhanced metadata must not change the stock game-variant identity.
    saved["cage4_enhanced_obs"] = False
    assert train_variant(saved) == train_variant(recipe)


def test_observed_files_persist_through_negative_analysis_and_clear_on_recovery(observed):
    state, const = observed
    host, slot = host_slot(const, 0)
    state = state.replace(
        host_compromised=state.host_compromised.at[host].set(2),
        host_file_artifact=state.host_file_artifact.at[host].set(2),
    )
    state = update_blue_observation_memory(state, const)
    assert not extras(state, const, 0)[1:].any()
    state = state.replace(blue_file_evidence=state.blue_file_evidence.at[host].set(2))
    state = update_blue_observation_memory(state, const)
    assert extras(state, const, 0)[1, slot] == 1
    assert extras(state, const, 0)[4, slot] == np.float32(1 / 3)
    state = state.replace(blue_file_evidence=jnp.zeros_like(state.blue_file_evidence))
    state = update_blue_observation_memory(state, const)
    assert extras(state, const, 0)[1, slot] == 1
    assert not extras(state, const, 1)[1:].any()
    state = state.replace(blue_recovered_this_step=state.blue_recovered_this_step.at[host].set(True))
    state = update_blue_observation_memory(state, const)
    assert not extras(state, const, 0)[1:, slot].any()


def test_remote_ioc_is_source_specific_bounded_delayed_and_persistent(observed):
    state, const = observed
    target, _ = host_slot(const, 0)
    origins = [host_slot(const, a) for a in (1, 2)]
    for event_slot, (host, _) in enumerate(origins):
        state = state.replace(blue_old_decoy_sources=state.blue_old_decoy_sources.at[target, event_slot].set(host))
    update = jax.jit(update_blue_observation_memory)
    state = update(state, const)
    assert int(state.blue_ioc_memory["delivered"].sum()) == 1
    assert int(state.blue_ioc_memory["pending"].sum()) == 1
    assert not np.asarray(state.blue_ioc_codes).any()
    # Expire events; the unsent IOC must remain queued.
    state = state.replace(blue_old_decoy_sources=jnp.full_like(state.blue_old_decoy_sources, -1))
    state = update(state, const)
    assert np.count_nonzero(np.asarray(state.blue_ioc_codes)) == 1
    state = update(state, const)
    for a, (_, slot) in zip((1, 2), origins):
        assert extras(state, const, a)[4, slot] == 1
    assert not extras(state, const, 0)[4].any()
    assert not np.asarray(state.blue_ioc_memory["pending"]).any()
    assert not np.asarray(state.blue_ioc_memory["delivered"]).any()
    # Reading observations must neither drain queues nor advance delivery.
    before = get_blue_obs(state, const, 1)
    np.testing.assert_array_equal(before, get_blue_obs(state, const, 1))


def test_local_decoy_ioc_and_file_priority_match_pretrained_contract(observed):
    state, const = observed
    source, slot = host_slot(const, 0)
    target, _ = host_slot(const, 0, 1)
    state = state.replace(blue_old_decoy_sources=state.blue_old_decoy_sources.at[target, 0].set(source))
    state = update_blue_observation_memory(state, const)
    # Detection may occur later in the subnet traversal than the source slot;
    # the released first-IOC traversal exposes it on the following observation.
    state = update_blue_observation_memory(state, const)
    assert extras(state, const, 0)[4, slot] == 1
    assert not np.asarray(state.blue_ioc_memory["delivered"]).any()
    state = state.replace(blue_file_evidence=state.blue_file_evidence.at[source].set(1))
    state = update_blue_observation_memory(state, const)
    assert extras(state, const, 0)[4, slot] == np.float32(2 / 3)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_reset_batch_and_auto_reset_keep_optional_memory_contract(observed, tmp_path, joint, enabled):
    from jaxborg.actions.encoding import BLUE_SLEEP
    from jaxborg.evaluation.jax_env_factory import make_jax_env, make_joint_jax_env
    from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
    from jaxborg.scenarios.cc4.topology import save_topology

    _, const = observed
    path = tmp_path / "topology.npz"
    save_topology(const, path)
    variant = replace(CC4_STOCK, cage4_enhanced_obs=enabled, num_steps=2)
    env = (make_joint_jax_env if joint else make_jax_env)(variant, topology_path=path)
    obs, state = env.reset(jax.random.PRNGKey(3))
    width = 450 if enabled else 210
    assert obs["blue_0"].shape == (width,)
    assert (state.state.blue_ioc_memory is not None) is enabled
    assert (state.state.blue_decoy_sources is not None) is enabled
    batched, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(4), 2))
    assert batched["blue_0"].shape == (2, width)
    actions = {a: jnp.int32(BLUE_SLEEP if a.startswith("blue") else 0) for a in env.agents}
    next_obs, next_state, *_ = jax.eval_shape(env.step, jax.random.PRNGKey(5), state, actions)
    assert next_obs["blue_0"].shape == (width,)
    assert (next_state.state.blue_ioc_memory is not None) is enabled
    if enabled:
        poisoned = state.state.replace(
            blue_ioc_memory=jax.tree.map(jnp.ones_like, state.state.blue_ioc_memory),
            blue_ioc_codes=jnp.ones_like(state.state.blue_ioc_codes),
        )
        owner = env._env if hasattr(env, "_env") else env
        fresh = owner._reset_state(state.replace(state=poisoned), jax.random.PRNGKey(8))
        assert not any(np.asarray(x).any() for x in jax.tree.leaves(fresh.state.blue_ioc_memory))
        assert not np.asarray(fresh.state.blue_ioc_codes).any()
        assert np.all(np.asarray(fresh.state.blue_decoy_sources) == -1)


def native(wrapper=EnhancedBlueFlatWrapper):
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=SleepAgent, red_agent_class=SleepAgent, steps=5
    )
    env = wrapper(CybORG(sg, "sim", seed=42), pad_spaces=True)
    obs, _ = env.reset(seed=42)
    return env, obs


@pytest.mark.parametrize("wrapper", [EnhancedBlueFlatWrapper, EnhancedEnterpriseMAE])
def test_native_new_default_shapes_and_episode_reset(wrapper):
    env, obs = native(wrapper)
    assert all(o.shape == (450,) for o in obs.values())
    assert all(env.observation_space(a).contains(o) for a, o in obs.items())
    obs, *_ = env.step({})
    assert all(o.shape == (450,) for o in obs.values())
    env._ioc_pending["blue_agent_0"].add(next(iter(env._ioc_owners)))
    env._evidence["blue_agent_0"][1:] = 1
    obs, _ = env.reset(seed=42)
    assert all(not o[258:].any() for o in obs.values())
    assert not any(env._ioc_pending.values())


def test_native_and_jax_remote_delivery_and_file_memory_match():
    from CybORG.Shared.Enums import TernaryEnum

    from jaxborg.parity.translate import build_mappings_from_cyborg
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    env, _ = native()
    const = build_const_from_cyborg(env.env).replace(cage4_enhanced_obs=True)
    state = initialize_blue_ioc(create_initial_state())
    mappings = build_mappings_from_cyborg(env.env)
    target_idx, _ = host_slot(const, 0)
    source_idx, slot = host_slot(const, 1)
    inverse = {v: k for k, v in mappings.hostname_to_idx.items()}
    target, source = inverse[target_idx], inverse[source_idx]
    native_state = env.env.environment_controller.state
    ip = next(ip for ip, name in native_state.ip_addresses.items() if name == source)
    # A visible connection to an owned decoy, with no Red session/compromise.
    decoy = SimpleNamespace(decoy_type=SimpleNamespace(name="EXPLOIT"), open_ports=[{"local_port": 9876}])
    native_state.hosts[target].processes.append(decoy)
    event = SimpleNamespace(local_port=9876, remote_address=ip)
    native_state.hosts[target].events.old_network_connections.append(event)
    state = state.replace(
        blue_old_decoy_sources=state.blue_old_decoy_sources.at[target_idx, 0].set(source_idx),
        old_host_activity_detected=state.old_host_activity_detected.at[target_idx].set(True),
    )
    state = update_blue_observation_memory(state, const)
    for agent in range(5):
        obs = env.observation_change(f"blue_agent_{agent}", {})
        np.testing.assert_array_equal(obs, get_blue_obs(state, const, agent))
    assert env._ioc_pending["blue_agent_0"] == {source}
    native_state.hosts[target].events.old_network_connections.clear()
    env.env.environment_controller.step_count += 1
    state = state.replace(
        blue_old_decoy_sources=jnp.full_like(state.blue_old_decoy_sources, -1),
        old_host_activity_detected=jnp.zeros_like(state.old_host_activity_detected),
    )
    state = update_blue_observation_memory(state, const)
    for agent in range(5):
        # Reverse iteration demonstrates that delivery is independent of conversion order.
        b = 4 - agent
        obs = env.observation_change(f"blue_agent_{b}", {})
        np.testing.assert_array_equal(obs, get_blue_obs(state, const, b))
    assert obs.shape == (450,)
    assert extras(state, const, 1)[4, slot] == 1
    action = next(a for a in env.actions("blue_agent_1") if type(a).__name__ == "Analyse" and a.hostname == source)
    obs = env.observation_change(
        "blue_agent_1",
        {
            "action": action,
            "success": TernaryEnum.TRUE,
            source: {"Files": [{"File Name": "escalate.sh", "Density": 0.9, "Signed": False}]},
        },
    )
    state = state.replace(blue_file_evidence=state.blue_file_evidence.at[source_idx].set(2))
    state = update_blue_observation_memory(state, const)
    np.testing.assert_array_equal(obs, get_blue_obs(state, const, 1))
    obs = env.observation_change("blue_agent_1", {"action": action, "success": TernaryEnum.TRUE, source: {"Files": []}})
    state = state.replace(blue_file_evidence=jnp.zeros_like(state.blue_file_evidence))
    state = update_blue_observation_memory(state, const)
    np.testing.assert_array_equal(obs, get_blue_obs(state, const, 1))


@pytest.mark.parametrize("architecture", ["recurrent", "mappo", "recurrent_mappo"])
def test_new_actor_and_centralized_critic_inference(observed, architecture, tmp_path):
    from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
    from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle
    from jaxborg.critic_observations import get_critic_obs
    from jaxborg.evaluation.jax_runner import load_jax_checkpoint
    from jaxborg.policies import init_policy_params, initial_carry, policy_from_arch, policy_step
    from jaxborg.recipe import team_recipe

    state, const = observed
    recipe = load("cotraining_lstm")
    recipe["arch"] = dict(name=architecture, hidden_dim=8, hidden_layers=1, activation="tanh")
    if architecture != "mappo":
        recipe["arch"]["cell"] = "lstm"
    if architecture != "recurrent":
        recipe["arch"]["critic_input"] = "joint_observations"
    arch = team_recipe(recipe, "blue")["arch"]
    module = policy_from_arch(arch, action_dim=BLUE_ALLOW_TRAFFIC_END)
    params = init_policy_params(module, jax.random.PRNGKey(0), 450)
    path = tmp_path / "model_v2.safetensors"
    save_jax_bundle(path, {"blue": PolicyBundleEntry(params, "blue", 450, BLUE_ALLOW_TRAFFIC_END, arch)})
    write_sidecar(tmp_path / "recipe_v2.yaml", recipe, seed=0, total_steps=1, backend="jax")
    module, params, saved = load_jax_checkpoint(path)
    assert recipe_blue_obs_size(saved) == 450
    obs = jnp.stack([get_blue_obs(state, const, a) for a in range(5)])
    kwargs = {} if architecture == "recurrent" else {"critic_obs": get_critic_obs(state, const, "joint_observations")}
    pi, value, _ = policy_step(
        module, params, obs, jnp.ones((5, BLUE_ALLOW_TRAFFIC_END), dtype=bool), carry=initial_carry(module, 5), **kwargs
    )
    assert np.isfinite(pi.logits).all()
    assert np.isfinite(value).all()


def test_native_joint_training_advances_the_same_delivery_clock():
    from jaxborg.cyborg_joint import BLUE_AGENT_IDS, RED_AGENT_IDS, CyborgJointAdapter
    from jaxborg.learned_red import RED_POLICY_SLEEP
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    env = CyborgJointAdapter(GameVariant(name="v2", cage4_enhanced_obs=True, num_steps=3), seed=42)
    obs, _ = env.reset(ep_seed=42)
    assert all(obs[a].shape == (450,) for a in BLUE_AGENT_IDS)
    source = next(h for h, (a, _) in env.blue_wrapper._ioc_owners.items() if a == "blue_agent_1")
    _, slot = env.blue_wrapper._ioc_owners[source]
    env.blue_wrapper._ioc_pending["blue_agent_0"].add(source)
    actions = {a: env._blue_sleep_index(a) for a in BLUE_AGENT_IDS}
    actions.update({a: RED_POLICY_SLEEP for a in RED_AGENT_IDS})
    obs, *_ = env.step(actions)
    assert obs["blue_agent_1"][402 + slot] == 1
    assert env.blue_wrapper._ioc_tick == 1
    assert not env.blue_wrapper._ioc_pending["blue_agent_0"]
