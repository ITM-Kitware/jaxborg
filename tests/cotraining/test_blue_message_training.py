"""PPO must train Blue's messages even when every action is busy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from test_ippo_jax_recurrent import _config as blue_config
from test_ippo_jax_recurrent import _one_update, _TinyBlueEnv
from test_jax_joint_trainer import _config as joint_config
from test_jax_joint_trainer import _TinyJointEnv

from jaxborg.policies import policy_from_arch
from scripts.train.algorithms import ippo_cyborg, ippo_jax, ippo_jax_joint


def changed(a, b):
    return any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


class MessageJointEnv(_TinyJointEnv):
    def reset(self, key):
        obs, state = super().reset(key)
        return obs, state.replace(state=state.state.replace(blue_pending_ticks=jnp.ones(1, dtype=jnp.int32)))

    def get_avail_actions(self, state):
        masks = super().get_avail_actions(state)
        masks["blue_0"] = jnp.array([True, False, False])
        return masks

    def step(self, key, state, actions):
        assert "red_messages" not in actions
        assert actions["blue_messages"].shape == (1, 8)
        obs, state, rewards, dones, info = super().step(key, state, actions)
        reward = actions["blue_messages"].sum() + state.state.time * 0.1
        return obs, state, {"blue_0": reward, "red_0": -reward}, dones, {**info, "reward_ria": reward}


@pytest.mark.parametrize("name", ["shared", "recurrent", "mappo", "recurrent_mappo"])
def test_joint_ppo_updates_busy_blue_message_head_and_keeps_red_frozen(monkeypatch, name):
    blue = policy_from_arch(dict(name=name, hidden_dim=8, hidden_layers=1, message_dim=8), action_dim=3)
    red = policy_from_arch(dict(name="shared", hidden_dim=8, hidden_layers=1), action_dim=5)
    env = MessageJointEnv(blue_obs_dim=4, red_obs_dim=6, blue_actions=3, red_actions=5)
    if name in ("mappo", "recurrent_mappo"):
        env.get_critic_obs = lambda state, mode, team: jnp.zeros((1, blue.critic_obs_dim))
    monkeypatch.setattr(ippo_jax_joint, "make_joint_jax_env", lambda *a, **kw: env)
    cfg = joint_config()
    cfg.update(NUM_ENVS=2, NUM_STEPS=4, TOTAL_TIMESTEPS=8, ENT_COEF=0.0, MSG_ENT_COEF=0.0)
    _, obs, state, init, update = ippo_jax_joint.make_joint_train(
        {"blue": dict(cfg), "red": dict(cfg)}, {"blue": blue, "red": red}, trainable_teams=("blue",)
    )
    params = init(jax.random.PRNGKey(3))
    before, before_red = params["blue"].params, params["red"].params
    norm = {t: ippo_jax_joint.initial_reward_norm_state(2) for t in ("blue", "red")}
    after, _, _, _, _, metrics = update(params, state, obs, jax.random.PRNGKey(7), norm)
    branch = "actor_head" if "mappo" in name else "actor_message"
    old, new = before["params"][branch], after["blue"].params["params"][branch]
    if "mappo" in name:
        old, new = old["message"], new["message"]
    assert changed(old, new)
    assert not changed(before_red, after["red"].params)
    assert float(metrics["blue"]["actor_fraction"]) == 1
    assert np.isfinite(float(metrics["blue"]["msg_entropy"]))
    assert "msg_bit_mean_7" in metrics["blue"]


@pytest.mark.parametrize("name", ["shared", "recurrent"])
def test_blue_only_jax_rollout_and_update_include_messages(monkeypatch, name):
    class MessageBlueEnv(_TinyBlueEnv):
        def step(self, key, state, actions):
            assert actions["blue_messages"].shape == (2, 8)
            return super().step(key, state, actions)

    monkeypatch.setattr(ippo_jax, "make_jax_env", lambda *a, **kw: MessageBlueEnv(4))
    monkeypatch.setattr(ippo_jax, "compute_blue_action_mask", lambda *a, **kw: jnp.ones(3, dtype=bool))
    network = policy_from_arch(dict(name=name, hidden_dim=8, hidden_layers=1, message_dim=8), action_dim=3)
    before, after, metrics = _one_update(blue_config(2, 4, 1), network)
    assert changed(before["params"]["actor_message"], after.params["params"]["actor_message"])
    assert metrics["msg_entropy"] > 0


@pytest.mark.parametrize("name", ["shared", "separate"])
def test_torch_joint_ppo_trains_busy_blue_messages(name):
    torch.manual_seed(9)
    arch = dict(name=name, hidden_dim=8, hidden_layers=1, message_dim=8)
    agent = policy_from_arch(arch, backend="cyborg", obs_dim=210, action_dim=242)
    cfg = dict(
        lr=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        msg_ent_coef=0.0,
        clip_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=1.0,
        anneal_lr=False,
        num_epochs=1,
        num_minibatches=1,
        norm_rewards=False,
    )
    runtime = ippo_cyborg.TorchTeamRuntime(
        "blue", cfg, agent, arch, True, {}, 1, 4, optimizer=torch.optim.Adam(agent.parameters(), lr=0.01)
    )
    obs = [{a: np.ones(210, dtype=np.float32) for a in runtime.agent_ids}]
    info = [
        {a: dict(action_mask=np.arange(242) == 0, actor_active=False, critic_active=True) for a in runtime.agent_ids}
    ]
    before = agent.message_head.weight.detach().clone()
    for step in range(4):
        actions = runtime.select_actions(step, obs, info)
        np.testing.assert_array_equal(actions, 0)
        assert runtime.outgoing_messages.shape == (1, 5, 8)
        assert runtime.rollout["actor_active"][step].all()
        runtime.rollout["rewards"][step] = float(runtime.outgoing_messages.sum())
    runtime.finish_rollout(obs, info)
    metrics = ippo_cyborg._ppo_update(runtime, 1, 1)
    assert not torch.equal(before, agent.message_head.weight)
    assert metrics["msg_entropy"] > 0


@pytest.mark.parametrize('enabled', [False, True])
def test_cyborg_global_recipe_flag_controls_both_runtime_and_legacy_training(monkeypatch, tmp_path, enabled):
    import json
    from types import SimpleNamespace
    from jaxborg.recipe import load, project_cleanrl
    from jaxborg.checkpoint import load_torch_bundle
    from jaxborg.evaluation import post_training

    recipe = load('cotraining')
    recipe.update(use_messages=enabled, cage4_enhanced_obs=False)
    recipe['arch'] = dict(name='shared', hidden_dim=8, hidden_layers=1)
    recipe['train']['team_overrides'] = {}
    recipe['eval'] = {}
    recipe['mlflow'] = {'checkpoint_eval': {'every_steps': 0}}
    cfg = project_cleanrl(recipe)
    cfg.update(num_envs=1, rollout_length=4, total_timesteps=4, num_rollouts_per_update=1,
               num_minibatches=1, num_epochs=1, norm_rewards=False)
    runtimes = ippo_cyborg._make_joint_runtimes(recipe, cfg, seed=4, num_envs=1, num_steps=4)
    assert runtimes['blue'].agent.message_dim == (8 if enabled else 0)
    assert runtimes['red'].agent.message_dim == 0

    names = ippo_cyborg.AGENT_IDS
    obs = [{a: np.ones(210, dtype=np.float32) for a in names}]
    info = [{a: dict(action_mask=np.arange(242) == 0, actor_active=False) for a in names}]
    class Envs:
        def __init__(self, *args, **kwargs):
            self.steps = 0
        def reset(self):
            return obs, info
        def step(self, actions):
            self.steps += 1
            assert ('blue_messages' in actions[0]) == enabled
            reward = float(actions[0]['blue_messages'].sum()) if enabled else float(self.steps)
            return obs, [dict.fromkeys(names, reward)], [self.steps == 4], info
        def close(self):
            pass
    monkeypatch.setattr(ippo_cyborg, 'ParallelEnvs', Envs)
    monkeypatch.setattr(ippo_cyborg, 'EXP_DIR', tmp_path)
    monkeypatch.setattr(ippo_cyborg, 'start_run', lambda *a, **kw: SimpleNamespace(info=SimpleNamespace(run_id='local-test')))
    monkeypatch.setattr(ippo_cyborg, 'MlflowCheckpointEvaluator', lambda *a: SimpleNamespace(due=lambda *a, **kw: False))
    monkeypatch.setattr(post_training, 'run_configured_evaluations_after_training', lambda *a, **kw: None)
    for method in ('log_metrics', 'log_artifact', 'end_run'):
        monkeypatch.setattr(ippo_cyborg.mlflow, method, lambda *a, **kw: None)
    recipe['train']['teams'] = 'blue'
    ippo_cyborg.train_legacy(SimpleNamespace(seed=4, tag='messages'), recipe, cfg)
    saved = tmp_path / 'ippo_cyborg/messages'
    entry = load_torch_bundle(saved / 'model_messages.pt').policies['blue']
    assert entry.arch.get('message_dim', 0) == (8 if enabled else 0)
    assert ('message_head.weight' in entry.weights) == enabled
    row = json.loads((saved / 'metrics.jsonl').read_text().splitlines()[0])
    assert ('team.blue.msg_entropy' in row) == enabled
