from __future__ import annotations

import numpy as np
import torch

from jaxborg.actions.red_policy import (
    RED_POLICY_ACTION_DIM,
    RED_POLICY_EXPLOIT_START,
    RED_POLICY_STEALTH_SCAN_START,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, RED_OBS_SIZE
from jaxborg.cyborg_joint import POLICY_AGENT_IDS, CyborgJointAdapter
from jaxborg.recipe import load, project_cleanrl
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.train.algorithms.ippo_cyborg import _make_joint_runtimes, compute_torch_ppo_loss


def _sleep_actions(env: CyborgJointAdapter) -> dict[str, int]:
    return {agent: env._blue_sleep_index(agent) if agent.startswith("blue") else 0 for agent in POLICY_AGENT_IDS}


def test_joint_adapter_contract_and_local_red_observation():
    env = CyborgJointAdapter(CC4_STOCK, seed=7)
    observations, infos = env.reset(ep_seed=11)
    try:
        assert len(observations) == 11
        assert observations["blue_agent_0"].shape == (210,)
        assert observations["red_agent_0"].shape == (RED_OBS_SIZE,)
        assert infos["red_agent_0"]["action_mask"].shape == (RED_POLICY_ACTION_DIM,)
        assert infos["red_agent_1"]["actor_active"] is False
        assert np.flatnonzero(infos["red_agent_1"]["action_mask"]).tolist() == [0]

        # The host planes begin after phase/time/status/identity/subnets.
        plane_start = 3 + 3 + 6 + 9
        discovered = observations["red_agent_0"][plane_start : plane_start + GLOBAL_MAX_HOSTS]
        inactive_discovered = observations["red_agent_1"][plane_start : plane_start + GLOBAL_MAX_HOSTS]
        assert inactive_discovered.sum() == 0
        scan_mask = infos["red_agent_0"]["action_mask"][
            RED_POLICY_STEALTH_SCAN_START : RED_POLICY_STEALTH_SCAN_START + GLOBAL_MAX_HOSTS
        ]
        np.testing.assert_array_equal(discovered, scan_mask)

        # Adding another agent's session cannot alter Red 0's local view.
        before = observations["red_agent_0"].copy()
        state = env.raw_env.environment_controller.state
        state.sessions["red_agent_1"][999] = state.sessions["red_agent_0"][0]
        after = env.red_observation(0)
        del state.sessions["red_agent_1"][999]
        np.testing.assert_array_equal(before, after)
    finally:
        env.close()


def test_joint_adapter_busy_rows_and_zero_sum_reward():
    env = CyborgJointAdapter(CC4_STOCK, seed=3)
    _observations, infos = env.reset(ep_seed=4)
    try:
        scan_candidates = np.flatnonzero(
            infos["red_agent_0"]["action_mask"][
                RED_POLICY_STEALTH_SCAN_START : RED_POLICY_STEALTH_SCAN_START + GLOBAL_MAX_HOSTS
            ]
        )
        assert len(scan_candidates) > 0
        actions = _sleep_actions(env)
        actions["blue_agent_0"] = 0  # valid two-tick Analyse action
        actions["red_agent_0"] = RED_POLICY_STEALTH_SCAN_START + int(scan_candidates[0])
        _obs, rewards, _term, _trunc, next_info = env.step(actions)

        assert rewards["red_agent_0"] == -rewards["blue_agent_0"]
        assert next_info["red_agent_0"]["actor_active"] is False
        assert np.flatnonzero(next_info["red_agent_0"]["action_mask"]).tolist() == [0]
        assert next_info["blue_agent_0"]["actor_active"] is False
        assert np.flatnonzero(next_info["blue_agent_0"]["action_mask"]).tolist() == [
            env._blue_sleep_index("blue_agent_0")
        ]
    finally:
        env.close()


def test_generic_red_exploit_is_native_and_session_zero():
    env = CyborgJointAdapter(CC4_STOCK, seed=9)
    env.reset(ep_seed=10)
    try:
        action = env.red_action_to_cyborg("red_agent_0", RED_POLICY_EXPLOIT_START)
        assert type(action).__name__ == "ExploitRemoteService"
        assert action.agent == "red_agent_0"
        assert action.session == 0
        assert type(action.exploit_action_selector).__name__ == "DefaultExploitActionSelector"
    finally:
        env.close()


def test_busy_rows_are_actor_excluded_but_value_included():
    new_logprob = torch.tensor([0.2, -0.1, 4.0], requires_grad=True)
    entropy = torch.tensor([0.5, 0.4, 100.0], requires_grad=True)
    new_value = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    loss, _parts = compute_torch_ppo_loss(
        new_logprob=new_logprob,
        entropy=entropy,
        new_value=new_value,
        old_logprob=torch.zeros(3),
        advantages=torch.tensor([1.0, -1.0, 1_000.0]),
        returns=torch.tensor([0.0, 0.0, 3.0]),
        actor_active=torch.tensor([True, True, False]),
        critic_active=torch.tensor([True, True, True]),
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )
    loss.backward()
    assert new_logprob.grad[2].item() == 0.0
    assert entropy.grad[2].item() == 0.0
    assert new_value.grad[2].item() != 0.0


def test_inactive_rows_are_excluded_from_actor_and_value_losses():
    new_logprob = torch.tensor([0.2, 7.0], requires_grad=True)
    entropy = torch.tensor([0.5, 99.0], requires_grad=True)
    new_value = torch.tensor([0.0, 10.0], requires_grad=True)
    loss, _parts = compute_torch_ppo_loss(
        new_logprob=new_logprob,
        entropy=entropy,
        new_value=new_value,
        old_logprob=torch.zeros(2),
        advantages=torch.tensor([1.0, 1_000.0]),
        returns=torch.tensor([1.0, -1_000.0]),
        actor_active=torch.tensor([True, False]),
        critic_active=torch.tensor([True, False]),
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )
    loss.backward()
    assert new_logprob.grad[1].item() == 0.0
    assert entropy.grad[1].item() == 0.0
    assert new_value.grad[1].item() == 0.0


def test_trainable_torch_policy_sources_record_fresh_seed():
    recipe = load("cotraining")
    cfg = project_cleanrl(recipe)
    runtimes = _make_joint_runtimes(
        recipe,
        cfg,
        seed=123,
        num_envs=1,
        num_steps=1,
    )

    assert runtimes["blue"].source == {"kind": "fresh", "seed": 123}
    assert runtimes["red"].source == {"kind": "fresh", "seed": 123}
