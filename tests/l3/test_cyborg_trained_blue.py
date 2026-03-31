"""L3 test: run a CybORG-trained CleanRL PPO policy through the differential harness.

Loads a PyTorch checkpoint trained in CybORG, runs inference using CybORG's
own observations and action masks (via BlueFlatWrapper), translates the chosen
CybORG action indices to JAX action indices, and feeds them to both environments
through the harness.

This tests the transfer direction: policy trained in CybORG, evaluated in both.

Usage:
    CYBORG_CHECKPOINT=/path/to/model_large_v7-singh-20m.pt uv run pytest tests/l3/test_cyborg_trained_blue.py -v -x -n auto
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from torch.distributions import Categorical

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.translate import cyborg_blue_to_jax
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

# --- CybORG PPO Agent (matches train_cleanrl_ppo.py) ---


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, act_dim)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, x, action_mask=None, action=None):
        features = self.features(x)
        logits = self.actor(features)
        if action_mask is not None:
            logits = logits + (action_mask.float() - 1.0) * 1e8
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features).squeeze(-1)

    def get_deterministic_action(self, x, action_mask):
        features = self.features(x)
        logits = self.actor(features)
        logits = logits + (action_mask.float() - 1.0) * 1e8
        return logits.argmax(dim=-1)


# --- Checkpoint discovery ---

_DEFAULT_CHECKPOINT_PATHS = [
    Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")) / "cleanrl_ppo" / "model_large_v7-singh-20m.pt",
    Path.home() / "src" / "cyber" / "jaxborg-exp" / "cleanrl_ppo" / "model_large_v7-singh-20m.pt",
]


_DEFAULT_CHECKPOINT_DIR = [
    Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")) / "cleanrl_ppo",
    Path.home() / "src" / "cyber" / "jaxborg-exp" / "cleanrl_ppo",
]


def _get_checkpoint_dir() -> Path | None:
    env_path = os.environ.get("CYBORG_CHECKPOINT_DIR")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None
    # Also accept a single file via CYBORG_CHECKPOINT
    env_file = os.environ.get("CYBORG_CHECKPOINT")
    if env_file:
        p = Path(env_file)
        if p.exists():
            return p.parent
    for p in _DEFAULT_CHECKPOINT_DIR:
        if p.exists():
            return p
    return None


def _load_cyborg_policy_large(checkpoint_dir: Path, tag="v7-singh-20m"):
    """Load the large CybORG-trained PyTorch policy (padded action/obs space)."""
    path = checkpoint_dir / f"model_large_{tag}.pt"
    if not path.exists():
        return None
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    obs_dim = state_dict["features.0.weight"].shape[1]
    act_dim = state_dict["actor.weight"].shape[0]
    agent = PPOAgent(obs_dim, act_dim)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, obs_dim, act_dim


# --- Test ---

CHECKPOINT_DIR = _get_checkpoint_dir()
SEEDS = list(range(10))
STEPS = 500

skip_reason = None
if CHECKPOINT_DIR is None:
    skip_reason = "No CybORG checkpoint dir found. Set CYBORG_CHECKPOINT_DIR=/path/to/cleanrl_ppo/"


def _run_cyborg_policy_episode(seed, max_steps, checkpoint_dir, strict=False):
    """Run episode with CybORG-trained policy driving blue via CybORG obs/masks."""
    result = _load_cyborg_policy_large(checkpoint_dir)
    if result is None:
        pytest.skip(f"No model_large .pt found in {checkpoint_dir}")
    policy_model, expected_obs_dim, expected_act_dim = result

    harness = CC4DifferentialHarness(
        seed=seed,
        max_steps=max_steps,
        blue_cls=SleepAgent,  # placeholder — we override actions
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
        strict_random_sync=False,
        check_obs=True,
        check_masks=True,
    )
    harness.reset()

    # We need the BlueFlatWrapper to get CybORG obs/masks
    wrapper = harness._blue_wrapper
    if wrapper is None:
        pytest.skip("BlueFlatWrapper not available (check_obs/check_masks must be True)")

    # Get wrapper's action-to-CybORG-object mapping for translation
    from tests.differential.blue_mask_projection import refresh_blue_wrapper_action_space

    for t in range(max_steps):
        # Get CybORG observations and action masks, pick actions, translate
        actions = {}
        for b in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{b}"

            # CybORG observation (flat vector from wrapper)
            cyborg_obs_dict = harness.cyborg_env.get_observation(agent_name)
            cyborg_obs = wrapper.observation_change(agent_name, cyborg_obs_dict)
            obs_tensor = torch.tensor(cyborg_obs, dtype=torch.float32).unsqueeze(0)

            # CybORG action mask — pad to expected_act_dim if smaller
            raw_mask = np.array(wrapper.action_mask(agent_name), dtype=np.float32)
            if len(raw_mask) < expected_act_dim:
                mask = np.zeros(expected_act_dim, dtype=np.float32)
                mask[: len(raw_mask)] = raw_mask
            else:
                mask = raw_mask
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            # Pad obs to expected_obs_dim if smaller
            raw_obs = np.array(cyborg_obs, dtype=np.float32)
            if len(raw_obs) < expected_obs_dim:
                obs_padded = np.zeros(expected_obs_dim, dtype=np.float32)
                obs_padded[: len(raw_obs)] = raw_obs
                obs_tensor = torch.tensor(obs_padded, dtype=torch.float32).unsqueeze(0)

            # Policy inference (deterministic)
            with torch.no_grad():
                cyborg_action_idx = int(policy_model.get_deterministic_action(obs_tensor, mask_tensor).item())

            # Translate: wrapper integer → CybORG action object → JAX index
            cyborg_action = wrapper.actions(agent_name)[cyborg_action_idx]
            jax_action_idx = cyborg_blue_to_jax(cyborg_action, agent_name, harness.mappings, const=harness.jax_const)
            actions[b] = jax_action_idx

        # Step both envs with the same blue actions
        result = harness.full_step(blue_actions=actions)
        # Refresh wrapper action space for next step
        refresh_blue_wrapper_action_space(wrapper)

        if strict:
            error_diffs = result.diffs
        else:
            error_diffs = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
        if error_diffs:
            d = error_diffs[0]
            detail = format_diffs(result.diffs)
            pytest.fail(
                f"Mismatch at seed={seed}, step={t}: "
                f"{d.field_name} [{d.host_or_agent}] "
                f"cyborg={d.cyborg_value} jax={d.jax_value}\n"
                f"Blue actions (JAX indices): {actions}\n"
                f"All diffs:\n{detail}"
            )


@pytest.mark.skipif(skip_reason is not None, reason=skip_reason or "")
class TestCyborgTrainedBlue:
    """Run CybORG-trained CleanRL PPO policy through differential harness.

    Tests the transfer direction: policy trained in CybORG, actions translated
    to JAX action space, both environments stepped identically.
    """

    @pytest.mark.parametrize("seed", SEEDS, ids=[f"seed_{s:02d}" for s in SEEDS])
    def test_episode(self, seed):
        _run_cyborg_policy_episode(seed=seed, max_steps=STEPS, checkpoint_dir=CHECKPOINT_DIR)
