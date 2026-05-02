"""L3 test: run a JAXborg-trained IPPO policy through the differential harness.

A trained policy exercises realistic blue action sequences (monitor → detect →
remove → restore) that random blue never would. This catches bugs in specific
action combinations that matter for training and transfer.

The policy observes JAX state (via get_blue_obs), computes action masks, runs
inference, and feeds the chosen actions to BOTH CybORG and JAX via the harness.
Any state divergence is a real bug.

Usage:
    JAXBORG_POLICY_CHECKPOINT=/path/to/model_<tag>.pkl \
        uv run pytest -o addopts="" -m "" tests/l3/test_trained_blue_policy.py -v -x -n auto
"""

import os
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_runner import load_jax_checkpoint
from jaxborg.observations import get_blue_obs
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import format_diffs

pytestmark = pytest.mark.slow


# --- Policy loading ---


@lru_cache(maxsize=4)
def _make_inference_fn(checkpoint_path: str):
    """Build a JIT-compiled deterministic inference function from a recipe checkpoint."""
    policy, params, _recipe = load_jax_checkpoint(Path(checkpoint_path))

    def _fwd(o, m):
        pi, _ = policy.apply(params, o, m)
        return pi.logits

    @jax.jit
    def batched_step(obs_stack, mask_stack):
        logits = jax.vmap(_fwd)(obs_stack, mask_stack)
        return jnp.argmax(logits, axis=-1)

    return batched_step


# --- Test ---

CHECKPOINT_ENV = "JAXBORG_POLICY_CHECKPOINT"
CHECKPOINT_PATH = Path(os.environ[CHECKPOINT_ENV]).expanduser() if os.environ.get(CHECKPOINT_ENV) else None
SEEDS = list(range(100))
STEPS = 500

skip_reason = None
if CHECKPOINT_PATH is None:
    skip_reason = f"Set {CHECKPOINT_ENV}=/path/to/model_<tag>.pkl"
elif not CHECKPOINT_PATH.exists():
    skip_reason = f"{CHECKPOINT_ENV} does not exist: {CHECKPOINT_PATH}"


def _run_trained_episode(seed, max_steps, checkpoint_path):
    """Run a single episode with the trained policy driving blue actions."""
    inference_fn = _make_inference_fn(str(checkpoint_path))

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

    for t in range(max_steps):
        # Get obs and masks from JAX state
        obs_stack = jnp.stack([get_blue_obs(harness.jax_state, harness.jax_const, i) for i in range(NUM_BLUE_AGENTS)])
        mask_stack = jnp.stack(
            [compute_blue_action_mask(harness.jax_const, i, harness.jax_state) for i in range(NUM_BLUE_AGENTS)]
        )

        # Policy inference (deterministic — argmax)
        actions_arr = inference_fn(obs_stack, mask_stack)
        actions = {i: int(actions_arr[i]) for i in range(NUM_BLUE_AGENTS)}

        # Step both envs with the same blue actions
        result = harness.full_step(blue_actions=actions)

        error_diffs = result.diffs
        if error_diffs:
            d = error_diffs[0]
            detail = format_diffs(result.diffs)
            pytest.fail(
                f"Mismatch at seed={seed}, step={t}: "
                f"{d.field_name} [{d.host_or_agent}] "
                f"cyborg={d.cyborg_value} jax={d.jax_value}\n"
                f"Blue actions: {actions}\n"
                f"All diffs:\n{detail}"
            )


@pytest.mark.skipif(skip_reason is not None, reason=skip_reason or "")
class TestTrainedBluePolicy:
    """Run trained IPPO policy through strict differential harness.

    The policy makes correlated action sequences that exercise specific
    blue action combinations (analyse→remove→restore, decoy placement,
    targeted monitoring). This catches bugs that random blue misses.
    """

    @pytest.mark.parametrize("seed", SEEDS, ids=[f"seed_{s:02d}" for s in SEEDS])
    def test_episode(self, seed):
        _run_trained_episode(seed=seed, max_steps=STEPS, checkpoint_path=CHECKPOINT_PATH)
