"""L3 test: run a JAXborg-trained IPPO policy through the differential harness.

A trained policy exercises realistic blue action sequences (monitor → detect →
remove → restore) that random blue never would. This catches bugs in specific
action combinations that matter for training and transfer.

The policy observes JAX state (via get_blue_obs), computes action masks, runs
inference, and feeds the chosen actions to BOTH CybORG and JAX via the harness.
Any state divergence is a real bug.

Usage:
    BLUE_CHECKPOINT=/path/to/checkpoint.pkl uv run pytest tests/l3/test_trained_blue_policy.py -v -x -n auto
"""

import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.observations import get_blue_obs
from jaxborg.policy import ActorCritic, LegacyActor
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

pytestmark = pytest.mark.skip(reason="checkpoint trained with old 776-action space")


# --- Checkpoint discovery ---

_DEFAULT_CHECKPOINT_DIRS = [
    Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")),
    Path.home() / "src" / "cyber" / "jaxborg-exp",
]


def _find_latest_checkpoint() -> Path | None:
    """Find the most recent .pkl checkpoint across experiment dirs."""
    best = None
    best_mtime = 0
    for exp_dir in _DEFAULT_CHECKPOINT_DIRS:
        if not exp_dir.exists():
            continue
        for pkl in exp_dir.rglob("checkpoint_*.pkl"):
            mtime = pkl.stat().st_mtime
            if mtime > best_mtime:
                best = pkl
                best_mtime = mtime
    return best


def _get_checkpoint_path() -> Path | None:
    """Get checkpoint from env var or auto-discover latest."""
    env_path = os.environ.get("BLUE_CHECKPOINT")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        return None
    return _find_latest_checkpoint()


# --- Policy loading ---


def _load_policy(checkpoint_path: Path):
    """Load policy and params from checkpoint. Returns (policy, params, kind)."""
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    nested_params = ckpt["params"].get("params", {})

    if "actor_head" in nested_params:
        policy = ActorCritic(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "current"

    if "Dense_0" in nested_params:
        policy = LegacyActor(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "legacy"

    raise ValueError(f"Unrecognized checkpoint: {sorted(nested_params.keys())}")


def _make_inference_fn(policy, params, policy_kind):
    """Build a JIT-compiled deterministic inference function."""
    if policy_kind == "current":

        def _fwd(o, m):
            return policy.apply(params, o, m, method=ActorCritic.actor).logits
    else:

        def _fwd(o, m):
            return policy.apply(params, o, m).logits

    @jax.jit
    def batched_step(obs_stack, mask_stack):
        logits = jax.vmap(_fwd)(obs_stack, mask_stack)
        return jnp.argmax(logits, axis=-1)

    return batched_step


# --- Test ---

CHECKPOINT_PATH = _get_checkpoint_path()
SEEDS = list(range(100))
STEPS = 500

skip_reason = None
if CHECKPOINT_PATH is None:
    skip_reason = "No checkpoint found. Set BLUE_CHECKPOINT=/path/to/checkpoint.pkl"


def _run_trained_episode(seed, max_steps, checkpoint_path, strict=False):
    """Run a single episode with the trained policy driving blue actions."""
    policy, params, kind = _load_policy(checkpoint_path)
    inference_fn = _make_inference_fn(policy, params, kind)

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

        # mismatch_mode="all" treats warnings as errors
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
                f"Blue actions: {actions}\n"
                f"All diffs:\n{detail}"
            )


@pytest.mark.skipif(skip_reason is not None, reason=skip_reason or "")
class TestTrainedBluePolicy:
    """Run trained IPPO policy through differential harness.

    The policy makes correlated action sequences that exercise specific
    blue action combinations (analyse→remove→restore, decoy placement,
    targeted monitoring). This catches bugs that random blue misses.
    """

    @pytest.mark.parametrize("seed", SEEDS, ids=[f"seed_{s:02d}" for s in SEEDS])
    def test_episode(self, seed):
        _run_trained_episode(seed=seed, max_steps=STEPS, checkpoint_path=CHECKPOINT_PATH)


@pytest.mark.skipif(skip_reason is not None, reason=skip_reason or "")
class TestTrainedBluePolicyStrict:
    """Same as above but warnings are treated as errors (mismatch_mode=all)."""

    @pytest.mark.parametrize("seed", list(range(10)), ids=[f"seed_{s:02d}" for s in range(10)])
    def test_episode_strict(self, seed):
        _run_trained_episode(seed=seed, max_steps=STEPS, checkpoint_path=CHECKPOINT_PATH, strict=True)
