"""Exercise retained-input analysis through observable probability changes."""

import importlib.util
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_RESTORE_START
from jaxborg.blue_observation_contract import CAGE4_ENHANCED_OBS_SIZE
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.policies.base import RecurrentPolicy
from jaxborg.policies.categorical import Categorical

SPEC = importlib.util.spec_from_file_location(
    "observation_value_sensitivity", Path(__file__).resolve().parents[1] / "studies/observation-value/sensitivity.py"
)
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


class EvidenceActor(nn.Module):
    def __call__(self, obs, avail_actions):
        logits = jnp.zeros((*obs.shape[:-1], 242))
        logits = logits.at[..., 0].set(obs[..., 60])
        logits = logits.at[..., BLUE_RESTORE_START].set(obs[..., BLUE_OBS_SIZE + 2 * 48])
        logits = logits.at[..., BLUE_RESTORE_START + 1].set(obs[..., BLUE_OBS_SIZE + 2 * 48 + 1])
        return Categorical(jnp.where(avail_actions, logits, -1e10)), jnp.zeros(obs.shape[:-1])


class HistoryActor(RecurrentPolicy):
    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))

    def __call__(self, carry, x, avail_actions=None, resets=None):
        logits = jnp.zeros((*x.shape[:-1], 242))
        logits = logits.at[..., BLUE_RESTORE_START].set(carry[1][None, :, 0])
        return carry, Categorical(jnp.where(avail_actions, logits, -1e10)), jnp.zeros(x.shape[:-1])


def retained(rows=1):
    observations = np.zeros((rows, 5, CAGE4_ENHANCED_OBS_SIZE), dtype=np.float32)
    observations[..., BLUE_OBS_SIZE : BLUE_OBS_SIZE + 2] = 1
    masks = np.zeros((rows, 5, 242), dtype=bool)
    masks[..., [0, BLUE_RESTORE_START, BLUE_RESTORE_START + 1]] = True
    genuine = np.zeros((rows, 5), dtype=bool)
    genuine[:, 0] = True
    return {
        "observations": observations,
        "masks": masks,
        "genuine": genuine,
        "phase": np.zeros(rows, dtype=np.int32),
        "topology_index": np.zeros(rows, dtype=np.int32),
        "n_subnets": np.ones((rows, 5), dtype=np.int32),
    }


def test_complete_host_evidence_moves_target_preference_without_changing_context():
    records = retained()
    obs = records["observations"]
    obs[..., 60:92] = 0.25
    obs[0, 0, 28] = 1
    obs[0, 0, 44] = 1
    for plane in (1, 2, 3, 4):
        obs[0, 0, BLUE_OBS_SIZE + plane * 48] = 1
    original = obs.copy()

    summary, measurements = sensitivity.analyze_inputs(EvidenceActor(), {}, records)

    assert summary["evidence_permutation"]["decisions"] == 1
    assert summary["evidence_permutation"]["tv_mean"] > 0
    assert summary["evidence_permutation"]["argmax_changed_mean"] == 1
    assert summary["evidence_permutation"]["source_Restore_probability_change_mean"] < 0
    assert summary["evidence_permutation"]["destination_Restore_probability_change_mean"] > 0
    np.testing.assert_array_equal(records["observations"], original)
    changed = measurements["evidence_observations"]
    np.testing.assert_array_equal(changed[..., :28], original[..., :28])
    np.testing.assert_array_equal(changed[..., 60:BLUE_OBS_SIZE], original[..., 60:BLUE_OBS_SIZE])
    np.testing.assert_array_equal(
        changed[..., BLUE_OBS_SIZE : BLUE_OBS_SIZE + 48], original[..., BLUE_OBS_SIZE : BLUE_OBS_SIZE + 48]
    )
    assert changed[0, 0, 28] == changed[0, 0, 44] == 0
    assert changed[0, 0, 29] == changed[0, 0, 45] == 1
    for plane in (1, 2, 3, 4):
        assert changed[0, 0, BLUE_OBS_SIZE + plane * 48] == 0
        assert changed[0, 0, BLUE_OBS_SIZE + plane * 48 + 1] == 1
    assert np.all(measurements["evidence_probabilities"][~records["masks"]] == 0)
    assert summary["carry_interpretation"] == "Feedforward actor; no recurrent carry."


def test_initial_carry_and_matched_actual_carries_are_reported_separately():
    records = retained(4)
    records["carry_c"] = np.zeros((4, 5, 1), dtype=np.float32)
    records["carry_h"] = np.broadcast_to(np.arange(1, 5, dtype=np.float32)[:, None, None], (4, 5, 1)).copy()
    records["topology_index"][2] = 1
    records["phase"][3] = 1
    records["observations"][3, :, 0] = 1
    actual_carry = records["carry_h"].copy()

    summary, measurements = sensitivity.analyze_inputs(HistoryActor(), {}, records)

    assert summary["evidence_permutation"]["decisions"] == 0
    assert summary["evidence_permutation"]["tv_mean"] is None
    assert summary["initial_carry_intervention"]["decisions"] == 4
    assert summary["initial_carry_intervention"]["tv_mean"] > 0
    assert summary["matched_carry_intervention"]["decisions"] == 2
    assert summary["matched_carry_intervention"]["tv_mean"] > 0
    assert measurements["donor_row"][0, 0] == 1
    assert measurements["donor_row"][1, 0] == 0
    assert measurements["donor_row"][2, 0] == measurements["donor_row"][3, 0] == -1
    np.testing.assert_array_equal(records["carry_h"], actual_carry)
    np.testing.assert_array_equal(
        measurements["original_probabilities"][2:], measurements["donor_carry_probabilities"][2:]
    )


def test_phase_mismatch_and_probability_replay_mismatch_are_rejected():
    records = retained()
    records["phase"][0] = 1
    with pytest.raises(ValueError, match="phases disagree"):
        sensitivity.analyze_inputs(EvidenceActor(), {}, records)
    records["phase"][0] = 0
    records["probabilities"] = np.zeros((1, 5, 242), dtype=np.float32)
    with pytest.raises(ValueError, match="Recomputed probabilities differ"):
        sensitivity.analyze_inputs(EvidenceActor(), {}, records)
