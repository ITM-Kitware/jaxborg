"""Check observable fitted distributions and matched-panel interpretations."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

METHODS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "studies/observation-value/baselines.py"))
fit_baseline = METHODS["fit_baseline"]
summarize_comparison = METHODS["summarize_comparison"]


def test_recovers_one_preference_vector_across_unequal_legal_masks():
    masks = np.array([[[1, 1, 0]], [[1, 1, 0]], [[1, 1, 0]], [[0, 1, 1]]], dtype=bool)
    teacher = softmax(np.where(masks, np.log([1.0, 3.0, 9.0]), -np.inf), axis=-1)
    fit = fit_baseline(teacher, masks, np.ones((4, 1), dtype=bool))
    held_out_masks = np.array([[[1, 1, 1]], [[1, 0, 1]]], dtype=bool)
    predicted = softmax(np.where(held_out_masks, np.array(fit["logits"]), -np.inf), axis=-1)
    expected = softmax(np.where(held_out_masks, np.log([1.0, 3.0, 9.0]), -np.inf), axis=-1)
    np.testing.assert_allclose(predicted, expected, atol=1e-6)
    assert fit["agents"][0]["teacher_to_baseline_kl_nats"] < 1e-10
    assert fit["agents"][0]["unique_legal_masks"] == 2
    assert fit["converged"]


def test_agents_fit_separately_and_busy_rows_cannot_change_preferences():
    teacher = np.array([[[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]], [[0.01, 0.99, 0.0], [0.99, 0.01, 0.0]]])
    masks = np.broadcast_to([True, True, False], teacher.shape)
    genuine = np.array([[True, True], [False, False]])
    fit = fit_baseline(teacher, masks, genuine)
    predicted = softmax(np.where(masks[0], np.array(fit["logits"]), -np.inf), axis=-1)
    np.testing.assert_allclose(predicted, teacher[0], atol=1e-6)
    assert [row["never_legal_actions"] for row in fit["agents"]] == [[2], [2]]
    assert [row["genuine_decisions"] for row in fit["agents"]] == [1, 1]


def test_teacher_illegal_mass_cannot_enter_fit():
    with pytest.raises(ValueError, match="illegal"):
        fit_baseline(np.array([[[0.4, 0.4, 0.2]]]), np.array([[[1, 1, 0]]]), np.array([[True]]))


def test_repeated_opponents_do_not_artificially_narrow_root_uncertainty():
    roots = np.array([0.0, 80.0, 120.0, 200.0])[:, None, None]
    small = summarize_comparison(roots)
    repeated = summarize_comparison(np.tile(roots, (1, 8, 4)))
    assert repeated["confidence_interval"] == pytest.approx(small["confidence_interval"])
    assert repeated["classification"] == "inconclusive"
    assert repeated["matched_episode_pairs"] == 128


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (150.0, "useful_full_policy_advantage"),
        (-150.0, "meaningful_baseline_advantage"),
        (20.0, "comparable_within_margin"),
    ],
)
def test_practical_margin_classification(difference, expected):
    report = summarize_comparison(np.full((8, 2, 4), difference))
    assert report["classification"] == expected
    assert report["mean_difference"] == difference


def test_wide_interval_cannot_establish_comparability():
    report = summarize_comparison(np.array([-500.0, 500.0, -500.0, 500.0])[:, None, None])
    assert report["mean_difference"] == 0
    assert report["classification"] == "inconclusive"


def test_missing_panel_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize_comparison(np.array([0.0, np.nan])[:, None, None])
