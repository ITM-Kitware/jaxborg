import pytest

from scripts.dev.parity.stats import tost_equivalence


def test_tost_equivalence_accepts_identical_zero_variance_samples():
    result = tost_equivalence([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], margin=0.5)

    assert result["equivalent"] is True
    assert result["mean_diff"] == 0.0


def test_tost_equivalence_rejects_zero_variance_gap_outside_margin():
    result = tost_equivalence([10.0, 10.0, 10.0], [1.0, 1.0, 1.0], margin=0.5)

    assert result["equivalent"] is False
    assert result["mean_diff"] == 9.0


def test_paired_tost_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        tost_equivalence([1.0, 2.0], [1.0, 2.0, 3.0], margin=1.0, paired=True)
