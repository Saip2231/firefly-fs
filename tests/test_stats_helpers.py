"""Tests for the statistics helpers in experiments/_common.py (exact values)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from _common import direction, holm  # noqa: E402


def test_holm_known_values():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(0.03, 2*0.02), max(0.04, 1*0.04)
    np.testing.assert_allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_holm_monotone_capped_and_nan_safe():
    adj = holm([0.5, np.nan, 0.001, 0.6])
    assert np.isnan(adj[1])
    # m = 3 non-NaN tests: 0.001*3, then max(.., 0.5*2)=1.0, then cap 1.0
    np.testing.assert_allclose(adj[[2, 0, 3]], [0.003, 1.0, 1.0])
    assert np.isnan(holm([np.nan, np.nan])).all()


def test_holm_never_below_raw_p():
    rng = np.random.default_rng(0)
    p = rng.random(20)
    assert np.all(holm(p) >= p)


def test_holm_nan_as_one_keeps_family_size():
    """PREREG policy: an all-tied comparison (p NaN) counts as p = 1, so m
    stays at the registered family size."""
    adj = holm([0.001, np.nan, 0.02, 0.5, 0.6, 0.7], nan_policy="as_one")
    # m = 6: 0.001*6 = 0.006, 0.02*5 = 0.1, then 0.5*4 = 2 -> capped 1.0
    np.testing.assert_allclose(adj, [0.006, 1.0, 0.1, 1.0, 1.0, 1.0])
    excl = holm([0.001, np.nan, 0.02, 0.5, 0.6, 0.7])   # default: m = 5
    np.testing.assert_allclose(excl[[0, 2]], [0.005, 0.08])
    assert np.isnan(excl[1])


def test_holm_bad_policy():
    import pytest

    with pytest.raises(ValueError):
        holm([0.1], nan_policy="drop")


def test_direction():
    assert direction(0.1) == "higher" and direction(-0.1) == "lower"
    assert direction(0.0) == "equal"
