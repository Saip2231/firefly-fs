"""Tests for Stage A1 (experiments/analysis/a1_rq1.py).

Synthetic families with known answers, cross-checked against independent
implementations (ANALYSIS.md rule 11): scipy.stats.wilcoxon for p,
statsmodels multipletests(method="holm") for the adjustment, plain pandas
for means.  Then the real results: 72 tests, PREREG structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from experiments._common import holm
from experiments.analysis import a1_rq1 as A1
from experiments.analysis import common as C

SEEDS = list(C.SEEDS)


def ser(values) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=SEEDS)


BASE = ser(np.linspace(0.80, 0.85, 10))
NOISE = np.array([1, 3, 2, 5, 4, 7, 6, 9, 8, 10]) * 1e-4       # distinct, no ties


def neutral(seed):
    """A comparator with mixed-sign differences around BASE (clearly not significant)."""
    signs = np.random.default_rng(seed).permutation([1, -1] * 5)
    return ser(BASE.values - signs * NOISE)


def family_with(name, comp) -> dict:
    comps = {c: neutral(i) for i, c in enumerate(C.COMPARATORS)}
    comps[name] = comp
    return {r["comparator"]: r for r in A1.family_rows(BASE, comps, dataset="toy", k=4, budget=800)}


# ---------------------------------------------------------------------------
# Known answers
# ---------------------------------------------------------------------------

def test_known_win():
    rows = family_with("anova", ser(BASE.values - 0.02 - NOISE))   # Firefly always higher
    r = rows["anova"]
    assert r["nonzero_pairs"] == 10 and np.isclose(r["mean_diff_pts"], 2.055)
    assert np.isclose(r["p_raw"], 2 / 1024)                      # exact two-sided minimum
    assert r["p_holm"] < 0.05 and r["label"] == "WIN"


def test_known_loss():
    rows = family_with("pca", ser(BASE.values + 0.03 + NOISE))
    r = rows["pca"]
    assert r["mean_diff_pts"] < 0 and r["p_holm"] < 0.05 and r["label"] == "LOSS"


def test_known_tie():
    rows = family_with("mi", neutral(42))
    assert rows["mi"]["label"] == "TIE" and rows["mi"]["p_holm"] >= 0.05


def test_all_zero_differences_are_tie_with_p_one():
    rows = family_with("l1", BASE.copy())
    r = rows["l1"]
    assert r["nonzero_pairs"] == 0 and r["mean_diff_pts"] == 0
    assert r["p_raw"] == 1.0 and r["p_holm"] == 1.0 and r["label"] == "TIE"


def test_undefined_test_keeps_family_size_six():
    """One all-zero comparator (p undefined -> 1) must not shrink m: the win's
    Holm p is 6 x 2/1024, not 5 x 2/1024."""
    comps = {c: neutral(i) for i, c in enumerate(C.COMPARATORS)}
    comps["anova"] = ser(BASE.values - 0.02 - NOISE)
    comps["l1"] = BASE.copy()
    rows = {r["comparator"]: r for r in A1.family_rows(BASE, comps, dataset="t", k=1, budget=800)}
    assert np.isclose(rows["anova"]["p_holm"], 6 * 2 / 1024)
    assert rows["l1"]["p_holm"] == 1.0


def test_win_needs_holm_not_raw_p():
    """Raw p < 0.05 but Holm p >= 0.05 -> TIE (correction actually applied).
    The two negative differences are the two smallest (ranks 1 and 2), so
    W- = 3 and the exact two-sided p = 2 * 5/1024 = 0.0098; Holm x6 = 0.059."""
    d = np.array([-1, 1, -1, 1, 1, 1, 1, 1, 1, 1]) * NOISE
    rows = family_with("rf", ser(BASE.values - d))
    r = rows["rf"]
    assert np.isclose(r["p_raw"], 10 / 1024)
    assert r["p_raw"] < 0.05 <= r["p_holm"] and r["label"] == "TIE"


def test_nan_accuracy_raises():
    bad = BASE.copy()
    bad.iloc[3] = np.nan
    sh = pd.DataFrame(dict(dataset="toy", seed=SEEDS, sha_all=["x"] * 10))
    with pytest.raises(ValueError, match="NaN"):
        A1.check_pair(BASE, bad, sh, "toy", "nan case")


def test_mismatched_seeds_raise():
    sh = pd.DataFrame(dict(dataset="toy", seed=SEEDS, sha_all=["x"] * 10))
    shifted = pd.Series(BASE.values, index=[s + 1 for s in SEEDS])
    with pytest.raises(ValueError, match="seeds"):
        A1.check_pair(BASE, shifted, sh, "toy", "shifted")
    with pytest.raises(ValueError, match="seeds"):
        A1.check_pair(BASE, BASE.iloc[:9], sh, "toy", "short")


def test_missing_split_hash_raises():
    sh = pd.DataFrame(dict(dataset="toy", seed=SEEDS[:9], sha_all=["x"] * 9))
    with pytest.raises(ValueError, match="split hashes"):
        A1.check_pair(BASE, BASE, sh, "toy", "hash")


def test_family_must_have_six_comparators():
    with pytest.raises(ValueError):
        A1.family_rows(BASE, {"anova": BASE}, dataset="t", k=1, budget=800)


# ---------------------------------------------------------------------------
# Cross-checks against independent implementations
# ---------------------------------------------------------------------------

def test_p_matches_scipy_and_means_match_pandas():
    rng = np.random.default_rng(7)
    for _ in range(20):
        a = ser(rng.integers(700, 900, 10) / 1000)
        b = ser(rng.integers(700, 900, 10) / 1000)
        r = A1.paired_test(a, b)
        diff = a - b
        if (diff != 0).any():
            ref = wilcoxon(a.values, b.values, zero_method="wilcox", alternative="two-sided")
            assert np.isclose(r["p_raw"], ref.pvalue)
        assert np.isclose(r["mean_diff_pts"], diff.mean() * 100)
        assert np.isclose(r["std_diff_pts"], diff.std(ddof=1) * 100)
        assert np.isclose(r["mean_firefly"], a.mean()) and r["nonzero_pairs"] == int((diff != 0).sum())


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.random(6) ** 3
        ref = multipletests(p, method="holm")[1]
        assert np.allclose(holm(p, nan_policy="as_one"), ref)
    p = np.array([0.001, np.nan, 0.02, 0.5, 0.6, 0.7])          # NaN counts as 1
    ref = multipletests(np.nan_to_num(p, nan=1.0), method="holm")[1]
    assert np.allclose(holm(p, nan_policy="as_one"), ref)


def test_labels():
    assert A1.label(0.01, 1.0) == "WIN" and A1.label(0.01, -1.0) == "LOSS"
    assert A1.label(0.05, 1.0) == "TIE" and A1.label(0.01, 0.0) == "TIE"


# ---------------------------------------------------------------------------
# Real results: structure only (no assertion on outcomes)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    return baselines, meta, A1.run_rq1(baselines, meta, C.load_split_hashes())


def test_real_rq1_structure(real):
    baselines, meta, t = real
    assert list(t.columns) == A1.RQ1_COLUMNS
    assert len(t) == C.prereg_family_count() * A1.FAMILY_SIZE == 72
    assert (t.n == 10).all() and set(t.label) <= {"WIN", "LOSS", "TIE"}
    assert t.groupby(["dataset", "k", "budget"]).size().eq(6).all()
    assert ((t.p_holm >= t.p_raw - 1e-12) & (t.p_holm <= 1)).all()


def test_real_uses_raw_masks_and_reuses_baselines(real):
    baselines, meta, t = real
    # firefly mean equals the plain-pandas mean of raw acc_lr
    m = meta["digits"]
    ref = m[(m.method == "firefly") & (m.mask_type == "raw") & (m.k == 16)
            & (m.budget == 800)].acc_lr.mean()
    row = t[(t.dataset == "digits") & (t.k == 16) & (t.budget == 800)].iloc[0]
    assert np.isclose(row.mean_firefly, ref)
    # the same baseline rows at both budgets
    a = t[(t.comparator == "anova")].pivot_table(index=["dataset", "k"], columns="budget",
                                                  values="mean_comparator")
    assert np.allclose(a[800], a[3000])
    b = baselines["digits"]
    assert np.isclose(a.loc[("digits", 16), 800], b[(b.method == "anova") & (b.k == 16)].acc.mean())


def test_descriptive_covers_all_methods(real):
    baselines, meta, _ = real
    d = A1.descriptive(baselines, meta)
    for ds in C.DATASETS:
        methods = set(d[d.dataset == ds].method)
        assert {"all", "random", "anova", "mi", "l1", "rf", "pca",
                "firefly", "pso", "firefly_seeded", "pso_seeded"} <= methods
    assert (d.n == 10).all()
