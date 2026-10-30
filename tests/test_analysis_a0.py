"""Tests for Stage A0 (experiments/analysis/common.py, a0_integrity.py,
style.py, takeaway.py).

A small synthetic grid passes every check; for each check a deliberately
broken copy must make that check fail.  The real result files must pass too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.analysis import a0_integrity as A0
from experiments.analysis import common as C
from experiments.analysis import style as S
from experiments.analysis import takeaway as T

D, KS, BUDGETS = 10, (2, 4), (800, 3000)


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------

def _idx(k, seed):
    return frozenset(int(i) for i in np.random.default_rng(seed).choice(D, k, replace=False))


def good_frames():
    b_rows = [dict(dataset="toy", method="all", seed=s, k=D, acc=0.9, idx=frozenset())
              for s in C.SEEDS]
    for m in ("random", "anova", "pca"):
        for k in KS:
            for s in C.SEEDS:
                b_rows.append(dict(dataset="toy", method=m, seed=s, k=k, acc=0.8,
                                   idx=frozenset() if m == "pca" else _idx(k, s)))
    m_rows = []
    for m in ("firefly", "pso"):
        for b in BUDGETS:
            for k in KS:
                for s in C.SEEDS:
                    for mt in C.MASK_TYPES:
                        m_rows.append(dict(dataset="toy", method=m, budget=b, seed=s, k=k,
                                           mask_type=mt, acc_lr=0.8, acc_knn=0.7, acc_svm=0.75,
                                           evals_used=b, idx=_idx(k, s + 100),
                                           config_hash="h" * 64, git_hash="abc1234"))
    split = pd.DataFrame(dict(dataset="toy", seed=list(C.SEEDS), sha_all=["x" * 64] * 10))
    return {"toy": pd.DataFrame(b_rows)}, {"toy": pd.DataFrame(m_rows)}, split


def checks_of(baselines, meta, split, stated=len(KS) * len(BUDGETS)):
    return {c.name: c for c in A0.run_checks(baselines, meta, split, stated)}


def test_good_frames_pass_everything():
    b, m, s = good_frames()
    res = checks_of(b, m, s)
    assert all(c.ok for c in res.values()), {n: c.detail for n, c in res.items() if not c.ok}
    assert res["git_hashes"].info


# ---------------------------------------------------------------------------
# One broken frame per check
# ---------------------------------------------------------------------------

def _fails(name, b, m, s, stated=len(KS) * len(BUDGETS)):
    res = checks_of(b, m, s, stated)
    assert not res[name].ok, f"{name} should fail: {res[name].detail}"
    return res


def test_row_counts_detects_missing_row():
    b, m, s = good_frames()
    m["toy"] = m["toy"].iloc[1:]
    _fails("row_counts", b, m, s)


def test_seeds_complete_detects_duplicate_seed():
    b, m, s = good_frames()
    df = m["toy"]
    df.loc[(df.seed == 9) & (df.method == "pso") & (df.k == 2) & (df.budget == 800)
           & (df.mask_type == "raw"), "seed"] = 8
    res = _fails("seeds_complete", b, m, s)
    assert res["row_counts"].ok                      # count unchanged, seeds wrong


def test_no_nan_acc_detects_nan():
    b, m, s = good_frames()
    m["toy"].loc[0, "acc_knn"] = np.nan
    _fails("no_nan_acc", b, m, s)
    b2, m2, s2 = good_frames()
    b2["toy"].loc[3, "acc"] = np.nan
    _fails("no_nan_acc", b2, m2, s2)


def test_evals_budget_detects_overrun():
    b, m, s = good_frames()
    m["toy"].loc[5, "evals_used"] = m["toy"].loc[5, "budget"] + 1
    _fails("evals_budget", b, m, s)


def test_idx_length_detects_short_and_out_of_range():
    b, m, s = good_frames()
    m["toy"].at[0, "idx"] = frozenset([0])           # k = 2 expected
    _fails("idx_length", b, m, s)
    b2, m2, s2 = good_frames()
    i = b2["toy"].index[b2["toy"].method == "anova"][0]
    b2["toy"].at[i, "idx"] = frozenset([0, D])       # index D is out of range
    _fails("idx_length", b2, m2, s2)


def test_idx_length_ignores_pca_and_all():
    b, m, s = good_frames()                          # pca/all have empty idx
    assert checks_of(b, m, s)["idx_length"].ok


def test_config_hash_detects_second_hash():
    b, m, s = good_frames()
    m["toy"].loc[7, "config_hash"] = "z" * 64
    _fails("config_hash", b, m, s)


def test_budgets_detects_smoke_budget_and_unknown_method():
    b, m, s = good_frames()
    m["toy"].loc[m["toy"].budget == 800, "budget"] = 150
    _fails("budgets", b, m, s)
    b2, m2, s2 = good_frames()
    m2["toy"].loc[0, "method"] = "ga"
    _fails("budgets", b2, m2, s2)


def test_seeds_match_detects_shifted_seeds():
    b, m, s = good_frames()
    m["toy"]["seed"] = m["toy"]["seed"] + 1
    _fails("seeds_match", b, m, s)


def test_split_hashes_detects_missing_and_empty():
    b, m, s = good_frames()
    _fails("split_hashes", b, m, s[s.seed != 3])
    s2 = s.copy()
    s2.loc[0, "sha_all"] = ""
    _fails("split_hashes", b, m, s2)


def test_families_must_match_prereg():
    b, m, s = good_frames()
    _fails("families", b, m, s, stated=12)
    _fails("families", b, m, s, stated=None)


# ---------------------------------------------------------------------------
# Loaders and helpers
# ---------------------------------------------------------------------------

def test_parse_idx():
    assert C.parse_idx("2,13,19") == frozenset({2, 13, 19})
    assert C.parse_idx("") == frozenset() and C.parse_idx(np.nan) == frozenset()
    assert C.parse_idx(None) == frozenset()


def test_unknown_baseline_label_raises():
    raw = pd.DataFrame(dict(method=["ANOVA F", "Genetic"], idx=["1,2", "3,4"]))
    with pytest.raises(ValueError):
        C.tidy_baselines(raw)


def test_label_mapping_covers_real_files():
    for ds in C.DATASETS:
        raw = pd.read_csv(C.RESULTS / f"baselines_{ds}.csv", usecols=["method"])
        assert set(raw.method) <= set(C.BASELINE_LABELS)


def test_loaders_on_real_files():
    b = C.load_baselines("digits")
    assert set(b.method) == {"all", "random", "anova", "mi", "l1", "rf", "pca"}
    assert C.dimension(b) == 64 and C.selection_ks(b) == [16, 32]
    m = C.load_meta("digits")
    assert all(isinstance(x, frozenset) for x in m.idx)
    assert C.prereg_family_count() == 12


def test_real_results_pass_integrity():
    """The committed Day-5 results must pass every A0 check."""
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    res = {c.name: c for c in A0.run_checks(baselines, meta, C.load_split_hashes(),
                                            C.prereg_family_count())}
    assert all(c.ok for c in res.values()), {n: c.detail for n, c in res.items() if not c.ok}


def test_validate_writes_report(tmp_path):
    path = tmp_path / "integrity_report.md"
    A0.validate(report_path=path)
    text = path.read_text(encoding="utf-8")
    assert "ALL CHECKS PASSED" in text and "| seeds_complete | PASS |" in text


def test_style_covers_every_method_and_labels_pca():
    for method in C.DISPLAY:
        st = S.style(method)
        assert {"color", "marker", "linestyle", "label"} <= set(st)
    assert "projection" in S.style("pca")["label"] and S.style("pca")["linestyle"] == "--"
    with pytest.raises(KeyError):
        S.style("ga")


def test_save_fig_writes_png_and_pdf(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    paths = S.save_fig(fig, "probe", tmp_path)
    plt.close(fig)
    assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.exists() for p in paths)


def test_takeaway_formatting():
    assert T.pts(0.0123) == "+1.2 pts" and T.pts(-0.005) == "-0.5 pts"
    assert T.pval(0.0004) == "p < 0.001" and T.pval(0.05) == "p = 0.050"
    assert T.pval(float("nan")) == "p = n/a"
    assert T.test_phrase(0.01, 0.002, 0.012, 10) == "+1.0 pts (p = 0.002, Holm p = 0.012, n = 10)"
    df = pd.DataFrame(dict(method=["anova", "pca"], mean=[0.8, 0.9]))
    assert T.best_mean(df) == ("PCA (projection, not a selector)", 0.9)
