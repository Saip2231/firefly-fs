"""Tests for Stage A6 (experiments/analysis/a6_transfer.py): known paired
differences and gaps on synthetic frames, pairing guard, and structure /
labels on the real results (no p-values by design)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.analysis import a6_transfer as A6
from experiments.analysis import common as C

SEEDS = list(C.SEEDS)


def _base(method, lr, knn, svm, k=4):
    return [dict(method=method, seed=s, k=k, acc_lr=lr, acc_knn=knn, acc_svm=svm,
                 idx=frozenset(range(k))) for s in SEEDS]


def _meta(method, lr, knn, svm, k=4, budget=800):
    return [dict(method=method, seed=s, k=k, budget=budget, mask_type=mt, acc_lr=lr,
                 acc_knn=knn + (0.0 if mt == "raw" else 0.5), acc_svm=svm)
            for s in SEEDS for mt in ("raw", "rerank")]


@pytest.fixture()
def toy():
    tr = pd.DataFrame(_base("anova", 0.80, 0.70, 0.78) + _base("mi", 0.75, 0.72, 0.74)
                      + _base("l1", 0.79, 0.69, 0.79) + _base("rf", 0.81, 0.75, 0.80))
    me = pd.DataFrame(_meta("firefly", 0.82, 0.74, 0.80) + _meta("pso", 0.78, 0.70, 0.77))
    return {"toy": tr}, {"toy": me}


def test_known_paired_differences_raw_only(toy):
    t = A6.transfer_table(*toy)
    r = t[(t.method == "firefly") & (t.comparator == "anova") & (t.classifier == "knn")].iloc[0]
    assert np.isclose(r.mean_diff_pts, 4.0)              # 0.74 - 0.70; rerank (+0.5) ignored
    assert r.seeds_higher == 10 and r.seeds_lower == 0 and r.n == 10
    r = t[(t.method == "pso") & (t.comparator == "rf") & (t.classifier == "svm")].iloc[0]
    assert np.isclose(r.mean_diff_pts, -3.0)
    assert len(t) == 1 * 2 * 4 * 3                        # family x methods x comparators x clf
    assert set(t[t.classifier != "lr"].role) == {"exploratory"}
    assert not any("p" == c or c.startswith("p_") for c in t.columns)   # no p-values


def test_gaps_known_values(toy):
    g = A6.gaps_table(*toy)
    ff = g[g.method == "firefly"].iloc[0]
    assert np.isclose(ff.gap_lr_knn_pts, 8.0) and np.isclose(ff.gap_lr_svm_pts, 2.0)
    an = g[g.method == "anova"].iloc[0]
    assert np.isclose(an.gap_lr_knn_pts, 10.0) and np.isnan(an.budget)


def test_mismatched_seeds_raise(toy):
    tr, me = toy
    tr = {"toy": tr["toy"][tr["toy"].seed != 3]}
    with pytest.raises(ValueError, match="seeds"):
        A6.transfer_table(tr, me)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("a6")
    transfer = {ds: C.load_baseline_transfer(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    return out, transfer, meta, A6.make_outputs(transfer, meta, out_dir=out, fig_dir=out)


def test_real_structure(real):
    _, transfer, meta, res = real
    t = res["transfer"]
    assert len(t) == C.prereg_family_count() * 2 * 4 * 3 == 288
    assert (t.n == 10).all() and (t.seeds_higher + t.seeds_lower <= 10).all()
    assert not t.comparator.isin(["pca", "random"]).any()


def test_real_values_match_pandas(real):
    _, transfer, meta, res = real
    t = res["transfer"]
    m, b = meta["cifar_r18"], transfer["cifar_r18"]
    ff = m[(m.method == "firefly") & (m.mask_type == "raw") & (m.k == 32)
           & (m.budget == 3000)].sort_values("seed").acc_knn.values
    an = b[(b.method == "anova") & (b.k == 32)].sort_values("seed").acc_knn.values
    r = t[(t.dataset == "cifar_r18") & (t.k == 32) & (t.budget == 3000) & (t.method == "firefly")
          & (t.comparator == "anova") & (t.classifier == "knn")].iloc[0]
    assert np.isclose(r.mean_diff_pts, (ff - an).mean() * 100)


def test_real_lr_reference_equals_a1(real):
    """The LR rows here must agree with the confirmatory A1 numbers."""
    from experiments.analysis import a1_rq1 as A1

    _, transfer, meta, res = real
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    a1 = A1.run_rq1(baselines, meta, C.load_split_hashes())
    t = res["transfer"]
    lr = t[(t.method == "firefly") & (t.classifier == "lr")].merge(
        a1, on=["dataset", "k", "budget", "comparator"], suffixes=("", "_a1"))
    assert len(lr) == 12 * 4 and np.allclose(lr.mean_diff_pts, lr.mean_diff_pts_a1)


def test_real_figures(real):
    out, _, _, res = real
    assert set(res["figures"]) == {f"F8_transfer_{ds}" for ds in C.DATASETS}
    for n, m in res["figures"].items():
        assert (out / f"{n}.png").exists() and "exploratory" in m["takeaway"]
        assert "No significance tests" in m["takeaway"]
