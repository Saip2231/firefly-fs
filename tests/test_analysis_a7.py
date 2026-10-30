"""Tests for Stage A7 (experiments/analysis/a7_ablations.py): known paired
differences on a synthetic grid, rerank cost / changed-mask counts, p-values
vs scipy, and structure and labels on the real results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from experiments.analysis import a7_ablations as A7
from experiments.analysis import common as C

SEEDS = list(C.SEEDS)
NOISE = np.array([1, 3, 2, 5, 4, 7, 6, 9, 8, 10]) * 1e-4


def _rows(method, budget, base, rerank_delta=0.0, change_mask=False):
    out = []
    for s in SEEDS:
        acc = base + NOISE[s]
        for mt in ("raw", "rerank"):
            idx = frozenset({0, 1}) if (mt == "raw" or not change_mask) else frozenset({0, 2})
            out.append(dict(method=method, budget=budget, k=2, seed=s, mask_type=mt,
                            acc_lr=acc + (rerank_delta if mt == "rerank" else 0.0),
                            evals_used=budget, rerank_fits=50 if mt == "rerank" else 0,
                            rerank_s=3.0 if mt == "rerank" else 0.0,
                            search_s=budget / 100, idx=idx))
    return out


@pytest.fixture()
def toy():
    rows = []
    for b, lift in ((800, 0.0), (3000, 0.02)):
        rows += _rows("firefly", b, 0.80 + lift, rerank_delta=0.01, change_mask=True)
        rows += _rows("firefly_seeded", b, 0.83 + lift)
        rows += _rows("pso", b, 0.78)
        rows += _rows("pso_seeded", b, 0.77)
    return {"toy": pd.DataFrame(rows)}


def test_seeded_vs_plain_known(toy):
    t = A7.ablation_table(toy)
    s = t[(t.ablation == "seeded_vs_plain") & (t.budget == 800)].set_index("method")
    assert np.isclose(s.loc["firefly", "mean_diff_pts"], 3.0)
    assert np.isclose(s.loc["pso", "mean_diff_pts"], -1.0)
    assert s.loc["firefly", "nonzero_pairs"] == 10 and s.loc["firefly", "seeds_higher"] == 10


def test_rerank_cost_and_changed_masks(toy):
    t = A7.ablation_table(toy)
    r = t[(t.ablation == "rerank_vs_raw") & (t.method == "firefly") & (t.budget == 800)].iloc[0]
    assert np.isclose(r.mean_diff_pts, 1.0) and r.masks_changed == 10
    assert r.total_fits_variant == 850 and r.total_fits_reference == 800
    assert r.rerank_s_mean == 3.0
    same = t[(t.ablation == "rerank_vs_raw") & (t.method == "pso") & (t.budget == 800)].iloc[0]
    assert same.masks_changed == 0 and same.nonzero_pairs == 0 and np.isnan(same.p_raw)


def test_budget_ablation_known(toy):
    t = A7.ablation_table(toy)
    b = t[(t.ablation == "budget_3000_vs_800")].set_index("method")
    assert np.isclose(b.loc["firefly", "mean_diff_pts"], 2.0)
    assert np.isclose(b.loc["pso", "mean_diff_pts"], 0.0)
    assert b.loc["firefly", "search_s_variant"] == 30 and b.loc["firefly", "search_s_reference"] == 8
    assert (t[t.ablation == "budget_3000_vs_800"].search_s_note.str.contains("parallel")).all()


def test_p_values_match_scipy(toy):
    t = A7.ablation_table(toy)
    df = toy["toy"]
    raw = df[(df.mask_type == "raw") & (df.budget == 3000)]
    a = raw[raw.method == "firefly"].sort_values("seed").acc_lr.values
    b = raw[raw.method == "firefly_seeded"].sort_values("seed").acc_lr.values
    ref = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided").pvalue
    r = t[(t.ablation == "seeded_vs_plain") & (t.method == "firefly") & (t.budget == 3000)].iloc[0]
    assert np.isclose(r.p_raw, ref)
    assert (t.label == A7.LABEL).all()


def test_mismatched_seeds_raise(toy):
    df = toy["toy"]
    df = df[~((df.method == "pso") & (df.seed == 4) & (df.mask_type == "raw"))]
    with pytest.raises(ValueError, match="seeds"):
        A7.ablation_table({"toy": df})


def test_levy_status(toy):
    assert "not run" in A7.levy_status(toy)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("a7")
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    return out, meta, A7.make_outputs(meta, baselines, out_csv=out / "ablations.csv", fig_dir=out)


def test_real_structure(real):
    _, meta, res = real
    t = res["table"]
    counts = t.ablation.value_counts()
    assert counts["seeded_vs_plain"] == 12 * 2           # 12 families x 2 pairs
    assert counts["rerank_vs_raw"] == 12 * 4             # x 4 methods
    assert counts["budget_3000_vs_800"] == 6 * 4         # 6 dataset x k x 4 methods
    assert "levy_vs_plain" not in counts and (t.n == 10).all()
    rr = t[t.ablation == "rerank_vs_raw"]
    assert (rr.total_fits_variant == rr.total_fits_reference + 50).all()
    assert rr.masks_changed.between(0, 10).all()


def test_real_value_matches_pandas(real):
    _, meta, res = real
    t = res["table"]
    m = meta["cifar_r18"]
    raw = m[m.mask_type == "raw"]
    a = raw[(raw.method == "firefly") & (raw.k == 16) & (raw.budget == 3000)].sort_values("seed")
    b = raw[(raw.method == "firefly") & (raw.k == 16) & (raw.budget == 800)].sort_values("seed")
    row = t[(t.ablation == "budget_3000_vs_800") & (t.dataset == "cifar_r18")
            & (t.method == "firefly") & (t.k == 16)].iloc[0]
    assert np.isclose(row.mean_diff_pts, (a.acc_lr.values - b.acc_lr.values).mean() * 100)


def test_real_figures_labelled(real):
    out, _, res = real
    expected = {f"{f}_{ds}" for f in ("F3_budget_curve", "F9_ablations") for ds in C.DATASETS}
    assert set(res["figures"]) == expected
    for n, m in res["figures"].items():
        assert (out / f"{n}.png").exists() and "not corrected" in m["takeaway"]
