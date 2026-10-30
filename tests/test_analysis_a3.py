"""Tests for Stage A3 (experiments/analysis/a3_cost.py): known answers on a
synthetic grid, then structure and labels on the real results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.analysis import a3_cost as A3
from experiments.analysis import common as C


def _rows(n=10, **cols):
    return [dict(seed=s, **cols) for s in range(n)]


@pytest.fixture()
def toy():
    b = pd.DataFrame(
        _rows(method="all", k=10, acc=0.90, search_s=0.0, infer_s=2e-6)
        + _rows(method="anova", k=2, acc=0.80, search_s=1.0, infer_s=1e-6)
        + _rows(method="pca", k=2, acc=0.85, search_s=0.5, infer_s=3e-6)      # slower -> never
        + _rows(method="mi", k=2, acc=0.70, search_s=4.0, infer_s=1e-6))      # less accurate
    m = pd.DataFrame(
        _rows(method="firefly", k=2, budget=800, mask_type="raw", acc_lr=0.82, search_s=10.0,
              rerank_s=0.0, infer_s=1e-6, evals_used=800, rerank_fits=0)
        + _rows(method="firefly", k=2, budget=800, mask_type="rerank", acc_lr=0.83,
                search_s=10.0, rerank_s=2.0, infer_s=1e-6, evals_used=800, rerank_fits=50))
    return {"toy": b}, {"toy": m}


def _row(t, method, mask="-"):
    return t[(t.method == method) & (t.mask_type == mask)].iloc[0]


def test_break_even_known_values(toy):
    t = A3.cost_table(*toy)
    raw, rr = _row(t, "firefly", "raw"), _row(t, "firefly", "rerank")
    assert np.isclose(raw.break_even_samples, 10.0 / 1e-6)          # 1e7
    assert np.isclose(rr.break_even_samples, 12.0 / 1e-6)           # rerank time included
    assert np.isclose(_row(t, "anova").break_even_samples, 1.0 / 1e-6)
    assert _row(t, "pca").break_even == "never" and np.isnan(_row(t, "pca").break_even_samples)
    assert _row(t, "all").break_even == "-"


def test_fits_and_totals(toy):
    t = A3.cost_table(*toy)
    raw, rr = _row(t, "firefly", "raw"), _row(t, "firefly", "rerank")
    assert raw.total_fits_mean == 800 and rr.total_fits_mean == 850
    assert raw.total_s_mean == 10.0 and rr.total_s_mean == 12.0 and raw.n == 10


def test_extra_cost_vs_anova(toy):
    t = A3.cost_table(*toy)
    raw = _row(t, "firefly", "raw")
    assert np.isclose(raw.mean_diff_vs_anova_pts, 2.0)
    assert np.isclose(raw.extra_s_per_pt_vs_anova, (10.0 - 1.0) / 2.0)
    rr = _row(t, "firefly", "rerank")
    assert np.isclose(rr.extra_s_per_pt_vs_anova, (12.0 - 1.0) / 3.0)
    mi = _row(t, "mi")
    assert mi.extra_cost_vs_anova == "no accuracy gain" and np.isclose(mi.mean_diff_vs_anova_pts, -10.0)
    assert _row(t, "anova").extra_cost_vs_anova == "-"


def test_labels_and_notes(toy):
    t = A3.cost_table(*toy)
    assert (t.analysis == "exploratory").all()
    assert t.search_s_note.str.contains("parallel load").all()
    assert t.infer_s_note.str.contains("indicative only").all()


# ---------------------------------------------------------------------------
# Real results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("a3")
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    res = A3.make_outputs(baselines, meta, out_csv=out / "rq2_cost.csv", fig_dir=out)
    return out, baselines, meta, res


def test_real_table_structure(real):
    out, baselines, meta, res = real
    t = res["table"]
    n_base = sum(b.groupby(["method", "k"]).ngroups for b in baselines.values())
    n_meta = sum(m.groupby(["method", "k", "budget", "mask_type"]).ngroups for m in meta.values())
    assert len(t) == n_base + n_meta and (t.n == 10).all()
    rr, raw = t[t.mask_type == "rerank"], t[t.mask_type == "raw"]
    assert (rr.total_fits_mean == rr.evals_used_mean + 50).all()
    assert (raw.total_fits_mean == raw.evals_used_mean).all()
    assert (raw.rerank_s_mean == 0).all() and (rr.rerank_s_mean > 0).all()
    sel = t[t.method != "all"]
    never = sel.break_even == "never"
    assert (never == (sel.infer_s_mean >= sel.infer_all_mean)).all()
    assert pd.read_csv(out / "rq2_cost.csv").shape[0] == len(t)


def test_real_means_match_pandas(real):
    _, _, meta, res = real
    t = res["table"]
    m = meta["cifar_r18"]
    ref = m[(m.method == "firefly") & (m.k == 64) & (m.budget == 3000)
            & (m.mask_type == "raw")].search_s
    row = t[(t.dataset == "cifar_r18") & (t.method == "firefly") & (t.k == 64)
            & (t.budget == 3000) & (t.mask_type == "raw")].iloc[0]
    assert np.isclose(row.search_s_mean, ref.mean()) and np.isclose(row.search_s_std, ref.std())


def test_real_figures_and_takeaways(real):
    out, _, _, res = real
    names = sorted(res["figures"])
    assert names == sorted(f"{f}_{ds}" for f in ("F4_acc_vs_search", "F5_break_even")
                           for ds in C.DATASETS)
    for n in names:
        assert (out / f"{n}.png").exists() and (out / f"{n}.pdf").exists()
        assert "exploratory" in res["figures"][n]["takeaway"]
    tk = pd.read_csv(out / "takeaways.csv")
    assert sorted(tk.figure) == names


def test_figure_titles_labelled_exploratory(real):
    import matplotlib.pyplot as plt

    t = real[3]["table"]
    for plot in (A3.plot_f4, A3.plot_f5):
        fig = plot(t, "digits")
        title = fig._suptitle.get_text()
        plt.close(fig)
        assert "exploratory" in title and "parallel load" in title
