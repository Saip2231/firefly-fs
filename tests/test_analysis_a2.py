"""Tests for Stage A2 (experiments/analysis/a2_figures.py).

Figures are checked through the dataframes they plot (plain-pandas
cross-checks), the files they write, and the computed takeaways; titles must
describe, never claim (ANALYSIS.md rule 8).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from experiments.analysis import a1_rq1 as A1
from experiments.analysis import a2_figures as A2
from experiments.analysis import common as C
from experiments.analysis import takeaway as T


@pytest.fixture(scope="module")
def data():
    return ({ds: C.load_baselines(ds) for ds in C.DATASETS},
            {ds: C.load_meta(ds) for ds in C.DATASETS})


@pytest.fixture(scope="module")
def made(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    return out, A2.make_figures(*data, out_dir=out)


def test_f1_frame_methods_and_means(data):
    baselines, meta = data
    f = A2.f1_frame(baselines, meta, "cifar_r18", 800)
    assert set(f.method) == set(A2.F1_METHODS) | {"all"}          # no seeded variants
    assert set(f.loc[f.method != "all", "k"]) == {16, 32, 64, 128}
    assert (f.n == 10).all()
    m = meta["cifar_r18"]
    ref = m[(m.method == "pso") & (m.mask_type == "raw") & (m.budget == 800) & (m.k == 64)].acc_lr
    row = f[(f.method == "pso") & (f.k == 64)].iloc[0]
    assert np.isclose(row["mean"], ref.mean()) and np.isclose(row["std"], ref.std(ddof=1))
    b = baselines["cifar_r18"]
    assert np.isclose(f[f.method == "all"]["mean"].iloc[0], b[b.method == "all"].acc.mean())


def test_f1_baselines_identical_across_budgets(data):
    baselines, meta = data
    f8 = A2.f1_frame(baselines, meta, "digits", 800).set_index(["method", "k"])
    f3 = A2.f1_frame(baselines, meta, "digits", 3000).set_index(["method", "k"])
    for m in ("random", "anova", "mi", "l1", "rf", "pca", "all"):
        assert np.allclose(f8.loc[m, "mean"], f3.loc[m, "mean"])
    assert not np.allclose(f8.loc["firefly", "mean"], f3.loc["firefly", "mean"])


def test_f1_takeaway_numbers_come_from_frame(data):
    baselines, meta = data
    f = A2.f1_frame(baselines, meta, "cifar_r18", 3000)
    text = A2.f1_takeaway(f)
    ff16 = f[(f.method == "firefly") & (f.k == 16)]["mean"].iloc[0]
    assert T.acc(ff16) in text
    assert T.acc(f[f.method == "all"]["mean"].iloc[0]) in text
    assert "rank" in text and "budget 3000" in text


def test_files_written_and_takeaways_recorded(made):
    out, figs = made
    names = [m["figure"] for m in figs]
    assert names == ["F1_acc_vs_k_digits_b800", "F1_acc_vs_k_digits_b3000",
                     "F1_acc_vs_k_cifar_r18_b800", "F1_acc_vs_k_cifar_r18_b3000",
                     "F2_rq1_grid"]
    for n in names:
        assert (out / f"{n}.png").stat().st_size > 10_000 and (out / f"{n}.pdf").exists()
    tk = pd.read_csv(out / "takeaways.csv")
    assert list(tk.columns) == T.TAKEAWAY_COLUMNS and sorted(tk.figure) == sorted(names)
    assert tk.takeaway.str.len().gt(40).all()


def test_rerun_does_not_duplicate_takeaways(data, made):
    out, _ = made
    A2.make_figures(*data, out_dir=out)
    assert len(pd.read_csv(out / "takeaways.csv")) == 5


def test_f2_frame_equals_a1(data, made):
    baselines, meta = data
    _, figs = made
    f2 = figs[-1]["frame"]
    ref = A1.run_rq1(baselines, meta, C.load_split_hashes())
    pd.testing.assert_frame_equal(f2.reset_index(drop=True), ref.reset_index(drop=True))
    assert len(f2) == 72 and "Holm within family, n=10" in A2.F2_LEGEND
    tk = figs[-1]["takeaway"]
    counts = f2.label.value_counts()
    assert f"{counts.get('WIN', 0)} WIN" in tk and f"{counts.get('LOSS', 0)} LOSS" in tk


def test_titles_describe_not_claim(data):
    import matplotlib.pyplot as plt

    baselines, meta = data
    figs = [A2.plot_f1(A2.f1_frame(baselines, meta, "digits", 800)),
            A2.plot_f2(A2.f2_frame(baselines, meta))]
    banned = re.compile(r"\b(win|wins|beats?|better|best|outperform\w*)\b", re.I)
    for fig in figs:
        title = fig.axes[0].get_title()
        assert title and not banned.search(title), title
        plt.close(fig)


def test_pca_labelled_projection_in_f1(data):
    import matplotlib.pyplot as plt

    baselines, meta = data
    fig = A2.plot_f1(A2.f1_frame(baselines, meta, "cifar_r18", 800))
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert any("projection, not a selector" in s for s in labels)
    assert not any("seeded" in s for s in labels)
