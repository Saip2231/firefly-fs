"""Dashboard content tests (B2-B8): every number a page shows equals a plain
pandas recomputation from the result files; replays and live search behave.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.lib import cost_view, data, facts, results_view, search_live, traces  # noqa: E402

RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"


def _rq1():
    return pd.read_csv(ANALYSIS / "rq1_tests.csv")


def test_verdict_counts_equal_pandas():
    rq1 = _rq1()
    first = facts.rq1_verdict(rq1)[0]
    vc = rq1["label"].value_counts()
    for lab in ("WIN", "LOSS", "TIE"):
        assert f"{int(vc.get(lab, 0))} {'are ' if lab == 'WIN' else ''}{lab}" in first
    assert f"Of {len(rq1)} pre-registered" in first
    assert math.isclose(facts.min_two_sided_p(10), 2 / 1024)


def test_every_family_panel_reproduces_rq1_mean_difference():
    """For all tests: per-seed differences recomputed from the raw CSVs have the
    mean and std that rq1_tests.csv reports, and the same non-zero count."""
    rq1 = _rq1()
    for ds in ("digits", "cifar_r18"):
        seeds = data.per_seed(ds)
        for r in rq1[rq1["dataset"] == ds].itertuples():
            d = results_view.paired_diffs(seeds, int(r.k), int(r.budget), r.comparator)
            assert len(d) == r.n
            assert math.isclose(d["diff_pts"].mean(), r.mean_diff_pts, abs_tol=1e-9)
            assert math.isclose(d["diff_pts"].std(), r.std_diff_pts, abs_tol=1e-9)
            assert int((d["diff_pts"].abs() > 1e-12).sum()) == r.nonzero_pairs


def test_grid_counts_equal_rq1():
    rq1 = _rq1()
    grid = results_view.grid_table(rq1)
    assert len(grid) == len(rq1) and grid["family"].nunique() * 6 == len(rq1)
    assert results_view.counts_by_label(grid) == {
        lab: int((rq1["label"] == lab).sum()) for lab in ("WIN", "LOSS", "TIE")}


def test_top_method_per_k_equals_pandas():
    desc = pd.read_csv(ANALYSIS / "descriptive.csv")
    top = facts.top_method_per_k(desc, "cifar_r18", 3000)
    d = desc[(desc.dataset == "cifar_r18") & (desc.method != "all")
             & (desc.budget.isna() | (desc.budget == 3000))]
    expected = d.loc[d.groupby("k")["mean"].idxmax(), "method"].tolist()
    assert top["method"].tolist() == expected


@pytest.mark.parametrize("dataset,method,k,budget,clf", [
    ("cifar_r18", "firefly", 16, 800, "knn"), ("cifar_r18", "anova", 64, None, "lr"),
    ("digits", "pso", 32, 3000, "knn")])
def test_break_even_calculator_equals_pandas(dataset, method, k, budget, clf):
    cost = pd.read_csv(ANALYSIS / "rq2_cost.csv")
    timing = pd.read_csv(ANALYSIS / "timing_idle.csv")
    factor = facts.contention_factor()
    r = cost_view.calculator(cost, timing, factor, dataset, method, k, budget, clf, 10_000)
    assert r["available"]
    c = cost[(cost.dataset == dataset) & (cost.method == method) & (cost.k == k)
             & cost.mask_type.isin(["raw", "-"])]
    c = c[c.budget.isna()] if budget is None else c[c.budget == budget]
    t = timing[(timing.dataset == dataset) & (timing.classifier == clf)]
    sel = t[(t.method == method) & (t.k == k)]
    sel = sel[sel.budget.isna()] if budget is None else sel[sel.budget == budget]
    idle = c.search_s_mean.iloc[0] / factor
    saving = (t[t.method == "all"].median_us.iloc[0] - sel.median_us.iloc[0]) * 1e-6
    assert math.isclose(r["search_idle_s"], idle, rel_tol=1e-12)
    assert math.isclose(r["saving_s"], saving, rel_tol=1e-12)
    if saving > 0:
        assert math.isclose(r["break_even"], idle / saving, rel_tol=1e-12)
    else:
        assert r["break_even"] == math.inf


def test_selection_counts():
    counts = results_view.selection_counts(["0,1,2", "1,2,3", ""], 5)
    assert counts.tolist() == [1, 2, 2, 1, 0]


def test_markdown_sections_from_prereg():
    text = (ROOT / "PREREG.md").read_text(encoding="utf-8")
    body = data.md_section(text, "Limitations accepted in advance")
    assert body and "n = 10" in body and "## " not in body
    assert data.md_section(text, "No such heading") is None


def test_every_shipped_trace_badge_is_true():
    index = traces.trace_index()
    assert len(index) > 0
    for file in index["file"]:
        t = traces.load_trace(file)
        assert traces.matches_logged(t), file
        assert len(t["frame_index"]) <= 60


def test_live_search_is_capped_and_deterministic():
    split = search_live.digits_split()
    a = search_live.run(split, k=8, n_pop=6, budget=1000, alpha=0.02, seed=1)
    b = search_live.run(split, k=8, n_pop=6, budget=1000, alpha=0.02, seed=1)
    assert a["budget"] == search_live.MAX_EVALS and a["evals"] <= search_live.MAX_EVALS
    assert np.array_equal(a["best_mask"], b["best_mask"]) and a["best_mask"].sum() == 8
    assert a["holdout_acc"] == b["holdout_acc"]
    assert set(split["search"]).isdisjoint(split["holdout"])
    label, proba = search_live.predict(a, split["X"][split["examples"][0]])
    assert 0 <= label <= 9 and math.isclose(proba.sum(), 1.0)


def test_toy_live_search_runs():
    res = search_live.run(search_live.toy_split(0), k=6, n_pop=6, budget=60, alpha=0.02,
                          seed=0)
    assert res["evals"] <= 60 and res["frames"]


@pytest.mark.parametrize("page", ["03_results.py", "05_stability.py", "07_watch_search.py"])
def test_interactive_pages_switch_dataset(page):
    at = AppTest.from_file(str(ROOT / "app" / "pages" / page), default_timeout=60).run()
    assert not at.exception
    if at.selectbox:            # change the first select box and rerun
        sb = at.selectbox[0]
        if len(sb.options) > 1:
            sb.select_index(1).run()
            assert not at.exception, [e.value for e in at.exception]
