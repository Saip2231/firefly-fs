"""Tests for the legacy dashboard prototype (app_legacy/, moved out of app/ in
Stage B0 of DASHBOARD.md; the new app is tested in test_dashboard_b0.py).
Streamlit is not needed: the data logic lives in app_legacy/dashboard_data.py
and streamlit_app.py is checked statically.

(a) the held-out split is named only inside dashboard_data._score_on_test,
    which goes through src.evaluate.evaluate; streamlit_app.py never loads
    data itself
(b) playground: hard cap of 300 evaluations, history / evaluation counts
    consistent
(c) explorer tables: missing meta file handled, PCA labelled as projection,
    non-LR classifiers exclude baselines
(d) every chart in streamlit_app.py has a takeaway sentence
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app_legacy"
sys.path.insert(0, str(APP))
import dashboard_data as dd  # noqa: E402


def _names(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tree, [(n.id if isinstance(n, ast.Name) else n.arg, n.lineno)
                  for n in ast.walk(tree)
                  if isinstance(n, (ast.Name, ast.arg))]


# (a) -----------------------------------------------------------------------

def test_test_split_only_in_score_on_test():
    tree, names = _names(APP / "dashboard_data.py")
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_score_on_test")
    hits = [ln for name, ln in names if name in {"Xte", "yte"}]
    assert hits and all(fn.lineno <= ln <= fn.end_lineno for ln in hits)
    imports = {(n.module, a.name) for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("src.evaluate", "evaluate") in imports
    src = (APP / "dashboard_data.py").read_text(encoding="utf-8")
    for bad in ("data[1]", "data[3]"):
        assert bad not in src


def test_streamlit_app_does_no_data_access_itself():
    src = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    _, names = _names(APP / "streamlit_app.py")
    assert not {"Xte", "yte"} & {n for n, _ in names}
    for bad in ("load_dataset", "read_csv", "open(", "np.load"):
        assert bad not in src, bad


# (b) -----------------------------------------------------------------------

def test_playground_respects_budget_and_history():
    r = dd.run_playground("toy", k=5, budget=60, n_fireflies=8, seed=0)
    assert r["budget"] == 60 and r["evals_used"] <= 60
    assert int(r["mask"].sum()) == 5
    assert len(r["history"]) == len(r["evals_at"]) > 0
    assert r["evals_at"] == sorted(r["evals_at"]) and r["evals_at"][-1] == r["evals_used"]
    assert all(r["history"][i] <= r["history"][i - 1] for i in range(1, len(r["history"])))
    assert r["init_best"] >= r["best_fit"]
    assert 0.0 <= r["test_acc"] <= 1.0


def test_playground_hard_cap():
    r = dd.run_playground("toy", k=5, budget=1000, n_fireflies=8, seed=1)
    assert r["budget"] == dd.BUDGET_CAP == 300
    assert r["evals_used"] <= 300
    src = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "dd.BUDGET_CAP" in src                     # slider max comes from the cap


# (c) -----------------------------------------------------------------------

def _base():
    return pd.DataFrame(dict(
        dataset="digits", method=["ANOVA F", "ANOVA F", "PCA", "PCA", "All Features"],
        seed=[0, 1, 0, 1, 0], k=[16, 16, 16, 16, 64], acc=[0.90, 0.92, 0.95, 0.94, 0.97]))


def _meta():
    rows = []
    for s in (0, 1):
        for mt, acc in (("raw", 0.91 + s / 100), ("rerank", 0.80)):
            rows.append(dict(method="firefly", budget=800, seed=s, k=16, mask_type=mt,
                             acc_lr=acc, acc_knn=0.5, acc_svm=0.6))
    return pd.DataFrame(rows)


def test_accuracy_table_without_meta():
    t = dd.accuracy_table(_base(), None)
    assert set(t.method) == {"ANOVA F", "PCA (projection)", "All Features"}
    s = dd.summarise(t)
    pca = s[s.method == "PCA (projection)"].iloc[0]
    assert np.isclose(pca["mean"], 0.945) and pca["n"] == 2
    assert "All Features" not in dd.takeaway_acc_vs_k(s).split(" at ")[0]
    assert "PCA (projection)" in dd.takeaway_acc_vs_k(s)


def test_accuracy_table_with_meta_and_classifier():
    t = dd.accuracy_table(_base(), _meta(), budget=800, mask_type="raw")
    ff = t[t.method == "firefly"]
    assert len(ff) == 2 and np.allclose(sorted(ff.acc), [0.91, 0.92])
    knn = dd.accuracy_table(_base(), _meta(), budget=800, metric="acc_knn")
    assert set(knn.source) == {"metaheuristic"} and np.allclose(knn.acc, 0.5)
    assert dd.accuracy_table(None, None).empty
    assert "seed-to-seed" in dd.takeaway_spread(t, 16)


def test_file_status_lists_all_four_files():
    fs = dd.file_status()
    assert len(fs) == 4 and set(fs.status) <= {"present", "not yet created"}


# (d) -----------------------------------------------------------------------

def test_every_chart_has_a_takeaway():
    """Each st.plotly_chart call is followed by a takeaway(...) call before
    the next chart (or the end of the file)."""
    src = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    body = src.split("def takeaway(", 1)[1]
    charts = [i for i in range(len(body)) if body.startswith("st.plotly_chart(", i)]
    assert charts
    for a, b in zip(charts, charts[1:] + [len(body)]):
        assert "takeaway(" in body[a:b], f"chart at offset {a} has no takeaway"
