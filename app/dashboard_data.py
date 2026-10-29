"""Data helpers for app/streamlit_app.py (no Streamlit import, so tests can
use them without Streamlit installed).

Reads only results/*.csv and calls src/ functions.  The test split is used
in exactly one place, _score_on_test, through src.evaluate.evaluate (the same
function the runners use); tests/test_app.py checks this statically.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifiers import final_clf  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.firefly import binary_firefly, make_fitness  # noqa: E402
from src.init_pop import top_k_mask  # noqa: E402

DATASETS = ["digits", "cifar_r18"]
BUDGET_CAP = 300                       # hard cap for the live playground
PROJECTION_METHODS = {"PCA"}           # projections, not selections
REFERENCE_METHODS = {"All Features"}


# ---------------------------------------------------------------------------
# CSV loading and summaries
# ---------------------------------------------------------------------------

def read_results_csv(name: str) -> pd.DataFrame | None:
    """results/<name> as a DataFrame, or None if the file does not exist."""
    path = RESULTS / name
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"idx": str})


def file_status() -> pd.DataFrame:
    """Which result files exist and how many rows they have."""
    rows = []
    for ds in DATASETS:
        for kind in ("baselines", "meta"):
            name = f"{kind}_{ds}.csv"
            df = read_results_csv(name)
            rows.append(dict(file=f"results/{name}",
                             status="present" if df is not None else "not yet created",
                             rows=0 if df is None else len(df)))
    return pd.DataFrame(rows)


def accuracy_table(baselines: pd.DataFrame | None, meta: pd.DataFrame | None,
                   budget: int | None = None, mask_type: str = "raw",
                   metric: str = "acc_lr") -> pd.DataFrame:
    """Per-seed accuracy rows in one tidy table: method, k, seed, acc, source.

    Baselines only have logistic-regression accuracy ("acc"), so they are
    included only for metric == "acc_lr".  "All Features" is kept (k = d) as
    a reference; PCA is labelled "PCA (projection)".
    """
    parts = []
    if baselines is not None and metric == "acc_lr":
        b = baselines[["method", "k", "seed", "acc"]].copy()
        b["source"] = "baseline"
        b.loc[b.method.isin(PROJECTION_METHODS), "method"] = "PCA (projection)"
        parts.append(b)
    if meta is not None and len(meta):
        m = meta[meta.mask_type == mask_type]
        if budget is not None:
            m = m[m.budget == budget]
        m = m[["method", "k", "seed", metric]].rename(columns={metric: "acc"})
        m["source"] = "metaheuristic"
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["method", "k", "seed", "acc", "source"])
    return pd.concat(parts, ignore_index=True)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """mean, std, n of acc per (source, method, k)."""
    if table.empty:
        return pd.DataFrame(columns=["source", "method", "k", "mean", "std", "n"])
    return (table.groupby(["source", "method", "k"], as_index=False)["acc"]
                 .agg(mean="mean", std="std", n="count"))


def takeaway_acc_vs_k(summary: pd.DataFrame) -> str:
    """One plain sentence from the numbers (no significance claim)."""
    s = summary[~summary.method.isin(REFERENCE_METHODS)]
    if s.empty:
        return "No results to summarise yet."
    ks = sorted(s.k.unique())
    k_lo, k_hi = ks[0], ks[-1]
    lo = s[s.k == k_lo].sort_values("mean", ascending=False).iloc[0]
    hi = s[s.k == k_hi].sort_values("mean", ascending=False).iloc[0]
    ref = summary[summary.method.isin(REFERENCE_METHODS)]
    ref_txt = (f"; all {int(ref.k.iloc[0])} features give {ref['mean'].iloc[0]:.3f}"
               if len(ref) else "")
    best = f"Highest mean accuracy is {lo.method} at k={k_lo} ({lo['mean']:.3f})"
    if k_hi != k_lo:
        best += f" and {hi.method} at k={k_hi} ({hi['mean']:.3f})"
    return (f"{best}{ref_txt}. Means over seeds only; see results/rq1_tests.csv "
            f"for the paired tests.")


def takeaway_spread(table: pd.DataFrame, k: int) -> str:
    """Sentence on the seed-to-seed spread at one k."""
    t = table[(table.k == k) & ~table.method.isin(REFERENCE_METHODS)]
    if t.empty:
        return f"No per-seed results at k={k}."
    g = t.groupby("method")["acc"]
    widest = (g.max() - g.min()).sort_values(ascending=False)
    return (f"At k={k} the seed-to-seed range is largest for {widest.index[0]} "
            f"({widest.iloc[0]:.3f}) and smallest for {widest.index[-1]} "
            f"({widest.iloc[-1]:.3f}); differences smaller than these ranges are "
            f"not reliable with 10 seeds.")


# ---------------------------------------------------------------------------
# Live playground
# ---------------------------------------------------------------------------

def _toy(seed: int):
    """Planted toy: 600 rows, 50 features, 5 informative (cols 0-4, shuffle
    off), split 450 search / 150 holdout.  Returns the same 4-tuple layout
    as load_dataset."""
    from sklearn.datasets import make_classification  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    X, y = make_classification(n_samples=600, n_features=50, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=seed)
    a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    return a, b, c, d


def _score_on_test(mask, data) -> float:
    """The ONLY use of the held-out split: src.evaluate.evaluate with the
    shared final classifier, exactly as in the runners."""
    Xtr, Xte, ytr, yte = data
    return float(evaluate(mask, Xtr, ytr, Xte, yte, final_clf())["accuracy"])


def run_playground(source: str, k: int, budget: int, n_fireflies: int, seed: int) -> dict:
    """One capped Firefly run.  budget is clamped to BUDGET_CAP."""
    budget = int(min(max(1, budget), BUDGET_CAP))
    data = load_dataset("digits", seed=seed) if source == "digits" else _toy(seed)
    Xtr, ytr = data[0], data[2]
    d = Xtr.shape[1]
    if not 1 <= k < d:
        raise ValueError(f"k must be in 1..{d - 1}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitness, state = make_fitness(Xtr, ytr, k=k, seed=seed)
        trace: dict = {}
        mask, best_fit, history = binary_firefly(
            fitness, state, d=d, k=k, budget=budget, n_fireflies=n_fireflies,
            seed=seed, trace=trace)
        search_evals = int(state["evals"])      # before the reference fit below
        # Reference only, after the search: may cost one extra fit, which is
        # not part of the search and not counted in evals_used.
        anova = top_k_mask(state["filter_scores"], k)
        anova_fit = float(fitness(anova))
    evals_at = list(np.cumsum([trace["init_evals"]]
                              + [s["new_evals"] for s in trace["sweeps"]])[1:])
    return dict(
        source=source, k=k, budget=budget, n_fireflies=n_fireflies, seed=seed, d=d,
        mask=mask, best_fit=float(best_fit), history=list(history),
        evals_at=[int(e) for e in evals_at], init_best=float(trace["init_best"]),
        init_evals=int(trace["init_evals"]), evals_used=search_evals,
        anova_fit=anova_fit, test_acc=_score_on_test(mask, data),
        anova_test_acc=_score_on_test(anova, data),
        warnings=sorted({w.category.__name__ for w in caught}),
    )
