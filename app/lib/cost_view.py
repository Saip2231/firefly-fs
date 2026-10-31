"""RQ2 cost logic: the break-even calculator (exploratory).

Search seconds come from rq2_cost.csv (A3, measured under 15-way parallel
load); an idle estimate divides them by the A4 contention factor.  Per-
prediction times are A4's idle single-sample medians (training rows, seed 0).
Break-even = idle search time / per-prediction saving versus all features;
"never" when the saving is zero or negative.
"""

from __future__ import annotations

import math

import pandas as pd


def _row(df: pd.DataFrame, method: str, k: int, budget: int | None) -> pd.DataFrame:
    d = df[(df["method"] == method) & (df["k"] == k)]
    return d[d["budget"].isna()] if budget is None else d[d["budget"] == budget]


def calculator(cost: pd.DataFrame, timing: pd.DataFrame, factor: float, dataset: str,
               method: str, k: int, budget: int | None, classifier: str,
               n_predictions: int) -> dict:
    """All numbers for one calculator setting (None where data is missing)."""
    c = cost[cost["dataset"] == dataset]
    if "mask_type" in c:
        c = c[c["mask_type"].isin(["raw", "-"]) | c["mask_type"].isna()]
    crow = _row(c, method, k, budget)
    t = timing[(timing["dataset"] == dataset) & (timing["classifier"] == classifier)]
    trow = _row(t, method, k, budget)
    tall = t[t["method"] == "all"]
    if crow.empty or trow.empty or tall.empty:
        return dict(available=False)
    search_logged = float(crow["search_s_mean"].iloc[0])
    search_idle = search_logged / factor if factor else search_logged
    all_us = float(tall["median_us"].iloc[0])
    sel_us = float(trow["median_us"].iloc[0])
    t_all, t_sel = all_us * 1e-6, sel_us * 1e-6
    saving = (all_us - sel_us) * 1e-6          # subtract in µs, then convert
    break_even = search_idle / saving if saving > 0 else math.inf
    return dict(available=True, search_logged_s=search_logged, search_idle_s=search_idle,
                t_all_s=t_all, t_selected_s=t_sel, saving_s=saving, break_even=break_even,
                net_s=n_predictions * saving - search_idle, n_predictions=n_predictions)


def sentence(r: dict, method_name: str, classifier_name: str) -> str:
    from app.lib.stats_view import format_break_even  # noqa: PLC0415

    if not r["available"]:
        return "No timing for this combination."
    if r["saving_s"] <= 0:
        return (f"With {classifier_name}, {method_name} saves nothing per prediction "
                f"({r['t_selected_s'] * 1e6:.1f} vs {r['t_all_s'] * 1e6:.1f} µs with all "
                "features), so the search never pays for itself.")
    return (f"With {classifier_name}, {method_name} saves {r['saving_s'] * 1e6:.1f} µs per "
            f"prediction; the estimated idle search time of {r['search_idle_s']:.1f} s is "
            f"repaid after {format_break_even(r['break_even'])} predictions.")
