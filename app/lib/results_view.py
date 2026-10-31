"""RQ1 explorer logic: the label grid and per-family paired differences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.lib.data import DataError
from app.lib.methods import COMPARATORS, display


def family_name(dataset: str, k: int, budget: int) -> str:
    return f"{dataset} k={int(k)} b={int(budget)}"


def grid_table(rq1: pd.DataFrame) -> pd.DataFrame:
    """rq1_tests rows with a family name and display comparator, in a fixed
    order (dataset, budget, k; comparators as in PREREG)."""
    g = rq1.copy()
    g["family"] = [family_name(r.dataset, r.k, r.budget) for r in g.itertuples()]
    g["comparator_name"] = g["comparator"].map(display)
    g["_c"] = g["comparator"].map({c: i for i, c in enumerate(COMPARATORS)})
    return (g.sort_values(["dataset", "budget", "k", "_c"]).drop(columns="_c")
             .reset_index(drop=True))


def paired_diffs(seeds: pd.DataFrame, k: int, budget: int, comparator: str) -> pd.DataFrame:
    """Per-seed Firefly (raw, this budget) minus comparator accuracy, in points.

    Baseline comparators have no budget; PSO uses the same budget.
    """
    ff = seeds[(seeds["method"] == "firefly") & (seeds["k"] == k)
               & (seeds["budget"] == budget)][["seed", "acc_lr"]]
    comp = seeds[(seeds["method"] == comparator) & (seeds["k"] == k)]
    comp = comp[comp["budget"] == budget] if comparator == "pso" else comp[comp["budget"].isna()]
    comp = comp[["seed", "acc_lr"]]
    if sorted(ff["seed"]) != sorted(comp["seed"]) or ff.empty:
        raise DataError(f"seeds do not pair for k={k}, budget={budget}, {comparator}.")
    out = ff.merge(comp, on="seed", suffixes=("_firefly", "_comparator")).sort_values("seed")
    out["diff_pts"] = (out["acc_lr_firefly"] - out["acc_lr_comparator"]) * 100
    return out.reset_index(drop=True)


def panel_sentence(row: pd.Series, diffs: pd.DataFrame) -> str:
    """One computed sentence for the family panel."""
    from app.lib.stats_view import format_p, label_text  # noqa: PLC0415

    higher = int((diffs["diff_pts"] > 0).sum())
    lower = int((diffs["diff_pts"] < 0).sum())
    return (f"{label_text(row['label'])}: Firefly minus {display(row['comparator'])} is "
            f"{row['mean_diff_pts']:+.2f} pts on average (std {row['std_diff_pts']:.2f}); "
            f"Firefly is higher on {higher} and lower on {lower} of {len(diffs)} seeds; "
            f"raw p {format_p(row['p_raw'])}, Holm-adjusted p {format_p(row['p_holm'])}, "
            f"n={int(row['n'])}.")


def explorer_takeaway(desc: pd.DataFrame) -> str:
    """Which method has the highest mean at each k in the filtered view."""
    d = desc[desc["method"] != "all"]
    if d.empty:
        return "No methods selected."
    top = d.loc[d.groupby("k")["mean"].idxmax()].sort_values("k")
    parts = [f"k={int(r.k)}: {display(r.method)} ({r.mean:.3f})" for r in top.itertuples()]
    return ("Highest mean accuracy among the shown methods, by k: " + "; ".join(parts)
            + ". Means over seeds only; the grid below holds the paired tests.")


def selection_counts(idx_values, d: int) -> np.ndarray:
    """How many of the given idx strings select each of the d features."""
    from app.lib.data import parse_idx  # noqa: PLC0415

    counts = np.zeros(d, dtype=int)
    for value in idx_values:
        for f in parse_idx(value):
            counts[f] += 1
    return counts


def counts_by_label(grid: pd.DataFrame) -> dict[str, int]:
    return {lab: int(np.sum(grid["label"] == lab)) for lab in ("WIN", "LOSS", "TIE")}
