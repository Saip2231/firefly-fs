"""Replay traces (built by experiments/build_app_assets.py): index, loading,
and the "final mask matches the logged run" badge, recomputed here from the
trace and results/meta_<dataset>.csv (not copied from the build output)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import data

TRACES = data.ASSETS / "traces"


@st.cache_data(show_spinner=False)
def trace_index() -> pd.DataFrame:
    rows = []
    for path in sorted(TRACES.glob("*.npz")):
        with np.load(path) as z:
            rows.append(dict(file=path.name, dataset=str(z["dataset"]), method=str(z["method"]),
                             k=int(z["k"]), budget=int(z["budget"]), seed=int(z["seed"])))
    return pd.DataFrame(rows, columns=["file", "dataset", "method", "k", "budget", "seed"])


@st.cache_data(show_spinner=False, max_entries=32)
def load_trace(file: str) -> dict:
    with np.load(TRACES / file) as z:
        return {key: z[key] for key in z.files}


def matches_logged(trace: dict) -> bool:
    """True when the trace's final mask selects exactly the CSV's logged idx."""
    meta = data.primary_meta(str(trace["dataset"]))
    row = meta[(meta["method"] == str(trace["method"])) & (meta["k"] == int(trace["k"]))
               & (meta["budget"] == int(trace["budget"])) & (meta["seed"] == int(trace["seed"]))]
    if len(row) != 1:
        return False
    logged = sorted(data.parse_idx(row["idx"].iloc[0]))
    return np.flatnonzero(trace["final_mask"]).tolist() == logged


def curve(trace: dict) -> pd.DataFrame:
    return pd.DataFrame(dict(evals=trace["curve_evals"], best=trace["curve_best"]))
