"""Data and embedding for the animated protocol diagram (Protocol page).

`build_payload()` turns the analysis tables into the JSON object the React
visualizer reads (contract: app/frontend/src/types.ts).  `render()` injects
it into the prebuilt single-file bundle app/assets/pipeline/index.html
(built from app/frontend with `npm run build`; no Node is needed at runtime)
and embeds it with st.components.v1.html.  Every number comes from the
tables; nothing here touches a dataset split.

Dev helper:  python -m app.lib.pipeline_view app/frontend/dev-data.json
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.lib import data
from app.lib.methods import display

BUNDLE = data.ASSETS / "pipeline" / "index.html"
PLACEHOLDER = "__PIPELINE_DATA__"
BASELINES = ("anova", "mi", "l1", "rf", "pca")
METAS = ("firefly", "pso")
SHORT = {"digits": "Digits", "cifar_r18": "CIFAR-10 (ResNet-18)"}
LABEL = {"digits": "Digits (8x8 pixels)",
         "cifar_r18": "CIFAR-10 subset, ResNet-18 embeddings"}
SOURCES = ("results/analysis/rq2_cost.csv", "results/meta_<dataset>.csv (val_fitness)")


def _num(x) -> float | None:
    """Finite float or None (JSON has no NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fitness_weight() -> float:
    """The size weight w in fitness = (1 - w) * error + w * k/d, read from the
    signature of src.firefly.make_fitness (the runner uses the default)."""
    from src.firefly import make_fitness  # noqa: PLC0415
    return float(inspect.signature(make_fitness).parameters["w"].default)


def inner_val_error(val_fitness: pd.Series, k: int, d: int, w: float) -> pd.Series:
    """Invert fitness = (1 - w) * error + w * k/d for exactly-k masks."""
    return (val_fitness - w * k / d) / (1.0 - w)


def _cell(row: pd.Series, inner_err: float | None) -> dict:
    return {
        "n": int(row["n"]),
        "acc_mean": _num(row["acc_mean"]),
        "acc_std": _num(row.get("acc_std")),
        "search_s_mean": _num(row.get("search_s_mean")),
        "evals_used_mean": _num(row.get("evals_used_mean")),
        "inner_val_error_mean": inner_err,
    }


def dataset_block(cost: pd.DataFrame, meta: pd.DataFrame | None, dataset: str,
                  w: float) -> dict | None:
    """One dataset's entry of the payload, from rq2_cost rows (raw masks and
    baselines) plus mean inner-validation error from the raw meta rows."""
    c = cost[(cost["dataset"] == dataset) & cost["mask_type"].isin(["raw", "-"])]
    if c.empty:
        return None
    all_rows = c[c["method"] == "all"]
    d = int(all_rows["k"].iloc[0]) if len(all_rows) else int(c["k"].max())
    sel = c[c["method"].isin(BASELINES + METAS)]
    k_values = sorted(int(k) for k in sel["k"].dropna().unique())
    budgets = sorted(int(b) for b in sel.loc[sel["method"].isin(METAS), "budget"].dropna().unique())

    inner: dict[tuple[str, int, int], float] = {}
    if meta is not None and "val_fitness" in meta.columns:
        m = meta[meta["method"].isin(METAS)]
        for (method, k, budget), g in m.groupby(["method", "k", "budget"]):
            err = inner_val_error(g["val_fitness"].astype(float), int(k), d, w)
            inner[(method, int(k), int(budget))] = _num(err.mean())

    results: dict[str, dict] = {}
    for _, r in sel.iterrows():
        k = int(r["k"])
        if r["method"] in METAS:
            b = int(r["budget"])
            results[f"{r['method']}|{k}|{b}"] = _cell(r, inner.get((r["method"], k, b)))
        else:
            results[f"{r['method']}|{k}|"] = _cell(r, None)

    all_features = None
    if len(all_rows):
        a = all_rows.iloc[0]
        all_features = {"n": int(a["n"]), "acc_mean": _num(a["acc_mean"]),
                        "acc_std": _num(a.get("acc_std"))}
    return {"label": LABEL.get(dataset, dataset), "short": SHORT.get(dataset, dataset),
            "d": d, "k_values": k_values, "budgets": budgets,
            "all_features": all_features, "results": results}


def _search_note(cost: pd.DataFrame) -> str:
    if "search_s_note" in cost.columns:
        notes = cost["search_s_note"].dropna().unique()
        if len(notes):
            return str(notes[0])
    return "search_s measured under parallel load; not idle-machine time"


@st.cache_data(show_spinner=False)
def build_payload() -> dict:
    """The full JSON payload for the visualizer (see app/frontend/src/types.ts)."""
    cost = data.load_analysis("rq2_cost")
    w = fitness_weight()
    datasets = {}
    for ds in data.DATASETS:
        try:
            meta = data.primary_meta(ds)
        except data.DataError:
            meta = None
        block = dataset_block(cost, meta, ds, w)
        if block:
            datasets[ds] = block
    return {
        "sources": list(SOURCES),
        "methods": {m: display(m) for m in BASELINES + METAS},
        "search_s_note": _search_note(cost),
        "inner_val_note": (f"Inner-val error = mean over seeds of the returned mask's "
                           f"inner-validation error, recovered from val_fitness = "
                           f"(1 - w)·error + w·k/d with w = {w:g}."),
        "datasets": datasets,
    }


def inject(html: str, payload: dict) -> str:
    """Put the payload into the bundle's <script type="application/json"> tag."""
    if PLACEHOLDER not in html:
        raise data.DataError(f"{BUNDLE.name}: data placeholder missing; rebuild the bundle.")
    blob = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return html.replace(PLACEHOLDER, blob.replace("</", "<\\/"), 1)


def bundle_html(path: Path = BUNDLE) -> str | None:
    return data.read_text(path)


def render(height: int = 1000) -> bool:
    """Embed the visualizer; False (nothing drawn) when the bundle is not built."""
    html = bundle_html()
    if html is None:
        return False
    import streamlit.components.v1 as components  # noqa: PLC0415

    from app.lib import theme  # noqa: PLC0415
    payload = {**build_payload(), "theme": theme.current_mode()}
    components.html(inject(html, payload), height=height, scrolling=True)
    return True


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("app/frontend/dev-data.json")
    out.write_text(json.dumps(build_payload(), indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out}")
