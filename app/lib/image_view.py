"""'Try an image' logic: per-method predictions with benchmark accuracy beside them."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.lib.methods import display
from app.lib.models import DemoModel, model_key, predict_proba

METHOD_ORDER = ("firefly", "pso", "firefly_seeded", "pso_seeded", "consensus",
                "anova", "mi", "l1", "rf", "pca", "all")
BUDGET = 800      # demo models use the seed-0 budget-800 masks (B1 item 3)


def benchmark_accuracy(desc: pd.DataFrame, method: str, k: int, d: int) -> float | None:
    """Mean test accuracy over seeds from descriptive.csv (None if not benchmarked)."""
    c = desc[desc["dataset"] == "cifar_r18"]
    if method == "all":
        row = c[(c["method"] == "all") & (c["k"] == d)]
    elif method in ("firefly", "pso", "firefly_seeded", "pso_seeded"):
        row = c[(c["method"] == method) & (c["k"] == k) & (c["budget"] == BUDGET)]
    else:
        row = c[(c["method"] == method) & (c["k"] == k) & c["budget"].isna()]
    return float(row["mean"].iloc[0]) if len(row) else None


def predictions(models: dict[str, DemoModel], x_scaled: np.ndarray, k: int,
                classes: list[str], desc: pd.DataFrame, d: int) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        key = model_key(method, d if method == "all" else k)
        if key not in models:
            continue
        m = models[key]
        t0 = time.perf_counter()
        p = predict_proba(m, x_scaled)[0]
        us = (time.perf_counter() - t0) * 1e6
        top = int(p.argmax())
        rows.append({"method": display(method), "key": method,
                     "features used": (str(m.k) if method != "pca"
                                       else f"{m.k} (projection of all {d})"),
                     "prediction": classes[top], "probability": float(p[top]),
                     "benchmark accuracy": benchmark_accuracy(desc, method, k, d),
                     "prediction time (µs)": round(us, 1)})
    return pd.DataFrame(rows)


def agreement_sentence(table: pd.DataFrame, true_label: str | None) -> str:
    counts = table["prediction"].value_counts()
    text = (f"{int(counts.iloc[0])} of {len(table)} methods predict '{counts.index[0]}'"
            + (f" ({len(counts)} different answers)" if len(counts) > 1 else ""))
    if true_label is not None:
        right = int((table["prediction"] == true_label).sum())
        text += f"; {right} of {len(table)} match the label '{true_label}'"
    return text + "."
