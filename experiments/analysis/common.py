"""Shared names, paths and loaders for the analysis stages (ANALYSIS.md section 3).

The ONE place that maps the labels used in the result CSVs to canonical
method names.  Baseline CSV labels (as written by run_baselines.py):

    "All Features" -> all      "Random"        -> random
    "ANOVA F"      -> anova    "Mutual Info"   -> mi
    "L1 Logistic"  -> l1       "RF Importance" -> rf
    "PCA"          -> pca      (projection, not a selector)

Meta CSV methods are already canonical: firefly, pso, firefly_seeded,
pso_seeded (firefly_levy if it was run).

Loaders read CSVs only, never the datasets, and return tidy frames with a
canonical `method` column, the original `method_label`, and `idx` parsed to
a frozenset of ints (empty for PCA and all).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS_DIR = RESULTS / "analysis"
FIGURES = ROOT / "figures"
PREREG = ROOT / "PREREG.md"

DATASETS = ("digits", "cifar_r18")
SEEDS = tuple(range(10))
BUDGETS = (800, 3000)
MASK_TYPES = ("raw", "rerank")

BASELINE_LABELS = {
    "All Features": "all",
    "Random": "random",
    "ANOVA F": "anova",
    "Mutual Info": "mi",
    "L1 Logistic": "l1",
    "RF Importance": "rf",
    "PCA": "pca",
}
META_METHODS = ("firefly", "pso", "firefly_seeded", "pso_seeded", "firefly_levy")
NO_IDX_METHODS = ("all", "pca")                 # no feature subset to check
COMPARATORS = ("anova", "mi", "l1", "rf", "pca", "pso")   # PREREG primary family
METHOD_UNDER_TEST = "firefly"

DISPLAY = {
    "all": "All features",
    "random": "Random",
    "anova": "ANOVA F",
    "mi": "Mutual information",
    "l1": "L1 logistic",
    "rf": "RF importance",
    "pca": "PCA (projection, not a selector)",
    "firefly": "Firefly",
    "pso": "PSO (random-key top-k, stall-swap)",
    "firefly_seeded": "Firefly, filter-seeded",
    "pso_seeded": "PSO, filter-seeded",
    "firefly_levy": "Firefly, long-jump mutation",
}
DATASET_DISPLAY = {
    "digits": "Digits (64-d)",
    "cifar_r18": "CIFAR-10 subset, ResNet-18 embeddings (512-d)",
}

META_ACC = ("acc_lr", "acc_knn", "acc_svm")


def parse_idx(value) -> frozenset[int]:
    """'2,13,19' -> frozenset({2, 13, 19}); NaN / '' -> frozenset()."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return frozenset()
    text = str(value).strip()
    if not text:
        return frozenset()
    return frozenset(int(t) for t in text.split(","))


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path, dtype={"idx": str})


def tidy_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical method names + parsed idx for a raw baseline frame."""
    unknown = sorted(set(df["method"]) - set(BASELINE_LABELS))
    if unknown:
        raise ValueError(f"unknown baseline method label(s) {unknown}; "
                         f"add them to BASELINE_LABELS in {Path(__file__).name}")
    out = df.copy()
    out["method_label"] = out["method"]
    out["method"] = out["method"].map(BASELINE_LABELS)
    out["idx"] = out["idx"].map(parse_idx) if "idx" in out else [frozenset()] * len(out)
    return out


def tidy_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Parsed idx for a raw meta frame (method names are already canonical)."""
    out = df.copy()
    out["method_label"] = out["method"]
    out["idx"] = out["idx"].map(parse_idx)
    return out


def load_baselines(dataset: str, path: Path | None = None) -> pd.DataFrame:
    return tidy_baselines(_read(path or RESULTS / f"baselines_{dataset}.csv"))


def load_meta(dataset: str, path: Path | None = None) -> pd.DataFrame:
    return tidy_meta(_read(path or RESULTS / f"meta_{dataset}.csv"))


def load_split_hashes(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RESULTS / "split_hashes.csv")


def dimension(baselines: pd.DataFrame) -> int:
    """d = k of the 'all' rows."""
    d = baselines.loc[baselines.method == "all", "k"].unique()
    if len(d) != 1:
        raise ValueError(f"expected one 'all' k, got {list(d)}")
    return int(d[0])


def selection_ks(baselines: pd.DataFrame) -> list[int]:
    """k values usable for selection (k < d), read from the baseline file."""
    d = dimension(baselines)
    return sorted(int(k) for k in baselines.loc[baselines.method != "all", "k"].unique() if k < d)


def prereg_family_count(path: Path = PREREG) -> int | None:
    """The number of families PREREG.md states, e.g. '(12 families: ...'."""
    m = re.search(r"\((\d+) families", path.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None
