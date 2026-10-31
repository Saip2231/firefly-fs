"""Manifest-driven, read-only loaders (DASHBOARD.md section 4).

Analysis tables are discovered from results/analysis/analysis_manifest.md
plus a glob of results/analysis/*.csv, read once (cached) and checked
against the required columns below.  Every problem becomes a DataError with
a one-line message for the UI.  This module reads CSV and markdown files
only; it never loads a dataset split.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
ASSETS = ROOT / "app" / "assets"
MANIFEST = ANALYSIS / "analysis_manifest.md"
INTEGRITY = ANALYSIS / "integrity_report.md"
DATASETS = ("digits", "cifar_r18")

# Required columns (extra columns are allowed).  Tables without an entry
# are loaded without a column check.
ANALYSIS_SCHEMAS: dict[str, tuple[str, ...]] = {
    "rq1_tests": ("dataset", "k", "budget", "comparator", "n", "mean_firefly",
                  "mean_comparator", "mean_diff_pts", "std_diff_pts",
                  "nonzero_pairs", "p_raw", "p_holm", "label"),
    "descriptive": ("dataset", "method", "display", "source", "k", "budget",
                    "metric", "n", "mean", "std"),
    "rq2_cost": ("dataset", "method", "k", "budget", "mask_type", "n", "acc_mean",
                 "search_s_mean", "infer_s_mean", "total_fits_mean", "break_even"),
    "timing_idle": ("dataset", "method", "budget", "k", "seed", "classifier",
                    "n_features", "median_us", "p25_us", "p75_us"),
    "rq3_stability": ("dataset", "method", "display", "k", "budget", "mask_type",
                      "jaccard_mean", "jaccard_std", "random_empirical",
                      "random_analytic"),
    "rq4_transfer": ("dataset", "k", "budget", "method", "comparator", "classifier",
                     "n", "mean_method", "mean_comparator", "mean_diff_pts",
                     "std_diff_pts"),
    "ablations": ("ablation", "dataset", "method", "k", "budget", "variant",
                  "reference", "n", "mean_variant", "mean_reference",
                  "mean_diff_pts", "std_diff_pts", "p_raw", "label"),
}

RAW_SCHEMAS: dict[str, tuple[str, ...]] = {
    "baselines": ("dataset", "method", "seed", "k", "acc", "idx"),
    "meta": ("dataset", "method", "budget", "seed", "k", "mask_type", "acc_lr",
             "acc_knn", "acc_svm", "search_s", "evals_used", "idx", "config_hash"),
    "baselines_transfer": ("dataset", "method", "seed", "k", "acc_knn", "acc_svm", "idx"),
}

_OUTPUT_ROW = re.compile(r"^\|\s*`(results/analysis/[^`]+)`\s*\|")


class DataError(RuntimeError):
    """A data file is missing, unreadable or lacks required columns."""


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def manifest_outputs(manifest: Path = MANIFEST) -> list[str]:
    """Repo-relative paths of the analysis outputs listed in the manifest."""
    if not manifest.exists():
        return []
    found = (_OUTPUT_ROW.match(line) for line in
             manifest.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(m.group(1) for m in found if m))


def discover_tables(analysis_dir: Path = ANALYSIS, manifest: Path = MANIFEST,
                    root: Path = ROOT) -> dict[str, Path]:
    """Table name (file stem) -> path, manifest entries first, then the glob."""
    tables: dict[str, Path] = {}
    for rel in manifest_outputs(manifest):
        path = root / rel
        if path.suffix == ".csv":
            tables[path.stem] = path
    for path in sorted(analysis_dir.glob("*.csv")):
        tables.setdefault(path.stem, path)
    return tables


def validate_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    """Raise DataError naming every required column that is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{label}: missing required column(s) {', '.join(missing)}. "
                        "Rebuild it with `python -m experiments.analyze`.")


def read_csv_checked(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read one CSV and validate it; all failures become DataError."""
    label = _rel(path)
    if not path.exists():
        raise DataError(f"{label}: file not found.")
    try:
        df = pd.read_csv(path, dtype={"idx": str})
    except Exception as exc:  # noqa: BLE001  (any parser failure is a data problem)
        raise DataError(f"{label}: could not be read ({type(exc).__name__}).") from exc
    if df.empty:
        raise DataError(f"{label}: file has no rows.")
    validate_columns(df, required, label)
    return df


@st.cache_data(show_spinner=False, max_entries=64)
def load_analysis(name: str) -> pd.DataFrame:
    """One analysis table by name, e.g. "rq1_tests"."""
    tables = discover_tables()
    if name not in tables:
        raise DataError(f"results/analysis/{name}.csv is not in the analysis "
                        "manifest or folder.")
    return read_csv_checked(tables[name], ANALYSIS_SCHEMAS.get(name, ()))


def load_optional(name: str) -> pd.DataFrame | None:
    """Like load_analysis, but a table that does not exist gives None.
    A table that exists but is broken still raises DataError."""
    if name not in discover_tables():
        return None
    return load_analysis(name)


@st.cache_data(show_spinner=False, max_entries=16)
def load_raw(kind: str, dataset: str) -> pd.DataFrame:
    """results/<kind>_<dataset>.csv (kind: baselines, meta, baselines_transfer)."""
    if kind not in RAW_SCHEMAS:
        raise DataError(f"unknown raw table kind {kind!r}.")
    if dataset not in DATASETS:
        raise DataError(f"unknown dataset {dataset!r}.")
    return read_csv_checked(RESULTS / f"{kind}_{dataset}.csv", RAW_SCHEMAS[kind])


def primary_meta(dataset: str) -> pd.DataFrame:
    """Meta rows used by the primary analysis (mask_type == "raw")."""
    meta = load_raw("meta", dataset)
    return meta[meta["mask_type"] == "raw"]


@st.cache_data(show_spinner=False, max_entries=16)
def read_text(path: Path) -> str | None:
    """A markdown/text file from the repo, or None if it does not exist."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def md_section(text: str | None, heading: str) -> str | None:
    """Body of the '## heading' section of a markdown document (None if absent)."""
    if not text:
        return None
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


def parse_idx(value) -> frozenset[int]:
    """'2,13,19' -> frozenset({2, 13, 19}); empty / NaN -> frozenset()."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return frozenset()
    text = str(value).strip()
    return frozenset(int(t) for t in text.split(",")) if text else frozenset()


@st.cache_data(show_spinner=False, max_entries=4)
def per_seed(dataset: str) -> pd.DataFrame:
    """One row per (method, k, budget, seed) used in the primary analysis.

    Baselines (canonical names, no budget, accuracy column `acc`) and meta
    rows with mask_type == "raw" (`acc_lr`), as columns method, k, budget
    (NaN for baselines), seed, acc_lr, idx, source.
    """
    from app.lib.methods import BASELINE_LABELS  # noqa: PLC0415

    base = load_raw("baselines", dataset)
    unknown = sorted(set(base["method"]) - set(BASELINE_LABELS))
    if unknown:
        raise DataError(f"baselines_{dataset}.csv: unknown method label(s) {unknown}.")
    b = pd.DataFrame(dict(method=base["method"].map(BASELINE_LABELS), k=base["k"],
                          budget=float("nan"), seed=base["seed"], acc_lr=base["acc"],
                          idx=base["idx"], source="baseline"))
    m = primary_meta(dataset)
    m = pd.DataFrame(dict(method=m["method"], k=m["k"], budget=m["budget"].astype(float),
                          seed=m["seed"], acc_lr=m["acc_lr"], idx=m["idx"],
                          source="metaheuristic"))
    return pd.concat([b, m], ignore_index=True)
