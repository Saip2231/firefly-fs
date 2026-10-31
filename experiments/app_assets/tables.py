"""Table copies for the app (B1 item 1).

Analysis tables and reports are copied byte for byte (so they can be checked
against the manifest); the raw result files are slimmed to the columns the
dashboard reads.
"""

from __future__ import annotations

import shutil

import pandas as pd

from .common import ANALYSIS, RESULTS, TABLES

ANALYSIS_FILES = ("rq1_tests.csv", "descriptive.csv", "rq2_cost.csv", "timing_idle.csv",
                  "contention.csv", "rq3_stability.csv", "rq3_feature_frequency.csv",
                  "rq4_transfer.csv", "rq4_gaps.csv", "ablations.csv",
                  "integrity_report.md", "analysis_manifest.md", "contention_factor.md",
                  "rq1_summary.md")
SLIM_COLUMNS = {
    "meta": ["dataset", "method", "budget", "seed", "k", "mask_type", "acc_lr", "acc_knn",
             "acc_svm", "n_feat", "search_s", "rerank_s", "infer_s", "evals_used",
             "rerank_fits", "val_fitness", "idx", "config_hash"],
    "baselines": ["dataset", "method", "seed", "k", "acc", "n_feat", "search_s",
                  "infer_s", "idx"],
    "baselines_transfer": ["dataset", "method", "seed", "k", "acc_knn", "acc_svm", "idx"],
}
DATASETS = ("digits", "cifar_r18")


def build() -> dict:
    """Copy / slim every table; returns {"copied": [...], "missing": [...]}."""
    TABLES.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in ANALYSIS_FILES:
        src = ANALYSIS / name
        if src.exists():
            shutil.copyfile(src, TABLES / name)
            copied.append(name)
        else:
            missing.append(f"results/analysis/{name}")
    for kind, cols in SLIM_COLUMNS.items():
        for ds in DATASETS:
            src = RESULTS / f"{kind}_{ds}.csv"
            if not src.exists():
                missing.append(f"results/{src.name}")
                continue
            df = pd.read_csv(src, dtype={"idx": str})
            df[[c for c in cols if c in df.columns]].to_csv(TABLES / src.name, index=False)
            copied.append(src.name)
    return dict(copied=copied, missing=missing)
