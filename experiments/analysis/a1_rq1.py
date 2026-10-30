"""Stage A1: RQ1 confirmatory tests, exactly as pre-registered (PREREG.md).

Primary metric: test accuracy of the final logistic regression, `acc_lr` from
the meta files (raw masks only) and `acc` from the baseline files (same
final_clf()).  For each of the 12 families (dataset x k x budget): plain
`firefly` vs each of anova, mi, l1, rf, pca, pso.  Baselines have no budget:
the same baseline rows are compared at budget 800 and at 3000.

Test (PREREG, verbatim rules)
  * paired two-sided Wilcoxon signed-rank by seed, n = 10, zero differences
    dropped (zero_method "wilcox"); scipy's default exact/approximate choice
  * all 10 differences zero -> TIE with p = 1 (undefined test)
  * Holm within each family of 6, family size fixed at 6: an undefined test
    counts as p = 1 (experiments._common.holm(..., nan_policy="as_one"))
  * WIN = Holm p < 0.05 and mean difference > 0; LOSS = Holm p < 0.05 and
    mean difference < 0; TIE otherwise.  No correction across families.
Pairing: before every test both sides must hold seeds 0-9 exactly once, and
results/split_hashes.csv must hold exactly one split hash per (dataset,
seed) -- the split both sides were evaluated on.  A NaN accuracy raises
(dropping it would break the pairing).

Outputs
  results/analysis/rq1_tests.csv     one row per family x comparator
  results/analysis/descriptive.csv   mean / std / n per method (incl. all,
                                     random, seeded variants; raw masks)
  results/analysis/rq1_summary.md    label pivot, mean differences, takeaway
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy

from experiments._common import holm, paired_wilcoxon
from experiments.analysis import a0_integrity
from experiments.analysis import common as C
from experiments.analysis import takeaway as T

ALPHA = 0.05
FAMILY_SIZE = len(C.COMPARATORS)
RQ1_COLUMNS = ["dataset", "k", "budget", "comparator", "n", "mean_firefly",
               "mean_comparator", "mean_diff_pts", "std_diff_pts", "nonzero_pairs",
               "p_raw", "p_holm", "label"]
OUT_TESTS = C.ANALYSIS_DIR / "rq1_tests.csv"
OUT_DESC = C.ANALYSIS_DIR / "descriptive.csv"
OUT_SUMMARY = C.ANALYSIS_DIR / "rq1_summary.md"
POWER_NOTE = ("n = 10 seeds limits power: the smallest two-sided Wilcoxon p is about 0.002, "
              "so a Holm-adjusted WIN or LOSS needs nearly all seeds to agree.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def primary_long(baselines: dict, meta: dict) -> pd.DataFrame:
    """Long table: dataset, method, k, budget (NaN for baselines), seed, acc.

    Meta rows: mask_type == "raw" only, acc = acc_lr.  Baseline rows: acc.
    """
    parts = []
    for ds, b in baselines.items():
        parts.append(b.assign(dataset=ds, budget=np.nan)[
            ["dataset", "method", "k", "budget", "seed", "acc"]])
    for ds, m in meta.items():
        raw = m[m.mask_type == "raw"]
        parts.append(raw.assign(dataset=ds).rename(columns={"acc_lr": "acc"})[
            ["dataset", "method", "k", "budget", "seed", "acc"]])
    return pd.concat(parts, ignore_index=True)


def seed_series(long: pd.DataFrame, dataset: str, method: str, k: int, budget) -> pd.Series:
    """Accuracy indexed by seed for one method in one family (baselines ignore budget)."""
    sel = (long.dataset == dataset) & (long.method == method) & (long.k == k)
    if method not in C.BASELINE_LABELS.values():
        sel &= long.budget == budget
    s = long.loc[sel].set_index("seed")["acc"].sort_index()
    if s.index.has_duplicates:
        raise ValueError(f"duplicate seeds for {dataset}/{method}/k={k}/b={budget}")
    return s


def check_pair(a: pd.Series, b: pd.Series, split_hashes: pd.DataFrame, dataset: str,
               what: str) -> None:
    """Assert identical seed sets 0-9, no NaN, one split hash per (dataset, seed)."""
    want = list(C.SEEDS)
    if list(a.index) != want or list(b.index) != want:
        raise ValueError(f"{what}: seeds {list(a.index)} vs {list(b.index)}, expected {want}")
    if a.isna().any() or b.isna().any():
        raise ValueError(f"{what}: NaN accuracy; refusing to drop pairs")
    sh = split_hashes[split_hashes.dataset == dataset]
    per_seed = sh.groupby("seed")["sha_all"].nunique()
    if sorted(per_seed.index) != want or (per_seed != 1).any():
        raise ValueError(f"{what}: split hashes for {dataset} are missing or not unique per seed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def paired_test(firefly: pd.Series, comparator: pd.Series) -> dict:
    """Paired Wilcoxon per PREREG; all-zero differences -> p = 1."""
    res = paired_wilcoxon(firefly.values, comparator.values)
    diff = firefly.values - comparator.values
    p = res["p"]
    return dict(n=res["n"], mean_firefly=float(firefly.mean()),
                mean_comparator=float(comparator.mean()),
                mean_diff_pts=float(diff.mean() * 100),
                std_diff_pts=float(diff.std(ddof=1) * 100) if len(diff) > 1 else np.nan,
                nonzero_pairs=res["n_nonzero"],
                p_raw=1.0 if np.isnan(p) else float(p),
                undefined=bool(np.isnan(p)))


def label(p_holm: float, mean_diff_pts: float) -> str:
    if p_holm < ALPHA and mean_diff_pts > 0:
        return "WIN"
    if p_holm < ALPHA and mean_diff_pts < 0:
        return "LOSS"
    return "TIE"


def family_rows(firefly: pd.Series, comparators: dict[str, pd.Series], **family) -> list[dict]:
    """Test firefly vs each comparator, Holm within this family of 6."""
    if len(comparators) != FAMILY_SIZE:
        raise ValueError(f"family needs {FAMILY_SIZE} comparators, got {list(comparators)}")
    rows = [dict(family, comparator=name, **paired_test(firefly, s))
            for name, s in comparators.items()]
    # an undefined test enters Holm as NaN -> treated as p = 1, m stays 6
    raw_for_holm = [np.nan if r["undefined"] else r["p_raw"] for r in rows]
    for r, ph in zip(rows, holm(raw_for_holm, nan_policy="as_one")):
        r["p_holm"] = float(ph)
        r["label"] = label(r["p_holm"], r["mean_diff_pts"])
    return rows


def families(meta: dict) -> list[tuple[str, int, int]]:
    return sorted({(ds, int(k), int(b)) for ds, m in meta.items()
                   for k, b in m[["k", "budget"]].drop_duplicates().itertuples(index=False)})


def run_rq1(baselines: dict, meta: dict, split_hashes: pd.DataFrame) -> pd.DataFrame:
    long = primary_long(baselines, meta)
    rows = []
    for ds, k, b in families(meta):
        ff = seed_series(long, ds, C.METHOD_UNDER_TEST, k, b)
        comps = {}
        for name in C.COMPARATORS:
            s = seed_series(long, ds, name, k, b)
            check_pair(ff, s, split_hashes, ds, f"{ds}/k={k}/b={b}: firefly vs {name}")
            comps[name] = s
        rows += family_rows(ff, comps, dataset=ds, k=k, budget=b)
    return pd.DataFrame(rows)[RQ1_COLUMNS]


def descriptive(baselines: dict, meta: dict) -> pd.DataFrame:
    """Mean / std / n of the primary accuracy for every method (raw masks)."""
    long = primary_long(baselines, meta)
    long["metric"] = np.where(long.budget.isna(), "acc", "acc_lr")
    g = (long.groupby(["dataset", "method", "k", "budget", "metric"], dropna=False)["acc"]
              .agg(n="count", mean="mean", std="std").reset_index())
    g["display"] = g.method.map(C.DISPLAY)
    g["source"] = np.where(g.budget.isna(), "baseline (no budget)", "metaheuristic, raw mask")
    return g[["dataset", "method", "display", "source", "k", "budget", "metric",
              "n", "mean", "std"]].sort_values(["dataset", "k", "budget", "method"],
                                               na_position="first")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pivot(tests: pd.DataFrame) -> pd.DataFrame:
    """families x comparators, cells 'LABEL +x.x' (mean difference in points)."""
    t = tests.assign(cell=tests.label + " " + tests.mean_diff_pts.map(lambda v: f"{v:+.1f}"),
                     family=tests.dataset + " k=" + tests.k.astype(str)
                     + " b=" + tests.budget.astype(str))
    order = list(dict.fromkeys(t.family))
    return t.pivot(index="family", columns="comparator", values="cell").loc[
        order, list(C.COMPARATORS)]


def rq1_takeaway(tests: pd.DataFrame) -> str:
    counts = tests.label.value_counts()
    n_win, n_loss, n_tie = (int(counts.get(x, 0)) for x in ("WIN", "LOSS", "TIE"))
    by_comp = (tests.groupby("comparator")["label"]
               .agg(lambda s: f"{(s == 'WIN').sum()}W/{(s == 'LOSS').sum()}L/{(s == 'TIE').sum()}T"))
    return (f"Across {tests[['dataset', 'k', 'budget']].drop_duplicates().shape[0]} families x "
            f"{FAMILY_SIZE} comparators ({len(tests)} tests), plain Firefly has {n_win} WIN, "
            f"{n_loss} LOSS and {n_tie} TIE (Holm within family, alpha {ALPHA}); per comparator: "
            + ", ".join(f"{c} {by_comp[c]}" for c in C.COMPARATORS)
            + f". Mean differences range from {T.pts(tests.mean_diff_pts.min() / 100)} to "
              f"{T.pts(tests.mean_diff_pts.max() / 100)}.")


def write_summary(tests: pd.DataFrame, path=OUT_SUMMARY) -> str:
    piv = pivot(tests)
    text = rq1_takeaway(tests)
    lines = ["# RQ1 confirmatory tests (Stage A1)", "",
             "- Pre-registered in PREREG.md: raw Firefly mask, `acc_lr`; paired two-sided "
             "Wilcoxon by seed (zero differences dropped, all-zero = TIE with p = 1); Holm "
             "within each family of 6 (undefined test counts as p = 1); no correction across "
             "the 12 families.",
             f"- scipy {scipy.__version__} `wilcoxon(zero_method='wilcox', "
             "alternative='two-sided')`, default exact/approximate method.",
             f"- {POWER_NOTE}",
             "- PCA is a projection, not a selector (full embedding + projection matrix at "
             "inference). Baselines have no search budget; the same baseline rows are used at "
             "budget 800 and 3000.", "",
             f"**Takeaway:** {text}", "",
             "## Labels and mean difference (Firefly minus comparator, accuracy points)", "",
             "| family | " + " | ".join(piv.columns) + " |",
             "|---|" + "---|" * len(piv.columns)]
    lines += [f"| {fam} | " + " | ".join(row) + " |" for fam, row in piv.iterrows()]
    lines += ["", "Full numbers (means, std of paired differences, non-zero pairs, raw and "
              "Holm p, n): `results/analysis/rq1_tests.csv`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text


def run() -> pd.DataFrame:
    a0_integrity.validate()                     # A0 must pass first
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    tests = run_rq1(baselines, meta, C.load_split_hashes())
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tests.to_csv(OUT_TESTS, index=False)
    descriptive(baselines, meta).to_csv(OUT_DESC, index=False)
    text = write_summary(tests)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("RQ1 (confirmatory): label and mean difference in accuracy points "
          "(Firefly minus comparator), raw masks, acc_lr")
    print(pivot(tests).to_string())
    print(f"\nTakeaway: {text}")
    print(POWER_NOTE)
    print(f"\nWrote {OUT_TESTS} ({len(tests)} rows)\nWrote {OUT_DESC}\nWrote {OUT_SUMMARY}")
    return tests
