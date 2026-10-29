"""Day 4 Stage 0 - before/after table of Day-3 p-values around the final_clf() rerun.

The tests are exactly those in results/day3_summary.md (make_day3_report's
stage2_tests on validate_clean_signal_rescored.csv and stage3_tests on
rerank_ablation.csv), so nothing here is a new analysis.

Usage (order matters)
---------------------
    python experiments/compare_day3_pvalues.py --snapshot   # BEFORE the reruns
    ... rerun validate_clean_signal, rescore_clean_signal, rerank_ablation ...
    python experiments/compare_day3_pvalues.py --compare    # AFTER the reruns

--snapshot writes results/day3_pvalues_before.csv (refuses to overwrite).
--compare writes results/day3_pvalues_before_after.csv (all tests) and prints
every test whose p or Holm-adjusted p changed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _common import RESULTS_DIR
from make_day3_report import read, stage2_tests, stage3_tests

BEFORE = RESULTS_DIR / "day3_pvalues_before.csv"
BOTH = RESULTS_DIR / "day3_pvalues_before_after.csv"
KEY = ["stage", "d", "n_search", "method", "metric"]
VALS = ["direction", "mean_diff", "n_nonzero", "p", "p_holm"]


def current() -> pd.DataFrame:
    s2 = stage2_tests(read("validate_clean_signal_rescored.csv")).assign(stage=2)
    s3 = stage3_tests(read("rerank_ablation.csv")).assign(stage=3, n_search=1000,
                                                          method="reranked-vs-raw")
    return pd.concat([s2, s3], ignore_index=True)[KEY + VALS]


def changed(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise 'differs', treating NaN == NaN as equal."""
    both_nan = a.isna() & b.isna()
    return ~both_nan & ~np.isclose(a.fillna(-1.0), b.fillna(-1.0), rtol=0, atol=1e-12)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--snapshot":
        if BEFORE.exists():
            sys.exit(f"{BEFORE} exists; refusing to overwrite.")
        df = current()
        df.to_csv(BEFORE, index=False)
        print(f"Saved {BEFORE} ({len(df)} tests)")
        return
    if mode != "--compare":
        sys.exit("usage: compare_day3_pvalues.py --snapshot | --compare")

    before = pd.read_csv(BEFORE)
    after = current()
    m = before.merge(after, on=KEY, suffixes=("_before", "_after"), how="outer",
                     indicator=True)
    assert (m._merge == "both").all(), "test sets differ before/after"
    m = m.drop(columns="_merge")
    m["p_changed"] = changed(m.p_before, m.p_after)
    m["p_holm_changed"] = changed(m.p_holm_before, m.p_holm_after)
    m.to_csv(BOTH, index=False)

    diff = m[m.p_changed | m.p_holm_changed]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = KEY + ["direction_before", "direction_after", "mean_diff_before", "mean_diff_after",
                  "p_before", "p_after", "p_holm_before", "p_holm_after"]
    print(f"{len(diff)} of {len(m)} tests changed p or Holm-adjusted p "
          f"(Holm families: stage 2 = {int((m.stage == 2).sum())} tests, "
          f"stage 3 = {int((m.stage == 3).sum())} tests)\n")
    print(diff[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for stage in (2, 3):
        g = m[m.stage == stage]
        print(f"\nstage {stage}: Holm p < 0.05 before {int((g.p_holm_before < 0.05).sum())}, "
              f"after {int((g.p_holm_after < 0.05).sum())} of {len(g)}")
    print(f"\nSaved {BOTH}")


if __name__ == "__main__":
    main()
