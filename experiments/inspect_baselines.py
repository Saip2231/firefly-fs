"""Day 4 Stage 0 - confirm the k lists from the baseline CSVs (read-only).

Prints, per results/baselines_<dataset>.csv: row count, seeds present, the
unique (method, k) pairs with rows per pair, and the k list per method
excluding "All Features".

Usage
-----
    python experiments/inspect_baselines.py
"""

from __future__ import annotations

import pandas as pd

from _common import RESULTS_DIR

DATASETS = ["digits", "cifar_r18"]


def main() -> None:
    pd.set_option("display.width", 200)
    for name in DATASETS:
        path = RESULTS_DIR / f"baselines_{name}.csv"
        df = pd.read_csv(path)
        print(f"=== {path.name} ===")
        print(f"rows: {len(df)}")
        print(f"seeds present: {sorted(df.seed.unique().tolist())}")
        pairs = (df.groupby(["method", "k"]).size().rename("rows").reset_index()
                   .sort_values(["method", "k"]))
        print(f"unique (method, k) pairs: {len(pairs)}")
        print(pairs.to_string(index=False))
        sel = df[df.method != "All Features"]
        print(f"k list (selection methods): {sorted(sel.k.unique().tolist())}\n")


if __name__ == "__main__":
    main()
