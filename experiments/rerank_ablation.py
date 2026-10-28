"""Stage 3 - Re-ranking ablation: raw Firefly best vs re-ranked best.

Data (per seed, random_state=seed), noisy toy with sklearn defaults:
  make_classification(n_samples=6000, n_features=d, n_informative=5,
                      n_redundant=0, n_repeated=0, shuffle=False)
  -> class_sep=1.0, n_clusters_per_class=2
  d in {50, 500}; 1000 search rows, independent 5000-row holdout (stratified).
Search: run_firefly(k=5, budget=800, max_rows=1000 -> 250 validation rows,
  n_fireflies=20, rerank_top_m=10, n_splits=5).  Re-ranking uses the 1000
  search rows only.  The holdout is used once per final mask.
Hits are reported two ways:
  hits      - selected columns among the 5 informative ones
  sep_hits  - selected columns among the SEPARATING informative ones
              (src.synthetic.separating_columns, threshold = 1.0/2 = 0.5,
              gaps measured on the search rows only)
Output: results/rerank_ablation.csv, entry in results/runs.json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from _common import RESULTS_DIR, paired_wilcoxon, warn_once

from src.firefly import run_firefly
from src.results_store import log_experiment
from src.synthetic import separating_columns

DS = [50, 500]
SEEDS = list(range(10))
N_INF, K, BUDGET, N_FIREFLIES = 5, 5, 800, 20
N_TOTAL, N_SEARCH = 6000, 1000
TOP_M, N_SPLITS = 10, 5
CLASS_SEP, N_CLUSTERS = 1.0, 2          # sklearn defaults


def holdout_acc(mask, Xs, ys, Xh, yh, seed):
    clf = LogisticRegression(max_iter=100, random_state=seed)  # as make_fitness
    clf.fit(Xs[:, mask], ys)
    return float(clf.score(Xh[:, mask], yh))


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, sep_info = [], []
    t_all = time.perf_counter()
    total_units = len(DS) * len(SEEDS)

    for di, d in enumerate(DS):
        for si, seed in enumerate(SEEDS):
            X, y = make_classification(
                n_samples=N_TOTAL, n_features=d, n_informative=N_INF,
                n_redundant=0, n_repeated=0, shuffle=False, random_state=seed)
            Xs, Xh, ys, yh = train_test_split(
                X, y, train_size=N_SEARCH, stratify=y, random_state=seed)

            s = separating_columns(Xs, ys, N_INF, CLASS_SEP, N_CLUSTERS)
            sep = set(s["sep_idx"].tolist())
            sep_info.append(dict(d=d, seed=seed, sep_idx=",".join(map(str, sorted(sep))),
                                 n_sep=len(sep), ambiguous=",".join(map(str, s["ambiguous"])),
                                 **{f"gap_{j}": g for j, g in enumerate(s["gaps"])}))

            t0 = time.perf_counter()
            r = run_firefly(Xs, ys, K, budget=BUDGET, seed=seed, max_rows=N_SEARCH,
                            rerank_top_m=TOP_M, n_splits=N_SPLITS, n_fireflies=N_FIREFLIES)
            secs = time.perf_counter() - t0

            cand_keys = [sc["mask"].tobytes() for sc in r["rerank_scores"]]
            changed = bool(not np.array_equal(r["raw_mask"], r["reranked_mask"]))
            for variant, mask in (("raw", r["raw_mask"]), ("reranked", r["reranked_mask"])):
                key = mask.tobytes()
                # 0 = lowest cached fitness; -1 = not among the top_m candidates
                # (possible for raw when more than top_m masks tie at its fitness)
                pos = cand_keys.index(key) if key in cand_keys else -1
                chosen = set(np.flatnonzero(mask).tolist())
                rows.append(dict(
                    d=d, seed=seed, variant=variant, k=K, budget=BUDGET,
                    hits=int(mask[:N_INF].sum()),
                    sep_hits=len(chosen & sep), n_sep=len(sep),
                    # single-split validation fitness from the search
                    val_fitness=(float(r["raw_fit"]) if variant == "raw"
                                 else r["rerank_scores"][pos]["cached_fitness"]),
                    cv_acc=r["rerank_scores"][pos]["cv_acc"] if pos >= 0 else np.nan,
                    candidate_rank=pos,
                    holdout_acc=holdout_acc(mask, Xs, ys, Xh, yh, seed),
                    changed=changed,
                    search_evals=r["search_evals"], rerank_fits=r["rerank_fits"],
                    seconds=secs, idx=",".join(map(str, sorted(chosen))),
                ))

            done = di * len(SEEDS) + si + 1
            el = time.perf_counter() - t_all
            print(f"[{done:>2}/{total_units}] d={d} seed {seed} done | elapsed {el / 60:.1f} min | "
                  f"ETA ~{el / done * (total_units - done) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    sep_df = pd.DataFrame(sep_info)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print(f"\nSEPARATING COLUMNS (threshold {CLASS_SEP / N_CLUSTERS:.2f}; search rows only)")
    print(sep_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Ambiguous labels: {int((sep_df.ambiguous != '').sum())} (expected 0)")

    print("\nPER-SEED ROWS")
    print(df[["d", "seed", "variant", "hits", "sep_hits", "n_sep", "val_fitness", "cv_acc",
              "candidate_rank", "holdout_acc", "changed", "search_evals", "rerank_fits"]]
          .to_string(index=False, float_format=fmt))

    print("\nMEAN +/- STD per (d, variant)   [n=10 seeds]")
    summ = (df.groupby(["d", "variant"])[["hits", "sep_hits", "holdout_acc", "val_fitness",
                                         "search_evals", "rerank_fits"]]
              .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    print(summ.to_string(index=False, float_format=fmt))

    print("\nRERANKED - RAW, paired Wilcoxon (two-sided, paired by seed, n=10; zero differences dropped)")
    tests = []
    for d, g in df.groupby("d"):
        raw = g[g.variant == "raw"].sort_values("seed")
        rr = g[g.variant == "reranked"].sort_values("seed")
        assert list(raw.seed) == list(rr.seed)
        for metric in ("hits", "sep_hits", "holdout_acc"):
            tests.append(dict(d=d, metric=metric,
                              **paired_wilcoxon(rr[metric].values, raw[metric].values)))
        tests[-1]["n_changed"] = int(rr.changed.sum())
    tdf = pd.DataFrame(tests)
    print(tdf.to_string(index=False, float_format=fmt))

    csv = RESULTS_DIR / "rerank_ablation.csv"
    df.to_csv(csv, index=False)
    sep_csv = RESULTS_DIR / "separating_columns_rerank_ablation.csv"
    sep_df.to_csv(sep_csv, index=False)
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nSaved {sep_csv}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="rerank_ablation",
        params=dict(ds=DS, n_informative=N_INF, ks=[K], budget=BUDGET,
                    n_fireflies=N_FIREFLIES, n_total=N_TOTAL, n_search=N_SEARCH,
                    n_holdout=N_TOTAL - N_SEARCH, max_rows=N_SEARCH,
                    rerank_top_m=TOP_M, n_splits=N_SPLITS, class_sep=CLASS_SEP,
                    n_clusters_per_class=N_CLUSTERS, data_random_state="seed"),
        seeds=SEEDS,
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (csv, sep_csv)],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
