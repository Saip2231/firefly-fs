"""Stage A6a: KNN and linear-SVM accuracy of the baseline subsets (ANALYSIS.md A6a).

The baseline CSVs hold logistic-regression accuracy only, so the ANOVA, MI,
L1, RF and random subsets have no KNN / SVM numbers for RQ4.  For every
baseline row with a subset (`idx`; PCA and 'all' have none) and k < d, this
script loads load_dataset(dataset, seed) and scores the stored subset with
the runner's own single evaluation function,
run_metaheuristics.evaluate_on_test (final_clf, knn_clf, svm_clf; one fit
each on the full training split, one score each on the test split).
No search, no tuning, no change to any existing file.

This module never handles the test arrays itself: the load_dataset tuple is
passed unopened to evaluate_on_test (tests/test_eval_baseline_transfer.py
checks this statically).  As a cross-check, the logistic-regression accuracy
returned by evaluate_on_test must equal the `acc` already in the baseline CSV
(same classifier, same split, same subset).

Output: results/baselines_transfer_<dataset>.csv
  (dataset, method, seed, k, acc_knn, acc_svm, idx, acc_lr, acc_lr_matches_baseline)
and an entry in results/runs.json.

Usage (from firefly-fs/, venv active):
    python experiments/eval_baseline_transfer.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS_DIR, git_hash, warn_once  # noqa: E402

import run_metaheuristics as rm  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from src.classifiers import clf_configs  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.results_store import log_experiment  # noqa: E402

COLUMNS = ["dataset", "method", "seed", "k", "acc_knn", "acc_svm", "idx", "acc_lr",
           "acc_lr_matches_baseline"]


def jobs_for(dataset: str, seeds=C.SEEDS) -> list[dict]:
    """Baseline rows with a subset and k < d, as plain dicts."""
    b = C.load_baselines(dataset)
    d = C.dimension(b)
    sel = b[~b.method.isin(C.NO_IDX_METHODS) & (b.k < d) & b.seed.isin(list(seeds))]
    return [dict(dataset=dataset, method=r.method, seed=int(r.seed), k=int(r.k), d=d,
                 idx=sorted(r.idx), acc_baseline=float(r.acc))
            for r in sel.sort_values(["method", "k", "seed"]).itertuples()]


def score_job(job: dict) -> dict:
    """Score one stored subset via the runner's evaluate_on_test."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = np.zeros(job["d"], dtype=bool)
        mask[job["idx"]] = True
        assert int(mask.sum()) == job["k"]
        res = rm.evaluate_on_test(mask, load_dataset(job["dataset"], seed=job["seed"]))
    return dict(dataset=job["dataset"], method=job["method"], seed=job["seed"], k=job["k"],
                acc_knn=res["acc_knn"], acc_svm=res["acc_svm"],
                idx=",".join(map(str, job["idx"])), acc_lr=res["acc_lr"],
                acc_lr_matches_baseline=bool(res["acc_lr"] == job["acc_baseline"]),
                warnings=sorted({w.category.__name__ for w in caught}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    warn_once()
    import eval_baseline_transfer as _self  # noqa: PLC0415  (workers import by name)

    t_all = time.perf_counter()
    paths, summary = [], {}
    for ds in args.datasets:
        jobs = jobs_for(ds)
        print(f"{ds}: {len(jobs)} baseline subsets (methods "
              f"{sorted({j['method'] for j in jobs})}, k {sorted({j['k'] for j in jobs})})",
              flush=True)
        with joblib.parallel_backend("loky", inner_max_num_threads=1):
            rows = joblib.Parallel(n_jobs=args.n_jobs)(
                joblib.delayed(_self.score_job)(j) for j in jobs)
        warn_types = sorted({w for r in rows for w in r.pop("warnings")})
        df = pd.DataFrame(rows)[COLUMNS]
        out = RESULTS_DIR / f"baselines_transfer_{ds}.csv"
        df.to_csv(out, index=False)
        paths.append(out)
        n_match = int(df.acc_lr_matches_baseline.sum())
        summary[ds] = dict(rows=len(df), lr_matches=n_match, warnings=warn_types)
        means = df.groupby(["method", "k"])[["acc_lr", "acc_knn", "acc_svm"]].mean().round(4)
        print(means.to_string())
        print(f"acc_lr equals the baseline CSV's acc in {n_match}/{len(df)} rows; "
              f"warning types: {warn_types or 'none'}\nWrote {out}", flush=True)
        if n_match < len(df):
            # Not fatal: the baselines ran without thread limits (multi-threaded
            # BLAS), which can flip a borderline prediction.  Reported, not hidden.
            bad = df[~df.acc_lr_matches_baseline]
            base = C.load_baselines(ds).set_index(["method", "seed", "k"])["acc"]
            bad = bad.assign(acc_baseline=[base[(r.method, r.seed, r.k)] for r in bad.itertuples()])
            print("MISMATCH rows (investigate before using A6):")
            print(bad[["method", "seed", "k", "acc_lr", "acc_baseline"]].to_string(index=False))
        print()

    total = time.perf_counter() - t_all
    print(f"Total runtime {total:.1f} s")
    log_experiment(
        name="eval_baseline_transfer",
        params=dict(datasets=args.datasets, evaluation="run_metaheuristics.evaluate_on_test",
                    classifiers=clf_configs(), n_jobs=args.n_jobs,
                    ks=sorted({int(k) for p in paths for k in pd.read_csv(p).k.unique()})),
        seeds=list(C.SEEDS),
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in paths],
        extra=dict(runtime_s=round(total, 1), summary=summary, git_hash=git_hash()),
    )


if __name__ == "__main__":
    main()
