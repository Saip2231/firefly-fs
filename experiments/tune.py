"""Day 4 Stage 4: one-factor-at-a-time hyperparameter tuning on SEPARATE seeds.

Protocol (CLAUDE.md section 8, Stage 4; fixed before any run)
  Seeds       100, 101, 102 only (never the evaluation seeds 0-9).
  Data        Xtr, _, ytr, _ = load_dataset(name, seed): the test split is
              discarded unread.  Xtr/ytr is split stratified 75/25
              (random_state=seed): the search sees ONLY the 75 %; the 25 % is
              the tuning holdout.
  Search      run_search(method) on make_fitness(X75, y75, k, seed) (its own
              inner split and max_rows=2000 subsample live inside the 75 %),
              search seed = tuning seed, budget 800.
  Score       tuning_acc = final_clf() (LogisticRegression(max_iter=500,
              random_state=0), shared with every final evaluation) trained on
              the full 75 % with the returned mask, scored on the 25 %.
  Conditions  (cifar_r18, 16), (cifar_r18, 64), (digits, 16).
  Configs     9 per method: the default and 8 one-factor variants (CONFIGS).
              Only plain firefly and pso are tuned; the frozen values then
              apply to firefly_levy / firefly_seeded and pso_seeded.
  Jobs        2 methods x 9 configs x 3 conditions x 3 seeds = 162.
Decision rule: experiments/tune_summary.py.

Execution
  joblib (loky) with n_jobs = os.cpu_count() - 1, 1 BLAS/OpenMP thread per
  worker.  The parent process appends each finished row to
  results/tuning_runs.csv immediately (single writer), so an interrupted run
  is resumed by rerunning the same command: jobs whose key (method,
  config_id, dataset, k, seed) is already in the CSV are skipped.

Usage (from firefly-fs/, venv active)
  python experiments/tune.py --dry-run   # default config of both methods, seed 100,
                                         # all 3 conditions -> results/tuning_runs_dryrun.csv
  python experiments/tune.py             # full grid -> results/tuning_runs.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from functools import lru_cache  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from _common import RESULTS_DIR, git_hash, warn_once  # noqa: E402

from src.classifiers import clf_configs, final_clf  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness  # noqa: E402
from src.results_store import log_experiment  # noqa: E402
from src.search_registry import run_search  # noqa: E402

TUNING_SEEDS = (100, 101, 102)
CONDITIONS = (("cifar_r18", 16), ("cifar_r18", 64), ("digits", 16))
BUDGET = 800
HOLDOUT_FRAC = 0.25

DEFAULTS = {
    "firefly": {"n_fireflies": 20, "beta0": 0.4, "gamma": 2.0, "alpha": 0.02},
    "pso":     {"n_particles": 20, "w_start": 0.9, "c1": 1.5, "c2": 1.5, "vmax": 0.5},
}
# (config_id, overrides of the default); order = report order, default first
_VARIANTS = {
    "firefly": [
        ("n_fireflies=10", {"n_fireflies": 10}), ("n_fireflies=40", {"n_fireflies": 40}),
        ("beta0=0.2", {"beta0": 0.2}), ("beta0=0.6", {"beta0": 0.6}),
        ("gamma=1", {"gamma": 1.0}), ("gamma=4", {"gamma": 4.0}),
        ("alpha=0.1", {"alpha": 0.1}), ("alpha=0.25", {"alpha": 0.25}),
    ],
    "pso": [
        ("n_particles=10", {"n_particles": 10}), ("n_particles=40", {"n_particles": 40}),
        ("c1=c2=1.0", {"c1": 1.0, "c2": 1.0}), ("c1=c2=2.0", {"c1": 2.0, "c2": 2.0}),
        ("vmax=0.25", {"vmax": 0.25}), ("vmax=1.0", {"vmax": 1.0}),
        ("w_start=0.7", {"w_start": 0.7}), ("w_start=1.0", {"w_start": 1.0}),
    ],
}
CONFIGS = {m: [("default", dict(DEFAULTS[m]))]
           + [(cid, {**DEFAULTS[m], **ov}) for cid, ov in _VARIANTS[m]]
           for m in DEFAULTS}

RUNS_CSV = RESULTS_DIR / "tuning_runs.csv"
DRY_CSV = RESULTS_DIR / "tuning_runs_dryrun.csv"
COLUMNS = ["method", "config_id", "config_json", "dataset", "k", "seed",
           "tuning_acc", "val_fitness", "evals_used", "search_s", "git_hash"]
KEY = ["method", "config_id", "dataset", "k", "seed"]


def build_jobs(dry_run: bool = False) -> list[dict]:
    """All (method, config, condition, seed) jobs; dry run = default configs, seed 100."""
    seeds = TUNING_SEEDS[:1] if dry_run else TUNING_SEEDS
    jobs = []
    for method, configs in CONFIGS.items():
        for config_id, hp in (configs[:1] if dry_run else configs):
            for dataset, k in CONDITIONS:
                for seed in seeds:
                    jobs.append(dict(method=method, config_id=config_id,
                                     config_json=json.dumps(hp, sort_keys=True),
                                     dataset=dataset, k=k, seed=seed))
    return jobs


def pending_jobs(jobs: list[dict], csv_path) -> list[dict]:
    """Jobs whose KEY is not yet in csv_path (resume)."""
    if not os.path.exists(csv_path):
        return list(jobs)
    done = pd.read_csv(csv_path)
    done_keys = {tuple(r) for r in done[KEY].astype(str).itertuples(index=False)}
    return [j for j in jobs if tuple(str(j[c]) for c in KEY) not in done_keys]


def tuning_split(Xtr, ytr, seed: int):
    """Stratified 75/25 split of the TRAINING data: (X_search, X_hold, y_search, y_hold)."""
    return train_test_split(Xtr, ytr, test_size=HOLDOUT_FRAC, stratify=ytr,
                            random_state=seed)


@lru_cache(maxsize=None)
def _tuning_data(dataset: str, seed: int):
    """Per-process cache of the tuning split; the test split is never kept."""
    Xtr, _, ytr, _ = load_dataset(dataset, seed=seed)
    return tuning_split(Xtr, ytr, seed)


def run_job(job: dict, git: str) -> dict:
    """One tuning run; returns a CSV row (plus "warnings", not written).

    Warnings are recorded here and returned, so the parent prints each
    distinct category once for the whole grid (rule 14) instead of once per
    worker process.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row = _run_job(job, git)
    row["warnings"] = [(w.category.__name__, str(w.message)) for w in caught]
    return row


def _run_job(job: dict, git: str) -> dict:
    Xs, Xh, ys, yh = _tuning_data(job["dataset"], job["seed"])
    d, k, seed = Xs.shape[1], job["k"], job["seed"]
    hp = json.loads(job["config_json"])
    t0 = time.perf_counter()
    fitness, state = make_fitness(Xs, ys, k=k, seed=seed)
    out = run_search(job["method"], fitness, state, d, k, BUDGET, seed, hp)
    search_s = time.perf_counter() - t0
    mask = out["best_mask"]
    assert int(mask.sum()) == k and out["evals_used"] <= BUDGET
    clf = final_clf()
    clf.fit(Xs[:, mask], ys)
    return {**job, "tuning_acc": float(clf.score(Xh[:, mask], yh)),
            "val_fitness": out["best_fit"], "evals_used": out["evals_used"],
            "search_s": round(search_s, 3), "git_hash": git}


def _append(row: dict, csv_path) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row[c] for c in COLUMNS})


def _parallel(n_jobs: int, tasks):
    """Results as each task finishes (unordered if this joblib supports it)."""
    try:
        return joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks)
    except ValueError:      # older joblib: ordered generator
        return joblib.Parallel(n_jobs=n_jobs, return_as="generator")(tasks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="default config of both methods, seed 100, all conditions")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path = DRY_CSV if args.dry_run else RUNS_CSV
    jobs = build_jobs(args.dry_run)
    todo = pending_jobs(jobs, csv_path)
    git = git_hash()
    print(f"{'DRY RUN' if args.dry_run else 'FULL GRID'}: {len(jobs)} jobs, "
          f"{len(jobs) - len(todo)} already in {csv_path.name}, {len(todo)} to run; "
          f"n_jobs={args.n_jobs}, git {git}", flush=True)

    # Workers must import run_job by module name.  Run as a script, this file
    # is __main__: loky then pickles run_job by value, while the lru_cache'd
    # _tuning_data it calls is pickled by reference to __main__, which the
    # workers do not have (AttributeError: Can't get attribute '_tuning_data'
    # on <module '__main__'>).  Importing the module as "tune" fixes that.
    import tune as _tune  # noqa: PLC0415
    worker = _tune.run_job

    t_all = time.perf_counter()
    if todo:
        with joblib.parallel_backend("loky", inner_max_num_threads=1):
            results = _parallel(args.n_jobs, (joblib.delayed(worker)(j, git) for j in todo))
            seen_warn: set[str] = set()
            for n, row in enumerate(results, 1):
                _append(row, csv_path)
                for cat, msg in row.pop("warnings"):
                    if cat not in seen_warn:
                        seen_warn.add(cat)
                        print(f"[warning, shown once] {cat}: {msg} "
                              f"(further {cat}s suppressed)", flush=True)
                print(f"[{n:>3}/{len(todo)}] {row['method']:<7} {row['config_id']:<15} "
                      f"{row['dataset']:<9} k={row['k']:<3} seed={row['seed']} "
                      f"tuning_acc={row['tuning_acc']:.4f} val={row['val_fitness']:.4f} "
                      f"evals={row['evals_used']} {row['search_s']:.1f}s "
                      f"| {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    total = time.perf_counter() - t_all

    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name}: {len(df)} rows (expected {len(jobs)})")
    if args.dry_run:
        pd.set_option("display.width", 200)
        print(df[["method", "config_id", "dataset", "k", "seed", "tuning_acc",
                  "val_fitness", "evals_used", "search_s"]]
              .sort_values(KEY).to_string(index=False))
        per_run = df.search_s.mean()
        full = len(build_jobs(False))
        print(f"\nMean search_s per run {per_run:.1f} s -> full grid ~{full * per_run / 60:.0f} "
              f"min on 1 worker, ~{full * per_run / 60 / args.n_jobs:.0f} min on "
              f"{args.n_jobs} workers (rough; cifar k=64 runs dominate)")
    print(f"Runtime this call: {total:.1f} s")

    log_experiment(
        name="tune_dryrun" if args.dry_run else "tune",
        params=dict(tuning_seeds=list(TUNING_SEEDS), conditions=[list(c) for c in CONDITIONS],
                    ks=sorted({k for _, k in CONDITIONS}), budget=BUDGET,
                    holdout_frac=HOLDOUT_FRAC, configs={m: dict(c) for m, c in CONFIGS.items()},
                    score="final_clf on tuning holdout (25 % of Xtr)",
                    final_clf=clf_configs()["final_clf"], n_jobs=args.n_jobs),
        seeds=list(TUNING_SEEDS[:1] if args.dry_run else TUNING_SEEDS),
        csv_paths=[str(csv_path.relative_to(RESULTS_DIR.parent))],
        extra=dict(runtime_s=round(total, 1), rows=len(df), jobs_run=len(todo),
                   git_hash=git, numpy=np.__version__, joblib=joblib.__version__),
    )


if __name__ == "__main__":
    main()
