"""Stage A4: idle-machine timing (ANALYSIS.md A4).  Computes; uses TRAINING rows only.

Part 1, inference timing
  For every stored subset of seed 0 (meta raw masks: each method x k x budget;
  baseline subsets: random, anova, mi, l1, rf; PCA with k components; and all
  features), fit final_clf() (logistic regression) and KNN (k=5) on the
  training rows of load_dataset(dataset, 0), then time single-sample
  prediction on training rows: 50 warm-up calls, then 1,000 timed calls, each
  on the next training row (cycling).  Each timed call starts from the full
  embedding row, so it includes the column slice (selectors) or the PCA
  projection (PCA): the deployment cost.  Reported: median and IQR in
  microseconds, and the saving against all features for the same classifier.
  The test split is never used: load_dataset's test arrays are discarded.

  Drift control (added after the first run, LOG.md): that run showed a step
  change half-way (single-sample LR ~300-600 us with wide IQR, then ~136 us
  with IQR ~3 us), i.e. the machine changed state during the run, so a
  subset timed early looked slower than one timed late.  Now (1) a burn-in
  keeps the CPU busy for BURN_IN_S seconds before any timing, and (2) every
  subset is bracketed by REF_REPEATS all-features calls just before and just
  after it with the same classifier; the saving is measured against these
  bracketing references, not against one 'all' measurement taken at another
  time.  Rows are flagged unstable when IQR > 25 % of the median or the two
  reference blocks differ by more than 20 %.

Part 2, contention check and determinism
  Re-run 10 search jobs sequentially on the idle machine (firefly and pso,
  cifar_r18, k=16, budget 800, seeds 0-4, config/frozen_config.yaml) with the
  runner's own make_fitness + run_search, timed exactly like the runner's
  search_s (make_fitness + search).  Compare with the same jobs' search_s in
  results/meta_cifar_r18.csv (measured under 15-way parallel load) and assert
  that every selected idx equals the logged idx.
  contention_factor = mean(logged search_s) / mean(idle search_s).

Outputs: results/analysis/timing_idle.csv, results/analysis/contention.csv,
results/analysis/contention_factor.md, an entry in results/runs.json.

Usage (from firefly-fs/, venv active, machine idle and plugged in):
    python experiments/time_final.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import git_hash, warn_once  # noqa: E402

import run_metaheuristics as rm  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from src.baselines import pca_transform  # noqa: E402
from src.classifiers import clf_configs, final_clf, knn_clf  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness  # noqa: E402
from src.results_store import log_experiment  # noqa: E402
from src.search_registry import run_search  # noqa: E402

SEED = 0
WARMUP, REPEATS = 50, 1000
REF_REPEATS = 100                 # all-features calls before and after each subset
BURN_IN_S = 20.0
MAX_IQR_FRAC, MAX_DRIFT = 0.25, 0.20
CONTENTION_JOBS = [("cifar_r18", m, 16, 800, s) for m in ("firefly", "pso") for s in range(5)]
OUT_TIMING = C.ANALYSIS_DIR / "timing_idle.csv"
OUT_CONT = C.ANALYSIS_DIR / "contention.csv"
OUT_FACTOR = C.ANALYSIS_DIR / "contention_factor.md"
CLASSIFIERS = {"lr": final_clf, "knn": knn_clf}


# ---------------------------------------------------------------------------
# Helpers (pure, tested)
# ---------------------------------------------------------------------------

def train_data(dataset: str, seed: int = SEED):
    """Training rows only; the test split is discarded unread."""
    Xtr, _, ytr, _ = load_dataset(dataset, seed=seed)
    return Xtr, ytr


def mask_from_idx(idx: frozenset, d: int) -> np.ndarray:
    m = np.zeros(d, dtype=bool)
    m[sorted(idx)] = True
    return m


def summarise_us(ns: np.ndarray) -> dict:
    """median / p25 / p75 / IQR in microseconds from nanosecond timings."""
    us = np.asarray(ns, dtype=float) / 1e3
    p25, med, p75 = np.percentile(us, [25, 50, 75])
    return dict(median_us=float(med), p25_us=float(p25), p75_us=float(p75),
                iqr_us=float(p75 - p25))


def time_single_predict(predict_one, X: np.ndarray, warmup: int = WARMUP,
                        repeats: int = REPEATS) -> np.ndarray:
    """Nanoseconds per call of predict_one(row) on successive rows of X."""
    n = len(X)
    for i in range(warmup):
        predict_one(X[i % n: i % n + 1])
    out = np.empty(repeats, dtype=np.int64)
    for i in range(repeats):
        row = X[(warmup + i) % n: (warmup + i) % n + 1]
        t0 = time.perf_counter_ns()
        predict_one(row)
        out[i] = time.perf_counter_ns() - t0
    return out


def bracketed_row(ns: np.ndarray, ref_before: np.ndarray, ref_after: np.ndarray) -> dict:
    """Subset timing summary plus its bracketing all-features references."""
    s = summarise_us(ns)
    b, a = np.median(ref_before) / 1e3, np.median(ref_after) / 1e3
    ref = float(np.median(np.concatenate([ref_before, ref_after])) / 1e3)
    drift = float(abs(a - b) / b) if b > 0 else np.nan
    return dict(**s, ref_all_median_us=ref, ref_drift=drift,
                saving_us_vs_all=ref - s["median_us"],
                stable=bool(s["iqr_us"] <= MAX_IQR_FRAC * s["median_us"]
                            and drift <= MAX_DRIFT))


def burn_in(seconds: float) -> None:
    """Keep one core busy so the CPU leaves its idle/low-clock state."""
    end = time.perf_counter() + seconds
    a = np.random.default_rng(0).random((200, 200))
    while time.perf_counter() < end:
        a = a @ a
        a /= np.abs(a).max()


def subsets(dataset: str, baselines: pd.DataFrame, meta: pd.DataFrame, seed: int = SEED):
    """(method, budget, k, kind, payload) for every subset of one seed."""
    d = C.dimension(baselines)
    ks = C.selection_ks(baselines)
    yield ("all", np.nan, d, "all", None)
    b = baselines[(baselines.seed == seed) & baselines.k.isin(ks)]
    for r in b.itertuples():
        if r.method == "pca":
            yield ("pca", np.nan, int(r.k), "pca", None)
        elif r.method != "all":
            yield (r.method, np.nan, int(r.k), "mask", mask_from_idx(r.idx, d))
    m = meta[(meta.seed == seed) & (meta.mask_type == "raw")]
    for r in m.itertuples():
        yield (r.method, int(r.budget), int(r.k), "mask", mask_from_idx(r.idx, d))


def contention_factor(logged_s, idle_s) -> float:
    return float(np.mean(logged_s) / np.mean(idle_s))


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def time_inference(dataset: str, warmup=WARMUP, repeats=REPEATS,
                   ref_repeats=REF_REPEATS) -> pd.DataFrame:
    baselines, meta = C.load_baselines(dataset), C.load_meta(dataset)
    Xtr, ytr = train_data(dataset)
    ref_clf = {c: make().fit(Xtr, ytr) for c, make in CLASSIFIERS.items()}
    rows = []
    for method, budget, k, kind, payload in subsets(dataset, baselines, meta):
        if kind == "pca":
            pca = pca_transform(Xtr, ytr, k, seed=SEED)
            Xfit, transform, includes = pca.transform(Xtr), pca.transform, "PCA projection"
        elif kind == "mask":
            Xfit, includes = Xtr[:, payload], "column slice"
            transform = (lambda x, m=payload: x[:, m])
        else:
            Xfit, transform, includes = Xtr, (lambda x: x), "none"
        for cname, make in CLASSIFIERS.items():
            clf = make().fit(Xfit, ytr)
            ref = ref_clf[cname].predict
            ref_before = time_single_predict(ref, Xtr, warmup=min(warmup, 10), repeats=ref_repeats)
            ns = time_single_predict(lambda x, c=clf, f=transform: c.predict(f(x)), Xtr,
                                     warmup, repeats)
            ref_after = time_single_predict(ref, Xtr, warmup=0, repeats=ref_repeats)
            rows.append(dict(dataset=dataset, method=method, budget=budget, k=k, seed=SEED,
                             classifier=cname, n_features=Xfit.shape[1], includes=includes,
                             warmup=warmup, repeats=repeats, ref_repeats=ref_repeats,
                             rows="training", **bracketed_row(ns, ref_before, ref_after)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def contention_check() -> pd.DataFrame:
    cfg, cfg_hash = rm.load_config(rm.FROZEN_CONFIG)
    meta = C.load_meta("cifar_r18")
    assert set(meta.config_hash) == {cfg_hash}, "frozen config differs from the logged one"
    Xtr, ytr = train_data("cifar_r18")
    rows = []
    for ds, method, k, budget, seed in CONTENTION_JOBS:
        if seed != SEED:
            Xtr_s, ytr_s = train_data(ds, seed)
        else:
            Xtr_s, ytr_s = Xtr, ytr
        t0 = time.perf_counter()
        fitness, state = make_fitness(Xtr_s, ytr_s, k=k, seed=seed)
        out = run_search(method, fitness, state, Xtr_s.shape[1], k, budget, seed,
                         rm.method_hp(cfg, method))
        idle_s = time.perf_counter() - t0
        logged = meta[(meta.method == method) & (meta.k == k) & (meta.budget == budget)
                      & (meta.seed == seed) & (meta.mask_type == "raw")].iloc[0]
        idx = frozenset(int(i) for i in np.flatnonzero(out["best_mask"]))
        rows.append(dict(dataset=ds, method=method, k=k, budget=budget, seed=seed,
                         idle_search_s=round(idle_s, 4), logged_search_s=logged.search_s,
                         ratio=logged.search_s / idle_s, evals_used=out["evals_used"],
                         idx_matches_logged=idx == logged.idx))
        print(f"  {method:<7} seed {seed}: idle {idle_s:6.1f} s, logged {logged.search_s:6.1f} s, "
              f"idx {'matches' if idx == logged.idx else 'DIFFERS'}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-contention", action="store_true")
    args = ap.parse_args(argv)
    warn_once()
    print("PRECONDITIONS (check before trusting the numbers): laptop plugged in, other apps "
          "closed, OneDrive paused, no other Python jobs.")
    print("Thread limits: " + ", ".join(f"{v}={os.environ[v]}" for v in
                                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                         "OPENBLAS_NUM_THREADS")))
    print(f"{platform.platform()}, {os.cpu_count()} logical CPUs; timing on TRAINING rows; "
          f"{WARMUP} warm-up + {REPEATS} timed single-sample predictions per subset\n")
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print(f"Burn-in: {BURN_IN_S:.0f} s of CPU work before timing ...", flush=True)
    burn_in(BURN_IN_S)
    timing = pd.concat([time_inference(ds) for ds in C.DATASETS], ignore_index=True)
    timing.to_csv(OUT_TIMING, index=False)
    for col, title in (("median_us", "Median single-sample prediction time, microseconds"),
                       ("saving_us_vs_all", "Saving vs bracketing all-features reference, "
                                            "microseconds (positive = faster)")):
        summary = (timing.groupby(["dataset", "classifier", "method"])[col].median()
                         .unstack("method").round(1))
        print(f"{title} (median over k and budget):\n{summary.to_string()}\n")
    n_unstable = int((~timing.stable).sum())
    print(f"Stability: {len(timing) - n_unstable}/{len(timing)} rows stable (IQR <= "
          f"{MAX_IQR_FRAC:.0%} of median, reference drift <= {MAX_DRIFT:.0%}); max reference "
          f"drift {timing.ref_drift.max():.1%}")
    if n_unstable:
        print("WARNING: unstable rows (machine not idle?):")
        print(timing.loc[~timing.stable, ["dataset", "method", "budget", "k", "classifier",
                                          "median_us", "iqr_us", "ref_drift"]].to_string(index=False))
    print(f"Wrote {OUT_TIMING} ({len(timing)} rows)\n")

    factor_line = "contention check skipped"
    if not args.skip_contention:
        print("Contention check (sequential, idle): cifar_r18 k=16 budget 800, seeds 0-4")
        cont = contention_check()
        cont.to_csv(OUT_CONT, index=False)
        factor = contention_factor(cont.logged_search_s, cont.idle_search_s)
        n_match = int(cont.idx_matches_logged.sum())
        factor_line = (f"contention_factor = {factor:.2f} (mean logged search_s under 15-way "
                       f"load / mean idle sequential search_s, {len(cont)} jobs); idx matches "
                       f"logged idx in {n_match}/{len(cont)} jobs")
        OUT_FACTOR.write_text(f"# Contention factor (Stage A4)\n\n{factor_line}\n\n"
                              f"Per job: `{OUT_CONT.relative_to(C.ROOT)}`\n", encoding="utf-8")
        print(f"\n{factor_line}\nWrote {OUT_CONT}, {OUT_FACTOR}")
        assert n_match == len(cont), "non-deterministic search: idx differs from the logged run"

    total = time.perf_counter() - t_all
    print(f"\nTotal runtime {total:.1f} s")
    log_experiment(
        name="time_final",
        params=dict(seed=SEED, warmup=WARMUP, repeats=REPEATS, ref_repeats=REF_REPEATS,
                    burn_in_s=BURN_IN_S, rows="training",
                    classifiers={k: clf_configs()[f"{k}_clf" if k == "knn" else "final_clf"]
                                 for k in CLASSIFIERS},
                    contention_jobs=[list(j) for j in CONTENTION_JOBS],
                    skip_contention=args.skip_contention),
        seeds=[SEED] if args.skip_contention else sorted({j[4] for j in CONTENTION_JOBS}),
        csv_paths=[str(p.relative_to(C.ROOT)) for p in (OUT_TIMING, OUT_CONT) if p.exists()],
        extra=dict(runtime_s=round(total, 1), contention=factor_line, git_hash=git_hash()),
    )


if __name__ == "__main__":
    main()
