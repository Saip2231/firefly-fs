"""One-off check of the Stage 2 outlier: d=500, n_search=1000, seed 1, where
firefly ended at val fitness 0.3723 with 0 hits (holdout 0.494).

Part 1 - exact replay (same data, same fitness seed, same algorithm seed).
  * confirms the replay reproduces the Stage 2 CSV values;
  * records every NEW evaluation and answers:
      did firefly ever evaluate a mask containing an informative column?
      if yes, was any such mask better than the returned best? (-> BUG)
Part 2 - how often does this happen?  Same data and same inner split
  (fitness seed 1), algorithm seeds 0-29 for firefly and random search.

Diagnosis only: src/ is not modified.  Uses search rows only (no holdout
except to repeat Stage 2's holdout accuracy for the replayed masks).
Outputs: results/check_d500_seed1_sweeps.csv, results/check_d500_seed1_freq.csv
"""

from __future__ import annotations

import time
from math import comb

import numpy as np
import pandas as pd

from _common import RESULTS_DIR, EvalRecorder, random_search, warn_once
from validate_clean_signal import BUDGET, K, N_FIREFLIES, N_INF, holdout_acc, make_splits

from src.firefly import binary_firefly, make_fitness
from src.results_store import log_experiment

D, N_SEARCH, DATA_SEED = 500, 1000, 1
ALGO_SEEDS = list(range(30))


def run_ff(Xs, ys, algo_seed):
    fitness, state = make_fitness(Xs, ys, k=K, seed=DATA_SEED, max_rows=len(ys))
    rec = EvalRecorder(fitness, state)
    trace: dict = {}
    mask, best, hist = binary_firefly(rec, state, d=D, k=K, budget=BUDGET,
                                      n_fireflies=N_FIREFLIES, seed=algo_seed, trace=trace)
    return mask, best, rec, trace, state


def run_rs(Xs, ys, algo_seed):
    fitness, state = make_fitness(Xs, ys, k=K, seed=DATA_SEED, max_rows=len(ys))
    rec = EvalRecorder(fitness, state)
    out = random_search(rec, state, d=D, k=K, budget=BUDGET, seed=algo_seed)
    return out["best_mask"], out["best_fit"], rec


def inf_stats(rec):
    M = np.asarray(rec.masks)[:, :N_INF]
    has_inf = M.any(axis=1)
    f = np.asarray(rec.fits)
    return dict(
        n_evals=len(f),
        masks_with_inf=int(has_inf.sum()),
        inf_cols_tried=int(M.any(axis=0).sum()),
        best_fit_with_inf=float(f[has_inf].min()) if has_inf.any() else np.nan,
        best_fit_without_inf=float(f[~has_inf].min()) if (~has_inf).any() else np.nan,
    )


def main():
    warn_once()
    t0 = time.perf_counter()
    searches, (Xh, yh) = make_splits(D, DATA_SEED)
    Xs, ys = searches[N_SEARCH]

    p_inf = 1 - comb(D - N_INF, K) / comb(D, K)
    print(f"Theory: P(random {K}-mask contains >=1 of {N_INF} informative cols, d={D}) = {p_inf:.4f}")
    print(f"        expected such masks in {BUDGET} random draws = {BUDGET * p_inf:.1f}")
    print(f"        P(none of {N_FIREFLIES} initial fireflies has one) = {(1 - p_inf) ** N_FIREFLIES:.3f}")

    # ---------------- Part 1: exact replay ---------------------------------
    print("\n=== PART 1: replay d=500, n_search=1000, seed 1 ===")
    s2 = pd.read_csv(RESULTS_DIR / "validate_clean_signal.csv")
    s2 = s2[(s2.d == D) & (s2.n_search == N_SEARCH) & (s2.seed == DATA_SEED)].set_index("method")

    mask_ff, best_ff, rec_ff, trace, state = run_ff(Xs, ys, DATA_SEED)
    mask_rs, best_rs, rec_rs = run_rs(Xs, ys, DATA_SEED)
    for name, m, b in (("firefly", mask_ff, best_ff), ("random_search", mask_rs, best_rs)):
        ref = s2.loc[name]
        same = np.isclose(b, ref.val_fitness, atol=1e-12) and \
            ",".join(map(str, np.flatnonzero(m))) == ref.idx
        print(f"  {name:<14} val_fit {b:.5f} (Stage 2: {ref.val_fitness:.5f}) "
              f"holdout {holdout_acc(m, Xs, ys, Xh, yh, DATA_SEED):.4f} "
              f"mask [{','.join(map(str, np.flatnonzero(m)))}] -> "
              f"{'REPRODUCED' if same else 'MISMATCH'}")

    for name, rec, best in (("firefly", rec_ff, best_ff), ("random_search", rec_rs, best_rs)):
        st = inf_stats(rec)
        print(f"\n  {name}: {st['n_evals']} new evals, {st['masks_with_inf']} contained an "
              f"informative column, distinct informative cols tried = {st['inf_cols_tried']}/{N_INF}")
        print(f"    best fitness among masks WITH informative col    : {st['best_fit_with_inf']:.5f}")
        print(f"    best fitness among masks WITHOUT informative col : {st['best_fit_without_inf']:.5f}")
        if st["masks_with_inf"] == 0:
            print("    -> NEVER TRIED an informative column (search coverage problem, not a bug)")
        elif st["best_fit_with_inf"] < best - 1e-12:
            print(f"    -> BUG: evaluated a better mask ({st['best_fit_with_inf']:.5f}) than it returned ({best:.5f})")
        else:
            print("    -> returned best is consistent with everything it evaluated")

    # per-sweep view, with how many evaluated masks contained an informative col
    has_inf = np.asarray(rec_ff.masks)[:, :N_INF].any(axis=1)
    pos = trace["init_evals"]
    print(f"\n  init: {trace['init_evals']} evals, best {trace['init_best']:.5f}, "
          f"masks with informative col: {int(has_inf[:pos].sum())}")
    sweep_rows = []
    for i, s in enumerate(trace["sweeps"]):
        n_inf = int(has_inf[pos:pos + s["new_evals"]].sum())
        pos += s["new_evals"]
        sweep_rows.append(dict(sweep=i, **s, masks_with_inf=n_inf))
    sw = pd.DataFrame(sweep_rows)
    print(sw.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    assert pos == state["evals"], "trace / recorder accounting mismatch"

    # ---------------- Part 2: frequency over algorithm seeds ---------------
    print(f"\n=== PART 2: same data & inner split, algorithm seeds {ALGO_SEEDS[0]}-{ALGO_SEEDS[-1]} ===")
    freq = []
    for a in ALGO_SEEDS:
        m, b, rec, _, _ = run_ff(Xs, ys, a)
        freq.append(dict(method="firefly", algo_seed=a, hits=int(m[:N_INF].sum()),
                         val_fitness=float(b), **inf_stats(rec)))
        m, b, rec = run_rs(Xs, ys, a)
        freq.append(dict(method="random_search", algo_seed=a, hits=int(m[:N_INF].sum()),
                         val_fitness=float(b), **inf_stats(rec)))
        print(f"  algo seed {a:>2} done | elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    fq = pd.DataFrame(freq)
    print(fq.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n  Summary over {len(ALGO_SEEDS)} algorithm seeds")
    for name, g in fq.groupby("method"):
        print(f"  {name:<14} zero-hit runs {int((g.hits == 0).sum())}/{len(g)} | "
              f"mean hits {g.hits.mean():.2f} | mean val_fit {g.val_fitness.mean():.4f} | "
              f"mean masks_with_inf {g.masks_with_inf.mean():.1f} | "
              f"runs that never tried one {int((g.masks_with_inf == 0).sum())}")

    csv_sw = RESULTS_DIR / "check_d500_seed1_sweeps.csv"
    csv_fq = RESULTS_DIR / "check_d500_seed1_freq.csv"
    sw.to_csv(csv_sw, index=False)
    fq.to_csv(csv_fq, index=False)
    total = time.perf_counter() - t0
    print(f"\nSaved {csv_sw}\nSaved {csv_fq}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="check_d500_seed1",
        params=dict(d=D, n_search=N_SEARCH, data_seed=DATA_SEED, fitness_seed=DATA_SEED,
                    ks=[K], budget=BUDGET, n_fireflies=N_FIREFLIES, class_sep=2.0),
        seeds=ALGO_SEEDS,
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (csv_sw, csv_fq)],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
