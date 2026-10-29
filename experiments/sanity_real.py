"""Day 4 Stage 3 sanity check on real data: all five registry methods.

Digits, load_dataset seed 0, k=16, budget 300, search seed 0, default
hyperparameters (tuning is Stage 4).  Only the training split is used: the
test split returned by load_dataset is discarded unread.  Test accuracy is
NOT computed or printed.

Per method (fresh make_fitness each run, so the same inner split): validation
fitness of the initial population (init_best) and at the end (best_fit),
evals used, sweeps / iterations, seconds.  Reference: validation fitness of
the ANOVA top-k mask built from state["filter_scores"] (inner-train rows,
same inner split).  Asserted: every seeded method starts at or below it
(member 0 of the seeded population IS that mask).

Validation fitness is in-sample (the rows that chose the mask); this is a
plumbing check, not a comparison of methods.

Output: results/sanity_real.csv, entry in results/runs.json.

Usage (from firefly-fs/, venv active):
    python experiments/sanity_real.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import pandas as pd  # noqa: E402

from _common import RESULTS_DIR, warn_once  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness  # noqa: E402
from src.init_pop import top_k_mask  # noqa: E402
from src.results_store import log_experiment  # noqa: E402
from src.search_registry import METHODS, run_search  # noqa: E402

DATASET, DATA_SEED, SEED, K, BUDGET = "digits", 0, 0, 16, 300


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    Xtr, _, ytr, _ = load_dataset(DATASET, seed=DATA_SEED)   # test split unused
    d = Xtr.shape[1]
    t_all = time.perf_counter()

    ref_fit, ref_state = make_fitness(Xtr, ytr, k=K, seed=SEED)
    anova = top_k_mask(ref_state["filter_scores"], K)
    anova_fit = float(ref_fit(anova))

    rows = []
    for method in METHODS:
        fitness, state = make_fitness(Xtr, ytr, k=K, seed=SEED)
        t0 = time.perf_counter()
        out = run_search(method, fitness, state, d, K, BUDGET, SEED, hp={})
        secs = time.perf_counter() - t0
        assert int(out["best_mask"].sum()) == K and out["evals_used"] <= BUDGET
        rows.append(dict(dataset=DATASET, data_seed=DATA_SEED, seed=SEED, k=K,
                         budget=BUDGET, method=method, init_fitness=out["init_best"],
                         final_fitness=out["best_fit"], evals_used=out["evals_used"],
                         iters=len(out["history"]),
                         diversity_last=out["diversity"][-1] if out["diversity"] else None,
                         seconds=secs, anova_fitness=anova_fit,
                         idx=",".join(map(str, out["best_mask"].nonzero()[0]))))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n{DATASET}, data seed {DATA_SEED}, k={K}, budget={BUDGET}, default hp; "
          f"validation fitness (lower is better, in-sample)")
    print(f"ANOVA top-k mask (inner-train scores, same inner split): {anova_fit:.4f}\n")
    print(df[["method", "init_fitness", "final_fitness", "evals_used", "iters",
              "diversity_last", "seconds"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    seeded = df[df.method.str.endswith("_seeded")]
    for _, r in seeded.iterrows():
        assert r.init_fitness <= anova_fit, (
            f"{r.method} starts at {r.init_fitness:.4f} > ANOVA {anova_fit:.4f}")
    print(f"\nCheck: seeded methods start at or below the ANOVA mask "
          f"({', '.join(f'{m} {v:.4f}' for m, v in zip(seeded.method, seeded.init_fitness))}): OK")

    csv = RESULTS_DIR / "sanity_real.csv"
    df.to_csv(csv, index=False)
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="sanity_real",
        params=dict(dataset=DATASET, data_seed=DATA_SEED, search_seed=SEED, ks=[K],
                    budget=BUDGET, methods=list(METHODS), hp="defaults",
                    reference="ANOVA top-k on inner-train rows"),
        seeds=[SEED],
        csv_paths=[str(csv.relative_to(RESULTS_DIR.parent))],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
