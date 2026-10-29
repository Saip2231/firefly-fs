"""Day 4 Stage 5 / Day 5: resumable runner for the metaheuristics.

One job = (dataset, method, budget, seed, k):
  1. Xtr, ytr from load_dataset(dataset, seed) (the test split is passed on,
     unread, to evaluate_on_test only).
  2. make_fitness(Xtr, ytr, k, seed): the inner split depends only on
     (dataset, seed), so it is identical for every method and budget.
  3. run_search(method, ...) with the hyperparameters from the config.
  4. rerank(top_m, n_splits-fold CV) on the TRAINING data only.
  5. evaluate_on_test: the raw best mask and the re-ranked mask are scored
     on the test split exactly once each with final_clf (acc_lr),
     knn_clf (acc_knn) and svm_clf (acc_svm).  This is the only function
     that names the test arrays (tests/test_runner.py checks it).
Two rows per job (mask_type raw | rerank).  Search cost is on both rows;
rerank_s / rerank_fits are 0 on the raw row (the raw mask does not need the
rerank) and the actual cost on the rerank row.

Resume: the main process appends each finished job's rows to
<out-dir>/meta_<dataset>.csv at once (single writer, flushed), so a crash
loses only in-flight jobs; on restart, keys already in the CSV are skipped.
--force reruns the selected jobs: their old rows are first moved to
meta_<dataset>_replaced.csv (append-only archive), never silently dropped.

k values: default = the k values of results/baselines_<dataset>.csv except
"All Features", and except k >= d (on digits k=64 = d is "all features";
binary_firefly needs k < d).  Skipped values are printed.

Usage (from firefly-fs/, venv active)
  python experiments/run_metaheuristics.py --smoke                 # digits, seeds 0-1, budget 150,
                                                                   # 2 ks, smoke config -> results/smoke/
  python experiments/run_metaheuristics.py --dataset digits --budgets 800 3000
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS_DIR, ROOT, git_hash, warn_once  # noqa: E402

from src.classifiers import clf_configs, final_clf, knn_clf, svm_clf  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.firefly import make_fitness, rerank  # noqa: E402
from src.results_store import log_experiment  # noqa: E402
from src.search_registry import run_search  # noqa: E402

DEFAULT_METHODS = ["firefly", "pso", "firefly_seeded", "pso_seeded"]
DEFAULT_BUDGETS = [800, 3000]
FROZEN_CONFIG = ROOT / "config" / "frozen_config.yaml"
SMOKE_CONFIG = ROOT / "config" / "smoke_config.yaml"
SMOKE = dict(dataset="digits", seeds=[0, 1], budgets=[150], n_ks=2)

SCHEMA = ["dataset", "method", "budget", "seed", "k", "mask_type", "acc_lr",
          "acc_knn", "acc_svm", "n_feat", "search_s", "rerank_s", "infer_s",
          "evals_used", "rerank_fits", "val_fitness", "diversity_last", "idx",
          "config_hash", "git_hash", "run_id"]
KEY = ["dataset", "method", "budget", "seed", "k"]
CONFIG_KEYS = {"firefly", "pso", "frac_seeded", "top_m", "n_splits"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path) -> tuple[dict, str]:
    """(config dict, SHA-256 of the file bytes).  YAML via PyYAML if present;
    otherwise the file must be in JSON flow style (valid YAML)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = json.loads(text)
    missing = CONFIG_KEYS - set(cfg)
    if missing:
        raise ValueError(f"{path}: missing config keys {sorted(missing)}")
    return cfg, hashlib.sha256(raw).hexdigest()


def method_hp(cfg: dict, method: str) -> dict:
    """Hyperparameters for run_search: the family block (+ frac_seeded)."""
    hp = dict(cfg["pso"] if method.startswith("pso") else cfg["firefly"])
    if method.endswith("_seeded"):
        hp["frac_seeded"] = cfg["frac_seeded"]
    return hp


def default_ks(dataset: str) -> tuple[list[int], list[int]]:
    """(usable ks, skipped ks) from results/baselines_<dataset>.csv."""
    df = pd.read_csv(RESULTS_DIR / f"baselines_{dataset}.csv")
    d = int(df.loc[df.method == "All Features", "k"].iloc[0])
    ks = sorted(int(k) for k in df.loc[df.method != "All Features", "k"].unique())
    return [k for k in ks if k < d], [k for k in ks if k >= d]


# ---------------------------------------------------------------------------
# Jobs and resume
# ---------------------------------------------------------------------------

def build_jobs(dataset, methods, budgets, seeds, ks) -> list[dict]:
    return [dict(dataset=dataset, method=m, budget=b, seed=s, k=k)
            for m in methods for b in budgets for s in seeds for k in ks]


def _key(row) -> tuple:
    return tuple(str(row[c]) for c in KEY)


def done_keys(csv_path) -> set:
    if not Path(csv_path).exists():
        return set()
    df = pd.read_csv(csv_path)
    return {_key(r) for _, r in df.iterrows()}


def archive_forced(csv_path, keys: set) -> int:
    """Move rows whose key is in *keys* to <name>_replaced.csv; return count."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or not keys:
        return 0
    df = pd.read_csv(csv_path, dtype={"idx": str})
    hit = df.apply(lambda r: _key(r) in keys, axis=1)
    if not hit.any():
        return 0
    arch = csv_path.with_name(csv_path.stem + "_replaced.csv")
    old = df[hit].assign(replaced_at=datetime.now().isoformat(timespec="seconds"))
    old.to_csv(arch, mode="a", header=not arch.exists(), index=False)
    df[~hit].to_csv(csv_path, index=False)
    return int(hit.sum())


def append_rows(rows: list[dict], csv_path) -> None:
    csv_path = Path(csv_path)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in SCHEMA})
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _data(dataset: str, seed: int):
    """Per-process cache of load_dataset(dataset, seed) (all four arrays)."""
    return load_dataset(dataset, seed=seed)


def evaluate_on_test(mask, data) -> dict:
    """The ONLY place the test split is used: one fit per classifier on the
    full training split with *mask*, scored once on the test split."""
    Xtr, Xte, ytr, yte = data
    lr = evaluate(mask, Xtr, ytr, Xte, yte, final_clf())
    knn = evaluate(mask, Xtr, ytr, Xte, yte, knn_clf())
    svm = evaluate(mask, Xtr, ytr, Xte, yte, svm_clf())
    return dict(acc_lr=lr["accuracy"], acc_knn=knn["accuracy"],
                acc_svm=svm["accuracy"], n_feat=lr["n_features"],
                infer_s=lr["inference_time_per_sample"])


def run_job(job: dict, cfg: dict, meta: dict) -> dict:
    """Run one job; returns {"rows": [raw, rerank], "warnings": [...]}."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = _run_job(job, cfg, meta)
    return {"rows": rows,
            "warnings": [(w.category.__name__, str(w.message)) for w in caught]}


def _run_job(job: dict, cfg: dict, meta: dict) -> list[dict]:
    data = _data(job["dataset"], job["seed"])
    Xtr, ytr = data[0], data[2]
    d, k, seed, budget = Xtr.shape[1], job["k"], job["seed"], job["budget"]

    t0 = time.perf_counter()
    fitness, state = make_fitness(Xtr, ytr, k=k, seed=seed)
    out = run_search(job["method"], fitness, state, d, k, budget, seed,
                     method_hp(cfg, job["method"]))
    search_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rr_mask, _, rr_fits = rerank(state, Xtr, ytr, k, top_m=cfg["top_m"],
                                 n_splits=cfg["n_splits"], seed=seed)
    rerank_s = time.perf_counter() - t0

    raw_mask = out["best_mask"]
    assert int(raw_mask.sum()) == k and int(rr_mask.sum()) == k
    assert out["evals_used"] <= budget
    common = dict(job, search_s=round(search_s, 4), evals_used=out["evals_used"],
                  diversity_last=out["diversity"][-1] if out["diversity"] else np.nan,
                  **meta)
    rows = []
    for mask_type, mask, rs, rf, vf in (
            ("raw", raw_mask, 0.0, 0, out["best_fit"]),
            ("rerank", rr_mask, round(rerank_s, 4), rr_fits,
             state["cache"][rr_mask.tobytes()])):
        rows.append(dict(common, mask_type=mask_type, rerank_s=rs, rerank_fits=rf,
                         val_fitness=float(vf),
                         idx=",".join(map(str, np.flatnonzero(mask))),
                         **evaluate_on_test(mask, data)))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    ap.add_argument("--seeds", type=int, default=10, help="N -> seeds 0..N-1")
    ap.add_argument("--seed-ids", nargs="+", type=int, default=None,
                    help="explicit seed list (overrides --seeds), e.g. for --force")
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--config", default=None,
                    help="default: config/frozen_config.yaml (smoke: smoke_config.yaml)")
    ap.add_argument("--n_jobs", "--n-jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/ (smoke: results/smoke/)")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write a runs.json entry (tests)")
    return ap.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    warn_once()

    dataset = SMOKE["dataset"] if args.smoke else args.dataset
    usable_ks, skipped_ks = default_ks(dataset)
    ks = args.ks or (usable_ks[:SMOKE["n_ks"]] if args.smoke else usable_ks)
    if args.ks is None and skipped_ks:
        print(f"Skipping k={skipped_ks} on {dataset} (k >= d)")
    if args.seed_ids is not None:
        seeds = args.seed_ids
    else:
        seeds = SMOKE["seeds"] if args.smoke else list(range(args.seeds))
    budgets = SMOKE["budgets"] if args.smoke and args.budgets == DEFAULT_BUDGETS else args.budgets
    config_path = Path(args.config) if args.config else (SMOKE_CONFIG if args.smoke else FROZEN_CONFIG)
    out_dir = Path(args.out_dir) if args.out_dir else (RESULTS_DIR / "smoke" if args.smoke else RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"meta_{dataset}.csv"

    cfg, cfg_hash = load_config(config_path)
    git = git_hash()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_meta_{dataset}"
    meta = dict(config_hash=cfg_hash, git_hash=git, run_id=run_id)

    jobs = build_jobs(dataset, args.methods, budgets, seeds, ks)
    if args.force:
        n_arch = archive_forced(csv_path, {_key(j) for j in jobs})
        todo = jobs
        print(f"--force: moved {n_arch} old rows to {csv_path.stem}_replaced.csv")
    else:
        done = done_keys(csv_path)
        todo = [j for j in jobs if _key(j) not in done]
    print(f"{run_id}: {len(jobs)} jobs ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"-> {csv_path}; config {config_path.name} sha256 {cfg_hash[:12]}; "
          f"git {git}; n_jobs={args.n_jobs}", flush=True)

    import run_metaheuristics as _self  # noqa: PLC0415  (workers import by name)
    t_all = time.perf_counter()
    seen_warn: set[str] = set()
    if todo:
        with joblib.parallel_backend("loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.n_jobs, return_as="generator")(
                joblib.delayed(_self.run_job)(j, cfg, meta) for j in todo)
            for n, res in enumerate(results, 1):
                append_rows(res["rows"], csv_path)
                for cat, msg in res["warnings"]:
                    if cat not in seen_warn:
                        seen_warn.add(cat)
                        print(f"[warning, shown once] {cat}: {msg} "
                              f"(further {cat}s suppressed)", flush=True)
                r = res["rows"][0]
                print(f"[{n:>4}/{len(todo)}] {r['method']:<15} b={r['budget']:<5} "
                      f"seed={r['seed']} k={r['k']:<4} evals={r['evals_used']} "
                      f"search {r['search_s']:.1f}s rerank {res['rows'][1]['rerank_s']:.1f}s "
                      f"| {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    total = time.perf_counter() - t_all

    n_rows = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
    print(f"\n{csv_path.name}: {n_rows} rows (this call: {2 * len(todo)} new); "
          f"runtime {total:.1f} s")
    if not args.no_log:
        log_experiment(
            name=f"meta_{dataset}" + ("_smoke" if args.smoke else ""),
            params=dict(dataset=dataset, methods=args.methods, budgets=budgets,
                        ks=ks, skipped_ks=skipped_ks, config_file=str(config_path.name),
                        config_hash=cfg_hash, config=cfg, force=args.force,
                        n_jobs=args.n_jobs, classifiers=clf_configs()),
            seeds=list(seeds),
            csv_paths=[str(csv_path.relative_to(ROOT))],
            extra=dict(run_id=run_id, git_hash=git, jobs_run=len(todo),
                       rows_total=n_rows, runtime_s=round(total, 1)),
        )
    return csv_path


if __name__ == "__main__":
    main()
