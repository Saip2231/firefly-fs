"""DASHBOARD.md Stage B1: build app/assets/ (the only writer of that folder).

Reads results/, LOG.md and the TRAINING split; writes only app/assets/.
Never reads a test split.  No runs.json entry: DASHBOARD.md rule 1 forbids
writes under results/; provenance goes to app/assets/facts.json.

Steps
  tables  slim copies of the analysis and result tables      (item 1)
  traces  re-run 18 logged searches with a population recorder  (item 2)
          and check the final mask against the logged idx (OK / FAIL)
  models  seed-0 demo models, numpy prediction               (item 3)
  digits  Digits training split, holdout carve, examples      (item 6)
  facts   facts.json with counts, provenance and SHA-256 of every asset
          (always rebuilt last)
Items 4 (ONNX) and 5 (gallery) come from experiments/day1/ (export_onnx.py,
build_gallery.py) after find_recipe.py has verified the Day-1 recipe; until
their files exist they are reported BLOCKED.

Usage (from firefly-fs/, venv active)
  python experiments/build_app_assets.py                  # everything
  python experiments/build_app_assets.py --only traces    # one step (+ facts)
  python experiments/build_app_assets.py --force          # redo existing traces
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import git_hash, warn_once  # noqa: E402

import run_metaheuristics as rm  # noqa: E402
from app_assets import common as A  # noqa: E402
from app_assets import digits, facts_build, models, tables, traces  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness  # noqa: E402

STEPS = ("tables", "traces", "models", "digits")
SEED = 0
TRACE_METHODS = ("firefly", "pso", "firefly_seeded")
TRACE_JOBS = [(ds, k, b, m)
              for ds, k in (("cifar_r18", 16), ("cifar_r18", 64), ("digits", 16))
              for b in (800, 3000) for m in TRACE_METHODS]
BLOCKED = {
    "onnx_export": ("BLOCKED: the Day-1 ResNet-18 embedding script (weights, resize, "
                    "normalisation, pooling) is not in the repo, and torch/torchvision "
                    "are not installed; an ONNX model cannot be shown to reproduce "
                    "data/cifar_r18_X.npy without them."),
    "gallery": ("BLOCKED: needs the CIFAR-10 test-set images (a download) and the "
                "Day-1 subset definition, to confirm the 10,000-image subset did not "
                "use the official test set and that labels follow the CIFAR-10 order."),
}


PRODUCED_BY = {   # blocked item -> (file that unblocks it, script that makes it)
    "onnx_export": ("resnet18_embed.onnx", "experiments/day1/export_onnx.py"),
    "gallery": ("gallery/gallery.json", "experiments/day1/build_gallery.py"),
}


def blocked_items() -> dict:
    """Items 4 and 5 stay BLOCKED until the Day-1 scripts have produced them."""
    return {name: reason for name, reason in BLOCKED.items()
            if not (A.ASSETS / PRODUCED_BY[name][0]).exists()}


def trace_path(ds: str, k: int, b: int, m: str):
    return A.TRACES / f"{ds}_k{k}_b{b}_{m}_s{SEED}.npz"


def environment() -> dict:
    """Interpreter and library versions (the logged runs used the project venv)."""
    import scipy  # noqa: PLC0415
    import sklearn  # noqa: PLC0415

    return dict(python=sys.executable, python_version=sys.version.split()[0],
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                in_project_venv=Path(sys.prefix).resolve() == (A.ROOT / ".venv").resolve())


def run_traces(force: bool, ks: list[int] | None = None) -> list[tuple[str, bool]]:
    cfg, cfg_hash = rm.load_config(rm.FROZEN_CONFIG)
    A.TRACES.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in TRACE_JOBS if ks is None or j[1] in ks]
    meta = {ds: pd.read_csv(A.RESULTS / f"meta_{ds}.csv", dtype={"idx": str})
            for ds in {j[0] for j in jobs}}
    data = {}
    results = []
    for n, (ds, k, b, m) in enumerate(jobs, 1):
        path = trace_path(ds, k, b, m)
        tag = f"[{n:>2}/{len(jobs)}] {ds:<9} k={k:<3} b={b:<4} {m:<15}"
        if path.exists() and not force:
            with np.load(path) as z:
                ok = bool(z["match"])
            print(f"{tag} {'OK' if ok else 'FAIL'} (existing file; --force to redo)")
            results.append((path.name, ok))
            continue
        df = meta[ds]
        row = df[(df.method == m) & (df.budget == b) & (df.seed == SEED) & (df.k == k)
                 & (df.mask_type == "raw")]
        if len(row) != 1:
            raise RuntimeError(f"{tag}: expected one logged raw row, found {len(row)}")
        row = row.iloc[0]
        if ds not in data:
            Xtr, _, ytr, _ = load_dataset(ds, seed=SEED)      # training split only
            data[ds] = (Xtr, ytr)
        Xtr, ytr = data[ds]

        t0 = time.perf_counter()
        fitness, state = make_fitness(Xtr, ytr, k=k, seed=SEED)
        out, rec = traces.record_search(m, fitness, state, Xtr.shape[1], k, b, SEED,
                                        rm.method_hp(cfg, m))
        secs = time.perf_counter() - t0
        arr = traces.trace_arrays(out, rec)
        idx_ok = traces.match_logged(arr["final_mask"], row.idx)
        fit_ok = bool(np.isclose(out["best_fit"], row.val_fitness, rtol=0, atol=1e-12))
        cfg_ok = cfg_hash == row.config_hash
        ok = idx_ok and fit_ok and cfg_ok and out["evals_used"] == row.evals_used
        np.savez_compressed(
            path, dataset=np.str_(ds), method=np.str_(m), k=np.int32(k),
            budget=np.int32(b), seed=np.int32(SEED), d=np.int32(Xtr.shape[1]),
            logged_idx=np.array(sorted(int(t) for t in row.idx.split(","))),
            logged_val_fitness=np.float64(row.val_fitness),
            logged_evals_used=np.int32(row.evals_used),
            logged_config_hash=np.str_(row.config_hash),
            logged_git_hash=np.str_(row.git_hash), config_hash=np.str_(cfg_hash),
            match=np.bool_(ok), replay_s=np.float64(secs), **arr)
        print(f"{tag} {'OK' if ok else 'FAIL'}  idx={'=' if idx_ok else '!='} "
              f"val_fitness={'=' if fit_ok else '!='} config={'=' if cfg_ok else '!='} "
              f"evals={out['evals_used']} frames={len(arr['frame_index'])}/"
              f"{int(arr['frames_total'])} {path.stat().st_size / 1e6:.2f} MB {secs:.1f} s",
              flush=True)
        results.append((path.name, ok))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--force", action="store_true", help="redo traces that exist")
    ap.add_argument("--ks", nargs="+", type=int, default=None,
                    help="limit the traces step to these k values, e.g. --ks 64")
    args = ap.parse_args(argv)
    warn_once()
    A.ASSETS.mkdir(parents=True, exist_ok=True)
    failures = []
    t_all = time.perf_counter()
    env = environment()
    print(f"python {env['python_version']} at {env['python']}; numpy {env['numpy']}, "
          f"scipy {env['scipy']}, scikit-learn {env['sklearn']}; project venv: "
          f"{env['in_project_venv']}", flush=True)
    if not env["in_project_venv"]:
        print("WARNING: not running in firefly-fs/.venv; the logged runs used the venv, "
              "so replays may not reproduce them.", flush=True)

    if "tables" in args.only:
        res = tables.build()
        print(f"tables: {len(res['copied'])} files -> {A.rel(A.TABLES)}"
              + (f"; missing {res['missing']}" if res["missing"] else ""))
        failures += [f"missing table {m}" for m in res["missing"]]
    if "traces" in args.only:
        res = run_traces(args.force, args.ks)
        n_ok = sum(ok for _, ok in res)
        print(f"traces: {n_ok}/{len(res)} OK (final mask matches the logged run)")
        failures += [f"trace FAIL {name}" for name, ok in res if not ok]
    if "models" in args.only:
        res = models.build(A.ASSETS / "demo_models.npz")
        print(f"models: {res['n_models']} demo models ({', '.join(res['methods'])}), "
              f"k={res['ks']}; {res['numpy_check']}")
    if "digits" in args.only:
        res = digits.build(A.ASSETS / "digits.npz")
        print(f"digits: {res['train_rows']} training rows -> {res['search_rows']} search / "
              f"{res['holdout_rows']} holdout, {res['examples']} examples")

    _, cfg_hash = rm.load_config(rm.FROZEN_CONFIG)
    blocked = blocked_items()
    facts = facts_build.build(git_hash(), cfg_hash, blocked, env)
    bc = facts["budgets_check"]
    print(f"facts: {A.rel(facts_build.FACTS)}; {len(facts['assets'])} assets, "
          f"{bc['assets_total_bytes'] / 1e6:.1f} MB total (cap "
          f"{bc['assets_max_bytes'] / 1e6:.0f} MB), budgets {'OK' if bc['ok'] else 'EXCEEDED'}; "
          f"config hash consistent: {facts['config_hash']['consistent']}")
    if not bc["ok"]:
        failures.append("asset size budget exceeded")
    if not facts["config_hash"]["consistent"]:
        failures.append("config hash differs between results and frozen config")
    for name, (file, script) in PRODUCED_BY.items():
        print(f"{name}: {blocked[name]}" if name in blocked
              else f"{name}: built by {script} (app/assets/{file})")
    print(f"done in {time.perf_counter() - t_all:.1f} s; "
          + ("FAILURES: " + "; ".join(failures) if failures else "all checks OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
