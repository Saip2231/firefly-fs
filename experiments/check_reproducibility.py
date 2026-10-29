"""Day 4 Stage 0 - split reproducibility across separate processes.

For each dataset in {digits, cifar_r18} and seeds 0-9, hash the outputs of
load_dataset(name, seed): one SHA-256 per array (Xtr, Xte, ytr, yte; dtype and
shape are hashed with the bytes) plus one over all four.

The parent process launches this script twice as separate worker processes,
each writing its own CSV to a temporary directory, asserts the two files are
byte-identical, and only then saves results/split_hashes.csv.  It also prints
the Xtr / Xte shapes per dataset (for LOG.md).

Test data are only hashed, never inspected or used.

Usage
-----
    python experiments/check_reproducibility.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import hashlib  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from _common import RESULTS_DIR  # noqa: E402

from src.data import load_dataset  # noqa: E402

DATASETS = ["digits", "cifar_r18"]
SEEDS = list(range(10))
FIELDS = ["dataset", "seed", "Xtr_shape", "Xte_shape",
          "sha_Xtr", "sha_Xte", "sha_ytr", "sha_yte", "sha_all"]
OUT = RESULTS_DIR / "split_hashes.csv"


def sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def worker(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for name in DATASETS:
            for seed in SEEDS:
                Xtr, Xte, ytr, yte = load_dataset(name, seed=seed)
                parts = {k: sha(a) for k, a in
                         (("Xtr", Xtr), ("Xte", Xte), ("ytr", ytr), ("yte", yte))}
                w.writerow(dict(
                    dataset=name, seed=seed,
                    Xtr_shape="x".join(map(str, Xtr.shape)),
                    Xte_shape="x".join(map(str, Xte.shape)),
                    **{f"sha_{k}": v for k, v in parts.items()},
                    sha_all=hashlib.sha256("".join(parts.values()).encode()).hexdigest(),
                ))


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = [Path(tmp) / f"run{i}.csv" for i in (1, 2)]
        for p in runs:
            subprocess.run([sys.executable, __file__, "--worker", str(p)], check=True)
        b1, b2 = (p.read_bytes() for p in runs)
        assert b1 == b2, "split hashes differ between the two processes"
        shutil.copyfile(runs[0], OUT)

    with open(OUT, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Two separate processes produced identical hashes "
          f"({len(rows)} rows, {len(b1)} bytes). Saved {OUT}")

    print("\nSHAPES (per dataset; identical across seeds is checked below)")
    for name in DATASETS:
        r = [x for x in rows if x["dataset"] == name]
        tr = {x["Xtr_shape"] for x in r}
        te = {x["Xte_shape"] for x in r}
        print(f"  {name:<10} Xtr {','.join(sorted(tr))}  Xte {','.join(sorted(te))}  "
              f"(same for all {len(r)} seeds: {len(tr) == len(te) == 1})")
        n_distinct = len({x['sha_all'] for x in r})
        print(f"  {'':<10} distinct splits across seeds: {n_distinct}/{len(r)}")


if __name__ == "__main__":
    main()
