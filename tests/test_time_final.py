"""Tests for experiments/time_final.py (Stage A4).  The full timing and the
contention check are run by the user on an idle machine; here: helpers,
a tiny real run on digits, and the training-rows-only rule."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import time_final as TF  # noqa: E402

from experiments.analysis import common as C  # noqa: E402


def test_summarise_us_known_values():
    ns = np.array([1000, 2000, 3000, 4000, 5000])          # 1..5 us
    s = TF.summarise_us(ns)
    assert s["median_us"] == 3.0 and s["p25_us"] == 2.0 and s["p75_us"] == 4.0
    assert s["iqr_us"] == 2.0


def test_time_single_predict_counts_and_cycles():
    seen = []
    X = np.arange(12).reshape(6, 2)
    out = TF.time_single_predict(lambda row: seen.append(int(row[0, 0])), X, warmup=3, repeats=7)
    assert len(out) == 7 and (out >= 0).all() and len(seen) == 10
    assert seen == [0, 2, 4, 6, 8, 10, 0, 2, 4, 6]           # successive rows, cycling


def test_mask_and_contention_factor():
    m = TF.mask_from_idx(frozenset({1, 3}), 5)
    assert m.tolist() == [False, True, False, True, False]
    assert TF.contention_factor([30, 30], [10, 20]) == 2.0


def test_subsets_cover_methods_and_skip_k_ge_d():
    b, m = C.load_baselines("digits"), C.load_meta("digits")
    subs = list(TF.subsets("digits", b, m))
    methods = {s[0] for s in subs}
    assert {"all", "random", "anova", "mi", "l1", "rf", "pca",
            "firefly", "pso", "firefly_seeded", "pso_seeded"} <= methods
    assert all(s[2] < 64 for s in subs if s[0] != "all")
    masks = [s for s in subs if s[3] == "mask"]
    assert all(int(s[4].sum()) == s[2] for s in masks)
    assert len(subs) == 1 + 6 * 2 + 4 * 2 * 2                 # all + baselines + meta raw


def test_bracketed_row_saving_drift_and_stability():
    us = lambda *v: np.array(v) * 1000                       # noqa: E731  us -> ns
    r = TF.bracketed_row(us(100, 100, 100, 100), us(300, 300), us(300, 300))
    assert r["ref_all_median_us"] == 300 and r["saving_us_vs_all"] == 200
    assert r["ref_drift"] == 0 and r["stable"]
    drifted = TF.bracketed_row(us(100, 100), us(300, 300), us(150, 150))   # machine changed
    assert np.isclose(drifted["ref_drift"], 0.5) and not drifted["stable"]
    noisy = TF.bracketed_row(us(50, 100, 150, 200), us(300), us(300))      # IQR > 25 % median
    assert not noisy["stable"]


def test_tiny_real_timing_run():
    df = TF.time_inference("digits", warmup=2, repeats=5, ref_repeats=3)
    assert set(df.classifier) == {"lr", "knn"} and (df.rows == "training").all()
    assert (df.median_us > 0).all() and (df.ref_all_median_us > 0).all()
    assert {"saving_us_vs_all", "ref_drift", "stable"} <= set(df.columns)
    assert set(df[df.method == "pca"].includes) == {"PCA projection"}
    assert set(df[df.method == "firefly"].includes) == {"column slice"}
    assert set(df[df.method == "all"].n_features) == {64}


def test_burn_in_runs_for_its_duration():
    import time

    t0 = time.perf_counter()
    TF.burn_in(0.2)
    assert time.perf_counter() - t0 >= 0.2


def test_never_touches_test_split():
    src = (ROOT / "experiments" / "time_final.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert not names & {"Xte", "yte", "X_test", "y_test", "X_te", "y_te"}
    for bad in ("evaluate_on_test", "data[1]", "data[3]", "from src.evaluate"):
        assert bad not in src, bad
    assert "Xtr, _, ytr, _ = load_dataset" in src
