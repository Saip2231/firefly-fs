"""Stage B1 tests (DASHBOARD.md): app asset build.

(a) the population recorder does not change any search result, and its
    frames agree with the search's own history
(b) helpers: frame cap, logged-idx match, consensus subset, training rows,
    numpy prediction, facts parsers
(c) static: the builders never name a test split
(d) built assets (skipped until `python experiments/build_app_assets.py` has
    run): every trace OK, budgets, checksums, numbers equal pandas
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_app_assets as B  # noqa: E402
from app_assets import common as A  # noqa: E402
from app_assets import facts_build, models, traces  # noqa: E402

from app.lib.models import DemoModel, load_models, model_key, predict_proba  # noqa: E402
from src.classifiers import final_clf  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness  # noqa: E402
from src.search_registry import run_search  # noqa: E402

FACTS = A.ASSETS / "facts.json"
HP = {"firefly": dict(n_fireflies=6), "pso": dict(n_particles=6),
      "firefly_seeded": dict(n_fireflies=6, frac_seeded=0.5)}


def _toy():
    X, y = make_classification(n_samples=300, n_features=30, n_informative=5,
                               n_redundant=0, shuffle=False, random_state=0)
    return X, y


# (a) -----------------------------------------------------------------------

@pytest.mark.parametrize("method", list(HP))
def test_recorder_does_not_change_the_search(method):
    X, y = _toy()
    f1, s1 = make_fitness(X, y, k=5, seed=3)
    plain = run_search(method, f1, s1, 30, 5, 90, 3, HP[method])
    f2, s2 = make_fitness(X, y, k=5, seed=3)
    rec_out, rec = traces.record_search(method, f2, s2, 30, 5, 90, 3, HP[method])
    assert np.array_equal(plain["best_mask"], rec_out["best_mask"])
    assert plain["best_fit"] == rec_out["best_fit"]
    assert plain["history"] == rec_out["history"]
    assert plain["diversity"] == rec_out["diversity"]
    assert plain["evals_used"] == rec_out["evals_used"]


@pytest.mark.parametrize("method", list(HP))
def test_frames_agree_with_history(method):
    X, y = _toy()
    f, s = make_fitness(X, y, k=5, seed=1)
    out, rec = traces.record_search(method, f, s, 30, 5, 90, 1, HP[method])
    fr = rec.frames
    assert len(fr) == len(out["history"]) + 1            # initial + one per sweep
    assert [x["best"] for x in fr[1:]] == out["history"]
    assert fr[0]["best"] == out["init_best"]
    assert all(x["pop"].shape == (6, 30) and (x["pop"].sum(axis=1) == 5).all() for x in fr)
    assert all(a["best"] >= b["best"] for a, b in zip(fr, fr[1:]))
    assert fr[-1]["evals"] == out["evals_used"] <= 90
    assert rec.value_kind in ("current fitness", "personal-best fitness")


def test_trace_arrays_cap_keeps_ends_and_full_curve():
    X, y = _toy()
    f, s = make_fitness(X, y, k=5, seed=0)
    out, rec = traces.record_search("firefly", f, s, 30, 5, 150, 0, HP["firefly"])
    arr = traces.trace_arrays(out, rec, cap=3)
    n = len(rec.frames)
    assert n > 3
    assert arr["frame_index"].tolist() == A.subsample_frames(n, 3).tolist()
    assert arr["frame_index"][0] == 0 and arr["frame_index"][-1] == n - 1
    assert arr["pop"].shape[0] == 3 and len(arr["curve_best"]) == n == int(arr["frames_total"])
    assert np.array_equal(arr["frame_best_mask"][-1], arr["final_mask"])


# (b) -----------------------------------------------------------------------

def test_subsample_frames():
    assert A.subsample_frames(5, 60).tolist() == [0, 1, 2, 3, 4]
    s = A.subsample_frames(1000, 60)
    assert len(s) <= 60 and s[0] == 0 and s[-1] == 999 and (np.diff(s) > 0).all()


def test_match_logged():
    m = np.zeros(10, dtype=bool)
    m[[1, 4, 7]] = True
    assert traces.match_logged(m, "1,4,7") and traces.match_logged(m, "7,1,4")
    assert not traces.match_logged(m, "1,4,8")


def test_consensus_subset_counts_and_ties():
    sets = [frozenset({0, 1, 2}), frozenset({1, 2, 3}), frozenset({2, 3, 4})]
    idx, counts = models.consensus_subset(sets, 3)
    assert idx.tolist() == [1, 2, 3] and counts.tolist() == [2, 3, 2]
    with pytest.raises(ValueError):
        models.consensus_subset([frozenset({0})], 2)


@pytest.mark.parametrize("seed", [0, 3])
def test_train_indices_reproduce_load_dataset(seed):
    Xraw, y = A.raw_features("digits")
    tr = A.train_indices(y, seed)
    Xtr, _, ytr, _ = load_dataset("digits", seed=seed)
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    assert np.array_equal(y[tr], ytr)
    assert np.allclose(StandardScaler().fit_transform(Xraw[tr]), Xtr)


def test_numpy_prediction_equals_sklearn():
    X, y = make_classification(n_samples=400, n_features=12, n_informative=6,
                               n_classes=4, random_state=0)
    idx = np.array([0, 3, 5, 9])
    clf = final_clf().fit(X[:, idx], y)
    model = DemoModel("anova", 4, clf.coef_, clf.intercept_, idx=idx)
    assert np.allclose(predict_proba(model, X), clf.predict_proba(X[:, idx]), atol=1e-10)


def test_facts_parsers():
    assert facts_build.last_pytest_count("231 passed ... later 261 passed in 3 s") == 261
    assert facts_build.last_pytest_count("nothing") is None
    assert facts_build.contention_factor("contention_factor = 2.95 (mean ...)") == 2.95


# (c) -----------------------------------------------------------------------

def test_builders_never_name_the_test_split():
    files = [ROOT / "experiments" / "build_app_assets.py",
             *sorted((ROOT / "experiments" / "app_assets").glob("*.py"))]
    for p in files:
        assert not re.search(r"\b(Xte|yte)\b", p.read_text(encoding="utf-8")), p


def test_trace_job_list():
    assert len(B.TRACE_JOBS) == len(set(B.TRACE_JOBS)) == 18


# (d) built assets ------------------------------------------------------------

built = pytest.mark.skipif(not FACTS.exists(),
                           reason="run python experiments/build_app_assets.py first")


@built
def test_every_trace_matches_the_logged_run_and_is_small():
    meta = {ds: pd.read_csv(A.RESULTS / f"meta_{ds}.csv", dtype={"idx": str})
            for ds in ("digits", "cifar_r18")}
    for ds, k, b, m in B.TRACE_JOBS:
        path = B.trace_path(ds, k, b, m)
        assert path.exists(), path
        assert path.stat().st_size <= A.TRACE_MAX_BYTES
        with np.load(path) as z:
            assert bool(z["match"]), path.name
            row = meta[ds].query("method == @m and budget == @b and seed == 0 "
                                 "and k == @k and mask_type == 'raw'").iloc[0]
            assert traces.match_logged(z["final_mask"], row.idx)
            assert len(z["frame_index"]) <= A.MAX_FRAMES
            assert (z["pop"].sum(axis=2) == k).all()
            assert int(z["curve_evals"][-1]) == int(z["evals_used"]) == int(row.evals_used)


@built
def test_facts_numbers_equal_pandas_and_checksums():
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    rq1 = pd.read_csv(A.ANALYSIS / "rq1_tests.csv")
    meta = pd.concat(pd.read_csv(A.RESULTS / f"meta_{ds}.csv") for ds in ("digits", "cifar_r18"))
    assert facts["rows"]["meta_total"] == len(meta)
    assert facts["rows"]["meta_raw"] == int((meta.mask_type == "raw").sum())
    assert facts["rq1"]["tests"] == len(rq1)
    assert facts["rq1"]["labels"] == {lab: int((rq1.label == lab).sum())
                                      for lab in ("WIN", "LOSS", "TIE")}
    assert facts["config_hash"]["consistent"] and facts["budgets_check"]["ok"]
    for relpath, info in facts["assets"].items():
        assert A.sha256(ROOT / relpath) == info["sha256"], relpath
    assert all(t["match"] for t in facts["traces"]) and len(facts["traces"]) == 18


@built
def test_demo_models_and_digits_assets():
    with np.load(A.ASSETS / "demo_models.npz") as z:
        ms = load_models(z)
        x = np.zeros((1, len(z["scaler_mean"])))
    for key, model in ms.items():
        p = predict_proba(model, x)
        assert p.shape == (1, 10) and np.isclose(p.sum(), 1.0), key
    meta = pd.read_csv(A.RESULTS / "meta_cifar_r18.csv", dtype={"idx": str})
    sets = [frozenset(int(t) for t in s.split(","))
            for s in meta.query("method == 'firefly' and budget == 800 and "
                                "mask_type == 'raw' and k == 16").idx]
    assert np.array_equal(ms[model_key("consensus", 16)].idx,
                          models.consensus_subset(sets, 16)[0])
    with np.load(A.ASSETS / "digits.npz") as d:
        s, h = set(d["search_pos"].tolist()), set(d["holdout_pos"].tolist())
        assert not s & h and len(s | h) == len(d["y_train"])
        assert set(d["example_pos"].tolist()) <= h
        assert d["X_train_raw"].max() <= 16
