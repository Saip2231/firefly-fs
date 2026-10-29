"""Tests for experiments/run_metaheuristics.py on the smoke settings
(digits, seeds 0-1, budget 150, two k values, config/smoke_config.yaml),
written to a temporary directory (results/ and runs.json are not touched).

(a) expected row count and exact column schema
(b) rerunning without --force adds zero rows
(c) --force on one job archives its old rows and reproduces identical idx
    and acc_lr
(d) evals_used <= budget on every row; n_feat == k; rerank cost only on
    rerank rows
(e) static check: Xte / yte appear only inside evaluate_on_test, and no
    other code indexes the test arrays out of the load_dataset tuple
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import run_metaheuristics as rm  # noqa: E402

N_JOBS = "2"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    csv_path = rm.main(["--smoke", "--out-dir", str(out), "--n_jobs", N_JOBS, "--no-log"])
    return out, csv_path


def _expected_rows():
    ks, _ = rm.default_ks("digits")
    return (len(rm.DEFAULT_METHODS) * len(rm.SMOKE["budgets"])
            * len(rm.SMOKE["seeds"]) * min(rm.SMOKE["n_ks"], len(ks)) * 2)


# (a) -----------------------------------------------------------------------

def test_rows_and_schema(smoke):
    _, csv_path = smoke
    df = pd.read_csv(csv_path, dtype={"idx": str})
    assert list(df.columns) == rm.SCHEMA
    assert len(df) == _expected_rows() == 32
    assert set(df.mask_type) == {"raw", "rerank"}
    assert df.groupby(rm.KEY).size().eq(2).all()            # one raw + one rerank
    assert set(df.k) == {16, 32} and set(df.seed) == {0, 1} and set(df.budget) == {150}
    assert df.config_hash.nunique() == 1 and df.run_id.nunique() == 1


# (b) -----------------------------------------------------------------------

def test_rerun_adds_zero_rows(smoke):
    out, csv_path = smoke
    before = pd.read_csv(csv_path, dtype={"idx": str})
    rm.main(["--smoke", "--out-dir", str(out), "--n_jobs", N_JOBS, "--no-log"])
    after = pd.read_csv(csv_path, dtype={"idx": str})
    pd.testing.assert_frame_equal(before, after)


# (c) -----------------------------------------------------------------------

def test_force_reproduces_one_job(smoke):
    out, csv_path = smoke
    rm.main(["--smoke", "--out-dir", str(out), "--n_jobs", "1", "--no-log",
             "--methods", "pso_seeded", "--seed-ids", "1", "--ks", "16", "--force"])
    df = pd.read_csv(csv_path, dtype={"idx": str})
    arch = pd.read_csv(out / "meta_digits_replaced.csv", dtype={"idx": str})
    assert len(df) == _expected_rows()                      # replaced, not duplicated
    assert len(arch) == 2
    sel = (df.method == "pso_seeded") & (df.seed == 1) & (df.k == 16)
    new = df[sel].sort_values("mask_type").reset_index(drop=True)
    old = arch.sort_values("mask_type").reset_index(drop=True)
    assert list(new.idx) == list(old.idx)
    assert list(new.acc_lr) == list(old.acc_lr)
    assert list(new.val_fitness) == list(old.val_fitness)


# (d) -----------------------------------------------------------------------

def test_budget_and_costs(smoke):
    _, csv_path = smoke
    df = pd.read_csv(csv_path, dtype={"idx": str})
    assert (df.evals_used <= df.budget).all()
    assert (df.n_feat == df.k).all()
    assert all(len(s.split(",")) == k for s, k in zip(df.idx, df.k))
    raw, rr = df[df.mask_type == "raw"], df[df.mask_type == "rerank"]
    assert (raw.rerank_fits == 0).all() and (raw.rerank_s == 0).all()
    assert (rr.rerank_fits == 10 * 5).all()
    for col in ("acc_lr", "acc_knn", "acc_svm"):
        assert df[col].between(0, 1).all()


# (e) -----------------------------------------------------------------------

def test_test_split_only_in_evaluate_on_test():
    src = (ROOT / "experiments" / "run_metaheuristics.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_on_test")
    inside = range(fn.lineno, fn.end_lineno + 1)
    hits = [n.lineno for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id in {"Xte", "yte"})
            or (isinstance(n, ast.arg) and n.arg in {"Xte", "yte"})]
    assert hits, "evaluate_on_test should unpack Xte, yte"
    assert all(ln in inside for ln in hits), f"Xte/yte outside evaluate_on_test: {hits}"
    # nobody pulls the test arrays out of the load_dataset tuple by position
    for bad in ("data[1]", "data[3]", "[1::2]"):
        assert bad not in src, bad


def test_config_loader_and_hp():
    cfg, h = rm.load_config(rm.SMOKE_CONFIG)
    assert len(h) == 64
    assert rm.method_hp(cfg, "pso_seeded")["frac_seeded"] == cfg["frac_seeded"]
    assert "frac_seeded" not in rm.method_hp(cfg, "firefly")
    assert rm.method_hp(cfg, "firefly")["n_fireflies"] == cfg["firefly"]["n_fireflies"]
    ks, skipped = rm.default_ks("digits")
    assert ks == [16, 32] and skipped == [64]
