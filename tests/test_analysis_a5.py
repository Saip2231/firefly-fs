"""Tests for Stage A5 (experiments/analysis/a5_stability.py): known Jaccard
values, the analytic random reference, frequency counts, and structure and
labels on the real results."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from experiments.analysis import a5_stability as A5
from experiments.analysis import common as C


def test_jaccard_known_values():
    assert A5.jaccard(frozenset({1, 2, 3}), frozenset({2, 3, 4})) == 2 / 4
    assert A5.jaccard(frozenset({1, 2}), frozenset({1, 2})) == 1.0
    assert A5.jaccard(frozenset({1}), frozenset({2})) == 0.0


def test_pairwise_mean_matches_plain_loop():
    sets = [frozenset({1, 2, 3}), frozenset({2, 3, 4}), frozenset({1, 2, 3})]
    j = A5.pairwise(sets)
    assert len(j) == 3
    ref = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    assert np.allclose(sorted(j), sorted(ref)) and np.isclose(j.mean(), (0.5 + 1.0 + 0.5) / 3)


def test_analytic_random_reference():
    # k = 16, d = 64: E|A&B| = 4, E|A|B| = 28 -> 1/7
    assert np.isclose(A5.analytic_random(16, 64), 4 / 28)
    # Monte-Carlo agreement is approximate (ratio of expectations): report, not a threshold
    rng = np.random.default_rng(0)
    sims = [A5.jaccard(frozenset(rng.choice(64, 16, replace=False)),
                       frozenset(rng.choice(64, 16, replace=False))) for _ in range(2000)]
    assert 0 < np.mean(sims) < 1


def test_frequency_counts():
    f = A5.frequency([frozenset({0, 1}), frozenset({1, 2}), frozenset({1})], d=4)
    assert f.tolist() == [1, 3, 1, 0] and f.sum() == 5


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("a5")
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    return out, baselines, meta, A5.make_outputs(baselines, meta, out_dir=out, fig_dir=out)


def test_real_stability_structure(real):
    _, baselines, meta, res = real
    t = res["stability"]
    assert (t.n_seeds == 10).all() and (t.n_pairs == 45).all()
    assert t.jaccard_mean.between(0, 1).all()
    assert not t.method.isin(["pca", "all"]).any()
    assert not ((t.dataset == "digits") & (t.k == 64)).any()          # k = d excluded
    assert (t.analysis == "exploratory").all() and t.caveat.str.contains("resampling").all()
    n_meta = sum(m.groupby(["method", "k", "budget", "mask_type"]).ngroups for m in meta.values())
    n_base = sum(b[~b.method.isin(["pca", "all"]) & b.k.isin(C.selection_ks(b))]
                 .groupby(["method", "k"]).ngroups for b in baselines.values())
    assert len(t) == n_meta + n_base


def test_real_value_matches_direct_computation(real):
    _, _, meta, res = real
    t = res["stability"]
    m = meta["cifar_r18"]
    sets = list(m[(m.method == "firefly") & (m.k == 16) & (m.budget == 800)
                  & (m.mask_type == "raw")].sort_values("seed").idx)
    ref = np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])
    row = t[(t.dataset == "cifar_r18") & (t.method == "firefly") & (t.k == 16)
            & (t.budget == 800) & (t.mask_type == "raw")].iloc[0]
    assert np.isclose(row.jaccard_mean, ref)
    assert np.isclose(row.random_analytic, A5.analytic_random(16, 512))


def test_real_frequency_table(real):
    _, _, _, res = real
    f = res["frequency"]
    assert (f.n_features_all10 <= f.n_features_ge5).all()
    assert (f.n_features_ge5 <= f.n_features_ever).all()
    assert (f.n_features_ever >= f.k).all() and (f.n_features_ever <= f.d).all()


def test_real_figures_and_takeaways(real):
    out, baselines, _, res = real
    expected = {f"F6_stability_{ds}" for ds in C.DATASETS}
    expected |= {f"F7_feature_frequency_{ds}_k{k}" for ds in C.DATASETS
                 for k in C.selection_ks(baselines[ds])}
    assert set(res["figures"]) == expected
    for n in expected:
        assert (out / f"{n}.png").exists() and (out / f"{n}.pdf").exists()
        assert "exploratory" in res["figures"][n]["takeaway"]
    assert set(pd.read_csv(out / "takeaways.csv").figure) == expected
