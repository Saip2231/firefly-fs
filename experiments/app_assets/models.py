"""Demo models (B1 item 3): seed-0 logistic regressions on each method's subset.

For every CIFAR k: the logged seed-0 masks of ANOVA, MI, L1, RF (baselines)
and of firefly, pso, firefly_seeded, pso_seeded (budget 800, raw mask), the
consensus subset, PCA (projection, not a selector) and all features.  Each
model is final_clf() fitted on the seed-0 TRAINING split, exactly as in the
benchmark; only coefficients and scaler statistics are saved.

Consensus subset: the k features plain Firefly selected most often across
seeds 0-9 at budget 800 (raw masks); ties go to the lower feature index.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from app.lib.models import DemoModel, model_key, predict_proba
from experiments.analysis import common as C
from src.baselines import pca_transform
from src.classifiers import final_clf
from src.data import load_dataset

from .common import RESULTS, raw_features, train_indices

DATASET = "cifar_r18"
SEED = 0
BUDGET = 800
BASELINE_SELECTORS = ("anova", "mi", "l1", "rf")
META_METHODS = ("firefly", "pso", "firefly_seeded", "pso_seeded")


def consensus_subset(idx_sets: list[frozenset[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
    """(top-k features by selection count, their counts); ties -> lower index."""
    counts: dict[int, int] = {}
    for s in idx_sets:
        for f in s:
            counts[f] = counts.get(f, 0) + 1
    ranked = sorted(counts, key=lambda f: (-counts[f], f))[:k]
    if len(ranked) < k:
        raise ValueError(f"only {len(ranked)} distinct features for k={k}")
    order = np.sort(np.array(ranked))
    return order, np.array([counts[f] for f in order])


def logged_masks(dataset: str = DATASET, seed: int = SEED, budget: int = BUDGET) -> dict:
    """(method, k) -> sorted idx array from the result CSVs (seed `seed`)."""
    base = C.load_baselines(dataset, RESULTS / f"baselines_{dataset}.csv")
    meta = C.load_meta(dataset, RESULTS / f"meta_{dataset}.csv")
    out = {}
    b0 = base[(base.seed == seed) & base.method.isin(BASELINE_SELECTORS)]
    for _, r in b0.iterrows():
        out[(r.method, int(r.k))] = np.array(sorted(r.idx))
    m0 = meta[(meta.seed == seed) & (meta.budget == budget) & (meta.mask_type == "raw")
              & meta.method.isin(META_METHODS)]
    for _, r in m0.iterrows():
        out[(r.method, int(r.k))] = np.array(sorted(r.idx))
    return out


def build(out_path) -> dict:
    """Fit and save every demo model; returns a summary for facts.json."""
    Xraw, y = raw_features(DATASET)
    tr = train_indices(y, SEED)
    scaler = StandardScaler().fit(Xraw[tr])
    Xtr, _, ytr, _ = load_dataset(DATASET, seed=SEED)
    if not (np.allclose(scaler.transform(Xraw[tr]), Xtr) and np.array_equal(y[tr], ytr)):
        raise RuntimeError("training rows / scaler do not reproduce load_dataset")
    d = Xtr.shape[1]

    meta = C.load_meta(DATASET, RESULTS / f"meta_{DATASET}.csv")
    ks = sorted(int(k) for k in meta.k.unique())
    masks = logged_masks()
    arrays: dict[str, np.ndarray] = {}
    keys: list[str] = []
    check_rows = Xtr[:200]

    def add(model: DemoModel, clf, extra: dict) -> None:
        key = model_key(model.method, model.k)
        if not np.allclose(predict_proba(model, check_rows), clf.predict_proba(
                model.features(check_rows)), atol=1e-8):
            raise RuntimeError(f"numpy prediction differs from sklearn for {key}")
        arrays[f"{key}__coef"], arrays[f"{key}__intercept"] = model.coef, model.intercept
        for name, value in extra.items():
            arrays[f"{key}__{name}"] = value
        keys.append(key)

    def add_subset(method: str, k: int, idx: np.ndarray, extra: dict | None = None) -> None:
        clf = final_clf().fit(Xtr[:, idx], ytr)
        add(DemoModel(method, k, clf.coef_, clf.intercept_, idx=idx), clf,
            dict(idx=idx, **(extra or {})))

    add_subset("all", d, np.arange(d))
    for k in ks:
        for method in BASELINE_SELECTORS + META_METHODS:
            add_subset(method, k, masks[(method, k)])
        sets = meta[(meta.method == "firefly") & (meta.budget == BUDGET)
                    & (meta.mask_type == "raw") & (meta.k == k)].idx.tolist()
        idx, counts = consensus_subset(sets, k)
        add_subset("consensus", k, idx, dict(counts=counts))
        pca = pca_transform(Xtr, ytr, k, seed=SEED)
        clf = final_clf().fit(pca.transform(Xtr), ytr)
        add(DemoModel("pca", k, clf.coef_, clf.intercept_, components=pca.components_,
                      pca_mean=pca.mean_), clf,
            dict(components=pca.components_, pca_mean=pca.mean_))

    np.savez_compressed(out_path, model_keys=np.array(keys), classes=np.unique(ytr),
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        dataset=np.str_(DATASET), seed=np.int32(SEED),
                        budget=np.int32(BUDGET), **arrays)
    return dict(dataset=DATASET, seed=SEED, meta_budget=BUDGET, ks=ks,
                n_models=len(keys), methods=sorted({k.split("__k")[0] for k in keys}),
                consensus_rule=("top-k features by selection count of plain firefly "
                                f"over seeds 0-9, budget {BUDGET}, raw masks; ties to "
                                "the lower index"),
                numpy_check="predict_proba equals sklearn on 200 training rows (atol 1e-8)")
