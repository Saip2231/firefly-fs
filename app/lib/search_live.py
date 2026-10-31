"""Capped live Firefly search (DASHBOARD.md rule 6) on the Digits training
split or a planted toy.  Never touches a test split.

Data: Digits rows come from app/assets/digits.npz (the seed-0 training split
and its search / holdout carve); the toy is make_classification with the
informative columns first.  The scaler is fitted on the search rows only, so
the holdout stays unseen until the final score.

Caps: at most 300 evaluations, 30 s wall-clock, BLAS limited to one thread.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from app.lib import data
from experiments.app_assets.traces import PopulationRecorder
from src.classifiers import final_clf
from src.firefly import make_fitness
from src.init_pop import top_k_mask
from src.search_registry import run_search

MAX_EVALS = 300
MAX_SECONDS = 30.0
FITNESS_W = 0.01            # make_fitness default sparsity weight
TOY = dict(n_samples=600, n_features=40, n_informative=6)


class TimeCap(Exception):
    """Raised inside the fitness wrapper when the wall-clock cap is reached."""


def digits_split() -> dict:
    with np.load(data.ASSETS / "digits.npz") as z:
        X = z["X_train_raw"].astype(float)
        y = z["y_train"]
        return dict(X=X, y=y, search=z["search_pos"], holdout=z["holdout_pos"],
                    examples=z["example_pos"], name="Digits training split")


def toy_split(seed: int) -> dict:
    X, y = make_classification(n_samples=TOY["n_samples"], n_features=TOY["n_features"],
                               n_informative=TOY["n_informative"], n_redundant=0,
                               shuffle=False, random_state=seed)
    s, h = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    return dict(X=X, y=y, search=np.sort(s), holdout=np.sort(h),
                examples=np.sort(h)[:0], name="planted toy")


def val_accuracy(fitness_value: float, k: int, d: int) -> float:
    """Inner validation accuracy from a fitness value (undo the constant w*k/d)."""
    return 1.0 - (fitness_value - FITNESS_W * k / d) / (1.0 - FITNESS_W)


def run(split: dict, k: int, n_pop: int, budget: int, alpha: float, seed: int,
        progress=None) -> dict:
    """One capped search; returns masks, frames, fitness and holdout scores."""
    budget = int(min(max(1, budget), MAX_EVALS))
    X, y = split["X"], split["y"]
    s, h = split["search"], split["holdout"]
    mean, scale = X[s].mean(axis=0), X[s].std(axis=0)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale
    d = X.shape[1]
    t0 = time.perf_counter()
    with threadpool_limits(1), warnings.catch_warnings():
        # Digits edge pixels are constant on the search rows; ANOVA F then
        # warns (constant features, 0/0).  Expected and harmless: those
        # pixels score 0 and are never preferred.
        warnings.filterwarnings("ignore", message="Features .* are constant",
                                category=UserWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                category=RuntimeWarning)
        fitness, state = make_fitness(Xs[s], y[s], k=k, seed=seed)

        def capped(mask):
            if time.perf_counter() - t0 > MAX_SECONDS:
                raise TimeCap
            value = fitness(mask)
            if progress is not None:
                progress(min(state["evals"], budget) / budget)
            return value

        stopped = None
        with PopulationRecorder() as rec:
            try:
                out = run_search("firefly", capped, state, d, k, budget, seed,
                                 dict(n_fireflies=n_pop, alpha=alpha))
                best_mask, best_fit = out["best_mask"], out["best_fit"]
            except TimeCap:
                stopped = f"stopped at the {MAX_SECONDS:.0f} s cap"
                if not rec.frames:
                    raise
                best_mask, best_fit = rec.frames[-1]["best_mask"], rec.frames[-1]["best"]
        evals = int(state["evals"])
        anova = top_k_mask(state["filter_scores"], k)
        anova_fit = float(fitness(anova))           # reference only, outside the budget

        def holdout(mask):
            clf = final_clf().fit(Xs[s][:, mask], y[s])
            return clf, float(clf.score(Xs[h][:, mask], y[h]))

        clf, acc = holdout(best_mask)
        _, anova_acc = holdout(anova)
    return dict(
        best_mask=best_mask, evals=evals, budget=budget, seconds=time.perf_counter() - t0,
        stopped=stopped, d=d, k=k,
        val_acc=val_accuracy(best_fit, k, d), holdout_acc=acc,
        anova_mask=anova, anova_val_acc=val_accuracy(anova_fit, k, d),
        anova_holdout_acc=anova_acc, model=clf, mean=mean, scale=scale,
        frames=[dict(best_mask=f["best_mask"], best=f["best"], evals=f["evals"])
                for f in rec.frames])


def predict(result: dict, x_raw: np.ndarray) -> tuple[int, np.ndarray]:
    """Class and probabilities for one raw row, using the found subset."""
    x = ((np.atleast_2d(x_raw) - result["mean"]) / result["scale"])[:, result["best_mask"]]
    proba = result["model"].predict_proba(x)[0]
    return int(result["model"].classes_[proba.argmax()]), proba
