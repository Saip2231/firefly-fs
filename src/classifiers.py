"""The two classifier configurations used in the benchmark (single source).

final_clf()
    Scores every FINAL or HOLDOUT accuracy: baselines runner, synthetic
    holdouts, rerank ablation, clean-signal validation, tuning holdout and the
    metaheuristic runner.  Every method is scored with this one estimator so
    accuracy differences come from the feature subset, not the classifier.
    random_state is fixed at 0; the lbfgs solver ignores it, so results are
    identical to the earlier LogisticRegression(max_iter=500).

search_clf(seed)
    The cheap classifier inside the wrapper fitness (make_fitness) and rerank.
    src/firefly.py builds it from SEARCH_CLF_PARAMS directly (so tests can
    monkeypatch src.firefly.LogisticRegression); behaviour is unchanged.

Experiment scripts must not construct LogisticRegression themselves;
tests/test_classifiers.py enforces this with a grep.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

FINAL_CLF_PARAMS: dict = {"max_iter": 500, "random_state": 0}
SEARCH_CLF_PARAMS: dict = {"max_iter": 100}       # random_state = search seed


def final_clf() -> LogisticRegression:
    """Fresh, unfitted classifier for final / holdout accuracy."""
    return LogisticRegression(**FINAL_CLF_PARAMS)


def search_clf(seed: int) -> LogisticRegression:
    """Fresh, unfitted classifier as used inside the wrapper fitness."""
    return LogisticRegression(**SEARCH_CLF_PARAMS, random_state=seed)


def clf_configs() -> dict:
    """Both configurations, for runs.json records."""
    return {
        "final_clf":  f"LogisticRegression({_fmt(FINAL_CLF_PARAMS)})",
        "search_clf": f"LogisticRegression({_fmt(SEARCH_CLF_PARAMS)}, random_state=seed)",
    }


def _fmt(params: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items())
