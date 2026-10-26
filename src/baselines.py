"""Standard feature-selection baselines for benchmarking."""

import warnings
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def select_all(X, y, k=None):
    """Select all features (no selection). k is ignored."""
    return np.ones(X.shape[1], dtype=bool)


def select_random(X, y, k, seed=42):
    """Randomly select exactly k features."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(X.shape[1], size=k, replace=False)
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[indices] = True
    return mask


def select_kbest_anova(X, y, k):
    """Select top k features by ANOVA F-test score."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    top_k = np.argsort(scores)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


def select_kbest_mi(X, y, k, seed=42):
    """Select top k features by mutual information."""
    scores = mutual_info_classif(X, y, random_state=seed)
    top_k = np.argsort(scores)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


def select_model_lasso(X, y, k, seed=42):
    """Select top k features by L1-penalised logistic regression coefficients."""
    lr = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        random_state=seed, C=1.0,
    )
    lr.fit(X, y)
    importance = np.mean(np.abs(lr.coef_), axis=0)
    top_k = np.argsort(importance)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


def pca_transform(X, y, k, seed=42):
    """Fit PCA with k components on X. Returns the fitted PCA transformer.

    Unlike the mask-based selectors, this returns a fitted PCA object.
    The evaluate() function handles both cases.
    """
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(X)
    return pca
