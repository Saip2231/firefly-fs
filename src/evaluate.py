"""Evaluation utilities for feature selection benchmark."""

import time
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score


def evaluate(mask_or_transform, X_train, y_train, X_test, y_test, clf):
    """Evaluate a classifier using a feature subset or transform.

    Parameters
    ----------
    mask_or_transform : np.ndarray or fitted transformer
        Boolean mask (1-D array) selecting columns, OR a fitted sklearn
        transformer with a .transform() method (e.g. fitted PCA).
    X_train, X_test : np.ndarray
        Training and test feature matrices.
    y_train, y_test : np.ndarray
        Training and test labels.
    clf : sklearn estimator
        Classifier to evaluate (will be cloned before fitting).

    Returns
    -------
    dict with keys:
        "accuracy"   : float – test set accuracy
        "n_features" : int   – number of features / components used
        "fit_time"   : float – seconds to fit the classifier
        "inference_time_per_sample" : float – seconds per test sample
    """
    # Apply mask or transform
    if isinstance(mask_or_transform, np.ndarray) and mask_or_transform.dtype == bool:
        X_tr = X_train[:, mask_or_transform]
        X_te = X_test[:, mask_or_transform]
        n_features = int(mask_or_transform.sum())
    elif hasattr(mask_or_transform, "transform"):
        X_tr = mask_or_transform.transform(X_train)
        X_te = mask_or_transform.transform(X_test)
        n_features = X_tr.shape[1]
    else:
        raise TypeError(
            "mask_or_transform must be a boolean numpy array or a fitted "
            "transformer with a .transform() method."
        )

    # Clone classifier to avoid side effects
    model = clone(clf)

    # Fit timing
    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    fit_time = time.perf_counter() - t0

    # Inference timing
    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    inference_time = time.perf_counter() - t0
    inference_time_per_sample = inference_time / len(X_te)

    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "n_features": n_features,
        "fit_time": fit_time,
        "inference_time_per_sample": inference_time_per_sample,
    }
