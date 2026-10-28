"""Ground-truth helpers for sklearn.make_classification toys.

Why "informative" != "useful"
-----------------------------
make_classification places each cluster centroid on a vertex of the
hypercube {-class_sep, +class_sep}^n_informative, adds standard-normal noise
mixed by a random per-cluster linear map, and shifts it to that vertex
(sklearn/datasets/_samples_generator.py).  A class's mean on an informative
column is therefore the average of its c = n_clusters_per_class vertex
coordinates, so the between-class mean gap on that column can only be a
multiple of 2*class_sep/c -- often exactly 0.  Columns with zero gap differ
only in covariance and carry (almost) no signal for a linear classifier.

separating_columns() labels an informative column "separating" when its
empirical class-mean gap exceeds half the smallest non-zero gap the
generator can produce: threshold = class_sep / c.  The threshold is derived
from the generator, never from search results.  Evaluation-only: nothing
here is used by the search.
"""

from __future__ import annotations

import numpy as np


def separating_columns(
    X: np.ndarray,
    y: np.ndarray,
    n_informative: int,
    class_sep: float = 1.0,
    n_clusters_per_class: int = 2,
    ambiguous_frac: float = 0.25,
) -> dict:
    """Return which of the first n_informative columns separate the classes.

    Requires binary y and shuffle=False data (informative cols 0..n_inf-1).

    Returns dict with
      sep_idx    : np.ndarray of separating column indices
      gaps       : |mean(class 1) - mean(class 0)| per informative column
      threshold  : class_sep / n_clusters_per_class
      ambiguous  : indices whose gap is within ambiguous_frac*threshold of
                   the threshold (should be empty; flags a doubtful label)
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("separating_columns supports binary labels only")
    Xi = X[:, :n_informative]
    gaps = np.abs(Xi[y == classes[1]].mean(axis=0) - Xi[y == classes[0]].mean(axis=0))
    threshold = class_sep / n_clusters_per_class
    ambiguous = np.flatnonzero(np.abs(gaps - threshold) < ambiguous_frac * threshold)
    return dict(sep_idx=np.flatnonzero(gaps > threshold), gaps=gaps,
                threshold=threshold, ambiguous=ambiguous)
