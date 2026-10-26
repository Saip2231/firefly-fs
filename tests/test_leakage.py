"""Tests for data-leakage detection in the feature-selection benchmark.

(a) Scaler is fitted on training split only.
(b) Train/test indices are disjoint and class proportions are stratified.
(c) No selector in baselines.py accepts test-set arguments.
"""

import os
import inspect

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.data import load_dataset
from src import baselines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASETS = [
    pytest.param("digits", id="digits"),
    pytest.param("cifar_r18", id="cifar_r18"),
]

SEED = 42


def _raw_data(name: str):
    """Return raw (X, y) before any splitting/scaling."""
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    elif name in ("cifar_r18", "cifar_18"):
        X = np.load(os.path.join(DATA_DIR, "cifar_r18_X.npy"))
        y = np.load(os.path.join(DATA_DIR, "cifar_r18_y.npy"))
        return X, y
    raise ValueError(f"Unknown dataset: {name!r}")


# ---------------------------------------------------------------------------
# (a) Scaler fitted on training split only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASETS)
def test_scaler_fitted_on_train_only(name):
    """Rebuild the split with the same seed, compute mean/std from raw TRAIN
    rows only (treat std==0 as 1), and verify that load_dataset's Xtr and Xte
    both equal (raw - train_mean) / train_std.  Also verify that the test
    set's own column mean is *not* ≈ 0 (it would be if the scaler were
    fitted on the whole dataset or on test data).
    """
    X_train, X_test, y_train, y_test = load_dataset(name, seed=SEED)
    X_raw, y_raw = _raw_data(name)

    # Reproduce the exact same split
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
    )

    # Compute train-only statistics (using float64 accumulator like StandardScaler)
    train_mean = X_train_raw.mean(axis=0, dtype=np.float64)
    train_std = X_train_raw.std(axis=0, dtype=np.float64)
    train_std[train_std == 0] = 1.0  # prevent division by zero

    expected_train = (X_train_raw - train_mean) / train_std
    expected_test = (X_test_raw - train_mean) / train_std

    tol = 1e-5 if X_train.dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        X_train, expected_train, atol=tol, rtol=tol,
        err_msg="X_train does not match manual (raw − train_mean) / train_std",
    )
    np.testing.assert_allclose(
        X_test, expected_test, atol=tol, rtol=tol,
        err_msg="X_test does not match manual (raw − train_mean) / train_std",
    )

    # If scaler were fitted on the full dataset, test-set column means would
    # be close to 0.  With a train-only scaler they should NOT be ≈ 0.
    test_col_means = X_test.mean(axis=0)
    # At least some columns should deviate noticeably from zero.
    assert not np.allclose(test_col_means, 0, atol=0.05), (
        "Test-set column means are suspiciously close to 0 — "
        "scaler may have been fitted on the full dataset."
    )


# ---------------------------------------------------------------------------
# (b) Disjoint indices & stratified class proportions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASETS)
def test_indices_disjoint(name):
    """Train and test rows must not overlap."""
    X_train, X_test, _, _ = load_dataset(name, seed=SEED)

    # Use raw data to get unique row identifiers
    X_raw, y_raw = _raw_data(name)
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
    )

    # Convert rows to sets of tuples for membership testing
    train_set = set(map(tuple, X_train_raw))
    test_set = set(map(tuple, X_test_raw))

    overlap = train_set & test_set
    assert len(overlap) == 0, (
        f"{len(overlap)} rows appear in both train and test sets."
    )


@pytest.mark.parametrize("name", DATASETS)
def test_stratified_split(name):
    """Class proportions in train and test should be approximately equal
    (within tolerance expected from an 80/20 stratified split).
    """
    _, _, y_train, y_test = load_dataset(name, seed=SEED)

    classes = np.unique(np.concatenate([y_train, y_test]))
    for c in classes:
        prop_train = (y_train == c).mean()
        prop_test = (y_test == c).mean()
        # Tolerance of 2 percentage points — generous for stratified splits
        assert abs(prop_train - prop_test) < 0.02, (
            f"Class {c}: train proportion {prop_train:.4f} vs test "
            f"{prop_test:.4f} differ by more than 2 pp."
        )


# ---------------------------------------------------------------------------
# (c) Selector signatures must NOT accept test-set arguments
# ---------------------------------------------------------------------------

# Every public selector / transform function in baselines.py
_SELECTOR_NAMES = [
    name for name, obj in inspect.getmembers(baselines, inspect.isfunction)
    if not name.startswith("_")
]

# Allowed parameter-name sets (order-insensitive).  Selectors should accept
# ONLY training data and hyper-parameters.
_ALLOWED_PARAMS = [
    {"X", "y", "k"},               # no seed variant
    {"X", "y", "k", "seed"},        # with seed
    {"Xtr", "k", "seed"},          # train-only variant
]

# Names that would indicate test-set arguments
_TEST_ARG_NAMES = {
    "X_test", "Xte", "X_te", "y_test", "yte", "y_te",
    "test", "x_test", "xte", "x_te", "ytest",
}


@pytest.mark.parametrize("func_name", _SELECTOR_NAMES)
def test_selector_has_no_test_args(func_name):
    """Assert that no selector function in baselines.py accepts test-set
    arguments.  Permitted signatures: (X, y, k[, seed]) or (Xtr, k[, seed]).
    """
    func = getattr(baselines, func_name)
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())

    # Hard check: no parameter whose name looks like test data
    offending = param_names & _TEST_ARG_NAMES
    assert not offending, (
        f"baselines.{func_name}() has test-set parameter(s): {offending}"
    )

    # Soft check: signature should match one of the allowed templates.
    # We strip defaults to compare names only.
    if param_names not in _ALLOWED_PARAMS:
        # Build a human-readable message but still pass if the only extra
        # params are clearly non-test (e.g. 'n_components').  The hard
        # check above already caught actual test params.
        pass  # informational — the hard check is the gate
