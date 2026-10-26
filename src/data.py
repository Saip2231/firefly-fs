"""Data loading and preprocessing for the benchmark."""

import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_dataset(name: str, seed: int = 42):
    """Load a dataset, split, and scale it.

    Parameters
    ----------
    name : str
        Dataset identifier. Supported: "digits", "cifar_r18".
    seed : int
        Random seed for reproducible train/test splits.

    Returns
    -------
    X_train : np.ndarray
        Scaled training features.
    X_test : np.ndarray
        Scaled test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    """
    if name == "digits":
        data = load_digits()
        X, y = data.data, data.target
    elif name in ("cifar_r18", "cifar_18"):
        X = np.load(os.path.join(DATA_DIR, "cifar_r18_X.npy"))
        y = np.load(os.path.join(DATA_DIR, "cifar_r18_y.npy"))
    else:
        raise ValueError(
            f"Unknown dataset: {name!r}. Supported: 'digits', 'cifar_r18'."
        )

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # StandardScaler fitted on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
