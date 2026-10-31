"""Paths, checksums and the training-row index helper shared by the B1 builders."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
ASSETS = ROOT / "app" / "assets"
TRACES = ASSETS / "traces"
TABLES = ASSETS / "tables"

TRACE_MAX_BYTES = 5 * 1024 * 1024
ASSETS_MAX_BYTES = 150 * 1024 * 1024
MAX_FRAMES = 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def raw_features(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled features and labels, exactly as src.data.load_dataset reads them."""
    from src.data import DATA_DIR  # noqa: PLC0415

    if dataset == "digits":
        d = load_digits()
        return d.data, d.target
    if dataset == "cifar_r18":
        return (np.load(os.path.join(DATA_DIR, "cifar_r18_X.npy")),
                np.load(os.path.join(DATA_DIR, "cifar_r18_y.npy")))
    raise ValueError(f"unknown dataset {dataset!r}")


def train_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Row indices of load_dataset's TRAINING split for this seed.

    Same call as src.data.load_dataset (stratified 80/20, random_state=seed)
    applied to row numbers, so the split is identical; the other part is
    discarded unread.
    """
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed,
                             stratify=y)
    return tr


def subsample_frames(n: int, cap: int = MAX_FRAMES) -> np.ndarray:
    """At most `cap` evenly spaced frame positions out of n, always keeping
    the first and the last."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))
