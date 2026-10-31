"""Digits assets (B1 item 6): the seed-0 TRAINING split only.

Saved: raw 8x8 pixel rows (uint8, 0-16) and labels of the training split,
the scaler statistics load_dataset uses, a stratified search / holdout carve
of the training split (the holdout is what the B8 live demo scores on), and
a few example digits taken from that holdout.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_dataset

from .common import raw_features, train_indices

SEED = 0
HOLDOUT_FRAC = 0.25
EXAMPLES_PER_CLASS = 2


def build(out_path) -> dict:
    Xraw, y = raw_features("digits")
    tr = train_indices(y, SEED)
    Xtr, _, ytr, _ = load_dataset("digits", seed=SEED)
    scaler = StandardScaler().fit(Xraw[tr])
    if not (np.allclose(scaler.transform(Xraw[tr]), Xtr) and np.array_equal(y[tr], ytr)):
        raise RuntimeError("digits training rows do not reproduce load_dataset")
    pixels = Xraw[tr].astype(np.uint8)
    if not np.array_equal(pixels, Xraw[tr]):
        raise RuntimeError("digits pixels are not integers in 0-16")

    search_pos, holdout_pos = train_test_split(np.arange(len(tr)), test_size=HOLDOUT_FRAC,
                                               random_state=SEED, stratify=ytr)
    rng = np.random.default_rng(SEED)
    examples = []
    for c in np.unique(ytr):
        cand = holdout_pos[ytr[holdout_pos] == c]
        examples.extend(sorted(rng.choice(cand, EXAMPLES_PER_CLASS, replace=False).tolist()))

    np.savez_compressed(out_path, X_train_raw=pixels, y_train=ytr,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        search_pos=np.sort(search_pos), holdout_pos=np.sort(holdout_pos),
                        example_pos=np.array(examples), seed=np.int32(SEED),
                        holdout_frac=np.float64(HOLDOUT_FRAC))
    return dict(seed=SEED, train_rows=len(tr), search_rows=len(search_pos),
                holdout_rows=len(holdout_pos), examples=len(examples))
