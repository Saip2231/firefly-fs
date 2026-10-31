"""Numpy demo models (built by experiments/build_app_assets.py, B1 item 3).

Prediction needs numpy only: standardise with the training-split scaler,
take the method's columns (or the PCA projection), then a multinomial
logistic regression (softmax of X @ coef.T + intercept).  The ONNX embedding
and image preprocessing live in app/lib/embedding.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemoModel:
    method: str
    k: int
    coef: np.ndarray                    # (n_classes, k)
    intercept: np.ndarray               # (n_classes,)
    idx: np.ndarray | None = None       # selected columns (selectors)
    components: np.ndarray | None = None  # (k, d) PCA projection
    pca_mean: np.ndarray | None = None    # (d,)

    def features(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.components is not None:
            return (x_scaled - self.pca_mean) @ self.components.T
        return x_scaled[:, self.idx]


def standardise(x_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(x_raw) - mean) / scale


def predict_proba(model: DemoModel, x_scaled: np.ndarray) -> np.ndarray:
    """Class probabilities, shape (n, n_classes)."""
    z = model.features(np.atleast_2d(x_scaled)) @ model.coef.T + model.intercept
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def model_key(method: str, k: int) -> str:
    return f"{method}__k{k}"


def load_models(npz) -> dict[str, DemoModel]:
    """All models stored in an opened demo_models.npz (key -> DemoModel)."""
    out = {}
    for key in (str(s) for s in npz["model_keys"]):
        method, k = key.split("__k")
        get = (lambda name: npz[f"{key}__{name}"] if f"{key}__{name}" in npz.files else None)
        out[key] = DemoModel(method=method, k=int(k), coef=npz[f"{key}__coef"],
                             intercept=npz[f"{key}__intercept"], idx=get("idx"),
                             components=get("components"), pca_mean=get("pca_mean"))
    return out
