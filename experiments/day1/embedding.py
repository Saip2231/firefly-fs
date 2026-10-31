"""Candidate Day-1 embedding recipes and the test that identifies the real one.

The script that produced data/cifar_r18_X.npy is not in the repository.  A
candidate recipe counts as the Day-1 recipe only if it reproduces stored rows
almost exactly: relative L2 distance to the nearest stored row below EXACT_REL,
with the stored label equal to the CIFAR-10 label.  A near miss (a similar
but different recipe) gives distances orders of magnitude larger.

Images: the official CIFAR-10 release via torchvision (downloaded once to
data/cifar10_raw/, git-ignored).  Nothing here reads the project's own
train/test splits of the stored embeddings except as a lookup table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "cifar10_raw"
LOGS = ROOT / "logs"
RECIPE_FILE = LOGS / "day1_recipe.json"
AFFINE_FILE = LOGS / "day1_affine.npz"
EXACT_REL = 1e-4            # same recipe, CPU float32 numerics
CLOSE_REL = 5e-3            # same recipe, different numerics (reported, not accepted)
IDENT_REL = 0.1             # "this stored row is this image" (non-members sit near 0.3)
NORMS = {"imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         "cifar": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "unit": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}         # ToTensor only
IMAGENET_MEAN, IMAGENET_STD = NORMS["imagenet"]

# resize: "pil" = torchvision Resize(size) on PIL images (bilinear), "tensor" =
# bilinear interpolation of the tensor, None = native 32x32 input.  ResNet-18
# has one ImageNet weight set (IMAGENET1K_V1).
_RESIZES = [("pil", 224), ("tensor", 224), ("pil", 128), ("tensor", 128),
            ("pil", 64), ("tensor", 64), (None, 32)]
RECIPES: dict[str, dict] = {
    f"{norm}_{resize or 'native'}{size}": dict(weights="IMAGENET1K_V1", resize=resize,
                                               size=size, norm=norm)
    for norm in NORMS for resize, size in _RESIZES
}
# Round 2 (after round 1 found the 224-pixel ImageNet recipes closest but not
# exact): the standard ImageNet eval transform, bicubic resizing, tensor
# resize without antialias, and train-mode BatchNorm (a missing model.eval()).
RECIPES.update({
    "imagenet_r256crop224": dict(weights="IMAGENET1K_V1", resize="pil_crop", size=224,
                                 norm="imagenet"),
    "imagenet_bicubic224": dict(weights="IMAGENET1K_V1", resize="pil_bicubic", size=224,
                                norm="imagenet"),
    "imagenet_tensor224_noaa": dict(weights="IMAGENET1K_V1", resize="tensor_noaa", size=224,
                                    norm="imagenet"),
    "imagenet_pil224_bntrain": dict(weights="IMAGENET1K_V1", resize="pil", size=224,
                                    norm="imagenet", bn="train"),
    "imagenet_r256crop224_bntrain": dict(weights="IMAGENET1K_V1", resize="pil_crop",
                                         size=224, norm="imagenet", bn="train"),
})
ROUND2 = [name for name in RECIPES if name.startswith("imagenet_") and (
    "crop" in name or "bicubic" in name or "noaa" in name or "bntrain" in name
    or name == "imagenet_pil224")]
# Round 3 (round 2: Resize(256)+CenterCrop(224) is the family, ~2% off, same
# feature norm): other implementations of that resize.
RECIPES.update({
    "imagenet_r256crop224_bicubic": dict(weights="IMAGENET1K_V1", resize="crop_pil_bicubic",
                                         size=224, norm="imagenet"),
    "imagenet_r256crop224_tensor": dict(weights="IMAGENET1K_V1", resize="crop_tensor_aa",
                                        size=224, norm="imagenet"),
    "imagenet_r256crop224_tensor_noaa": dict(weights="IMAGENET1K_V1",
                                             resize="crop_tensor_noaa", size=224,
                                             norm="imagenet"),
    "imagenet_r256crop224_nearest": dict(weights="IMAGENET1K_V1", resize="crop_pil_nearest",
                                         size=224, norm="imagenet"),
})
ROUND3 = ["imagenet_r256crop224", "imagenet_r256crop224_bicubic",
          "imagenet_r256crop224_tensor", "imagenet_r256crop224_tensor_noaa",
          "imagenet_r256crop224_nearest"]


def stored() -> tuple[np.ndarray, np.ndarray]:
    """The stored Day-1 embeddings (float64 for exact distances) and labels."""
    return (np.load(DATA / "cifar_r18_X.npy").astype(np.float64),
            np.load(DATA / "cifar_r18_y.npy"))


def cifar(train: bool):
    """(uint8 images (n, 32, 32, 3), labels, class names) of the official release."""
    from torchvision.datasets import CIFAR10  # noqa: PLC0415

    ds = CIFAR10(root=str(RAW), train=train, download=True)
    return np.asarray(ds.data), np.asarray(ds.targets), list(ds.classes)


def load_model(weights: str):
    import torch  # noqa: PLC0415
    from torchvision.models import resnet18  # noqa: PLC0415

    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()         # 512-d global-average-pooled features
    return model.eval()


def preprocess(images: np.ndarray, recipe: dict):
    """uint8 (n, 32, 32, 3) -> normalised float tensor (n, 3, S, S)."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    size, resize = recipe["size"], recipe["resize"]
    if resize in ("pil", "pil_bicubic"):   # torchvision Resize on a PIL image
        method = Image.BICUBIC if resize == "pil_bicubic" else Image.BILINEAR
        images = np.stack([np.asarray(Image.fromarray(im).resize((size, size), method))
                           for im in images])
    big = round(size * 256 / 224)          # Resize(256) before CenterCrop(224)
    off = (big - size) // 2
    if resize in ("pil_crop", "crop_pil_bicubic", "crop_pil_nearest"):
        method = {"pil_crop": Image.BILINEAR, "crop_pil_bicubic": Image.BICUBIC,
                  "crop_pil_nearest": Image.NEAREST}[resize]
        images = np.stack([np.asarray(Image.fromarray(im).resize((big, big), method))
                           [off:off + size, off:off + size] for im in images])
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    if resize in ("tensor", "tensor_noaa"):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=resize == "tensor")
    if resize in ("crop_tensor_aa", "crop_tensor_noaa"):
        x = F.interpolate(x, size=(big, big), mode="bilinear", align_corners=False,
                          antialias=resize == "crop_tensor_aa")[:, :, off:off + size,
                                                                off:off + size]
    m, s = NORMS[recipe.get("norm", "imagenet")]
    mean = torch.tensor(m).view(1, 3, 1, 1)
    std = torch.tensor(s).view(1, 3, 1, 1)
    return (x - mean) / std


def embed(model, images: np.ndarray, recipe: dict, batch: int = 64) -> np.ndarray:
    """Embeddings in eval mode, or with train-mode BatchNorm for bn="train"
    (batch statistics; the running statistics are restored afterwards)."""
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    net = model
    if recipe.get("bn") == "train":
        net = copy.deepcopy(model).train()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out.append(net(preprocess(images[i:i + batch], recipe)).numpy())
    return np.concatenate(out).astype(np.float64)


def nearest(X: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest stored row for each query and the relative distance
    ||q - x|| / ||x|| to it."""
    x2 = (X ** 2).sum(axis=1)
    d2 = (Q ** 2).sum(axis=1)[:, None] + x2[None, :] - 2.0 * Q @ X.T
    idx = d2.argmin(axis=1)
    dist = np.sqrt(np.maximum(d2[np.arange(len(Q)), idx], 0.0))
    return idx, dist / np.sqrt(x2[idx])


def describe(X: np.ndarray) -> dict:
    """What post-processing the stored rows look like they had."""
    norms = np.linalg.norm(X, axis=1)
    col_mean, col_std = X.mean(axis=0), X.std(axis=0)
    info = dict(min=float(X.min()), max=float(X.max()), frac_zero=float((X == 0).mean()),
                row_norm_mean=float(norms.mean()), row_norm_std=float(norms.std()),
                col_mean_abs=float(np.abs(col_mean).mean()), col_std_mean=float(col_std.mean()))
    info["l2_normalised"] = bool(abs(info["row_norm_mean"] - 1) < 1e-3
                                 and info["row_norm_std"] < 1e-3)
    info["standardised"] = bool(info["col_mean_abs"] < 1e-2 and abs(info["col_std_mean"] - 1) < 5e-2)
    info["non_negative"] = bool(info["min"] >= 0)
    return info


def affine_fit(E_: np.ndarray, X_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column least squares X ≈ a * E + b over matched pairs (rows)."""
    e_mean, x_mean = E_.mean(axis=0), X_.mean(axis=0)
    de, dx = E_ - e_mean, X_ - x_mean
    var = (de ** 2).sum(axis=0)
    a = np.where(var > 0, (de * dx).sum(axis=0) / np.where(var > 0, var, 1), 0.0)
    return a, x_mean - a * e_mean


def to_stored_space(emb: np.ndarray, info: dict) -> np.ndarray:
    """Apply the verified post-processing so embeddings compare with stored rows."""
    post = info.get("post", "raw")
    if post == "l2":
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if post == "affine":
        with np.load(AFFINE_FILE) as z:
            return z["a"] * emb + z["b"]
    return emb


def verified_recipe(accept_approximate: bool = False) -> dict:
    """The recipe find_recipe.py established; raises if there is none.
    APPROXIMATE (identified, not bit-exact) only with accept_approximate."""
    if not RECIPE_FILE.exists():
        raise SystemExit(f"{RECIPE_FILE} missing: run experiments/day1/find_recipe.py first")
    info = json.loads(RECIPE_FILE.read_text(encoding="utf-8"))
    allowed = {"MATCH", "APPROXIMATE"} if accept_approximate else {"MATCH"}
    if info.get("verdict") not in allowed:
        hint = " (add --accept-approximate)" if info.get("verdict") == "APPROXIMATE" else ""
        raise SystemExit(f"no accepted recipe ({info.get('verdict')}){hint}; not exporting")
    return info


def stored_threshold(info: dict) -> float:
    """Distance below which an embedding is taken to BE a stored row."""
    return EXACT_REL if info["verdict"] == "MATCH" else IDENT_REL
