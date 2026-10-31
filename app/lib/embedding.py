"""Image -> 512-d ResNet-18 embedding for the "Try an image" page.

Steps (each shown on the page):
  1. square centre crop of the photo
  2. downscale to 32x32 with antialiasing (CIFAR resolution)
  3. the reconstructed Day-1 transform (app/assets/embedding_recipe.json):
     PIL bilinear Resize(256) -> CenterCrop(224) -> /255 -> ImageNet mean/std
  4. ResNet-18 trunk in ONNX (onnxruntime) -> 512-d average-pooled features
  5. the recorded post-processing (none for the stored CIFAR features)

The recipe was reconstructed (the Day-1 script is lost); embedding_recipe.json
says how closely it reproduces the stored features.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

CIFAR_SIZE = 32


def square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def to_cifar(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """(square crop, 32x32 antialiased) of an RGB image."""
    sq = square_crop(img)
    return sq, sq.resize((CIFAR_SIZE, CIFAR_SIZE), Image.LANCZOS)


def day1_input(img32: np.ndarray, recipe: dict) -> np.ndarray:
    """uint8 (32, 32, 3) -> float32 (1, 3, S, S), as in the Day-1 recipe."""
    if recipe["resize"] != "pil_crop":
        raise ValueError(f"unsupported recipe resize {recipe['resize']!r}")
    size = int(recipe["size"])
    big = round(size * 256 / 224)
    off = (big - size) // 2
    up = np.asarray(Image.fromarray(img32).resize((big, big), Image.BILINEAR))
    x = up[off:off + size, off:off + size].astype(np.float32) / 255.0
    x = (x - np.asarray(recipe["mean"], np.float32)) / np.asarray(recipe["std"], np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def upscaled_view(img32: np.ndarray, recipe: dict) -> np.ndarray:
    """The 224x224 picture the network sees (before normalisation), for display."""
    size = int(recipe["size"])
    big = round(size * 256 / 224)
    off = (big - size) // 2
    up = np.asarray(Image.fromarray(img32).resize((big, big), Image.BILINEAR))
    return up[off:off + size, off:off + size]


class Embedder:
    """ONNX ResNet-18 trunk plus the recorded post-processing."""

    def __init__(self, onnx_path: Path, recipe: dict, affine_path: Path | None = None):
        import onnxruntime as ort  # noqa: PLC0415

        self.recipe = recipe
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.name_in = self.session.get_inputs()[0].name
        self.name_out = self.session.get_outputs()[0].name
        self.affine = None
        if recipe.get("post") == "affine" and affine_path is not None:
            with np.load(affine_path) as z:
                self.affine = (z["a"], z["b"])

    def raw(self, img32: np.ndarray) -> np.ndarray:
        out = self.session.run([self.name_out], {self.name_in: day1_input(img32, self.recipe)})
        return out[0][0].astype(np.float64)

    def __call__(self, img32: np.ndarray) -> tuple[np.ndarray, float]:
        """(512-d embedding in the stored-feature space, seconds)."""
        t0 = time.perf_counter()
        e = self.raw(img32)
        if self.recipe.get("post") == "l2":
            e = e / np.linalg.norm(e)
        elif self.affine is not None:
            e = self.affine[0] * e + self.affine[1]
        return e, time.perf_counter() - t0
