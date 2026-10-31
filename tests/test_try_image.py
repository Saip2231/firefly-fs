"""B7 tests: upload safety, ONNX vs torch agreement, reproducible gallery
predictions, and the page itself."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import safety  # noqa: E402

ASSETS = ROOT / "app" / "assets"
onnx_built = pytest.mark.skipif(not (ASSETS / "resnet18_embed.onnx").exists(),
                                reason="run experiments/day1/export_onnx.py first")


def _png(size=(40, 30), exif: bool = False, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", size, (200, 30, 30))
    buf = io.BytesIO()
    if exif:
        ex = Image.Exif()
        ex[0x010F] = "SecretCameraMaker"          # Make
        img.save(buf, format="JPEG", exif=ex.tobytes())
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


# -- safety ---------------------------------------------------------------------

def test_accepts_png_jpg_webp():
    for fmt, name in (("PNG", "a.png"), ("JPEG", "a.jpg"), ("WEBP", "a.webp")):
        img = safety.load_upload(_png(fmt=fmt), name)
        assert img.mode == "RGB" and img.size == (40, 30)


def test_strips_exif():
    data = _png(exif=True)
    with Image.open(io.BytesIO(data)) as raw:
        assert raw.getexif().get(0x010F) == "SecretCameraMaker"
    clean = safety.load_upload(data, "photo.jpg")
    assert not clean.getexif() and not clean.info.get("exif")


# Data is built inside the test: a 5 MB parameter would end up in the test ID,
# which pytest copies into an environment variable (Windows caps those at 32,767
# characters).
BAD_UPLOADS = {
    "too-big": (lambda: b"x" * (safety.MAX_BYTES + 1), "big.png", "limit"),
    "not-an-image": (lambda: b"not an image at all", "a.png", "decoded"),
    "wrong-extension": (lambda: _png(), "a.gif", "not accepted"),
}


@pytest.mark.parametrize("case", list(BAD_UPLOADS))
def test_rejects_bad_uploads(case):
    make, name, match = BAD_UPLOADS[case]
    with pytest.raises(safety.UploadError, match=match):
        safety.load_upload(make(), name)


def test_rejects_other_formats_with_allowed_extension():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="BMP")
    with pytest.raises(safety.UploadError, match="format BMP"):
        safety.load_upload(buf.getvalue(), "sneaky.png")


# -- ONNX and gallery -------------------------------------------------------------

@onnx_built
def test_onnx_matches_torch_reference_with_app_preprocessing():
    """App preprocessing + ONNX reproduce the build's torch embeddings (1e-4)."""
    pytest.importorskip("onnxruntime")
    from app.lib.embedding import Embedder  # noqa: PLC0415

    recipe = json.loads((ASSETS / "embedding_recipe.json").read_text(encoding="utf-8"))
    emb = Embedder(ASSETS / "resnet18_embed.onnx", recipe)
    with np.load(ASSETS / "onnx_check.npz") as z:
        images, ref, tol = z["images"], z["torch_embedding"], float(z["tol"])
    out = np.stack([emb.raw(im) for im in images])
    assert out.shape == ref.shape and float(np.abs(out - ref).max()) <= tol


@onnx_built
def test_gallery_is_reproducible_and_from_the_test_set():
    pytest.importorskip("onnxruntime")
    from app.lib.embedding import Embedder  # noqa: PLC0415
    from app.lib.models import load_models, model_key, predict_proba, standardise  # noqa: PLC0415

    recipe = json.loads((ASSETS / "embedding_recipe.json").read_text(encoding="utf-8"))
    gallery = json.loads((ASSETS / "gallery" / "gallery.json").read_text(encoding="utf-8"))
    stored = np.load(ASSETS / "gallery" / "embeddings.npy")
    assert "test set" in gallery["source"] and len(gallery["images"]) == 30
    assert all(e["nearest_stored_rel"] > 0.1 for e in gallery["images"])
    emb = Embedder(ASSETS / "resnet18_embed.onnx", recipe)
    with np.load(ASSETS / "demo_models.npz") as z:
        models = load_models(z)
        scaler = (z["scaler_mean"], z["scaler_scale"])
    m = models[model_key("firefly", 16)]
    for i, entry in enumerate(gallery["images"][:5]):
        img = np.asarray(Image.open(ASSETS / "gallery" / entry["file"]).convert("RGB"))
        e1, _ = emb(img)
        assert float(np.abs(e1 - stored[i]).max()) <= 1e-4
        p1 = predict_proba(m, standardise(stored[i], *scaler))
        p2 = predict_proba(m, standardise(stored[i], *scaler))
        assert np.array_equal(p1, p2) and np.isclose(p1.sum(), 1.0)


@onnx_built
def test_try_image_page_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415

    at = AppTest.from_file(str(ROOT / "app" / "pages" / "08_try_image.py"),
                           default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("One image is an anecdote" in w.value for w in at.warning)
    at.selectbox(key="ti_pick").select_index(7).run()
    assert not at.exception
