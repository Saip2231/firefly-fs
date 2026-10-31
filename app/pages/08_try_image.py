"""Playground: try the seed-0 demo models on one image (gallery first, then upload)."""

import json

import numpy as np
import streamlit as st
from PIL import Image

from app.components.callout import data_error, takeaway
from app.lib import charts, data, facts
from app.lib.embedding import Embedder, to_cifar, upscaled_view
from app.lib.image_view import agreement_sentence, predictions
from app.lib.methods import display
from app.lib.models import load_models, model_key, standardise
from app.lib.safety import ALLOWED_EXTENSIONS, MAX_BYTES, UploadError, load_upload

st.title("Try an image")
st.markdown(
    "Run the demo models on an image from the official CIFAR-10 test set or your own. "
    "Feature selection happened once on training data; it does not run on a new image.")

A = data.ASSETS
NEEDED = [A / "resnet18_embed.onnx", A / "embedding_recipe.json", A / "demo_models.npz",
          A / "gallery" / "gallery.json", A / "gallery" / "embeddings.npy"]
missing = [p.name for p in NEEDED if not p.exists()]
if missing:
    st.warning(f"Not available: missing assets {missing}. Build them with "
               "experiments/day1/ and experiments/build_app_assets.py.", icon=":material/block:")
    st.stop()


@st.cache_resource(show_spinner=False)
def assets():
    recipe = json.loads((A / "embedding_recipe.json").read_text(encoding="utf-8"))
    gallery = json.loads((A / "gallery" / "gallery.json").read_text(encoding="utf-8"))
    with np.load(A / "demo_models.npz") as z:
        models = load_models(z)
        scaler = (z["scaler_mean"], z["scaler_scale"])
    embedder = Embedder(A / "resnet18_embed.onnx", recipe, A / "embedding_affine.npz")
    return recipe, gallery, np.load(A / "gallery" / "embeddings.npy"), models, scaler, embedder


try:
    recipe, gallery, gallery_emb, models, scaler, embedder = assets()
    desc = data.load_analysis("descriptive")
except (data.DataError, OSError, ValueError) as err:
    data_error(data.DataError(str(err)))
    st.stop()
classes = recipe["classes"]
rel = recipe.get("stored_median_rel")
st.info(
    "The ResNet-18 embedding step was reconstructed (the original Day-1 script is lost): "
    "Resize(256), CenterCrop(224), ImageNet normalisation, average-pooled features. "
    + (f"It reproduces the benchmark's stored features to within about {rel:.0%} "
       "(relative), with the correct labels, but not bit-exactly." if rel else ""),
    icon=":material/info:")

source = st.segmented_control("Image source", ["Gallery", "Upload"], default="Gallery",
                              key="ti_src") or "Gallery"
true_label = None
if source == "Gallery":
    items = gallery["images"]
    pick = st.selectbox("Gallery image (official CIFAR-10 test set, never used by the project)",
                        list(range(len(items))),
                        format_func=lambda i: f"{items[i]['class_name']} "
                                              f"(test image #{items[i]['cifar_test_index']})",
                        key="ti_pick")
    entry = items[pick]
    img32 = np.asarray(Image.open(A / "gallery" / entry["file"]).convert("RGB"))
    emb = gallery_emb[pick].astype(np.float64)
    onnx_emb, secs = embedder(img32)
    true_label = entry["class_name"]
    steps = [("CIFAR test image, 32x32", img32)]
    check = float(np.abs(onnx_emb - emb).max())
else:
    up = st.file_uploader(f"Upload an image ({', '.join(ALLOWED_EXTENSIONS)}, at most "
                          f"{MAX_BYTES / 1e6:.0f} MB; not stored)", type=list(ALLOWED_EXTENSIONS),
                          key="ti_upload")
    if up is None:
        st.stop()
    try:
        img = load_upload(up.getvalue(), up.name)
    except UploadError as err:
        st.error(f"Upload refused: {err}", icon=":material/error:")
        st.stop()
    square, small = to_cifar(img)
    img32 = np.asarray(small)
    emb, secs = embedder(img32)
    check = None
    steps = [("Your image (metadata removed)", np.asarray(img)),
             ("Square centre crop", np.asarray(square)),
             ("32x32, antialiased", img32)]

st.subheader("Preprocessing")
steps.append(("What the network sees: Resize(256), CenterCrop(224)", upscaled_view(img32, recipe)))
for col, (caption, arr) in zip(st.columns(4), steps):   # fixed width: no wide gaps
    shown = Image.fromarray(arr)
    if shown.size[0] <= 32:
        shown = shown.resize((160, 160), Image.NEAREST)   # crisp pixels
    else:
        shown.thumbnail((480, 480))                        # display copy only
    col.image(shown, caption=caption, width=160)
st.caption(f"ONNX embedding time: {secs * 1000:.0f} ms"
           + (f"; the ONNX embedding agrees with the stored gallery embedding to "
              f"{check:.1e}." if check is not None else "."))

st.subheader("Predictions per method")
ks = sorted({m.k for key, m in models.items() if m.method != "all"})
k = st.segmented_control("k (features kept)", ks, default=ks[0], key="ti_k") or ks[0]
d = int(models[next(key for key in models if key.startswith("all__"))].k)
x_scaled = standardise(emb, *scaler)
table = predictions(models, x_scaled, k, classes, desc, d)
st.dataframe(table.drop(columns=["key"]), hide_index=True, column_config={
    "probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
    "benchmark accuracy": st.column_config.NumberColumn(
        format="%.3f", help="mean test accuracy over seeds (descriptive.csv); the "
                            "consensus subset was not benchmarked")})
takeaway(agreement_sentence(table, true_label))
seeds = facts.asset_facts().get("seeds", [])
test_rows = facts.asset_facts().get("split_rows", {}).get("cifar_r18", {}).get("test")
st.warning(
    ("One image is an anecdote. The benchmark accuracy beside each method comes from "
     f"{test_rows:,} test images over {len(seeds)} seeds." if test_rows and seeds else
     "One image is an anecdote. The benchmark accuracy beside each method comes from the "
     "full test split over all seeds.")
    + f" The classifier knows only the {len(classes)} CIFAR-10 classes and forces any "
    "photo into one of them.", icon=":material/warning:")

st.subheader("Which embedding dimensions each method uses")
method = st.selectbox("Method", list(table["key"]), format_func=display, key="ti_method")
m = models[model_key(method, d if method == "all" else k)]
st.plotly_chart(charts.embedding_strip(
    emb, m.idx, f"{len(emb)}-d embedding; highlighted: dimensions used by {display(method)}"
    + (" (PCA uses all of them through a projection)" if method == "pca" else "")),
    key="ti_strip")
st.caption("Firefly does not run on this image: its subset was chosen once, on training "
           "data (seed 0), and the logistic regression here uses only those dimensions.")
