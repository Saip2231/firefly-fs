"""B1 item 4: export the verified Day-1 ResNet-18 to ONNX and check it.

Requires logs/day1_recipe.json with verdict MATCH (find_recipe.py).
Writes only app/assets/: resnet18_embed.onnx, embedding_recipe.json and
onnx_check.npz (100 images, their torch embeddings, and stored-row indices,
so tests can compare ONNX with torch and with data/cifar_r18_X.npy without
torch installed).

Checks (printed OK / FAIL):
  ONNX vs torch, 100 images: max |diff| <= 1e-4
  ONNX vs stored Day-1 rows for the images found in the stored subset: rel < 1e-4
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.day1 import embedding as E  # noqa: E402

ASSETS = E.ROOT / "app" / "assets"
ONNX = ASSETS / "resnet18_embed.onnx"
TOL = 1e-4


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accept-approximate", action="store_true",
                    help="export an APPROXIMATE (identified, not bit-exact) recipe")
    args = ap.parse_args(argv)
    info = E.verified_recipe(args.accept_approximate)
    recipe = info["params"]
    model = E.load_model(recipe["weights"])
    ASSETS.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, recipe["size"], recipe["size"])
    try:      # classic TorchScript exporter (no extra packages)
        torch.onnx.export(model, dummy, str(ONNX), input_names=["image"],
                          output_names=["embedding"], opset_version=17, dynamo=False)
    except Exception as exc:  # noqa: BLE001  (removed or failing in this torch version)
        print(f"classic ONNX exporter unavailable ({type(exc).__name__}: {exc}); "
              "trying the dynamo exporter (needs: pip install onnxscript)")
        torch.onnx.export(model, (dummy,), str(ONNX), input_names=["image"],
                          output_names=["embedding"], dynamo=True)

    X, y = E.stored()
    rng = np.random.default_rng(1)
    parts = []
    for train in (True, False):          # 50 + 50, so stored rows are found either way
        imgs, labs, _ = E.cifar(train)
        p = rng.choice(len(imgs), 50, replace=False)
        parts.append((imgs[p], labs[p], p if train else p + 1_000_000))
    images = np.concatenate([a for a, _, _ in parts])
    labels = np.concatenate([b for _, b, _ in parts])
    pick = np.concatenate([c for _, _, c in parts])   # >= 1e6: test-set index + 1e6
    x_in = E.preprocess(images, recipe)
    with torch.no_grad():
        ref = model(x_in).numpy()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    name_in, name_out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    # one image per call: the app embeds one image at a time, and the graph's
    # batch dimension is fixed at 1
    arr = x_in.numpy()
    out = np.concatenate([sess.run([name_out], {name_in: arr[i:i + 1]})[0]
                          for i in range(len(arr))])
    max_diff = float(np.abs(out - ref).max())

    nn, rel = E.nearest(X, E.to_stored_space(out.astype(np.float64), info))
    in_subset = rel < E.stored_threshold(info)
    stored_ok = bool(in_subset.any() and (y[nn][in_subset] == labels[in_subset]).all())

    sha = hashlib.sha256(ONNX.read_bytes()).hexdigest()
    if info["post"] == "affine":          # the app applies the same transform
        with np.load(E.AFFINE_FILE) as z:
            np.savez(ASSETS / "embedding_affine.npz", a=z["a"], b=z["b"])
    mean, std = E.NORMS[recipe.get("norm", "imagenet")]
    (ASSETS / "embedding_recipe.json").write_text(json.dumps(dict(
        recipe=info["recipe"], **recipe, mean=mean, std=std, post=info["post"],
        output="512-d global average pool (fc removed)", onnx_sha256=sha,
        classes=info["classes"], torch_vs_onnx_max_abs_diff=max_diff,
        found_in_stored_subset=int(in_subset.sum()), verdict=info["verdict"],
        stored_median_rel=float(np.median(rel[in_subset])) if in_subset.any() else None,
        note=("reconstructed: reproduces rows of data/cifar_r18_X.npy exactly"
              if info["verdict"] == "MATCH" else
              "reconstructed APPROXIMATELY: identifies rows of data/cifar_r18_X.npy with "
              "correct labels, but differs from the original features by "
              "stored_median_rel (relative); not bit-exact")),
        indent=2), encoding="utf-8")
    np.savez_compressed(ASSETS / "onnx_check.npz", images=images, torch_embedding=ref,
                        cifar_index=pick, stored_row=np.where(in_subset, nn, -1),
                        tol=np.float64(TOL))

    print(f"ONNX written: {ONNX.relative_to(E.ROOT)} ({ONNX.stat().st_size / 1e6:.1f} MB)")
    print(f"[{'OK' if max_diff <= TOL else 'FAIL'}] ONNX vs torch on 100 images: "
          f"max |diff| {max_diff:.2e} (tolerance {TOL})")
    kind = "reproduces" if info["verdict"] == "MATCH" else "approximately reproduces"
    med = f", median rel {np.median(rel[in_subset]):.2e}" if in_subset.any() else ""
    print(f"[{'OK' if stored_ok else 'FAIL'}] ONNX {kind} {int(in_subset.sum())} stored "
          f"Day-1 rows among the 100 images, labels agree{med}")
    return 0 if max_diff <= TOL and stored_ok else 1


if __name__ == "__main__":
    sys.exit(main())
