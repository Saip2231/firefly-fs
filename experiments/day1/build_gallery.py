"""B1 item 5: gallery of 30 official CIFAR-10 TEST images (3 per class).

Each candidate is embedded with the verified Day-1 recipe and kept only if it
is NOT one of the stored rows (nearest stored row clearly farther than an
exact match), so the "never used by the project" claim holds per image.
Writes only app/assets/gallery/: 30 PNGs, gallery.json, embeddings.npy
(torch embeddings, so gallery predictions are reproducible without ONNX).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.day1 import embedding as E  # noqa: E402

OUT = E.ROOT / "app" / "assets" / "gallery"
PER_CLASS = 3
NOT_STORED_REL = E.IDENT_REL   # members sit near 2e-2 (approximate) or 1e-6 (exact)


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accept-approximate", action="store_true")
    info = E.verified_recipe(ap.parse_args(argv).accept_approximate)
    recipe = info["params"]
    model = E.load_model(recipe["weights"])
    X, _ = E.stored()
    imgs, labels, classes = E.cifar(train=False)
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    entries, embs, rejected = [], [], 0
    for c, name in enumerate(classes):
        cand = rng.permutation(np.flatnonzero(labels == c))
        kept = 0
        for start in range(0, len(cand), 16):
            batch = cand[start:start + 16]
            emb = E.to_stored_space(E.embed(model, imgs[batch], recipe), info)
            _, rel = E.nearest(X, emb)
            for i, r, e in zip(batch, rel, emb):
                if r <= NOT_STORED_REL:
                    rejected += 1
                    continue
                file = f"{len(entries):02d}_{name}.png"
                Image.fromarray(imgs[i]).save(OUT / file)
                entries.append(dict(file=file, label=int(c), class_name=name,
                                    cifar_test_index=int(i), nearest_stored_rel=float(r)))
                embs.append(e)
                kept += 1
                if kept == PER_CLASS:
                    break
            if kept == PER_CLASS:
                break
    np.save(OUT / "embeddings.npy", np.asarray(embs, dtype=np.float32))
    (OUT / "gallery.json").write_text(json.dumps(dict(
        source="official CIFAR-10 test set (torchvision CIFAR10, train=False)",
        recipe=info["recipe"], post=info["post"],
        embeddings="embeddings.npy rows are in the stored-feature space (post-processing applied)",
        classes=classes, rejected_as_stored=rejected,
        images=entries), indent=2), encoding="utf-8")
    print(f"gallery: {len(entries)} images ({PER_CLASS} per class) -> "
          f"{OUT.relative_to(E.ROOT)}; {rejected} candidates rejected because they are "
          "stored rows; smallest distance to a stored row "
          f"{min(e['nearest_stored_rel'] for e in entries):.2e} (relative)")
    return 0 if len(entries) == PER_CLASS * len(classes) else 1


if __name__ == "__main__":
    sys.exit(main())
