"""Identify the Day-1 embedding recipe by reproducing stored rows.

1. Describe the stored rows: raw (non-negative pooled ReLU features),
   L2-normalised rows, or standardised columns.
2. For each candidate recipe, embed a sample of official CIFAR-10 train and
   test images and map them into the stored space:
     raw          as is
     l2           divide each embedding by its norm
     standardised (e - mu) / sigma with mu, sigma estimated on reference
                  images, then an EXACT per-column affine fit on the best
                  candidate pairs (recovers the transform that was used)
3. MATCH = sampled images reproduce stored rows to relative distance < 1e-4
   with the correct label.  CLOSE (< 5e-3) is reported, not accepted.

Also counts sampled official TEST images found in the stored subset (the
gallery's "never used" claim depends on it).  Writes logs/day1_recipe.json
and, for an affine match, logs/day1_affine.npz; nothing under results/.

Usage (from firefly-fs/, venv active):
  python experiments/day1/find_recipe.py
  python experiments/day1/find_recipe.py --recipes imagenet_pil224 --n 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.day1 import embedding as E  # noqa: E402

N_REF = 1000          # reference images for mu / sigma (standardised case)
N_FIT = 40            # best candidate pairs used for the affine fit
CAND_REL = 0.2        # candidate threshold before the exact fit


def sample(n_total: int, n: int) -> np.ndarray:
    """First 64 indices (detects 'first N images' subsets) plus a fixed random rest."""
    first = np.arange(min(64, n, n_total))
    rest = np.random.default_rng(0).choice(np.arange(len(first), n_total),
                                           size=max(0, n - len(first)), replace=False)
    return np.sort(np.concatenate([first, rest]))


def evaluate(X, y, q, labels, idx) -> dict:
    nn, rel = E.nearest(X, q)
    exact, close, ident = rel < E.EXACT_REL, rel < E.CLOSE_REL, rel < E.IDENT_REL
    return dict(exact=int(exact.sum()), label_agree=int((y[nn][exact] == labels[exact]).sum()),
                ident=int(ident.sum()),
                ident_label_agree=int((y[nn][ident] == labels[ident]).sum()),
                ident_median_rel=float(np.median(rel[ident])) if ident.any() else float("nan"),
                close=int(close.sum()), close_label_agree=int((y[nn][close] == labels[close]).sum()),
                first64_exact=int(exact[idx < 64].sum()), median_rel=float(np.median(rel)),
                min_rel=float(rel.min()), nn=nn, rel=rel)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=256, help="images per split")
    ap.add_argument("--recipes", nargs="+", default=E.ROUND3, choices=list(E.RECIPES),
                    help="default: the round-3 candidates (see embedding.ROUND3)")
    ap.add_argument("--all", action="store_true", help="try every recipe")
    args = ap.parse_args(argv)
    if args.all:
        args.recipes = list(E.RECIPES)

    X, y = E.stored()
    desc = E.describe(X)
    space = "l2" if desc["l2_normalised"] else "standardised" if desc["standardised"] else "raw"
    print("stored rows: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in desc.items()))
    print(f"comparison space: {space}", flush=True)

    splits, ref_imgs = {}, None
    for train in (True, False):
        imgs, labels, classes = E.cifar(train)
        idx = sample(len(imgs), args.n)
        splits["train" if train else "test"] = (imgs[idx], labels[idx], idx)
        if train:
            ref_imgs = imgs[np.random.default_rng(2).choice(len(imgs), N_REF, replace=False)]
    model = E.load_model("IMAGENET1K_V1")

    rows, best_info = [], None
    for name in args.recipes:
        recipe = E.RECIPES[name]
        t0 = time.perf_counter()
        embs = {s: E.embed(model, v[0], recipe) for s, v in splits.items()}
        affine = None
        if space == "l2":
            maps = {s: e / np.linalg.norm(e, axis=1, keepdims=True) for s, e in embs.items()}
        elif space == "standardised":
            ref = E.embed(model, ref_imgs, recipe)
            mu, sd = ref.mean(axis=0), ref.std(axis=0)
            sd[sd == 0] = 1.0
            maps = {s: (e - mu) / sd for s, e in embs.items()}
            # exact stage: affine fit on the best candidate pairs of both splits
            E_all = np.concatenate(list(embs.values()))
            nn, rel = E.nearest(X, np.concatenate(list(maps.values())))
            best = np.argsort(rel)[:N_FIT]
            best = best[rel[best] < CAND_REL]
            if len(best) >= 10:
                a, b = E.affine_fit(E_all[best], X[nn[best]])
                affine = (a, b)
                maps = {s: a * e + b for s, e in embs.items()}
        else:
            maps = embs
        for split, (imgs, labels, idx) in splits.items():
            r = evaluate(X, y, maps[split], labels, idx)
            r.update(recipe=name, split=split, n=len(imgs), affine=affine is not None,
                     p5_rel=float(np.percentile(r["rel"], 5)),
                     norm_mean=float(np.linalg.norm(embs[split], axis=1).mean()),
                     seconds=round(time.perf_counter() - t0, 1))
            rows.append(r)
            print(f"{name:<32} {split:<5} exact {r['exact']:>3}/{r['n']} "
                  f"close {r['close']:>3} identified {r['ident']:>3} (labels agree "
                  f"{r['ident_label_agree']}, median rel {r['ident_median_rel']:.2e})  rel min "
                  f"{r['min_rel']:.2e} p5 {r['p5_rel']:.2e} median {r['median_rel']:.2e}  "
                  f"norm {r['norm_mean']:.1f} (stored {desc['row_norm_mean']:.1f})"
                  f"{'  [affine]' if affine else ''}", flush=True)
        pair = rows[-2:]
        ident_rel = [r["ident_median_rel"] for r in pair if r["ident"]]
        # rank: exact matches, then close ones, then the tightest identification
        score = (sum(r["exact"] for r in pair), sum(r["close"] for r in pair),
                 -min(ident_rel) if ident_rel else -9.0)
        if best_info is None or score > best_info[1]:
            best_info = (name, score, affine)

    name, score, affine = best_info
    matched = [r for r in rows if r["recipe"] == name]
    ok = score[0] > 0 and all(r["label_agree"] == r["exact"] for r in matched)
    n_close = sum(r["close"] for r in matched)
    close = not ok and n_close > 0 and all(r["close_label_agree"] == r["close"] for r in matched)
    n_ident = sum(r["ident"] for r in matched)
    approx = (not ok and not close and n_ident >= 5
              and all(r["ident_label_agree"] == r["ident"] for r in matched))
    verdict = "MATCH" if ok else "CLOSE" if close else "APPROXIMATE" if approx else "NO MATCH"
    ok = ok or approx           # APPROXIMATE is recorded; export needs --accept-approximate
    post = "affine" if affine is not None else space
    if ok and affine is not None:
        np.savez(E.AFFINE_FILE, a=affine[0], b=affine[1])
    clean = [{k: v for k, v in r.items() if k not in ("nn", "rel")} for r in rows]
    info = dict(verdict=verdict, recipe=name if ok else None,
                params=E.RECIPES[name] if ok else None, post=post if ok else None,
                stored=desc, threshold=E.EXACT_REL, classes=classes, results=clean,
                train_exact=next(r["exact"] for r in matched if r["split"] == "train"),
                test_exact=next(r["exact"] for r in matched if r["split"] == "test"),
                n_per_split=args.n)
    E.LOGS.mkdir(exist_ok=True)
    E.RECIPE_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")

    best_rel = min(r["median_rel"] for r in rows)
    if close:
        print(f"\nCLOSE (not exact): {name} has {n_close} stored rows within "
              f"{E.CLOSE_REL} with correct labels, none within {E.EXACT_REL}. Likely the "
              "right recipe with different numerics; nothing is exported until you decide.")
        return 1
    if not ok:
        closest = min(rows, key=lambda r: r["min_rel"] if r["split"] == "train" else 9.0)
        train = [r for r in rows if r["split"] == "train"]
        bn = [r["min_rel"] for r in train if "bntrain" in r["recipe"]]
        ev = [r["min_rel"] for r in train if "bntrain" not in r["recipe"]]
        print(f"\nNO MATCH among {len(args.recipes)} recipes (best median rel "
              f"{best_rel:.2e}; closest single train image {closest['min_rel']:.2e} with "
              f"{closest['recipe']}). Nothing will be exported.")
        if bn and ev and min(bn) < 0.5 * min(ev):
            print("Train-mode BatchNorm is much closer than eval mode: Day 1 probably ran "
                  "without model.eval(), so its features depend on batch composition and "
                  "cannot be reproduced exactly.")
        return 1
    train_id = next(r["ident"] for r in matched if r["split"] == "train")
    test_id = next(r["ident"] for r in matched if r["split"] == "test")
    info.update(identified_train=train_id, identified_test=test_id,
                identified_median_rel=float(np.nanmedian([r["ident_median_rel"]
                                                          for r in matched])))
    E.RECIPE_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if verdict == "APPROXIMATE":
        print(f"\nAPPROXIMATE: {name} identifies stored rows unambiguously (train "
              f"{train_id}/{args.n}, test {test_id}/{args.n} sampled images, all labels "
              f"agree) but not bit-exactly: median relative distance "
              f"{info['identified_median_rel']:.2e} for identified images vs about 3e-1 for "
              "non-members. Likely the same recipe computed with different numerics "
              "(e.g. GPU). Export only if you accept that: add --accept-approximate.")
    else:
        print(f"\nMATCH: {name} (post-processing: {post}) reproduces stored rows exactly "
              f"(train {info['train_exact']}/{args.n}, test {info['test_exact']}/{args.n} "
              "sampled images found in the stored subset).")
    if info["test_exact"] or test_id:
        print("WARNING: the stored subset contains official CIFAR-10 TEST images; "
              "build_gallery.py excludes every image found in the stored rows.")
    print(f"written {E.RECIPE_FILE.relative_to(E.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
