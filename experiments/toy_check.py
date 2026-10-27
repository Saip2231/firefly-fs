"""
Toy benchmark check: Firefly vs Random Search on planted synthetic features.
Evaluates both d=50 and d=500 configurations across 5 seeds.
Compares informative feature hits, validation fitness, and holdout accuracy.
"""

import sys
import os
import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.firefly import make_fitness, binary_firefly


def random_search(fitness, state, d, k, budget, seed, ni):
    """Random search evaluating valid k-sparse masks within budget."""
    rng = np.random.default_rng(seed)
    best_fit = np.inf
    best_mask = None
    sample_hits = []

    while state["evals"] < budget:
        mask = np.zeros(d, dtype=bool)
        chosen = rng.choice(d, size=k, replace=False)
        mask[chosen] = True
        sample_hits.append(int(mask[:ni].sum()))
        f = fitness(mask)
        if f < best_fit:
            best_fit = f
            best_mask = mask.copy()

    mean_sampled_hits = float(np.mean(sample_hits)) if sample_hits else 0.0
    return best_mask, best_fit, mean_sampled_hits


def evaluate_config(d, ni=5, k=5, budget=600, seeds=(0, 1, 2, 3, 4)):
    print("=" * 80)
    print(f"EXPERIMENT CONFIG: D={d}, Informative={ni}, K={k}, Budget={budget}, Seeds={list(seeds)}")
    print("=" * 80)

    # 6000 synthetic rows: 1000 for search (stratified inner train/val), 5000 independent holdout
    X, y = make_classification(
        n_samples=6000,
        n_features=d,
        n_informative=ni,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    Xs, Xh, ys, yh = train_test_split(X, y, train_size=1000, stratify=y, random_state=0)

    true_mask = np.zeros(d, dtype=bool)
    true_mask[:ni] = True

    def holdout_acc(mask):
        clf = LogisticRegression(max_iter=500).fit(Xs[:, mask], ys)
        return float(clf.score(Xh[:, mask], yh))

    true_holdout = holdout_acc(true_mask)
    print(f"Ground Truth (first {ni} features): Holdout Accuracy = {true_holdout:.4f}")
    print(f"Random Search theoretical expected hits per draw: {k * (ni / d):.4f}")
    print("-" * 80)

    fa_results = []
    rnd_results = []

    header = f"{'Seed':<5} | {'Firefly Hits':<12} {'FA ValFit':<10} {'FA HoldAcc':<10} | {'Rnd BestHit':<12} {'Rnd ValFit':<10} {'Rnd HoldAcc':<10} {'Rnd MeanHit':<12}"
    print(header)
    print("-" * len(header))

    for s in seeds:
        # --- 1. Firefly ---
        fit_fa, st_fa = make_fitness(Xs, ys, k=k, seed=s)
        mask_fa, fit_val_fa, _ = binary_firefly(
            fit_fa, st_fa, d=d, k=k,
            budget=budget, n_fireflies=20,
            seed=s,
        )
        hits_fa = int(mask_fa[:ni].sum())
        acc_fa = holdout_acc(mask_fa)

        fa_results.append({
            "seed": s,
            "hits": hits_fa,
            "val_fit": fit_val_fa,
            "holdout_acc": acc_fa,
            "evals": st_fa["evals"],
        })

        # --- 2. Random Search ---
        fit_rnd, st_rnd = make_fitness(Xs, ys, k=k, seed=s)
        mask_rnd, fit_val_rnd, mean_sampled = random_search(
            fit_rnd, st_rnd, d=d, k=k, budget=budget, seed=s, ni=ni
        )
        hits_rnd = int(mask_rnd[:ni].sum())
        acc_rnd = holdout_acc(mask_rnd)

        rnd_results.append({
            "seed": s,
            "hits": hits_rnd,
            "val_fit": fit_val_rnd,
            "holdout_acc": acc_rnd,
            "mean_sampled_hits": mean_sampled,
            "evals": st_rnd["evals"],
        })

        print(
            f"{s:<5} | {hits_fa:>2}/{ni:<9} {fit_val_fa:>10.4f} {acc_fa:>10.4f} | "
            f"{hits_rnd:>2}/{ni:<9} {fit_val_rnd:>10.4f} {acc_rnd:>10.4f} {mean_sampled:>12.4f}"
        )

    print("-" * len(header))

    fa_hits = [r["hits"] for r in fa_results]
    fa_fits = [r["val_fit"] for r in fa_results]
    fa_accs = [r["holdout_acc"] for r in fa_results]

    rnd_hits = [r["hits"] for r in rnd_results]
    rnd_fits = [r["val_fit"] for r in rnd_results]
    rnd_accs = [r["holdout_acc"] for r in rnd_results]
    rnd_sampled = [r["mean_sampled_hits"] for r in rnd_results]

    summary = (
        f"{'Mean':<5} | {np.mean(fa_hits):>4.2f}/{ni:<7} {np.mean(fa_fits):>10.4f} {np.mean(fa_accs):>10.4f} | "
        f"{np.mean(rnd_hits):>4.2f}/{ni:<7} {np.mean(rnd_fits):>10.4f} {np.mean(rnd_accs):>10.4f} {np.mean(rnd_sampled):>12.4f}\n"
        f"{'Std':<5} | {np.std(fa_hits):>12.2f} {np.std(fa_fits):>10.4f} {np.std(fa_accs):>10.4f} | "
        f"{np.std(rnd_hits):>12.2f} {np.std(rnd_fits):>10.4f} {np.std(rnd_accs):>10.4f} {np.std(rnd_sampled):>12.4f}"
    )
    print(summary)
    print()
    return fa_results, rnd_results


def main():
    parser = argparse.ArgumentParser(description="Toy benchmark: Firefly vs Random Search")
    parser.add_argument("--d", type=int, choices=[50, 500], default=None,
                        help="Dimension to evaluate (default runs both 50 and 500)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run (default: 5)")
    parser.add_argument("--budget", type=int, default=600,
                        help="Evaluation budget (default: 600)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    if args.d is not None:
        evaluate_config(d=args.d, budget=args.budget, seeds=seeds)
    else:
        evaluate_config(d=50, budget=args.budget, seeds=seeds)
        evaluate_config(d=500, budget=args.budget, seeds=seeds)


if __name__ == "__main__":
    main()
