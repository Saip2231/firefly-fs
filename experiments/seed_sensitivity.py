"""
Seed-sensitivity check: Random vs ANOVA vs MI at k=4 and k=8 over 10 seeds.
Verifies that the single-seed anomaly (Random beating ANOVA at k=4) is noise.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.neighbors import KNeighborsClassifier
from src.data import load_dataset
from src.baselines import select_random, select_kbest_anova, select_kbest_mi
from src.evaluate import evaluate

SEEDS = list(range(10))
K_VALUES = [4, 8]
CLF = KNeighborsClassifier()

SELECTORS = {
    "Random":       select_random,
    "ANOVA F-test": select_kbest_anova,
    "Mutual Info":  select_kbest_mi,
}


def main():
    # Accumulate accuracies[method][k] = [acc_seed0, acc_seed1, ...]
    results = {name: {k: [] for k in K_VALUES} for name in SELECTORS}

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = load_dataset("digits", seed=seed)
        for name, selector in SELECTORS.items():
            for k in K_VALUES:
                if name in ("Random",):
                    mask = selector(X_train, y_train, k, seed=seed)
                elif name == "ANOVA F-test":
                    mask = selector(X_train, y_train, k)
                else:  # Mutual Info
                    mask = selector(X_train, y_train, k, seed=seed)
                res = evaluate(mask, X_train, y_train, X_test, y_test, CLF)
                results[name][k].append(res["accuracy"])

    # Print summary table
    header = f"{'Method':<20} {'k':>4} {'Mean Acc':>10} {'Std Acc':>9} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        for name in SELECTORS:
            accs = np.array(results[name][k])
            print(f"{name:<20} {k:>4} {accs.mean():>10.4f} {accs.std():>9.4f} "
                  f"{accs.min():>8.4f} {accs.max():>8.4f}")
        print()


if __name__ == "__main__":
    main()
