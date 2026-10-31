"""Method names: CSV labels -> canonical names -> display names (one place)."""

from __future__ import annotations

BASELINE_LABELS: dict[str, str] = {
    "All Features": "all", "Random": "random", "ANOVA F": "anova",
    "Mutual Info": "mi", "L1 Logistic": "l1", "RF Importance": "rf", "PCA": "pca",
}
META = ("firefly", "pso", "firefly_seeded", "pso_seeded")
SELECTORS = ("random", "anova", "mi", "l1", "rf")
COMPARATORS = ("anova", "mi", "l1", "rf", "pca", "pso")          # PREREG primary family
ORDER = ("firefly", "pso", "firefly_seeded", "pso_seeded", "anova", "mi", "l1", "rf",
         "pca", "random", "all")

DISPLAY: dict[str, str] = {
    "all": "All features",
    "random": "Random",
    "anova": "ANOVA F",
    "mi": "Mutual information",
    "l1": "L1 logistic",
    "rf": "RF importance",
    "pca": "PCA (projection, not a selector)",
    "firefly": "Firefly",
    "pso": "PSO (random-key top-k, stall-swap)",
    "firefly_seeded": "Firefly, filter-seeded",
    "pso_seeded": "PSO, filter-seeded",
    "consensus": "Firefly consensus subset",
}
DATASET_DISPLAY = {
    "digits": "Digits (64 pixels)",
    "cifar_r18": "CIFAR-10 subset, ResNet-18 embeddings (512-d)",
}


def display(method: str) -> str:
    return DISPLAY.get(method, method)


def sort_key(method: str) -> int:
    return ORDER.index(method) if method in ORDER else len(ORDER)
