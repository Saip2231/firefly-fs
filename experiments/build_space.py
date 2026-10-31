"""DASHBOARD.md B9: assemble a Hugging Face Space (Streamlit SDK) in dist/space/.

Copies exactly what the dashboard reads at runtime: app/ (with app/assets/),
src/ (live search), experiments/app_assets/ (population recorder),
the result tables under results/, PREREG.md and .streamlit/config.toml.
Never copies data/ (embeddings, CIFAR images) or logs/.  Writes the Space
README (YAML header) and requirements.txt (requirements-app.txt without
streamlit, which the Space installs from sdk_version).

This script does not upload anything.  Deploying is a separate, manual step.

Usage (from firefly-fs/, venv active):
  python experiments/build_space.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "space"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")

TREES = ["app", "src", "experiments/app_assets", "results/analysis"]
FILES = ["PREREG.md", ".streamlit/config.toml", "results/split_hashes.csv"] + [
    f"results/{kind}_{ds}.csv" for kind in ("baselines", "meta", "baselines_transfer")
    for ds in ("digits", "cifar_r18")]

README = """---
title: Firefly Feature Selection Benchmark
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: {streamlit}
app_file: app/streamlit_app.py
python_version: "3.11"
pinned: false
---

# Firefly feature selection: an honest benchmark

Interactive dashboard for a leakage-safe benchmark of binary Firefly feature
selection against standard baselines on Digits and a CIFAR-10 subset with
ResNet-18 embeddings: pre-registered paired tests, cost, stability, transfer,
ablations, replays of logged searches, a capped live search and a "Try an
image" playground. Every number shown is read from the result files.
"""


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for tree in TREES:
        shutil.copytree(ROOT / tree, OUT / tree, ignore=SKIP)
    for file in FILES:
        (OUT / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / file, OUT / file)

    reqs = (ROOT / "requirements-app.txt").read_text(encoding="utf-8").splitlines()
    streamlit = next(line.split("==")[1] for line in reqs if line.startswith("streamlit=="))
    (OUT / "requirements.txt").write_text(
        "\n".join(line for line in reqs if not line.startswith("streamlit==")) + "\n",
        encoding="utf-8")
    (OUT / "README.md").write_text(README.format(streamlit=streamlit), encoding="utf-8")

    files = [p for p in OUT.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    big = [p.relative_to(OUT).as_posix() for p in files if p.stat().st_size > 10 * 1024 * 1024]
    assert not (OUT / "data").exists()
    print(f"space: {OUT.relative_to(ROOT)} ({len(files)} files, {size / 1e6:.1f} MB); "
          f"streamlit {streamlit} via sdk_version")
    if big:
        print(f"files over 10 MB (uploaded through Git LFS / Xet by the CLI): {big}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
