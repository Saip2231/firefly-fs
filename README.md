# Firefly feature selection: an honest benchmark

Binary Firefly feature selection benchmarked against standard baselines (ANOVA F,
mutual information, L1 logistic, RF importance, PCA as a projection, PSO with
random-key (top-k) decoding and stall-swap fallback) on Digits and a CIFAR-10
subset with ResNet-18 embeddings. Leakage-safe splits, 10 seeds, matched search
budgets, a pre-registered primary analysis (`PREREG.md`), and exploratory
cost, stability, transfer and ablation analyses.

The project rules and stage specs are in `CLAUDE.md`, `ANALYSIS.md` and
`DASHBOARD.md`; every run is recorded in `LOG.md`.

## Reproduce the analysis

```powershell
.venv\Scripts\Activate.ps1
python -m experiments.analyze                 # tables in results/analysis/, figures in figures/
python -m experiments.analyze --rebuild-check # regenerate and compare
pytest tests/ -q
```

Always use the project venv: some replays only reproduce the logged runs with
its exact library builds (`LOG.md`, Stage B1).

## Dashboard

A multi-page Streamlit app over the finished results. Every number, verdict and
takeaway on it is computed from the files in `results/` and `app/assets/`;
nothing is typed in, and the app never loads the project's test splits.

| Page | What it shows |
|---|---|
| Home | KPIs, the computed RQ1 verdict, integrity banner |
| Protocol | split / selection / inner validation / single test use; PREREG design; deviations register |
| Results explorer (RQ1) | accuracy vs k with filters; clickable WIN/LOSS/TIE grid; per-seed paired differences, CSV download |
| Cost (RQ2) | accuracy vs search time; break-even calculator (exploratory) |
| Stability (RQ3) | Jaccard across seeds vs random; feature frequency; Digits pixel map |
| Transfer and ablations | KNN / SVM transfer, seeding, re-ranking, budget (exploratory) |
| Watch a real search | replays of logged searches, checked against the logged masks |
| Try an image | gallery (official CIFAR-10 test set) or upload; per-method predictions |
| Digits live and build your own | capped live search (300 evaluations, 30 s) with an overfitting meter |
| Limits | what the benchmark does not show |

### Run locally

```powershell
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

### Rebuild the dashboard assets

```powershell
python experiments/build_app_assets.py      # tables, replay traces, demo models, Digits, facts.json
# Day-1 embedding (needs requirements-build.txt; downloads CIFAR-10 to data/):
python experiments/day1/find_recipe.py
python experiments/day1/export_onnx.py --accept-approximate
python experiments/day1/build_gallery.py --accept-approximate
python experiments/build_app_assets.py --only tables
```

The Day-1 embedding script was lost; `experiments/day1/` reconstructed the
recipe (Resize(256), CenterCrop(224), ImageNet normalisation, ResNet-18
average-pooled features) by reproducing stored rows. It identifies them with
the correct labels but is not bit-exact; `app/assets/embedding_recipe.json`
records how close it is, and the "Try an image" page says so.

### Static fallback and deployment

```powershell
python experiments/export_fallback.py   # fallback/index.html: figures, takeaways, verdict, offline
python experiments/build_space.py       # dist/space/: Hugging Face Space (Streamlit SDK) bundle
```

Uploading the Space bundle is a manual step (see `docs/walkthrough.md` for the
demo script).
