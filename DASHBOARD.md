# DASHBOARD.md: rules and stage specs for the showcase dashboard (B0 to B9)

Import from CLAUDE.md with the line `@DASHBOARD.md`. This is the specification for everything under `app/`. Each session you are told "Run Stage B<n> of DASHBOARD.md". CLAUDE.md and ANALYSIS.md still apply; this file adds dashboard rules and never overrides them.

## 1. What we are building

A multi-page Streamlit app that showcases a finished, honest benchmark: leakage-safe protocol, 960 result rows, 72 pre-registered RQ1 tests (mostly ties and losses for plain Firefly, a few wins at CIFAR k=16), cost, stability, transfer, and ablations. It must let an evaluator (1) grasp the findings in one minute, (2) explore every result interactively, (3) watch real searches replay, and (4) try the models on new inputs in a playground. Evidence first, honest always, fast and reproducible.

## 2. Rules (always on)

1. **Read-only on results.** Do not edit `src/`, `config/`, `PREREG.md`, `results/**`, or `figures/**`. The app never recomputes experiment results. Everything it shows comes from data files through the data layer (`app/lib/data.py`). Derived assets are produced only by `experiments/build_app_assets.py` into `app/assets/`.
2. **No hard-coded findings.** Every number, verdict sentence, KPI, chart takeaway, and "WIN/LOSS/TIE" count is computed from data at load time. If a sentence contains a number, a function produced it.
3. **Test-split discipline.** The app never loads the project's test splits. The "Try an image" gallery uses images from the **official CIFAR-10 test set** (`train=False`), which the project never used. The Digits live demo searches on the Digits training split only, and scores on a holdout carved from that training split.
4. **Honest copy.** Never write "proves", "beats", "wins" (except as the pre-registered label WIN with its adjusted p), "state of the art", or "outperforms" without the test and effect size beside it. Say what Firefly does not do: it does not run on a new image; selection happens once on training data. State n=10 and limited power where tests appear. PCA is labelled "projection, not a selector". Use "long-jump mutation", "PSO with random-key (top-k) decoding and stall-swap fallback", "CIFAR-10 subset with ResNet-18 embeddings".
5. **Single images are anecdotes.** Every per-image prediction view carries the line "One image is an anecdote. The benchmark accuracy beside each method comes from 2,000 test images over 10 seeds." The classifier knows 10 CIFAR classes and forces any photo into one of them; say so.
6. **Live compute caps.** Live searches: at most 300 evaluations, 30 s wall-clock, single thread, fixed seed unless the user changes it, with a visible progress bar and a Stop button. No unbounded loops. Use `st.cache_data` and `st.cache_resource` for loaders and models.
7. **Input safety.** Uploads: jpg, png, webp, at most 5 MB, strip EXIF, decode with PIL, never store or log images. URL fetch (optional): http and https only, resolve and block private, loopback and link-local addresses, 5 s timeout, 5 MB cap, content-type check, no redirects to blocked hosts.
8. **Performance budgets.** Cold start to first page under 3 s locally; page switch under 1 s after first load; each trace file under 5 MB; `app/assets/` under 150 MB total; no runtime network calls except the optional URL fetch; no external fonts or scripts (self-hosted or system fonts only).
9. **Accessibility and design.** Colour-blind-safe palette, never colour alone (always text or pattern too), contrast at least 4.5:1, alt text on every figure, body text at least 16 px, animations user-controlled (Play, Pause, Step, speed) with a "reduce motion" toggle that shows static frames.
10. **Structure.** Thin pages, logic in `app/lib/`, reusable UI in `app/components/`, type hints, no file over 400 lines, one concern per module.
11. **Tests.** Streamlit `AppTest` smoke test for every page (no exceptions, key elements present), number-consistency tests (every KPI equals a pandas recomputation), a grep test that no file under `app/` references `Xte` or `yte`, ONNX vs torch embedding agreement, trace-vs-CSV match, and a load-time test.
12. **Stage gates.** Explain the plan in a few lines, implement, run `pytest tests/ -q` and the app, show exact output plus a plain-language description of each screen (layout, controls, what changes on interaction), summarise in 5 lines, then **STOP**. Wait for "continue".
13. **Commits.** After each green stage: `app: B<n> <what>`. `LOG.md` gets a short entry per stage.
14. **Ask before:** changing anything in `src/`, `config/`, or results; adding a dependency; switching away from Streamlit; adding runtime network calls; deploying or pushing.

## 3. Architecture

```
app/
  streamlit_app.py          # entry: set_page_config, theme CSS, st.navigation
  pages/                    # one thin file per page (01_home.py ... )
  lib/
    data.py                 # manifest-driven loaders (cached), schema validation
    facts.py                # KPIs, verdict text, takeaways computed from data
    stats_view.py           # label colours, formatting of p, points, "never"
    traces.py               # load/validate replay traces, match badge
    models.py               # numpy demo models, ONNX embedding, preprocessing
    search_live.py          # capped live search (Digits, toy), generator of frames
    safety.py               # upload and URL validation
    theme.py                # tokens, plotly template, css
  components/               # swarm_grid.py, label_grid.py, kpi_strip.py, callout.py
  assets/                   # built by experiments/build_app_assets.py
  tests/                    # AppTest, consistency, safety, asset checks
.streamlit/config.toml      # base theme
requirements-app.txt        # pinned
```

## 4. Data contracts

- Discover analysis outputs from `results/analysis/analysis_manifest.md` (and glob `results/analysis/*.csv`). Expected: `rq1_tests.csv` (72 rows: dataset, k, budget, comparator, n, mean_firefly, mean_comparator, mean_diff_pts, std_diff_pts, nonzero_pairs, p_raw, p_holm, label), `descriptive.csv`, `rq2_cost.csv`, `timing_idle.csv`, `stability.csv`, transfer and ablation tables, `integrity_report.md`, `analysis_manifest.md`.
- Raw rows: `results/baselines_<dataset>.csv`, `results/meta_<dataset>.csv` (primary `mask_type == "raw"`), `results/baselines_transfer_<dataset>.csv`, `results/split_hashes.csv`.
- Facts file `app/assets/facts.json` (built from the above): counts of rows, seeds, tests, labels, budgets, k lists, config hash, git hashes, test count from the last pytest log, load-inflation factor from A4.
- Every loader validates required columns and raises a clear error shown in the UI (not a stack trace). A missing optional file hides the dependent widget with a one-line note.
- Search time (`search_s`) is always labelled "measured under 15-way parallel load" except in the idle-timing views.

## 5. Design tokens

- **Method colours (Okabe-Ito):** firefly `#E69F00`, firefly_seeded `#F0E442`, pso `#0072B2`, pso_seeded `#56B4E9`, anova `#009E73`, mi `#CC79A7`, l1 `#D55E00`, rf `#8C564B`, pca black dashed, all and random grey `#7F7F7F`.
- **Labels:** WIN `#009E73` with a check mark, LOSS `#D55E00` with a cross, TIE grey `#BDBDBD` with an equals sign.
- **Surfaces:** dark default (`#0E1117` background, `#161B22` cards, border `#30363D`), light theme supported through the same tokens. Accent glow `#E69F00` used sparingly for fireflies and focus states.
- **Type:** system sans stack, 16 px body, 1.25 scale for headings, tabular numerals in tables.
- **Motion:** transitions at most 600 ms, ease-out; replay frame rate user-controlled from 1 to 30 frames per second; no autoplay on page load.
- **Plotly template:** shared template in `theme.py` (transparent paper, subtle grid, method colours, readable hover with units).

## 6. Stage specs

**B0 Foundation.** Create the structure in section 3, theme and CSS, `st.navigation` with the page list from the playbook, the manifest-driven data layer with schema validation, `facts.py`, a placeholder per page, and the AppTest harness. Acceptance: app launches; every page opens; integrity banner (from `integrity_report.md`) shows PASS on Home; tests green.

**B1 Asset build** (`experiments/build_app_assets.py` plus helpers; allowed to read results, writes only `app/assets/`).
1. `facts.json` and slim CSV copies of the tables the app needs.
2. Demo traces: re-run 6 to 8 combinations from the frozen config with `trace=` (CIFAR k=16 and 64 seed 0 budget 800 and 3000 for firefly, pso, firefly_seeded; Digits k=16 seed 0), save `.npz` (cap about 60 sweeps), and **assert the final mask equals the logged `idx`**; print OK or FAIL for each. Optional: seeds 0 to 9 for CIFAR k=16 and 64 budget 800 for mean convergence curves.
3. Demo models for seed 0 and the consensus subset: masks, scaler statistics, logistic-regression weights, PCA projection, saved as `demo_models.npz`. Prediction works with numpy only.
4. ResNet-18 exported to ONNX with a test that ONNX and torch embeddings agree within 1e-4 on 100 images.
5. Gallery: 30 images from the **official CIFAR-10 test set** (3 per class) saved as small PNGs with labels.
6. Digits assets: training split arrays and a handful of example digits.
Acceptance: all assets under budget, match checks OK, checksums recorded in `facts.json`.

**B2 Home, Protocol, Limits pages.** See the playbook for content. Acceptance: KPI strip and verdict card computed from data; protocol diagram shows split, selection on train only, inner validation, test used once; deviations register rendered from a markdown asset; consistency tests pass.

**B3 Results explorer (RQ1).** Interactive accuracy vs k with filters (dataset, budget, methods), a clickable WIN/LOSS/TIE grid (12 families by 6 comparators) that opens a per-family panel with paired per-seed differences, adjusted p, effect size in points, and a CSV download. Acceptance: grid counts equal `rq1_tests.csv`; each cell shows label as text plus symbol.

**B4 Cost and Stability pages.** Cost versus accuracy scatter (log x), break-even calculator (inference count slider, classifier LR or KNN, uses A3 and A4 numbers, "never" shown plainly), load-inflation note; stability bars with random reference, feature-frequency chart, and on Digits an 8x8 heatmap of selection frequency per method (valid because features are pixels). Acceptance: calculator results equal a pandas recomputation; no heatmap is drawn for CIFAR embeddings.

**B5 Transfer and Ablations page.** Transfer under KNN and SVM, budget curve, seeding effect, re-rank effect with `total_fits` and seconds. Every panel labelled exploratory. Acceptance: numbers match the analysis tables.

**B6 Watch a real search + concept animation.** Replay page: pick dataset, k, seed, budget, method(s); animated population grid (rows fireflies or particles, columns features, lit cells selected, brightness by fitness), best-fitness curve, evaluation counter, speed slider, Play, Pause, Step, and the badge "final mask matches the logged run" computed from the trace and the CSV. Concept animation: self-contained HTML/JS in `st.components.v1.html` on a toy with about 12 features, labelled illustrative. Acceptance: badge true for every shipped trace; frames capped at 60.

**B7 Playground: Try an image.** Gallery first, then upload or URL. Show the preprocessing steps (square crop, 32x32 antialiased, upscaled as in extraction), the embedding strip with each method's chosen subset highlighted, a table of predictions and probabilities per method, benchmark accuracy beside each, inference time, and the anecdote and 10-class disclaimers. Acceptance: gallery predictions on the sample images are reproducible; safety tests pass; no `Xte` references.

**B8 Playground: Digits live and Build-your-own search.** Digits live: pick or draw a digit, run a capped live search, animate the selected pixels on the 8x8 grid, classify with the subset. Build-your-own: sliders (k, population, budget up to 300, alpha) on Digits or a planted toy, live convergence, comparison with ANOVA at the same k, and an **overfitting meter** showing inner validation fitness versus accuracy on a holdout carved from the training split. Acceptance: runs finish within the cap, Stop works, results deterministic for a seed.

**B9 Polish, QA, deploy, backups.** Responsive check, reduce-motion mode, empty and error states, load-time test, README section, `requirements-app.txt` pinned, deployment to Hugging Face Spaces (Streamlit SDK), a recorded 3-minute walkthrough, and a static export of the key figures as a fallback. Acceptance: all tests green, deployed link works from a fresh browser profile, fallback assets saved.

## 7. Reviewer pass (fresh session, read-only, after B3 and after B9)

Check: hard-coded numbers or findings; overclaiming copy; use of test splits; unsafe input handling; unbounded compute; mismatched figures versus CSVs; missing exploratory labels; contrast and alt text; performance budgets; animation controls; dead widgets; consistency between pages (same numbers everywhere); reproducibility of live demos for a fixed seed. Report severity (BLOCKER, MAJOR, MINOR), file:line, why, minimal fix.

## 8. Definition of done (per stage)

Tests pass, screens described, takeaways computed, exploratory labels present, performance budgets met, reviewer BLOCKER and MAJOR items resolved, commit made.