# Three-minute walkthrough (recording script)

Record the screen with the app running locally (`streamlit run app/streamlit_app.py`)
or on the Space. Read the numbers from the screen; do not memorise them, so the
recording always matches the data.

| Time | Page | Do | Say |
|---|---|---|---|
| 0:00-0:25 | Home | Point at the integrity banner, then the KPI strip | "Every number here is read from the result files. The integrity checks pass." Read the WIN / LOSS / TIE counts. |
| 0:25-0:45 | Home | Read the verdict card | Read its first sentence and the power note: n = 10 seeds, TIE is not equivalence. |
| 0:45-1:00 | Protocol | Point along the diagram | "The test split is locked; selection only sees training rows; the test split is used once." |
| 1:00-1:35 | Results explorer | Switch dataset to CIFAR; click a WIN cell, then a LOSS cell | Read the panel sentence: mean difference in points, adjusted p, seeds higher and lower. |
| 1:35-1:55 | Cost | Pick Firefly, KNN, move the prediction slider | Read the break-even sentence, then switch to logistic regression and read it again. |
| 1:55-2:20 | Watch a real search | CIFAR, k=16, budget 800; press Play | Point at the green badge: "the replay reproduces the logged run exactly". |
| 2:20-2:45 | Try an image | Gallery image, then k=16 | Read the anecdote warning; "the embedding is a reconstruction, within about 3 percent". |
| 2:45-3:00 | Digits live | Run the live search | Read the overfitting meter: inner validation vs holdout. Close on Limits. |

Before recording: run the app once and open every page (fills the caches), turn
off notifications, and keep the browser at 100% zoom.
