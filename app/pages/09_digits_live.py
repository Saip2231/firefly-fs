"""Playground: capped live Firefly search on the Digits training split
(Digits live) or on Digits / a planted toy with your own settings."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.callout import takeaway
from app.components.frame_player import frame_player
from app.lib import charts, search_live
from app.lib.methods import display

st.title("Digits live and build your own")
st.markdown(
    "Run a small, capped search live on the Digits training split and score it on a "
    "holdout carved from that training split.")
st.caption(f"Caps: at most {search_live.MAX_EVALS} evaluations and "
           f"{search_live.MAX_SECONDS:.0f} s per run, one thread. Clicking any control "
           "while a run is in progress (Stop included) stops it.")

MODES = ["Digits live", "Build your own"]
mode = st.segmented_control("Mode", MODES, default=MODES[0], key="dl_mode") or MODES[0]


@st.cache_data(show_spinner=False)
def digits() -> dict:
    return search_live.digits_split()


def run_search(split: dict, k: int, n_pop: int, budget: int, alpha: float, seed: int) -> dict:
    bar = st.progress(0.0, text="Searching...")
    st.button("Stop", key="dl_stop", icon=":material/stop:")
    res = search_live.run(split, k, n_pop, budget, alpha, seed,
                          progress=lambda f: bar.progress(f, text=f"Searching... {f:.0%}"))
    bar.empty()
    return res


def overfitting_meter(res: dict) -> None:
    df = pd.DataFrame(dict(
        name=["Firefly: inner validation", "Firefly: holdout",
              "ANOVA: inner validation", "ANOVA: holdout"],
        method=["firefly", "firefly", "anova", "anova"],
        acc=[res["val_acc"], res["holdout_acc"], res["anova_val_acc"],
             res["anova_holdout_acc"]]))
    st.plotly_chart(charts.bars_with_reference(
        df, "acc", None, "Overfitting meter: inner validation vs holdout accuracy",
        "accuracy"), key=f"dl_meter_{mode}")
    gap = res["val_acc"] - res["holdout_acc"]
    takeaway(f"Firefly's subset scores {res['val_acc']:.3f} on the inner validation rows "
             f"it was chosen on and {res['holdout_acc']:.3f} on the unseen holdout "
             f"({gap * 100:+.1f} pts gap); ANOVA at the same k scores "
             f"{res['anova_holdout_acc']:.3f} on the holdout. One run, one seed.")


def frames_view(res: dict, key: str) -> None:
    frames = res["frames"]

    def render(i: int) -> None:
        f = frames[i]
        st.plotly_chart(charts.pixel_grid(
            f["best_mask"].astype(int),
            f"Best subset after {f['evals']} evaluations (frame {i + 1}/{len(frames)})",
            zmax=1), key=f"{key}_px_{i}")

    frame_player(len(frames), key, render)


if mode == "Digits live":
    split = digits()
    ex = split["examples"]
    c1, c2, c3 = st.columns(3)
    pick = c1.selectbox("Digit to classify (from the holdout)", list(range(len(ex))),
                        format_func=lambda i: f"Example {i + 1}: a {int(split['y'][ex[i]])}",
                        key="dl_pick")
    k = c2.slider("k (pixels kept)", 4, 48, 16, key="dl_k")
    budget = c3.slider("Budget (evaluations)", 30, search_live.MAX_EVALS, 150, step=10,
                       key="dl_budget")
    x = split["X"][ex[pick]]
    st.plotly_chart(charts.pixel_grid(x, "The digit (8x8 pixel intensities)", zmax=16),
                    key="dl_digit")
    if st.button("Run live search", type="primary", icon=":material/play_arrow:"):
        st.session_state["dl_res"] = (k, budget, run_search(split, k, 10, budget, 0.02, 0))
    saved = st.session_state.get("dl_res")
    if saved and saved[:2] == (k, budget):
        res = saved[2]
        if res["stopped"]:
            st.warning(res["stopped"])
        st.markdown(f"Used {res['evals']} of {res['budget']} evaluations in "
                    f"{res['seconds']:.1f} s (seed 0).")
        frames_view(res, "dl_frames")
        label, proba = search_live.predict(res, x)
        st.metric("Prediction with the selected pixels",
                  f"{label} ({proba.max():.0%})",
                  help=f"true label {int(split['y'][ex[pick]])}")
        st.caption(f"One digit is an anecdote. The holdout accuracy below comes from "
                   f"{len(split['holdout'])} held-out training digits.")
        overfitting_meter(res)
    elif saved:
        st.info("Settings changed; run the search again.")
else:
    c1, c2, c3 = st.columns(3)
    source = c1.segmented_control("Data", ["digits", "toy"], default="digits",
                                  format_func={"digits": "Digits training split",
                                               "toy": "Planted toy"}.get,
                                  key="by_src") or "digits"
    seed = int(c2.number_input("Seed", 0, 99, 0, key="by_seed"))
    split = digits() if source == "digits" else search_live.toy_split(seed)
    d = split["X"].shape[1]
    k = c3.slider("k", 2, d // 2, min(16, d // 4), key=f"by_k_{source}")
    c1, c2, c3 = st.columns(3)
    n_pop = c1.slider("Population (fireflies)", 4, 30, 10, key="by_pop")
    budget = c2.slider("Budget (evaluations)", 30, search_live.MAX_EVALS, 200, step=10,
                       key="by_budget")
    alpha = c3.slider("alpha (mutation scale)", 0.0, 0.5, 0.02, step=0.01, key="by_alpha")
    params = (source, seed, k, n_pop, budget, alpha)
    if st.button("Run search", type="primary", icon=":material/play_arrow:"):
        st.session_state["by_res"] = (params, run_search(split, k, n_pop, budget, alpha, seed))
    saved = st.session_state.get("by_res")
    if saved and saved[0] == params:
        res = saved[1]
        if res["stopped"]:
            st.warning(res["stopped"])
        curve = pd.DataFrame(dict(evals=[f["evals"] for f in res["frames"]],
                                  best=[f["best"] for f in res["frames"]]))
        st.plotly_chart(charts.fitness_curves({"firefly": curve}, None,
                                              "Live convergence: best validation fitness"),
                        key="by_curve")
        overfitting_meter(res)
        if source == "toy":
            hits = int(res["best_mask"][:search_live.TOY["n_informative"]].sum())
            st.markdown(f"Selected columns: {np.flatnonzero(res['best_mask']).tolist()}. "
                        f"{hits} of the {search_live.TOY['n_informative']} informative "
                        "columns (the first ones) were picked; informative columns are not "
                        "always separating.")
        else:
            frames_view(res, "by_frames")
        st.caption(f"Compared with {display('anova')} at the same k, chosen on the same "
                   "inner-train rows. Same seed and settings give the same result.")
    elif saved:
        st.info("Settings changed; run the search again.")
