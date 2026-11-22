"""Watch a real search: frame-by-frame replay of logged Firefly / PSO runs."""

import streamlit as st

from app.components.bento import card, golden, hero, kv, nerd, row
from app.components.callout import takeaway
from app.components.frame_player import frame_player
from app.lib import charts, traces
from app.lib.methods import DATASET_DISPLAY, DATASET_SHORT, display, sort_key

hero("Watch a real search",
     "Replay a logged Firefly or PSO search, frame by frame.",
     eyebrow="✨ Demos",
     more="Frame-by-frame replays of logged Firefly and PSO searches. Playback starts only "
          "when you press Play.")

index = traces.trace_index()
if index.empty:
    with card("empty", "No traces yet"):
        st.info("No traces in app/assets/traces/. Build them with "
                "`python experiments/build_app_assets.py`.")
    st.stop()

with card("run", "Pick a logged run", eyebrow="Controls"):
    c1, c2, c3, c4 = row([1.3, 1, 1, 1.2])
    datasets = sorted(index["dataset"].unique())
    dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_SHORT.get,
                                   default=datasets[-1], key="w_ds") or datasets[-1]
    sub = index[index["dataset"] == dataset]
    ks = sorted(sub["k"].unique())
    k = c2.segmented_control("k", ks, default=ks[0], key=f"w_k_{dataset}") or ks[0]
    budgets = sorted(sub["budget"].unique())
    budget = c3.segmented_control("Budget", budgets, default=budgets[0],
                                  key="w_b") or budgets[0]
    sub = sub[(sub["k"] == k) & (sub["budget"] == budget)]
    methods = sorted(sub["method"].unique(), key=sort_key)
    method = c4.selectbox("Method", methods, format_func=display, key="w_m")
seed = int(sub["seed"].iloc[0])

trace = traces.load_trace(sub[sub["method"] == method]["file"].iloc[0])
curves = {m: traces.curve(traces.load_trace(f))
          for m, f in zip(sub["method"], sub["file"])}
n_frames = len(trace["frame_index"])
member = "firefly" if method.startswith("firefly") else "particle"
final = curves[method]


def render(i: int) -> None:
    sweep = int(trace["frame_index"][i])
    evals = int(trace["frame_evals"][i])
    m = st.columns(3)
    m[0].metric("Evaluations used", f"{evals} / {int(trace['budget'])}", border=True)
    m[1].metric("Best validation fitness", f"{float(trace['frame_best'][i]):.4f}", border=True)
    m[2].metric("Sweep", "initial swarm" if sweep == 0 else f"{sweep} of "
                f"{int(trace['frames_total']) - 1}", border=True)
    st.plotly_chart(charts.population(
        trace["pop"][i], trace["pop_value"][i],
        f"Frame {i + 1}/{n_frames}: each row is a {member}, lit cells are selected "
        f"features (brighter = lower {trace['value_kind']})"))
    st.plotly_chart(charts.fitness_curves(
        curves, (display(method), evals, float(trace["frame_best"][i])),
        f"Best validation fitness vs evaluations (frame {i + 1})"))


with card("player", f"{display(method)} · {DATASET_DISPLAY[dataset]}",
          eyebrow=f"✨ Replay · k = {k} · budget {budget}"):
    frame_player(n_frames, f"w_frame_{dataset}_{k}_{budget}_{method}", render)

left, right = golden("check")
with left, card("summary", "What the replay shows", eyebrow="Summary"):
    takeaway(f"{display(method)} improved its best validation fitness from "
             f"{final['best'].iloc[0]:.4f} to {final['best'].iloc[-1]:.4f} over "
             f"{int(final['evals'].iloc[-1])} evaluations; the curves compare the methods "
             f"run with the same seed, k={k} and budget {budget}.")
    with nerd("summary-note"):
        st.caption("Frames are evenly spaced and always include the first and last. "
                   "Validation fitness is what the search optimises, so it is optimistic; "
                   "test accuracy is on the Results page.")
with right, card("check", "Replay check", eyebrow="🔒 Matches the logged run"):
    if traces.matches_logged(trace):
        st.success(f"✓ Final mask matches the logged run (results/meta_{dataset}.csv, "
                   f"seed {seed}, raw mask).", icon=":material/verified:")
    else:
        st.error("✗ Final mask does NOT match the logged run.", icon=":material/error:")
    with nerd("check-detail"):
        kv([("Frames shown", f"{n_frames} of {int(trace['frames_total'])}"),
            ("Fitness, start", f"{final['best'].iloc[0]:.4f}"),
            ("Fitness, end", f"{final['best'].iloc[-1]:.4f}")])
