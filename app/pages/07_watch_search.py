"""Watch a real search: frame-by-frame replay of logged Firefly / PSO runs."""

import streamlit as st

from app.components.callout import takeaway
from app.components.frame_player import frame_player
from app.lib import charts, traces
from app.lib.methods import DATASET_DISPLAY, display, sort_key

st.title("Watch a real search")
st.markdown(
    "Frame-by-frame replays of logged Firefly and PSO searches. Playback starts only "
    "when you press Play.")

index = traces.trace_index()
if index.empty:
    st.info("No traces in app/assets/traces/. Build them with "
            "`python experiments/build_app_assets.py`.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
datasets = sorted(index["dataset"].unique())
dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_DISPLAY.get,
                               default=datasets[-1], key="w_ds") or datasets[-1]
sub = index[index["dataset"] == dataset]
ks = sorted(sub["k"].unique())
k = c2.segmented_control("k", ks, default=ks[0], key=f"w_k_{dataset}") or ks[0]
budgets = sorted(sub["budget"].unique())
budget = c3.segmented_control("Budget", budgets, default=budgets[0], key="w_b") or budgets[0]
sub = sub[(sub["k"] == k) & (sub["budget"] == budget)]
methods = sorted(sub["method"].unique(), key=sort_key)
method = c4.selectbox("Method", methods, format_func=display, key="w_m")
seed = int(sub["seed"].iloc[0])

trace = traces.load_trace(sub[sub["method"] == method]["file"].iloc[0])
if traces.matches_logged(trace):
    st.success(f"✓ Final mask matches the logged run (results/meta_{dataset}.csv, "
               f"seed {seed}, raw mask).", icon=":material/verified:")
else:
    st.error("✗ Final mask does NOT match the logged run.", icon=":material/error:")

curves = {m: traces.curve(traces.load_trace(f))
          for m, f in zip(sub["method"], sub["file"])}
n_frames = len(trace["frame_index"])
member = "firefly" if method.startswith("firefly") else "particle"


def render(i: int) -> None:
    sweep = int(trace["frame_index"][i])
    evals = int(trace["frame_evals"][i])
    m = st.columns(3)
    m[0].metric("Evaluations used", f"{evals} / {int(trace['budget'])}")
    m[1].metric("Best validation fitness", f"{float(trace['frame_best'][i]):.4f}")
    m[2].metric("Sweep", "initial swarm" if sweep == 0 else f"{sweep} of "
                f"{int(trace['frames_total']) - 1}")
    st.plotly_chart(charts.population(
        trace["pop"][i], trace["pop_value"][i],
        f"Frame {i + 1}/{n_frames}: each row is a {member}, lit cells are selected "
        f"features (brighter = lower {trace['value_kind']})"))
    st.plotly_chart(charts.fitness_curves(
        curves, (display(method), evals, float(trace["frame_best"][i])),
        f"Best validation fitness vs evaluations (frame {i + 1})"))


frame_player(n_frames, f"w_frame_{dataset}_{k}_{budget}_{method}", render)
st.caption(f"Frames are capped at {n_frames} of {int(trace['frames_total'])} recorded "
           "states, evenly spaced, always including the first and last. Validation "
           "fitness is what the search optimises, so it is optimistic; test accuracy is "
           "on the Results page.")
final = curves[method]
takeaway(f"{display(method)} improved its best validation fitness from "
         f"{final['best'].iloc[0]:.4f} to {final['best'].iloc[-1]:.4f} over "
         f"{int(final['evals'].iloc[-1])} evaluations; the curves compare the methods "
         f"run with the same seed, k={k} and budget {budget}.")
