"""User-controlled frame player: Step (slider and button), Play, Pause, speed,
and a reduce-motion toggle that disables playback (DASHBOARD.md rule 9).

Playback redraws one placeholder in a loop.  Any click (Pause included)
starts a new script run, which ends the loop; the frame reached is kept.
Nothing plays on page load.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import streamlit as st


def _step(key: str, n_frames: int) -> None:
    st.session_state[key] = min(st.session_state.get(key, 0) + 1, n_frames - 1)


def frame_player(n_frames: int, key: str, render: Callable[[int], None]) -> None:
    pending = f"{key}__pending"
    if pending in st.session_state:
        st.session_state[key] = st.session_state.pop(pending)
    if st.session_state.get(key, 0) > n_frames - 1:
        st.session_state[key] = n_frames - 1

    reduce = st.toggle("Reduce motion (static frames, no playback)", key=f"{key}__reduce")
    cols = st.columns([4, 1, 1, 1, 2], vertical_alignment="bottom")
    frame = cols[0].slider("Frame", 0, max(n_frames - 1, 1), key=key,
                           disabled=n_frames < 2)
    frame = min(frame, n_frames - 1)
    cols[1].button("Step", key=f"{key}__step", icon=":material/skip_next:",
                   on_click=_step, args=(key, n_frames), disabled=frame >= n_frames - 1)
    play = cols[2].button("Play", key=f"{key}__play", icon=":material/play_arrow:",
                          disabled=reduce or frame >= n_frames - 1)
    cols[3].button("Pause", key=f"{key}__pause", icon=":material/pause:")
    fps = cols[4].slider("Speed (frames per second)", 1, 30, 4, key=f"{key}__fps",
                         disabled=reduce)

    slot = st.empty()
    with slot.container():
        render(frame)
    if play and not reduce:
        for i in range(frame + 1, n_frames):
            time.sleep(1.0 / fps)
            st.session_state[pending] = i
            with slot.container():
                render(i)
