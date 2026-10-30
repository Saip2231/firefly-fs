"""Playground: try the demo models on one image."""

import streamlit as st

from app.components.callout import placeholder

st.title("Try an image")
st.markdown(
    "Run the demo models on an image from the official CIFAR-10 test set or your own. "
    "Feature selection happened once on training data; it does not run on a new image.")
placeholder("B7", "gallery, upload, preprocessing steps and per-method predictions")
