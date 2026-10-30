"""Entry point: page config, theme CSS and navigation (DASHBOARD.md section 3).

Run from firefly-fs/ with the venv active:
    streamlit run app/streamlit_app.py

Pages live in app/pages/ and are registered in app/lib/navigation.py; all
logic lives in app/lib/ and app/components/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from app.lib import theme  # noqa: E402
from app.lib.navigation import PAGE_SPECS  # noqa: E402

st.set_page_config(page_title="Firefly benchmark", page_icon=":material/insights:",
                   layout="wide")
st.markdown(f"<style>{theme.css()}</style>", unsafe_allow_html=True)

sections: dict[str, list] = {}
for i, spec in enumerate(PAGE_SPECS):
    sections.setdefault(spec.section, []).append(
        st.Page(APP_DIR / "pages" / spec.file, title=spec.title, icon=spec.icon,
                url_path=spec.url_path, default=(i == 0)))

st.navigation(sections).run()
