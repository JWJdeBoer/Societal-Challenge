# ui_state.py
from __future__ import annotations
import streamlit as st


def init_session_state() -> None:
    defaults = {
        "step": 1,
        "nav_step": 1,          # sidebar radio state
        "pending_step": None,   # button-driven navigation (applied before widgets)

        "data_source": "default",
        "input_csv_path": "combined.csv",
        "df_preview": None,
        "df_fingerprint": None,
        "validation": {"errors": [], "warnings": [], "info": []},
        "cfg_hash": None,
        "points_proj": None,
        "clusters_wgs84": None,
        "selected_cluster_id": None,
        "solutions": None,
        "recommendations": None,
        "toolbox_toggles": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def request_step(step: int) -> None:
    """Request a step change. Safe: will be applied before widgets are created."""
    st.session_state.pending_step = int(step)


def apply_pending_navigation() -> None:
    """Call at top of app BEFORE sidebar widgets exist."""
    ps = st.session_state.get("pending_step")
    if ps is None:
        return
    ps = int(ps)
    st.session_state.step = ps
    st.session_state.nav_step = ps  # safe here (radio not instantiated yet)
    st.session_state.pending_step = None


def get_step() -> int:
    return int(st.session_state.get("step", 1))


def clear_results() -> None:
    st.session_state.cfg_hash = None
    st.session_state.points_proj = None
    st.session_state.clusters_wgs84 = None
    st.session_state.selected_cluster_id = None
    st.session_state.recommendations = None
