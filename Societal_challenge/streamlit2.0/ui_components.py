# ui_components.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st


def step_header(title: str, description: str) -> None:
    st.header(title)
    st.caption(description)
    st.divider()


def show_messages(errors: Sequence[str], warnings: Sequence[str], info: Sequence[str]) -> None:
    if info:
        with st.expander("Wat is er ingelezen?", expanded=False):
            for line in info:
                st.write(f"• {line}")

    if errors:
        st.error("Er zijn problemen gevonden in de CSV. Los dit eerst op:")
        for e in errors:
            st.write(f"• {e}")

    if warnings:
        st.warning("Let op (niet blokkeren):")
        for w in warnings:
            st.write(f"• {w}")


def kpi_row(items: List[tuple]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def dataframe_section(title: str, df: pd.DataFrame, *, height: int = 320) -> None:
    st.subheader(title)
    st.dataframe(df, use_container_width=True, height=height)


def solution_card(sol: dict, *, why: str) -> None:
    """Render a calm 'card-like' layout for a toolbox solution."""
    st.markdown(f"### {sol.get('name', 'Oplossing')}")
    cat = sol.get("category", {}) or {}
    main_cat = cat.get("main")
    cong = cat.get("congestion_type")
    meta = " • ".join([x for x in [main_cat, cong] if x])
    if meta:
        st.caption(meta)

    desc = sol.get("description")
    if desc:
        st.write(desc)

    st.markdown("**Waarom deze oplossing nu in beeld komt**")
    st.write(why)

    cols = st.columns(2)
    pros = sol.get("pros") or []
    cons = sol.get("cons") or []
    with cols[0]:
        if pros:
            st.markdown("**Pluspunten**")
            for p in pros[:8]:
                st.write(f"• {p}")
    with cols[1]:
        if cons:
            st.markdown("**Aandachtspunten**")
            for c in cons[:8]:
                st.write(f"• {c}")

    checklist = sol.get("checklist") or []
    if checklist:
        st.markdown("**Checklist (snelle check)**")
        for item in checklist[:10]:
            st.write(f"• {item}")

    rules = sol.get("exclusion_rules") or []
    if rules:
        with st.expander("Niet toepasbaar als…", expanded=False):
            for r in rules:
                cond = r.get("condition", "")
                reason = r.get("reason", "")
                st.write(f"• {cond} → {reason}".strip())

    notes = sol.get("notes") or {}
    if notes:
        with st.expander("Notities", expanded=False):
            for k, v in notes.items():
                st.write(f"**{k}**: {v}")
