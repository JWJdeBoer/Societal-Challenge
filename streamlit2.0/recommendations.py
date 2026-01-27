# recommendations.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from toolbox_recommender import build_cluster_context, recommend_for_cluster, load_toolbox, Recommendation


@st.cache_resource(show_spinner=False)
def get_toolbox_solutions(yaml_path: str) -> List[dict]:
    return load_toolbox(yaml_path)


@st.cache_data(show_spinner=False)
def compute_recommendations(
    clusters_df: pd.DataFrame,
    *,
    yaml_path: str,
    top_k: int,
    toggles: Dict[str, Any],
    toggles_fingerprint: str,  # <-- nieuw: alleen voor cache-key
) -> Dict[int, List[Recommendation]]:
    solutions = get_toolbox_solutions(yaml_path)

    out: Dict[int, List[Recommendation]] = {}
    if clusters_df is None or clusters_df.empty:
        return out

    for _, row in clusters_df.iterrows():
        cid = int(row.get("cluster_id"))
        ctx = build_cluster_context(
            row,
            grid_constraints_present=bool(toggles.get("grid_constraints_present", True)),
            feed_in_capacity_available=bool(toggles.get("feed_in_capacity_available", True)),
            flexible_load_present=bool(toggles.get("flexible_load_present", False)),
            physical_space_available=bool(toggles.get("physical_space_available", True)),
            suitable_wind_location_present=bool(toggles.get("suitable_wind_location_present", False)),
            spatial_opportunity_available=bool(toggles.get("spatial_opportunity_available", True)),
            sufficient_thermal_demand_present=bool(toggles.get("sufficient_thermal_demand_present", False)),
            thermal_demand_is_low=bool(toggles.get("thermal_demand_is_low", False)),
        )
        out[cid] = recommend_for_cluster(ctx, solutions, top_k=top_k)

    return out
