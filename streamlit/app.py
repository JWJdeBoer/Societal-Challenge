# app.py
import io
import os
import streamlit as st
import pandas as pd
import geopandas as gpd

from clustering_pipeline import (
    ClusterConfig,
    run_return,
    plot_clusters,
    plot_top_n_clusters,
)

st.set_page_config(page_title="Clustering Tool", layout="wide")
st.title("Geospatial clustering (DBSCAN / HDBSCAN)")

st.sidebar.header("Input")

# --- INPUT: upload of pad ---
use_upload = st.sidebar.checkbox("Upload CSV i.p.v. pad", value=True)

uploaded = None
input_csv_path = None

if use_upload:
    uploaded = st.sidebar.file_uploader("Upload combined.csv", type=["csv"])
    if uploaded is None:
        st.info("Upload een CSV om te starten.")
        st.stop()
else:
    input_csv_path = st.sidebar.text_input("Pad naar combined.csv", value="combined.csv")

st.sidebar.header("Clustering instellingen")

algorithm = st.sidebar.selectbox("Algorithm", ["dbscan", "hdbscan"], index=0)
eps_m = st.sidebar.slider("eps (meters)", min_value=50, max_value=20000, value=5000, step=50)
min_samples = st.sidebar.slider("min_samples", min_value=1, max_value=50, value=3, step=1)
min_cluster_size = st.sidebar.slider("min_cluster_size (filter achteraf)", min_value=1, max_value=50, value=3, step=1)
target_n_clusters = st.sidebar.slider("Aantal clusters (top-N selectie)", min_value=1, max_value=50, value=10, step=1)

st.sidebar.header("Filters")

exclude_sources_txt = st.sidebar.text_input(
    "exclude_sources (komma-gescheiden)",
    value="Bouwwerk,Bovenregionaal"
)

# min_per_source als simpele invoer: "Locatiespecifiek=1,Energieproject=1"
min_per_source_txt = st.sidebar.text_input(
    "min_per_source (bv: Locatiespecifiek=1,Energieproject=1)",
    value="Locatiespecifiek=1,Energieproject=1"
)

make_plot = st.sidebar.checkbox("Maak plot", value=True)
top_n_plot = st.sidebar.slider("Top-N plot", min_value=1, max_value=20, value=5, step=1)

# ---------- Helpers ----------
def parse_exclude_sources(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_min_per_source(s: str):
    out = {}
    s = s.strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v.isdigit():
            out[k] = int(v)
    return out

# ---------- Load CSV (upload -> tijdelijke file) ----------
tmp_csv_path = None
if uploaded is not None:
    # Streamlit file_uploader geeft bytes; we schrijven naar temp file zodat jouw pipeline het kan lezen
    tmp_csv_path = os.path.join(st.session_state.get("tmpdir", "."), "uploaded_combined.csv")
    with open(tmp_csv_path, "wb") as f:
        f.write(uploaded.getbuffer())
    input_csv_path = tmp_csv_path

# ---------- Build config ----------
cfg = ClusterConfig(
    input_csv=input_csv_path,
    algorithm=algorithm,
    eps_m=float(eps_m),
    min_samples=int(min_samples),
    min_cluster_size=int(min_cluster_size),
    target_n_clusters=int(target_n_clusters),
    exclude_sources=parse_exclude_sources(exclude_sources_txt),
    min_per_source=parse_min_per_source(min_per_source_txt),
    make_plot=False,  # we doen plotting hier, niet inside run()
)

st.subheader("Config preview")
st.json({
    "input_csv": cfg.input_csv,
    "algorithm": cfg.algorithm,
    "eps_m": cfg.eps_m,
    "min_samples": cfg.min_samples,
    "min_cluster_size": cfg.min_cluster_size,
    "target_n_clusters": cfg.target_n_clusters,
    "exclude_sources": cfg.exclude_sources,
    "min_per_source": cfg.min_per_source,
})

# ---------- Run button ----------
run_btn = st.button("Run clustering")

if run_btn:
    with st.spinner("Clustering draait..."):
        points_proj, clusters_wgs84 = run_return(cfg)

    st.success("Klaar!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster summary (top-N)")
        if clusters_wgs84.empty:
            st.warning("Geen clusters gevonden na filtering/top-N.")
        else:
            show_cols = [c for c in [
                "rank","cluster_id","n_points","score","score_norm","centroid_lat","centroid_lon"
            ] if c in clusters_wgs84.columns]
            # voeg n_<source> kolommen toe
            show_cols += [c for c in clusters_wgs84.columns if c.startswith("n_")]
            st.dataframe(clusters_wgs84[show_cols].sort_values("rank"))

    with col2:
        st.subheader("Punten (preview)")
        st.dataframe(points_proj.drop(columns=["geometry"], errors="ignore").head(50))

    # ---------- Plot ----------
    if make_plot and not clusters_wgs84.empty:
        st.subheader("Plot clusters (alle geselecteerde clusters)")
        fig_all = plot_clusters(points_proj, cfg)
        st.pyplot(fig_all)

        st.subheader(f"Plot top {top_n_plot} clusters")
        fig_top = plot_top_n_clusters(points_proj, clusters_wgs84, n=top_n_plot)
        st.pyplot(fig_top)

    # ---------- Downloads ----------
    st.subheader("Downloads")

    # points CSV (wkt)
    points_wgs84 = points_proj.to_crs(epsg=4326).copy()
    points_wgs84["lon"] = points_wgs84.geometry.x
    points_wgs84["lat"] = points_wgs84.geometry.y
    points_wgs84["geometry_wkt"] = points_wgs84.geometry.apply(lambda g: g.wkt if g is not None else None)
    points_csv = points_wgs84.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download points_with_clusters.csv",
        data=points_csv,
        file_name="points_with_clusters.csv",
        mime="text/csv"
    )

    # clusters CSV (wkt)
    clusters_out = clusters_wgs84.copy()
    clusters_out["geometry_wkt"] = clusters_out.geometry.apply(lambda g: g.wkt if g is not None else None)
    clusters_csv = clusters_out.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download cluster_summary.csv",
        data=clusters_csv,
        file_name="cluster_summary.csv",
        mime="text/csv"
    )
