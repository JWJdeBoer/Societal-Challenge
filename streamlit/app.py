# app.py
import os
import streamlit as st
import pandas as pd
from toolbox_recommender import (
    load_toolbox,
    build_cluster_context,
    recommend_for_cluster,
)
from clustering_pipeline import (
    ClusterConfig,
    OptimizationRule,
    run_return,
    plot_clusters,
    plot_top_n_clusters,
)

st.set_page_config(page_title="Clustering Tool", layout="wide")
st.title("Geospatial clustering voor het Rijksvastgoedbedrijf")

# -----------------------------
# Constants / UI options
# -----------------------------
SOURCE_OPTIONS = ["Bouwwerk", "Bovenregionaal", "Energieproject", "Locatiespecifiek"]
DIR_OPTIONS = ["max", "min"]

# Alleen deze features mogen in optimizer (en alleen voor Bouwwerk)
BOUWWERK_FEATURES = [
    "Contractcapaciteit",
    "Aangevraagde Contractcapaciteit",
    "Toekomstige contractcapaciteit",
    "Maximaal piekvermogen",
    "Verhoogde basislast",
    "Vermogen laadpunten",
    "WP vermogen",
    "Max vermogen verbruik",
    "Max ruimte verbruik",
    "Oordeel verbruik",
    "WP Zonnepanelen",
    "Max ruimte opwek",
    "Oordeel opwek",
]


# -----------------------------
# Sidebar: Input
# -----------------------------
st.sidebar.header("Input")
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
    if not os.path.exists(input_csv_path):
        st.warning("Pad bestaat (nog) niet. Vul een correct pad in of gebruik upload.")
        st.stop()


# -----------------------------
# Read minimal header/columns for UI (no preview table)
# -----------------------------
if uploaded is not None:
    df_head = pd.read_csv(uploaded, nrows=5)
    uploaded.seek(0)
else:
    df_head = pd.read_csv(input_csv_path, nrows=5)

available_cols = set(df_head.columns)
available_bouwwerk_features = [c for c in BOUWWERK_FEATURES if c in available_cols]


# -----------------------------
# Sidebar: Clustering settings
# -----------------------------
st.sidebar.header("Clustering instellingen")
algorithm = st.sidebar.selectbox("Algorithm", ["dbscan", "hdbscan"], index=0)

eps_m = st.sidebar.slider(
    "Maximale afstand tussen gebouwen in meters",
    min_value=50, max_value=20000, value=5000, step=50
)

min_samples = st.sidebar.slider(
    "Minimaal aantal gebouwen voor een cluster",
    min_value=1, max_value=40, value=3, step=1
)

target_n_clusters = st.sidebar.slider(
    "Laat de top N clusters zien",
    min_value=1, max_value=20, value=10, step=1
)

# Afgeleide waarden (jouw wens)
min_cluster_size = int(min_samples)           # filter achteraf altijd gelijk aan min_samples
top_n_plot = int(target_n_clusters)           # Top-N plot altijd gelijk aan target_n_clusters

show_plots = st.sidebar.checkbox("Toon plots", value=True)


# -----------------------------
# Sidebar: Filters
# -----------------------------
st.sidebar.header("Filters")

exclude_sources = st.sidebar.multiselect(
    "Dit type locaties niet meenemen",
    options=SOURCE_OPTIONS,
    default=["Bovenregionaal"],
)

st.sidebar.caption("Minimaal aantal locaties per locatietype binnen een cluster.")
min_per_source = {}
for s in SOURCE_OPTIONS:
    val = st.sidebar.number_input(
        f"minimaal aantal: {s}",
        min_value=0,
        max_value=999,
        value=0 if s in ["Bovenregionaal"] else 1,
        step=1,
    )
    if int(val) > 0:
        min_per_source[s] = int(val)


# -----------------------------
# Sidebar: Optimization rule (alleen Bouwwerk)
# - agg altijd SUM
# - weight via belangrijkheid (0.5/1.0/1.5)
# -----------------------------
st.sidebar.header("Optimalisatie (alleen Bouwwerk)")
use_opt_rule = st.sidebar.checkbox("Gebruik optimalisatie op", value=False)

opt_rules = []
if use_opt_rule:
    if not available_bouwwerk_features:
        st.sidebar.warning("Geen Bouwwerk-feature kolommen gevonden in je CSV.")
    else:
        opt_column = st.sidebar.selectbox(
            "Bouwwerk feature (kolom)",
            options=available_bouwwerk_features,
        )

        opt_dir = st.sidebar.selectbox("Richting (max/min)", options=DIR_OPTIONS, index=0)

        importance_label = st.sidebar.radio(
            "Belangrijkheid",
            options=["Niet belangrijk", "Belangrijk", "Zeer belangrijk"],
            index=1,
            horizontal=True
        )
        importance_map = {
            "Niet belangrijk": 0.5,
            "Belangrijk": 1.0,
            "Zeer belangrijk": 1.5
        }
        opt_weight = importance_map[importance_label]

        opt_rules = [
            OptimizationRule(
                source="Bouwwerk",
                column=opt_column,
                agg="sum",           # vast
                direction=opt_dir,   # type: ignore
                weight=float(opt_weight),
            )
        ]
st.sidebar.header("Toolbox context (aanname)")
grid_constraints_present = st.sidebar.checkbox("Netcongestie aanwezig", value=True)
feed_in_capacity_available = st.sidebar.checkbox("Invoedingsruimte beschikbaar (teruglever)", value=True)
flexible_load_present = st.sidebar.checkbox("Flexibele / stuurbare vraag aanwezig", value=False)
physical_space_available = st.sidebar.checkbox("Fysieke ruimte beschikbaar (installaties)", value=True)

sufficient_thermal_demand_present = st.sidebar.checkbox("Voldoende warmtevraag aanwezig", value=False)
thermal_demand_is_low = st.sidebar.checkbox("Warmtevraag is laag", value=False)
suitable_wind_location_present = st.sidebar.checkbox("Windlocatie geschikt", value=False)


# -----------------------------
# Handle upload -> temp file (pipeline verwacht pad)
# -----------------------------
if uploaded is not None:
    tmp_csv_path = "uploaded_combined.csv"
    with open(tmp_csv_path, "wb") as f:
        f.write(uploaded.getbuffer())
    input_csv_path = tmp_csv_path


# -----------------------------
# Build config
# -----------------------------
cfg = ClusterConfig(
    input_csv=input_csv_path,
    algorithm=algorithm,
    eps_m=float(eps_m),
    min_samples=int(min_samples),
    min_cluster_size=int(min_cluster_size),
    target_n_clusters=int(target_n_clusters),
    exclude_sources=exclude_sources,
    min_per_source=min_per_source,
    optimization_rules=opt_rules,
    make_plot=False,  # plots doen we in Streamlit
)


# -----------------------------
# Run
# -----------------------------
run_btn = st.button("Start clustering")

if run_btn:
    with st.spinner("Clustering draait..."):
        points_proj, clusters_wgs84 = run_return(cfg)

    st.success("De clusters zijn gevonden!")

    # ---- OUTPUT: cluster summary ----
    st.subheader("De gevonden clusters")
    if clusters_wgs84.empty:
        st.warning("Geen clusters gevonden na filtering/top-N.")
    else:
        show_cols = [c for c in [
            "rank", "cluster_id", "n_points", "score", "score_norm", "centroid_lat", "centroid_lon"
        ] if c in clusters_wgs84.columns]
        show_cols += [c for c in clusters_wgs84.columns if c.startswith("n_")]

        extra_cols = [
            "sum_pairwise_dist_m",
            "mean_pairwise_dist_m",
            "mean_opwek_to_others_m",
            "sum_opwek_kw",
            "sum_max_ruimte_verbruik_kw",
            "vrije_ruimte_kw",
        ]
        show_cols += [c for c in extra_cols if c in clusters_wgs84.columns]

        energy_cols = [
            "sum_opwek_kw",
            "sum_bouwwerk_kw",
            "mean_bouwwerk_demand_kw_used",
            "future_demand_kw",
            "vrije_ruimte_kw",
        ]

        show_cols += [c for c in energy_cols if c in clusters_wgs84.columns]

        # verwijder dubbele kolommen vóór display (Streamlit/PyArrow eis)
        clusters_wgs84 = clusters_wgs84.loc[:, ~clusters_wgs84.columns.duplicated()].copy()

        # dedupe show_cols
        seen = set()
        show_cols = [c for c in show_cols if not (c in seen or seen.add(c))]

        st.dataframe(
            clusters_wgs84[show_cols].sort_values("rank"),
            use_container_width=True
        )

    # ---- OUTPUT: punten per cluster (expanders), excl noise (-1) ----
    st.subheader("Punten per cluster")

    points_table = points_proj.drop(columns=["geometry"], errors="ignore").copy()

    if "cluster_id" not in points_table.columns:
        st.warning("Geen cluster_id kolom gevonden in punten output.")
    else:
        clustered_only = points_table[points_table["cluster_id"] != -1].copy()

        if clustered_only.empty:
            st.info("Geen punten in clusters (alles is noise of weggefilterd).")
        else:
            clustered_only["cluster_id"] = clustered_only["cluster_id"].astype(int)
            cluster_ids = sorted(clustered_only["cluster_id"].unique().tolist())

            # cluster_id vooraan
            cols = ["cluster_id"] + [c for c in clustered_only.columns if c != "cluster_id"]
            clustered_only = clustered_only[cols]

            for cid in cluster_ids:
                df_c = clustered_only[clustered_only["cluster_id"] == cid].copy()

                # Verwijder lege kolommen per cluster
                df_c = df_c.dropna(axis=1, how="all")

                n = len(df_c)
                src_summary = ""
                if "source" in df_c.columns:
                    vc = df_c["source"].value_counts(dropna=False)
                    src_summary = " | " + ", ".join([f"{k}: {int(v)}" for k, v in vc.items()])

                with st.expander(f"Cluster {cid} — {n} punten{src_summary}", expanded=False):
                    st.dataframe(df_c, use_container_width=True)
    solutions = load_toolbox("toolbox_solutions.yaml")

    st.subheader("Aanbevolen oplossingen per cluster (Toolbox)")



    if not clusters_wgs84.empty:
        # Loop per cluster
        for _, crow in clusters_wgs84.sort_values("rank").iterrows():
            cid = int(crow["cluster_id"])

            ctx = build_cluster_context(
                crow,
                grid_constraints_present=grid_constraints_present,
                feed_in_capacity_available=feed_in_capacity_available,
                flexible_load_present=flexible_load_present,
                physical_space_available=physical_space_available,
                suitable_wind_location_present=suitable_wind_location_present,
                sufficient_thermal_demand_present=sufficient_thermal_demand_present,
                thermal_demand_is_low=thermal_demand_is_low,
            )

            recs = recommend_for_cluster(ctx, solutions, top_k=5)

            with st.expander(f"Cluster {cid} — top {len(recs)} oplossingen", expanded=False):

                st.markdown(
                    f"""
                    **🏢 Clusterkenmerken**
                    - **Aantal gebouwen:** {ctx['number_of_buildings']}

                    **⚡ Elektrische balans**
                    - 🔋 **Opwek:** {crow['sum_opwek_kw']:.1f} kW
                    - 🏠 **Bestaande bouwwerken:** {crow['sum_bouwwerk_kw']:.1f} kW
                    - ➖ **Toekomstige vraag:** −{crow['future_demand_kw']:.1f} kW  
                      _(gemiddeld {crow['mean_bouwwerk_demand_kw_used']:.1f} kW per locatie)_
                    - ✅ **Vrije ruimte:** **{crow['vrije_ruimte_kw']:.1f} kW**
                    """
                )

                st.divider()  # optioneel maar netjes

                if not recs:
                    st.info("Geen oplossingen toepasbaar op basis van huidige toggles/regels.")
                else:
                    for r in recs:
                        sol = r.solution
                        st.markdown(f"### {r.name}  \n**Score:** {r.score:.2f}  \n**Waarom:** {r.reason}")

                        # compact kaartje met pro/cons
                        pros = sol.get("pros", [])
                        cons = sol.get("cons", [])
                        if pros:
                            st.markdown("**Voordelen:**")
                            st.write(pros)
                        if cons:
                            st.markdown("**Nadelen:**")
                            st.write(cons)

                        # wanneer wel/niet (regels)
                        if sol.get("preconditions"):
                            st.markdown("**Voorwaarden:**")
                            st.write(sol["preconditions"])
                        if sol.get("exclusion_rules"):
                            st.markdown("**Niet toepasbaar als:**")
                            st.write([{"condition": x.get("condition"), "reason": x.get("reason")} for x in
                                      sol["exclusion_rules"]])

    # ---- PLOTS (optioneel) ----
    if show_plots and not clusters_wgs84.empty:
        st.subheader("Plots")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Alle geselecteerde clusters**")
            fig_all = plot_clusters(points_proj, cfg)
            fig_all.set_size_inches(6, 6)
            st.pyplot(fig_all, use_container_width=True)

        with colB:
            st.markdown(f"**Top {top_n_plot} clusters**")
            fig_top = plot_top_n_clusters(points_proj, clusters_wgs84, n=top_n_plot)
            fig_top.set_size_inches(6, 6)
            st.pyplot(fig_top, use_container_width=True)

    # ---- DOWNLOADS ----
    st.subheader("Downloads")

    points_wgs84 = points_proj.to_crs(epsg=4326).copy()
    points_wgs84["lon"] = points_wgs84.geometry.x
    points_wgs84["lat"] = points_wgs84.geometry.y
    points_wgs84["geometry_wkt"] = points_wgs84.geometry.apply(lambda g: g.wkt if g is not None else None)
    points_csv = points_wgs84.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download points_with_clusters.csv",
        data=points_csv,
        file_name="points_with_clusters.csv",
        mime="text/csv",
    )

    clusters_out = clusters_wgs84.copy()
    clusters_out["geometry_wkt"] = clusters_out.geometry.apply(lambda g: g.wkt if g is not None else None)
    clusters_csv = clusters_out.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download cluster_summary.csv",
        data=clusters_csv,
        file_name="cluster_summary.csv",
        mime="text/csv",
    )
