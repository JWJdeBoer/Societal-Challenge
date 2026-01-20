# app.py
import os
import io
import hashlib
from typing import List

import pandas as pd
import streamlit as st

from clustering_pipeline import ClusterConfig, OptimizationRule, run_return
from recommendations import compute_recommendations
from data_access import save_uploaded_csv, load_csv_preview, file_fingerprint
from validators import validate_dataframe
from ui_state import init_session_state, request_step, get_step, clear_results, apply_pending_navigation
from ui_components import step_header, show_messages, dataframe_section, kpi_row, solution_card


# -----------------------------
# Constants (kept close to original for diff-friendliness)
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
    "Opwek",
    "Max ruimte verbruik",
    "Huidig energiegebruik",
    "Toekomstig energiegebruik",
]


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="RVB — Netcongestie Toolbox (clustering)",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
# BELANGRIJK: altijd vóór de sidebar widgets
apply_pending_navigation()

st.title("Netcongestie Toolbox — clustering & oplossingsadvies")
st.caption(
    "Deze tool groepeert locaties in clusters op basis van afstand en energie-informatie, "
    "en geeft per cluster oplossingsopties uit de RVB-toolbox."
)
st.divider()


# -----------------------------
# Sidebar: navigation + data source
# -----------------------------
with st.sidebar:
    st.header("Navigatie")

    def _on_nav_change():
        st.session_state.step = int(st.session_state.nav_step)

    st.radio(
        "Waar wil je naartoe?",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1) Data laden",
            2: "2) Instellingen",
            3: "3) Clusters",
            4: "4) Oplossingen & export",
        }[x],
        key="nav_step",
        on_change=_on_nav_change,
    )

    # Safety net: maak step gelijk aan radio
    st.session_state.step = int(st.session_state.nav_step)

    st.divider()
    st.header("Data")

    data_source = st.radio(
        "CSV bron",
        options=["default", "upload"],
        format_func=lambda x: "Gebruik standaard combined.csv" if x == "default" else "Upload een CSV",
        index=0 if st.session_state.data_source == "default" else 1,
    )
    st.session_state.data_source = data_source

    if data_source == "upload":
        uploaded = st.file_uploader("Upload combined.csv", type=["csv"])
        if uploaded is not None:
            path, fp = save_uploaded_csv(uploaded)
            st.session_state.input_csv_path = path
            st.session_state.df_fingerprint = fp
    else:
        st.session_state.input_csv_path = "combined.csv"
        st.session_state.df_fingerprint = file_fingerprint(st.session_state.input_csv_path)

    if not os.path.exists(st.session_state.input_csv_path):
        st.error("Het CSV-bestand is niet gevonden. Controleer de locatie of gebruik upload.")
        st.stop()

    if st.button("Resultaten wissen"):
        clear_results()
        st.toast("Resultaten gewist.", icon=None)

    st.divider()
    st.caption("Tip: wijzig instellingen in stap 2 en klik pas daarna op **Start analyse**.")


# -----------------------------
# Step 1: Load + validate
# -----------------------------
def render_step_1() -> None:
    step_header(
        "Stap 1 — Data laden en controleren",
        "We lezen de CSV in en controleren of de verplichte kolommen aanwezig zijn. "
        "Als er fouten zijn, tonen we wat je moet aanpassen.",
    )

    try:
        df_prev = load_csv_preview(st.session_state.input_csv_path, nrows=200)
    except Exception as e:
        st.error("Kon de CSV niet inlezen. Controleer of het bestand een geldige CSV is.")
        st.exception(e)
        st.stop()

    st.session_state.df_preview = df_prev

    vr = validate_dataframe(df_prev, geometry_col="geometry", source_col="source")
    st.session_state.validation = {"errors": vr.errors, "warnings": vr.warnings, "info": vr.info}
    show_messages(vr.errors, vr.warnings, vr.info)

    st.subheader("Voorbeeld van de data")
    st.dataframe(df_prev.head(20), use_container_width=True)

    if vr.errors:
        st.info("Los eerst de fouten op. Daarna kun je verder naar stap 2.")
        return

    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Ga naar stap 2 →"):
            request_step(2)
            st.rerun()


# -----------------------------
# Step 2: Settings + run
# -----------------------------
def render_step_2() -> None:
    step_header(
        "Stap 2 — Instellingen kiezen",
        "Kies de clusteringinstellingen. Gebruik bij twijfel de standaardwaarden. "
        "Klik daarna op **Start analyse**.",
    )

    val = st.session_state.validation or {}
    if val.get("errors"):
        st.warning("Er zijn nog fouten in de data. Ga terug naar stap 1 om dit op te lossen.")
        if st.button("← Terug naar stap 1"):
            request_step(1)
            st.rerun()
        return

    df_prev = st.session_state.df_preview
    available_bouwwerk_features = []
    if isinstance(df_prev, pd.DataFrame):
        available_bouwwerk_features = [c for c in BOUWWERK_FEATURES if c in df_prev.columns]

    with st.form("settings_form", clear_on_submit=False):
        st.subheader("Basisinstellingen")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            algorithm = st.selectbox("Clustering algoritme", options=["dbscan", "hdbscan"], index=0)
        with c2:
            eps_m = st.number_input(
                "Maximale afstand tussen locaties (meter)",
                min_value=50.0,
                max_value=50000.0,
                value=5000.0,
                step=50.0,
            )
        with c3:
            min_samples = st.number_input(
                "Minimaal aantal locaties per cluster (algoritme)",
                min_value=1,
                max_value=999,
                value=3,
                step=1,
            )
        with c4:
            min_cluster_size = st.number_input(
                "Minimaal aantal locaties per cluster (filter)",
                min_value=1,
                max_value=999,
                value=3,
                step=1,
            )

        target_n_clusters = st.number_input(
            "Aantal clusters om te tonen (Top N)",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
        )

        with st.expander("Geavanceerde opties", expanded=False):
            st.markdown("**Bronnen**")
            exclude_sources = st.multiselect(
                "Locatietypes uitsluiten",
                options=SOURCE_OPTIONS,
                default=["Bovenregionaal"],
                help="Uitsluiten betekent: deze locaties tellen niet mee in clustering.",
            )

            st.caption("Minimaal aantal locaties per locatietype binnen een cluster (0 = geen eis).")
            min_per_source = {}
            cols = st.columns(2)
            for i, s in enumerate(SOURCE_OPTIONS):
                with cols[i % 2]:
                    default_val = 0 if s in ["Bovenregionaal"] else 1
                    v = st.number_input(
                        f"minimaal aantal: {s}",
                        min_value=0,
                        max_value=999,
                        value=default_val,
                        step=1,
                        help="Als je dit op 0 zet, stellen we geen minimum-eis.",
                    )
                    if int(v) > 0:
                        min_per_source[s] = int(v)

            st.markdown("**Optimalisatie (alleen Bouwwerk)**")
            use_opt_rule = st.checkbox(
                "Gebruik optimalisatie",
                value=False,
                help="Kiest Top-N clusters op basis van een Bouwwerk-kenmerk.",
            )
            opt_rules: List[OptimizationRule] = []
            if use_opt_rule:
                if not available_bouwwerk_features:
                    st.warning("Geen Bouwwerk-feature kolommen gevonden in je CSV (van de bekende lijst).")
                else:
                    opt_column = st.selectbox("Bouwwerk feature (kolom)", options=available_bouwwerk_features)
                    opt_dir = st.selectbox("Richting (max/min)", options=DIR_OPTIONS, index=0)
                    importance = st.select_slider(
                        "Belang (weging)",
                        options=["laag", "normaal", "hoog"],
                        value="normaal",
                        help="Hoe zwaar telt dit mee in de selectie van Top-N clusters?",
                    )
                    weight_map = {"laag": 0.5, "normaal": 1.0, "hoog": 2.0}
                    opt_rules = [
                        OptimizationRule(column=opt_column, agg="sum", direction=opt_dir, weight=weight_map[importance])
                    ]

            st.markdown("**Toolbox-context (voor stap 4)**")
            st.caption("Deze vragen helpen om oplossingen te filteren. Je kunt dit later aanpassen.")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                grid_constraints_present = st.checkbox("Netcongestie/knelpunt aanwezig", value=True)
            with t2:
                feed_in_capacity_available = st.checkbox("Teruglevering mogelijk", value=True)
            with t3:
                flexible_load_present = st.checkbox("Stuurbaar verbruik aanwezig", value=False)
            with t4:
                physical_space_available = st.checkbox("Fysieke ruimte beschikbaar", value=True)

            with st.expander("Aanvullende context (optioneel)", expanded=False):
                suitable_wind_location_present = st.checkbox("Geschikte windlocatie", value=False)
                spatial_opportunity_available = st.checkbox("Ruimtelijke kans aanwezig", value=True)
                sufficient_thermal_demand_present = st.checkbox("Voldoende warmtevraag", value=False)
                thermal_demand_is_low = st.checkbox("Warmtevraag is laag", value=False)

        submitted = st.form_submit_button("Start analyse")

    if not submitted:
        st.info("Kies instellingen en klik op **Start analyse** om clustering uit te voeren.")
        return

    cfg = ClusterConfig(
        input_csv=st.session_state.input_csv_path,
        algorithm=str(algorithm),
        eps_m=float(eps_m),
        min_samples=int(min_samples),
        min_cluster_size=int(min_cluster_size),
        target_n_clusters=int(target_n_clusters),
        exclude_sources=list(exclude_sources),
        min_per_source=dict(min_per_source),
        optimization_rules=list(opt_rules),
        make_plot=False,
    )
    cfg_hash = hashlib.sha256(repr(cfg).encode("utf-8")).hexdigest()

    st.session_state.toolbox_toggles = {
        "grid_constraints_present": bool(grid_constraints_present),
        "feed_in_capacity_available": bool(feed_in_capacity_available),
        "flexible_load_present": bool(flexible_load_present),
        "physical_space_available": bool(physical_space_available),
        "suitable_wind_location_present": bool(suitable_wind_location_present),
        "spatial_opportunity_available": bool(spatial_opportunity_available),
        "sufficient_thermal_demand_present": bool(sufficient_thermal_demand_present),
        "thermal_demand_is_low": bool(thermal_demand_is_low),
    }

    need_recompute = (
        (st.session_state.cfg_hash != cfg_hash)
        or (st.session_state.points_proj is None)
        or (st.session_state.clusters_wgs84 is None)
    )

    if need_recompute:
        clear_results()
        st.session_state.cfg_hash = cfg_hash

        with st.status("Analyse draait…", expanded=True) as status:
            status.write("Data wordt geladen…")
            try:
                points_proj, clusters_wgs84 = run_return(cfg)
            except Exception as e:
                status.update(label="Fout bij clustering", state="error", expanded=True)
                st.error("Er ging iets mis tijdens clustering. Controleer je data en instellingen.")
                st.exception(e)
                return

            st.session_state.points_proj = points_proj
            st.session_state.clusters_wgs84 = clusters_wgs84
            status.write("Clustering afgerond.")
            status.update(label="Analyse afgerond", state="complete", expanded=False)

    st.success("Clusteringresultaten zijn klaar.")
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Ga naar stap 3 →"):
            request_step(4)
            st.rerun()


# -----------------------------
# Step 3: Clusters view
# -----------------------------
def render_step_3() -> None:
    step_header(
        "Stap 3 — Clusters bekijken",
        "Bekijk de gevonden clusters. Kies een cluster om detailinformatie te zien.",
    )

    clusters = st.session_state.clusters_wgs84
    points = st.session_state.points_proj

    if clusters is None or points is None:
        st.warning("Er zijn nog geen resultaten. Ga naar stap 2 en klik op **Start analyse**.")
        if st.button("← Naar stap 2"):
            request_step(2)
            st.rerun()
        return

    try:
        n_points = int(len(points))
        n_clusters = int(clusters["cluster_id"].nunique()) if "cluster_id" in clusters.columns else int(len(clusters))
        n_noise = int((points["cluster_id"] == -1).sum()) if "cluster_id" in points.columns else 0
        noise_pct = f"{(100.0 * n_noise / max(n_points, 1)):.1f}%"
    except Exception:
        n_points, n_clusters, n_noise, noise_pct = 0, 0, 0, "—"

    kpi_row(
        [
            ("Locaties (punten)", f"{n_points:,}".replace(",", "."), "Aantal locaties in de input."),
            ("Clusters", f"{n_clusters:,}".replace(",", "."), "Aantal gevonden clusters (na filtering)."),
            ("Noise / niet-geclusterd", f"{n_noise:,}".replace(",", "."), "Locaties die niet in een cluster vielen."),
            ("% noise", noise_pct, "Aandeel locaties zonder cluster."),
        ]
    )

    clusters_df = pd.DataFrame(clusters).copy()
    if "geometry" in clusters_df.columns:
        clusters_df = clusters_df.drop(columns=["geometry"])

    sort_col = "cluster_score_norm" if "cluster_score_norm" in clusters_df.columns else None
    if sort_col:
        clusters_df = clusters_df.sort_values(sort_col, ascending=False)

    dataframe_section("Cluster-overzicht", clusters_df, height=380)

    ids = (
        sorted([int(x) for x in clusters_df["cluster_id"].dropna().unique().tolist()])
        if "cluster_id" in clusters_df.columns
        else []
    )
    if not ids:
        st.info("Geen clusters gevonden. Tip: vergroot de afstand (eps) of verlaag minima.")
        return

    selected = st.selectbox("Kies een cluster voor detail", options=ids, index=0)
    st.session_state.selected_cluster_id = int(selected)

    row = clusters_df[clusters_df["cluster_id"] == int(selected)].iloc[0]
    st.subheader(f"Cluster {int(selected)} — details")
    detail_cols = [c for c in row.index if c not in ["cluster_id"]]
    st.dataframe(
        pd.DataFrame({"kenmerk": detail_cols, "waarde": [row[c] for c in detail_cols]}),
        use_container_width=True,
        height=260,
    )

    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Ga naar stap 4 →"):
            request_step(4)
            st.rerun()


# -----------------------------
# Step 4: Recommendations + export
# -----------------------------
def render_step_4() -> None:
    step_header(
        "Stap 4 — Oplossingen & export",
        "Op basis van de clusters tonen we oplossingen uit de toolbox. "
        "Daarna kun je resultaten downloaden.",
    )

    clusters = st.session_state.clusters_wgs84
    points = st.session_state.points_proj

    if clusters is None or points is None:
        st.warning("Er zijn nog geen resultaten. Ga naar stap 2 en klik op **Start analyse**.")
        if st.button("← Naar stap 2"):
            request_step(2)
            st.rerun()
        return

    clusters_df = pd.DataFrame(clusters).copy()
    if "geometry" in clusters_df.columns:
        clusters_df = clusters_df.drop(columns=["geometry"])

    selected_cluster_id = st.session_state.selected_cluster_id
    if selected_cluster_id is None and "cluster_id" in clusters_df.columns and not clusters_df.empty:
        selected_cluster_id = int(clusters_df["cluster_id"].iloc[0])
        st.session_state.selected_cluster_id = selected_cluster_id

    st.subheader("Toolbox-instellingen")
    toggles = st.session_state.get(
        "toolbox_toggles",
        {
            "grid_constraints_present": True,
            "feed_in_capacity_available": True,
            "flexible_load_present": False,
            "physical_space_available": True,
            "suitable_wind_location_present": False,
            "spatial_opportunity_available": True,
            "sufficient_thermal_demand_present": False,
            "thermal_demand_is_low": False,
        },
    )

    with st.expander("Context aanpassen (optioneel)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            toggles["grid_constraints_present"] = st.checkbox(
                "Netcongestie/knelpunt aanwezig",
                value=bool(toggles.get("grid_constraints_present", True)),
            )
        with c2:
            toggles["feed_in_capacity_available"] = st.checkbox(
                "Teruglevering mogelijk",
                value=bool(toggles.get("feed_in_capacity_available", True)),
            )
        with c3:
            toggles["flexible_load_present"] = st.checkbox(
                "Stuurbaar verbruik aanwezig",
                value=bool(toggles.get("flexible_load_present", False)),
            )
        with c4:
            toggles["physical_space_available"] = st.checkbox(
                "Fysieke ruimte beschikbaar",
                value=bool(toggles.get("physical_space_available", True)),
            )

    st.session_state.toolbox_toggles = toggles

    top_k = st.slider("Aantal oplossingen per cluster", min_value=1, max_value=10, value=5, step=1)

    mode = st.radio(
        "Welke clusters wil je bekijken?",
        options=["selected", "topn"],
        format_func=lambda x: "Alleen geselecteerde cluster" if x == "selected" else "Top N clusters",
        index=0,
        horizontal=True,
    )

    if mode == "selected":
        view_df = clusters_df[clusters_df["cluster_id"] == int(selected_cluster_id)].copy()
    else:
        view_df = clusters_df.copy()

    yaml_path = "toolbox_solutions.yaml"
    try:
        recs = compute_recommendations(view_df, yaml_path=yaml_path, top_k=int(top_k), toggles=toggles)
    except Exception as e:
        st.error("Kon toolbox-aanbevelingen niet berekenen.")
        st.exception(e)
        return

    st.session_state.recommendations = recs

    for cid, lst in recs.items():
        st.markdown(f"## Cluster {cid}")
        if not lst:
            st.info("Geen toepasselijke oplossingen gevonden (op basis van de huidige context/voorwaarden).")
            continue
        for r in lst:
            with st.container(border=True):
                solution_card(r.solution, why=r.reason)

    st.divider()
    st.subheader("Resultaten downloaden")

    try:
        points_df = pd.DataFrame(points).copy()
        clusters_export_df = pd.DataFrame(clusters).copy()

        if "geometry" in points_df.columns:
            points_df["geometry"] = points_df["geometry"].astype(str)
        if "geometry" in clusters_export_df.columns:
            clusters_export_df["geometry"] = clusters_export_df["geometry"].astype(str)

        points_csv = points_df.to_csv(index=False).encode("utf-8")
        clusters_csv = clusters_export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download punten (CSV)",
            data=points_csv,
            file_name="points_with_clusters.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download clusters (CSV)",
            data=clusters_csv,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error("Export naar CSV ging mis.")
        st.exception(e)

    try:
        import openpyxl  # noqa: F401

        rec_rows = []
        for cid, lst in (st.session_state.recommendations or {}).items():
            for r in lst:
                rec_rows.append(
                    {
                        "cluster_id": cid,
                        "solution_id": r.solution_id,
                        "name": r.name,
                        "score": r.score,
                        "reason": r.reason,
                        "category_main": (r.solution.get("category") or {}).get("main"),
                        "congestion_type": (r.solution.get("category") or {}).get("congestion_type"),
                    }
                )
        rec_df = pd.DataFrame(rec_rows)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            points_df.to_excel(writer, sheet_name="points", index=False)
            clusters_export_df.to_excel(writer, sheet_name="clusters", index=False)
            rec_df.to_excel(writer, sheet_name="recommendations", index=False)

        st.download_button(
            "Download alles (Excel)",
            data=buf.getvalue(),
            file_name="rvb_toolbox_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("Excel-export is niet beschikbaar (openpyxl ontbreekt). CSV-export werkt altijd.")


# -----------------------------
# Router
# -----------------------------
if get_step() == 1:
    render_step_1()
elif get_step() == 2:
    render_step_2()
elif get_step() == 3:
    render_step_3()
else:
    render_step_4()
