# app.py
import os
import io
import hashlib
from typing import List
import json
import hashlib

import pandas as pd
import streamlit as st

from clustering_pipeline import ClusterConfig, OptimizationRule, run_return
from recommendations import compute_recommendations
from data_access import save_uploaded_csv, load_csv_preview, file_fingerprint
from validators import validate_dataframe
from ui_state import init_session_state, request_step, get_step, clear_results, apply_pending_navigation
from ui_components import step_header, show_messages, dataframe_section, kpi_row, solution_card
import base64
from pathlib import Path
#
# def render_fixed_logo():
#     logo_path = Path("afbeeldingen/logo-rijksvastgoedbedrijf-clipart-lg.png")
#
#     if not logo_path.exists():
#         return
#
#     encoded = base64.b64encode(logo_path.read_bytes()).decode()
#
#     st.markdown(
#         f"""
#         <style>
#         .rvb-logo {{
#             position: fixed;
#             top: 12px;
#             right: 20px;
#             z-index: 1000;
#         }}
#         </style>
#
#         <div class="rvb-logo">
#             <img src="data:image/png;base64,{encoded}" width="140">
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )



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

st.title("Verlichting netcongestie Tool — clustering & oplossingsadvies")

st.caption(
    "Deze tool groepeert locaties in clusters op basis van afstand en energie-informatie, "
    "en geeft per cluster oplossingsopties uit de RVB-toolbox."
)
st.divider()


# -----------------------------
# Sidebar: navigation + data source
# -----------------------------
with st.sidebar:
    with st.sidebar:
        st.image(
            "afbeeldingen/Logo_rijksvastgoedbedrijf.svg",
            width=300,
        )
        st.markdown("---")
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

# render_fixed_logo()


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

    # st.session_state.df_preview = df_prev
    #
    # vr = validate_dataframe(df_prev, geometry_col="geometry", source_col="source")
    # st.session_state.validation = {"errors": vr.errors, "warnings": vr.warnings, "info": vr.info}
    # show_messages(vr.errors, vr.warnings, vr.info)


    # if vr.errors:
    #     st.info("Los eerst de fouten op. Daarna kun je verder naar stap 2.")
    #     return
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        st.image(
            "afbeeldingen/ChatGPT Image Jan 25, 2026 at 05_03_10 PM.png",
            caption="source: ChatGPT",
            width=700,
        )

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
        "Stap 2 — Clustering keuzes instellen",
        "In deze stap kan je de clustering keuzes instellen. "
        "Klik daarna op **Start analyse**.",
    )

    val = st.session_state.validation or {}
    if val.get("errors"):
        st.warning("Er zijn nog fouten in de data. Ga terug naar stap 1 om dit op te lossen.")
        if st.button("← Terug naar stap 1"):
            request_step(1)
            st.rerun()
        return



    try:
        header_df = pd.read_csv(st.session_state.input_csv_path, nrows=0)
        csv_columns = [str(c).strip() for c in header_df.columns]
    except Exception:
        csv_columns = []

    # Match met whitelist (BOUWWERK_FEATURES)
    available_bouwwerk_features = [c for c in BOUWWERK_FEATURES if c in csv_columns]

    # -----------------------------
    # Basisinstellingen (geen form -> direct zichtbaar/aanpasbaar)
    # -----------------------------
    st.subheader("Basisinstellingen")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.text_input(
            "Clustering algoritme",
            value="DBSCAN",
            disabled=True,
        )

    with c2:
        eps_m = st.number_input(
            "Maximale afstand tussen locaties (meter)",
            min_value=50.0,
            max_value=50000.0,
            value=float(st.session_state.get("eps_m", 5000.0)),
            step=50.0,
            key="eps_m",
        )

    with c3:
        min_cluster_n = st.number_input(
            "Minimaal aantal locaties per cluster",
            min_value=1,
            max_value=999,
            value=int(
                st.session_state.get(
                    "min_cluster_n",
                    st.session_state.get("min_samples", 3),
                )
            ),
            step=1,
            key="min_cluster_n",
        )

    # Koppel expliciet: beide krijgen dezelfde waarde
    min_samples = int(min_cluster_n)
    min_cluster_size = int(min_cluster_n)

    target_n_clusters = st.number_input(
        "Aantal clusters om te tonen (Top N)",
        min_value=1,
        max_value=200,
        value=int(st.session_state.get("target_n_clusters", 10)),
        step=1,
        key="target_n_clusters",
    )

    # -----------------------------
    # Geavanceerd
    # -----------------------------
    with st.expander("Geavanceerde opties", expanded=False):
        st.markdown("**Bronnen**")
        exclude_sources = st.multiselect(
            "Locatietypes uitsluiten",
            options=SOURCE_OPTIONS,
            default=st.session_state.get("exclude_sources", ["Bovenregionaal"]),
            help="Uitsluiten betekent: deze locaties tellen niet mee in clustering.",
            key="exclude_sources",
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
                    value=int(st.session_state.get(f"min_per_source_{s}", default_val)),
                    step=1,
                    help="Als je dit op 0 zet, stellen we geen minimum-eis.",
                    key=f"min_per_source_{s}",
                )
                if int(v) > 0:
                    min_per_source[s] = int(v)

        st.markdown("**Optimalisatie (alleen Bouwwerk)**")
        use_opt_rule = st.checkbox(
            "Gebruik optimalisatie",
            value=bool(st.session_state.get("use_opt_rule", False)),
            help="Kiest Top-N clusters op basis van een Bouwwerk-kenmerk.",
            key="use_opt_rule",
        )

        opt_rules: List[OptimizationRule] = []
        if use_opt_rule:
            if not available_bouwwerk_features:
                st.warning("Geen Bouwwerk-feature kolommen gevonden in je CSV (van de bekende lijst).")
            else:
                opt_column = st.selectbox(
                    "Bouwwerk feature (kolom)",
                    options=available_bouwwerk_features,
                    index=0,
                    key="opt_column",
                )
                opt_dir = st.selectbox(
                    "Richting (max/min)",
                    options=DIR_OPTIONS,
                    index=0,
                    key="opt_dir",
                )
                importance = st.selectbox(
                    "Belang (weging)",
                    options=["laag", "normaal", "hoog"],
                    index=["laag", "normaal", "hoog"].index(
                        st.session_state.get("importance", "normaal")
                    ),
                    key="importance",
                )
                weight_map = {"laag": 0.5, "normaal": 1.0, "hoog": 2.0}
                opt_rules = [
                    OptimizationRule(
                        column=str(opt_column),
                        agg="sum",
                        direction=str(opt_dir),
                        weight=float(weight_map[str(importance)]),
                    )
                ]

        # Toolbox context is bewust weg uit stap 2 (komt pas in stap 4)

    st.divider()

    # -----------------------------
    # Start analyse
    # -----------------------------
    start = st.button("Start analyse", type="primary")

    if not start:
        st.info("Pas instellingen aan en klik op **Start analyse** om clustering uit te voeren.")
        return

    cfg = ClusterConfig(
        input_csv=st.session_state.input_csv_path,
        algorithm="dbscan",  # vast
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
            request_step(3)  # <-- dit was de bug in jouw app
            st.rerun()




# -----------------------------
# Step 3: Clusters view
# -----------------------------
def render_step_3() -> None:
    # Force scroll naar boven bij openen van stap 3
    st.markdown('<div id="top-of-step-3"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        const el = window.parent.document.getElementById("top-of-step-3");
        if (el) { el.scrollIntoView({behavior: "instant", block: "start"}); }
        </script>
        """,
        unsafe_allow_html=True,
    )

    step_header(
        "Stap 3 — Clusters bekijken",
        "Bekijk de gevonden clusters. Je kunt de top 3 clusters tonen of één specifiek cluster kiezen "
        ".",
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
        [("Aantal clusters", f"{n_clusters:,}".replace(",", "."), "Aantal gevonden clusters."),
            ("Totaal aantal locaties in de analyse", f"{n_points:,}".replace(",", "."), "Aantal locaties in de analyse."),


        ]
    )

    clusters_df = pd.DataFrame(clusters).copy()
    if "geometry" in clusters_df.columns:
        clusters_df = clusters_df.drop(columns=["geometry"])

    # --- Alleen deze details tonen + labels zoals jij wil ---
    DETAIL_LABELS = {
        "rank": "rank",
        "cluster_id": "Cluster ID",
        "n_points": "Aantal locaties",
        "sum_pairwise_dist_m": "Totale afstand tussen locaties",
        "mean_pairwise_dist_m": "Gemiddelde afstand tussen de locaties",
        "mean_opwek_to_others_m": "Energie opwek (MWp)",
        "sub_opwek_kw": "Energie opwek (kw)",
        "sum_bouwwerk_kw": "Vraag bouwwerken (kw)",
        "future_demand_kw": "Toekomstige vraag VKA's",
        "vrije_ruimte_kw": "Vrije ruimte energie in het cluster (kw)",
        "n_Energieproject": "Aantal energieprojecten",
        "n_Bouwwerk": "Aantal bouwwerken",
        "n_locatiespecifiek": "Aantal locatiespecifiek VKA",
        "n_Bovenregionaal": "Aantal bovenregionaal VKA",
    }

    # Cluster-overzicht: alleen gewenste kolommen
    keep_cols = [c for c in DETAIL_LABELS.keys() if c in clusters_df.columns]
    clusters_overview = clusters_df[keep_cols].copy()
    # Verwijder lege rijen (alles NaN/leeg) en rijen zonder cluster_id
    clusters_overview = clusters_overview.dropna(how="all")

    if "cluster_id" in clusters_overview.columns:
        clusters_overview = clusters_overview.dropna(subset=["cluster_id"])

    # Sorteren op rank in overzicht
    if "rank" in clusters_overview.columns:
        try:
            clusters_overview["rank"] = pd.to_numeric(clusters_overview["rank"], errors="coerce")
            clusters_overview = clusters_overview.sort_values("rank", ascending=True)
        except Exception:
            clusters_overview = clusters_overview.sort_values("rank", ascending=True)
    else:
        st.info("Kolom 'rank' ontbreekt; cluster-overzicht wordt niet op rank gesorteerd.")

    # Hernoem kolommen voor overzicht
    clusters_overview = clusters_overview.rename(columns=DETAIL_LABELS)

    dataframe_section("Gevonden clusters", clusters_overview, height=220)

    ids = (
        sorted([int(x) for x in clusters_df["cluster_id"].dropna().unique().tolist()])
        if "cluster_id" in clusters_df.columns
        else []
    )
    if not ids:
        st.info("Geen clusters gevonden. Tip: vergroot de afstand (eps) of verlaag minima.")
        return

    # -----------------------------
    # Default selectie = rank 1
    # -----------------------------
    default_cluster_id = ids[0]
    if "rank" in clusters_df.columns:
        try:
            ranked = clusters_df.dropna(subset=["rank", "cluster_id"]).copy()
            ranked["rank_num"] = pd.to_numeric(ranked["rank"], errors="coerce")
            ranked = ranked.dropna(subset=["rank_num"]).sort_values("rank_num", ascending=True)
            if not ranked.empty:
                default_cluster_id = int(ranked.iloc[0]["cluster_id"])  # beste/rank=1
        except Exception:
            default_cluster_id = ids[0]

    # -----------------------------
    # Keuze: top 3 of specifiek
    # -----------------------------
    mode = st.radio(
        "Weergave",
        options=["Laat top 3 zien", "Laat specifiek cluster zien"],
        index=0,  # default: specifiek
        horizontal=True,
        key="step3_view_mode",
    )

    def render_cluster_block(cluster_id: int) -> None:
        row = clusters_df[clusters_df["cluster_id"] == int(cluster_id)].iloc[0]
        st.subheader(f"Cluster {int(cluster_id)}")

        # 1) Eerst: locaties in dit cluster
        st.markdown("#### Locaties in dit cluster")
        if "cluster_id" not in points.columns:
            st.info("Geen puntinformatie met cluster_id beschikbaar om locatienamen te tonen.")
        elif "LocatieNaam" not in points.columns:
            st.info(
                "Kolom **LocatieNaam** ontbreekt in de data. Voeg deze kolom toe om locatienamen per cluster te tonen.")
        else:
            pts_df = pd.DataFrame(points)
            pts_cluster = pts_df[pts_df["cluster_id"] == int(cluster_id)].copy()

            cols_to_show = ["LocatieNaam"]
            if "source" in pts_cluster.columns:
                cols_to_show.append("source")

            show_df = pts_cluster[cols_to_show].copy()
            show_df["LocatieNaam"] = show_df["LocatieNaam"].astype(str).map(lambda x: x.strip())
            show_df = show_df[show_df["LocatieNaam"].notna() & (show_df["LocatieNaam"] != "")]
            show_df = show_df.sort_values("LocatieNaam")

            st.caption(
                f"{int(show_df['LocatieNaam'].nunique())} unieke locaties"
            )
            st.dataframe(show_df, use_container_width=True, height=300, hide_index=True)

        # 2) Daarna: clusterdetails
        st.markdown("#### Details")
        detail_rows = []
        for key, label in DETAIL_LABELS.items():
            if key in row.index:
                detail_rows.append({"Kenmerk": label, "Waarde": row[key]})

        details_df = pd.DataFrame(detail_rows)
        st.dataframe(details_df, use_container_width=True, height=260, hide_index=True)

    if mode == "Laat top 3 zien":
        if "rank" in clusters_df.columns:
            top3_ids = (
                clusters_df.dropna(subset=["rank", "cluster_id"])
                .assign(rank_num=lambda d: pd.to_numeric(d["rank"], errors="coerce"))
                .dropna(subset=["rank_num"])
                .sort_values("rank_num", ascending=True)
                .head(3)["cluster_id"]
                .astype(int)
                .tolist()
            )
        else:
            top3_ids = ids[:3]

        if not top3_ids:
            st.info("Geen clusters om te tonen.")
        else:
            st.info(f"Top {len(top3_ids)} clusters.")
            for cid in top3_ids:
                render_cluster_block(int(cid))
                st.divider()

            st.session_state.selected_cluster_id = int(top3_ids[0])

    else:
        default_index = ids.index(default_cluster_id) if default_cluster_id in ids else 0

        selected = st.selectbox(
            "Kies een cluster voor detail",
            options=ids,
            index=default_index,
            key="step3_selected_cluster_id",
        )
        st.session_state.selected_cluster_id = int(selected)
        render_cluster_block(int(selected))

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

    # Zorgt dat toggles nooit None zijn (fix van eerder blijft)
    DEFAULT_TOGGLES = {
        "grid_constraints_present": True,
        "feed_in_capacity_available": True,
        "flexible_load_present": False,
        "physical_space_available": True,
        "suitable_wind_location_present": False,
        "spatial_opportunity_available": True,
        "sufficient_thermal_demand_present": False,
        "thermal_demand_is_low": False,
    }

    toggles = st.session_state.get("toolbox_toggles")
    if not isinstance(toggles, dict):
        toggles = DEFAULT_TOGGLES.copy()
    else:
        for k, v in DEFAULT_TOGGLES.items():
            toggles.setdefault(k, v)

    # --- widgets altijd zichtbaar ---
    c1, c2 = st.columns(2)

    with c1:
        toggles["grid_constraints_present"] = st.checkbox(
            "Netcongestie / netbeperking aanwezig",
            value=bool(toggles.get("grid_constraints_present", True)),
            key="toggle_grid_constraints_present",
        )
        toggles["feed_in_capacity_available"] = st.checkbox(
            "Terugleverruimte beschikbaar",
            value=bool(toggles.get("feed_in_capacity_available", True)),
            key="toggle_feed_in_capacity_available",
        )
        toggles["flexible_load_present"] = st.checkbox(
            "Flexibele/stuurbare vraag aanwezig",
            value=bool(toggles.get("flexible_load_present", True)),
            key="toggle_flexible_load_present",
        )
        toggles["physical_space_available"] = st.checkbox(
            "Fysieke ruimte beschikbaar (bijv. container/ruimte voor techniek)",
            value=bool(toggles.get("physical_space_available", True)),
            key="toggle_physical_space_available",
        )

    with c2:
        toggles["suitable_wind_location_present"] = st.checkbox(
            "Geschikte windlocatie aanwezig",
            value=bool(toggles.get("suitable_wind_location_present", False)),
            key="toggle_suitable_wind_location_present",
        )
        toggles["spatial_opportunity_available"] = st.checkbox(
            "Ruimtelijke mogelijkheid aanwezig (gebied/ruimte/bodem)",
            value=bool(toggles.get("spatial_opportunity_available", True)),
            key="toggle_spatial_opportunity_available",
        )
        toggles["sufficient_thermal_demand_present"] = st.checkbox(
            "Voldoende thermische vraag aanwezig",
            value=bool(toggles.get("sufficient_thermal_demand_present", True)),
            key="toggle_sufficient_thermal_demand_present",
        )
        toggles["thermal_demand_is_low"] = st.checkbox(
            "Thermische vraag is laag",
            value=bool(toggles.get("thermal_demand_is_low", False)),
            key="toggle_thermal_demand_is_low",
        )

    # altijd terugschrijven
    st.session_state.toolbox_toggles = toggles

    top_k = st.selectbox(
        "Maximaal aantal oplossingen",
        options=list(range(1, 11)),  # 1 t/m 10
        index=4,  # standaard = 5
    )

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

        toggles_fingerprint = hashlib.sha256(
            json.dumps(toggles, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        recs = compute_recommendations(
            view_df,
            yaml_path=yaml_path,
            top_k=int(top_k),
            toggles=dict(toggles),  # copy: voorkomt gedoe met mutatie
            toggles_fingerprint=toggles_fingerprint,
        )

    except Exception as e:
        st.error("Kon toolbox-aanbevelingen niet berekenen.")
        st.exception(e)
        return

    st.session_state.recommendations = recs

    for cid, lst in recs.items():
        st.markdown(f"## Cluster {cid}")
        # --- LocatieNaam + source onder cluster header (zoals stap 3) ---
        if "cluster_id" in points.columns and "LocatieNaam" in points.columns:
            pts_df = pd.DataFrame(points)
            pts_cluster = pts_df[pts_df["cluster_id"] == int(cid)].copy()

            cols_to_show = ["LocatieNaam"]
            if "source" in pts_cluster.columns:
                cols_to_show.append("source")

            show_df = pts_cluster[cols_to_show].copy()
            show_df["LocatieNaam"] = show_df["LocatieNaam"].astype(str).map(lambda x: x.strip())
            show_df = show_df[show_df["LocatieNaam"].notna() & (show_df["LocatieNaam"] != "")]
            show_df = show_df.sort_values("LocatieNaam")

            st.caption(
                f"{int(show_df['LocatieNaam'].nunique())} unieke locatiena(a)m(en) "
                f"(totaal punten in cluster: {len(pts_cluster)})"
            )
            st.dataframe(show_df, use_container_width=True, height=220, hide_index=True)
        else:
            # Stil falen (geen spam): alleen tonen als je wil debuggen
            pass

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
