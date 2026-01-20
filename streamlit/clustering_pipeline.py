from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
from sklearn.cluster import DBSCAN

# HDBSCAN optioneel
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


# =========================
# Config dataclasses
# =========================

@dataclass
class RequiredPoint:
    """Punt dat NIET als noise (-1) mag eindigen."""
    # optie A: uniek identificeren via kolom
    column: Optional[str] = None
    value: Optional[Any] = None

    # optie B: coordinaten (in projectie-CRS, meestal EPSG:28992)
    x: Optional[float] = None
    y: Optional[float] = None
    tol_m: float = 1.0


Agg = Literal["sum", "mean", "max", "min", "count_nonnull"]
Direction = Literal["max", "min"]


@dataclass
class OptimizationRule:
    """
    Definieert hoe clusters gescoord worden o.b.v. (subset van) punten.
    Voorbeeld: maximaliseer som van "Max ruimte verbruik" voor source="Bouwwerk".
    """
    source: Optional[str] = None
    column: Optional[str] = None
    agg: Agg = "sum"
    direction: Direction = "max"
    weight: float = 1.0


@dataclass
class ClusterConfig:
    # Input
    input_csv: str = "combined.csv"
    geometry_col: str = "geometry"
    source_col: str = "source"
    # Proxy voor toekomstige locaties (Bovenregionaal/Locatiespecifiek)
    future_consumer_sources: Tuple[str, ...] = ("Bovenregionaal", "Locatiespecifiek")
    global_mean_bouwwerk_verbruik_kw: Optional[float] = None
    assumed_input_crs_epsg: int = 28992  # RD New

    # Clustering
    algorithm: str = "dbscan"            # "dbscan" of "hdbscan"
    eps_m: float = 5000.0
    min_samples: int = 3
    min_cluster_size: int = 3            # harde filter achteraf

    # Constraints
    min_per_source: Dict[str, int] = field(default_factory=dict)
    exclude_sources: List[str] = field(default_factory=list)
    future_consumer_sources: Tuple[str, ...] = ("Bovenregionaal", "Locatiespecifiek")
    global_mean_bouwwerk_demand_kw: Optional[float] = None

    required_points: List[RequiredPoint] = field(default_factory=list)
    attach_required_max_dist_m: float = 2000.0

    # Output selectie: exact N clusters
    target_n_clusters: Optional[int] = 10

    # Score/optimalisatie regels (gebruikt bij top-N selectie)
    # Als leeg -> default score = n_points + compactness bonus
    optimization_rules: List[OptimizationRule] = field(default_factory=list)

    # Exports
    out_dir: str = "outputs"
    out_points_csv: str = "points_with_clusters.csv"
    out_clusters_csv: str = "cluster_summary.csv"
    out_gpkg: str = "clusters.gpkg"

    # Visualisatie (alleen voor CLI run)
    make_plot: bool = True


# =========================
# Load + preprocess
# =========================

def load_combined_csv(cfg: ClusterConfig) -> gpd.GeoDataFrame:
    if not os.path.exists(cfg.input_csv):
        raise FileNotFoundError(f"Input CSV bestaat niet: {cfg.input_csv}")

    df = pd.read_csv(cfg.input_csv)

    if cfg.geometry_col not in df.columns:
        raise ValueError(f"Kolom '{cfg.geometry_col}' ontbreekt in CSV.")

    # Drop typische indexkolom
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df[cfg.geometry_col] = df[cfg.geometry_col].astype(str).apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=cfg.geometry_col, crs=f"EPSG:{cfg.assumed_input_crs_epsg}")
    return gdf


def ensure_single_point_per_feature(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Forceer: elke feature (Point/Line/Polygon/Multi*) wordt exact één Point.
    - Polygon/MultiPolygon -> representative_point() (ligt binnen geometrie)
    - LineString/MultiLineString -> interpolate(0.5, normalized=True) (midden op lijn)
    - Point -> blijft Point
    - Anders -> centroid fallback
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    def to_one_point(geom):
        gt = geom.geom_type
        if gt == "Point":
            return geom
        if gt in ("Polygon", "MultiPolygon"):
            return geom.representative_point()
        if gt in ("LineString", "MultiLineString"):
            try:
                return geom.interpolate(0.5, normalized=True)
            except Exception:
                return geom.centroid
        return geom.centroid

    gdf["geometry"] = gdf.geometry.apply(to_one_point)
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    return gdf


def to_projected_for_meters(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Any]:
    """Projecteer naar UTM als input lat/lon zou zijn; RD blijft RD."""
    gdf = gdf.copy()
    if gdf.crs is None:
        raise ValueError("CRS ontbreekt. Zet CRS op input of gebruik assumed_input_crs_epsg.")

    original_crs = gdf.crs
    if original_crs.is_geographic:
        utm = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm)
    return gdf, original_crs


# =========================
# Clustering
# =========================

def cluster_points(gdf_proj: gpd.GeoDataFrame, cfg: ClusterConfig) -> gpd.GeoDataFrame:
    gdf_proj = gdf_proj.copy()
    coords = np.vstack([gdf_proj.geometry.x.values, gdf_proj.geometry.y.values]).T

    algo = cfg.algorithm.lower().strip()
    if algo == "dbscan":
        model = DBSCAN(eps=cfg.eps_m, min_samples=cfg.min_samples)
        labels = model.fit_predict(coords)
    elif algo == "hdbscan":
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan is niet geïnstalleerd. Doe: pip install hdbscan")
        model = hdbscan.HDBSCAN(
            min_cluster_size=max(cfg.min_cluster_size, 2),
            min_samples=cfg.min_samples,
        )
        labels = model.fit_predict(coords)
    else:
        raise ValueError("algorithm moet 'dbscan' of 'hdbscan' zijn.")

    gdf_proj["cluster_id"] = labels
    return gdf_proj


# =========================
# Required points constraint
# =========================

def _find_required_index(gdf: gpd.GeoDataFrame, rp: RequiredPoint) -> Optional[int]:
    if rp.column and rp.value is not None:
        if rp.column not in gdf.columns:
            raise ValueError(f"RequiredPoint column '{rp.column}' bestaat niet.")
        matches = gdf.index[gdf[rp.column] == rp.value].tolist()
        return int(matches[0]) if matches else None

    if rp.x is not None and rp.y is not None:
        p = Point(rp.x, rp.y)
        dists = gdf.geometry.distance(p)
        best_idx = int(dists.idxmin())
        if float(dists.loc[best_idx]) <= float(rp.tol_m):
            return best_idx
        return None

    raise ValueError("RequiredPoint moet óf (column+value) óf (x+y) hebben.")


def enforce_required_points(gdf_proj: gpd.GeoDataFrame, cfg: ClusterConfig) -> gpd.GeoDataFrame:
    """
    Zorgt dat required points niet als noise (-1) eindigen.
    - als noise: attach naar dichtstbijzijnde cluster binnen attach_required_max_dist_m
      anders: maak eigen cluster-id
    """
    gdf = gdf_proj.copy()
    if not cfg.required_points:
        return gdf

    existing = sorted([c for c in gdf["cluster_id"].unique().tolist() if c != -1])
    next_cluster_id = (max(existing) + 1) if existing else 0

    for rp in cfg.required_points:
        idx = _find_required_index(gdf, rp)
        if idx is None:
            print(f"[WARN] RequiredPoint niet gevonden: {rp}")
            continue

        if int(gdf.loc[idx, "cluster_id"]) != -1:
            continue

        required_geom = gdf.loc[idx, "geometry"]
        non_noise = gdf[gdf["cluster_id"] != -1].copy()

        if non_noise.empty:
            gdf.loc[idx, "cluster_id"] = next_cluster_id
            next_cluster_id += 1
            continue

        dists = non_noise.geometry.distance(required_geom)
        best_idx = int(dists.idxmin())
        best_dist = float(dists.loc[best_idx])
        best_cluster = int(non_noise.loc[best_idx, "cluster_id"])

        if best_dist <= float(cfg.attach_required_max_dist_m):
            gdf.loc[idx, "cluster_id"] = best_cluster
        else:
            gdf.loc[idx, "cluster_id"] = next_cluster_id
            next_cluster_id += 1

    return gdf


# =========================
# Scoring + cluster summary
# =========================

def filter_clusters(cluster_summary: gpd.GeoDataFrame, cfg: ClusterConfig) -> gpd.GeoDataFrame:
    out = cluster_summary.copy()
    out = out[out["n_points"] >= int(cfg.min_cluster_size)].copy()

    for src, min_n in cfg.min_per_source.items():
        col = f"n_{src}"
        if col not in out.columns:
            return out.iloc[0:0].copy()
        out = out[out[col] >= int(min_n)].copy()

    return out


def _apply_agg(series: pd.Series, agg: Agg) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if agg == "sum":
        return float(s.fillna(0).sum())
    if agg == "mean":
        return float(s.dropna().mean()) if s.dropna().size else 0.0
    if agg == "max":
        return float(s.dropna().max()) if s.dropna().size else 0.0
    if agg == "min":
        return float(s.dropna().min()) if s.dropna().size else 0.0
    if agg == "count_nonnull":
        return float(s.notna().sum())
    raise ValueError(f"Onbekende agg: {agg}")


def compute_cluster_compactness(grp: gpd.GeoDataFrame) -> float:
    """Gemiddelde afstand (m) van punten tot centroid. Lager = compacter."""
    if grp.empty:
        return float("inf")
    centroid = grp.geometry.union_all().centroid
    dists = grp.geometry.distance(centroid)
    return float(dists.mean())


def score_clusters(points_proj: gpd.GeoDataFrame, cluster_ids: List[int], cfg: ClusterConfig) -> pd.DataFrame:
    """
    Score per cluster.
    - Als optimization_rules leeg: score = n_points + compactness bonus
    - Anders: som van (rule.weight * signed(metric))
    """
    rows = []
    for cid in cluster_ids:
        grp = points_proj[points_proj["cluster_id"] == cid].copy()
        if grp.empty:
            continue

        if not cfg.optimization_rules:
            n_points = float(len(grp))
            compactness = compute_cluster_compactness(grp)
            compactness_score = 1.0 / (compactness + 1e-6)

            alpha = 1.0
            beta = 1000.0

            score = alpha * n_points + beta * compactness_score

            rows.append({
                "cluster_id": int(cid),
                "score": float(score),
                "n_points": float(n_points),
                "compactness_m": float(compactness),
            })
            continue

        total_score = 0.0
        for rule in cfg.optimization_rules:
            sub = grp
            if rule.source is not None:
                sub = sub[sub[cfg.source_col] == rule.source] if cfg.source_col in sub.columns else sub.iloc[0:0]
            if sub.empty:
                val = 0.0
            else:
                if rule.column is None:
                    val = 0.0
                else:
                    val = _apply_agg(sub[rule.column], rule.agg)

            signed = val if rule.direction == "max" else -val
            total_score += float(rule.weight) * signed

        rows.append({"cluster_id": int(cid), "score": float(total_score)})

    return pd.DataFrame(rows)

def _pairwise_sum_mean_m(grp: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Retourneert:
      - sum_pairwise_dist_m: som van afstanden over alle unieke puntparen (i<j)
      - mean_pairwise_dist_m: gemiddelde afstand over alle unieke puntparen (i<j)

    Let op: O(n^2). Voor grote clusters wordt gesampled om het beheersbaar te houden.
    """
    n = len(grp)
    if n < 2:
        return 0.0, 0.0

    coords = np.column_stack([grp.geometry.x.values, grp.geometry.y.values])

    # sampling om O(n^2) te beperken bij hele grote clusters
    MAX_N = 1500
    if n > MAX_N:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=MAX_N, replace=False)
        coords = coords[idx]
        n = MAX_N

    chunk = 500
    sum_dist = 0.0
    count = 0

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        A = coords[i0:i1]  # (a,2)

        diff = A[:, None, :] - coords[None, :, :]   # (a,n,2)
        dist = np.sqrt((diff ** 2).sum(axis=2))     # (a,n)

        # alleen bovenste driehoek i<j
        for local_i, global_i in enumerate(range(i0, i1)):
            if global_i + 1 < n:
                upper = dist[local_i, global_i + 1:]
                if upper.size:
                    sum_dist += float(upper.sum())
                    count += int(upper.size)

    mean_dist = (sum_dist / count) if count > 0 else 0.0
    return float(sum_dist), float(mean_dist)

def _mean_distance_from_energy_to_others_m(grp: gpd.GeoDataFrame, source_col: str) -> Optional[float]:
    """
    Als er Energieproject-punten aanwezig zijn:
      gemiddelde afstand van Energieproject-punten naar alle NIET-Energieproject punten.

    Retourneert None als er geen Energieproject punten zijn, of geen 'andere' punten.
    """
    if source_col not in grp.columns:
        return None

    energy = grp[grp[source_col] == "Energieproject"]
    others = grp[grp[source_col] != "Energieproject"]

    if energy.empty or others.empty:
        return None

    E = np.column_stack([energy.geometry.x.values, energy.geometry.y.values])
    O = np.column_stack([others.geometry.x.values, others.geometry.y.values])

    # chunken voor geheugen (E x O kan groot worden)
    chunk = 500
    sum_dist = 0.0
    count = 0

    for i0 in range(0, len(E), chunk):
        i1 = min(i0 + chunk, len(E))
        A = E[i0:i1]  # (a,2)

        diff = A[:, None, :] - O[None, :, :]   # (a, m, 2)
        dist = np.sqrt((diff ** 2).sum(axis=2))  # (a, m)

        sum_dist += float(dist.sum())
        count += int(dist.size)

    return float(sum_dist / count) if count > 0 else None



def _free_electric_space_kw(
    grp: gpd.GeoDataFrame,
    source_col: str,
    pv_capacity_factor: float = 0.12,
    future_consumer_sources: Tuple[str, ...] = ("Bovenregionaal", "Locatiespecifiek"),
    global_mean_bouwwerk_demand_kw: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Vrije elektriciteitsruimte in kW.

    - Energieproject 'opwek' is MWp => effectieve opwek (kW) = MWp * 1000 * CF
    - Bouwwerk 'Max ruimte verbruik' is kW (kan negatief zijn)
    - Bovenregionaal/Locatiespecifiek zijn toekomstige locaties met extra vraag:
        toekomstige_vraag_kw = n_future * global_mean_bouwwerk_demand_kw
      Deze vraag gaat VAN de vrije ruimte af.

    Returns:
      sum_effective_opwek_kw,
      sum_bouwwerk_kw,
      mean_bouwwerk_demand_kw_used,
      future_demand_kw,
      vrije_ruimte_kw
    """
    if source_col not in grp.columns:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # --- opwek (MWp -> kW) ---
    sum_effective_opwek_kw = 0.0
    ep = grp[grp[source_col] == "Energieproject"]
    if not ep.empty and "opwek" in ep.columns:
        opwek_mwp = pd.to_numeric(ep["opwek"], errors="coerce").fillna(0.0)
        sum_effective_opwek_kw = float((opwek_mwp * 1000.0 * pv_capacity_factor).sum())

    # --- bouwwerk som (ruimte/verbruik, kW) ---
    sum_bouwwerk_kw = 0.0
    bw = grp[grp[source_col] == "Bouwwerk"]
    if not bw.empty and "Max ruimte verbruik" in bw.columns:
        bw_vals = pd.to_numeric(bw["Max ruimte verbruik"], errors="coerce").fillna(0.0)
        sum_bouwwerk_kw = float(bw_vals.sum())

    # --- toekomstige vraag (gaat eraf) ---
    mean_demand_kw = float(global_mean_bouwwerk_demand_kw) if global_mean_bouwwerk_demand_kw is not None else 0.0
    n_future = int(grp[grp[source_col].isin(future_consumer_sources)].shape[0])
    future_demand_kw = float(n_future * mean_demand_kw)

    vrije_ruimte_kw = sum_effective_opwek_kw + sum_bouwwerk_kw - future_demand_kw

    return (
        float(sum_effective_opwek_kw),
        float(sum_bouwwerk_kw),
        float(mean_demand_kw),
        float(future_demand_kw),
        float(vrije_ruimte_kw),
    )





def summarize_clusters(points_proj: gpd.GeoDataFrame, cfg: ClusterConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Maakt cluster summary (centroids in WGS84) en filtert + selecteert top-N.
    Retourneert:
      - points_proj met cluster_id (-1 als noise of weggefilterd)
      - clusters_wgs84 met centroid_lat/lon, n_points, n_<source>, score
    """
    points = points_proj.copy()
    clustered = points[points["cluster_id"] != -1].copy()

    if clustered.empty:
        return points, gpd.GeoDataFrame(columns=["cluster_id"], geometry=[], crs="EPSG:4326")

    rows = []
    for cid, grp in clustered.groupby("cluster_id"):
        hull = grp.geometry.union_all().convex_hull
        centroid_proj = hull.centroid
        minx, miny, maxx, maxy = grp.total_bounds



        # (2) Vrije elektriciteitsruimte (kW)
        sum_opwek_kw, sum_bouwwerk_kw, mean_demand_used_kw, future_demand_kw, vrije_ruimte_kw = _free_electric_space_kw(
            grp,
            cfg.source_col,
            future_consumer_sources=cfg.future_consumer_sources,
            global_mean_bouwwerk_demand_kw=cfg.global_mean_bouwwerk_demand_kw,
        )

        src_counts = grp[cfg.source_col].value_counts(dropna=False).to_dict() if cfg.source_col in grp.columns else {}

        sum_pairwise_m, mean_pairwise_m = _pairwise_sum_mean_m(grp)
        mean_opwek_to_others_m = _mean_distance_from_energy_to_others_m(grp, cfg.source_col)

        rows.append(
            {
                "cluster_id": int(cid),
                "n_points": int(len(grp)),
                "sum_pairwise_dist_m": float(sum_pairwise_m),

                "mean_opwek_to_others_m": float(
                    mean_opwek_to_others_m) if mean_opwek_to_others_m is not None else np.nan,

                # bounding box in projected CRS
                "bbox_minx": float(minx),
                "bbox_miny": float(miny),
                "bbox_maxx": float(maxx),
                "bbox_maxy": float(maxy),

                # centroid (projected)
                "geometry": centroid_proj,

                # nieuwe afstandsattributen
                "mean_pairwise_dist_m": float(mean_pairwise_m),


                # nieuwe energie-attributen
                "sum_opwek_kw": float(sum_opwek_kw),
"sum_bouwwerk_kw": float(sum_bouwwerk_kw),
"mean_bouwwerk_demand_kw_used": float(mean_demand_used_kw),
"future_demand_kw": float(future_demand_kw),
"vrije_ruimte_kw": float(vrije_ruimte_kw),



                "src_counts": src_counts,
            }
        )

    cluster_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=points.crs)

    # expand source counts -> n_<source>
    all_sources = set()
    for d in cluster_gdf["src_counts"].tolist():
        all_sources.update(d.keys())
    for s in sorted(all_sources):
        cluster_gdf[f"n_{s}"] = cluster_gdf["src_counts"].apply(lambda d: int(d.get(s, 0)))
    cluster_gdf = cluster_gdf.drop(columns=["src_counts"])

    # centroid lat/lon in WGS84
    cluster_wgs84 = cluster_gdf.to_crs(epsg=4326)
    cluster_wgs84["centroid_lon"] = cluster_wgs84.geometry.x
    cluster_wgs84["centroid_lat"] = cluster_wgs84.geometry.y

    # filters
    cluster_wgs84 = filter_clusters(cluster_wgs84, cfg)

    # score + top-N selectie
    if not cluster_wgs84.empty:
        candidate_ids = cluster_wgs84["cluster_id"].astype(int).tolist()
        scores = score_clusters(points, candidate_ids, cfg)

        # ✅ voorkomen dubbele kolommen bij merge (n_points bestaat al in cluster_wgs84)
        if "n_points" in scores.columns:
            scores = scores.drop(columns=["n_points"])

        cluster_wgs84 = cluster_wgs84.merge(scores, on="cluster_id", how="left")

        if cfg.target_n_clusters is not None:
            keep_n = int(cfg.target_n_clusters)
            cluster_wgs84 = cluster_wgs84.sort_values("score", ascending=False).head(keep_n).copy()

    # points buiten de geselecteerde clusters -> noise
    keep_ids = set(cluster_wgs84["cluster_id"].astype(int).tolist())
    points.loc[~points["cluster_id"].isin(keep_ids), "cluster_id"] = -1

    return points, cluster_wgs84


def add_normalized_scores(clusters_summary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Voeg score_norm (0-1) en rank toe."""
    out = clusters_summary.copy()
    if out.empty:
        return out

    base_col = "score" if "score" in out.columns else "n_points"
    s = pd.to_numeric(out[base_col], errors="coerce").fillna(0.0)

    s_min, s_max = float(s.min()), float(s.max())
    if s_max == s_min:
        out["score_norm"] = 1.0
    else:
        out["score_norm"] = (s - s_min) / (s_max - s_min)

    out["rank"] = out[base_col].rank(method="dense", ascending=False).astype(int)
    return out


# =========================
# Export
# =========================

def export_results(points_proj: gpd.GeoDataFrame, clusters_wgs84: gpd.GeoDataFrame, cfg: ClusterConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    points_wgs84 = points_proj.to_crs(epsg=4326).copy()
    points_wgs84["lon"] = points_wgs84.geometry.x
    points_wgs84["lat"] = points_wgs84.geometry.y

    points_csv = os.path.join(cfg.out_dir, cfg.out_points_csv)
    clusters_csv = os.path.join(cfg.out_dir, cfg.out_clusters_csv)
    gpkg = os.path.join(cfg.out_dir, cfg.out_gpkg)

    # CSV: geometry als WKT
    p_out = points_wgs84.copy()
    p_out["geometry_wkt"] = p_out.geometry.apply(lambda g: g.wkt if g is not None else None)
    p_out.drop(columns=["geometry"], inplace=True, errors="ignore")
    p_out.to_csv(points_csv, index=False)

    c_out = clusters_wgs84.copy()
    c_out["geometry_wkt"] = c_out.geometry.apply(lambda g: g.wkt if g is not None else None)
    c_out.drop(columns=["geometry"], inplace=True, errors="ignore")
    c_out.to_csv(clusters_csv, index=False)

    # GeoPackage
    points_proj.to_file(gpkg, layer="points", driver="GPKG")
    clusters_wgs84.to_file(gpkg, layer="cluster_centroids_wgs84", driver="GPKG")

    print(f"[OK] {points_csv}")
    print(f"[OK] {clusters_csv}")
    print(f"[OK] {gpkg}")


# =========================
# Plotting (Streamlit-friendly: return fig)
# =========================

def plot_clusters(points_proj: gpd.GeoDataFrame, cfg: ClusterConfig):
    g = points_proj.copy()
    clustered = g[g["cluster_id"] != -1]
    noise = g[g["cluster_id"] == -1]

    fig, ax = plt.subplots(figsize=(10, 10))
    if not noise.empty:
        noise.plot(ax=ax, markersize=5, alpha=0.2, label="noise")
    if not clustered.empty:
        clustered.plot(ax=ax, column="cluster_id", markersize=10, legend=True)

    ax.set_title(f"Clusters ({cfg.algorithm}) eps={cfg.eps_m}m min_samples={cfg.min_samples}")
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def plot_top_n_clusters(points_proj: gpd.GeoDataFrame, clusters_summary: gpd.GeoDataFrame, n: int = 5, title: Optional[str] = None):
    if clusters_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Geen clusters om te plotten.")
        ax.set_axis_off()
        return fig

    sort_col = "score" if "score" in clusters_summary.columns else "n_points"
    top_clusters = clusters_summary.sort_values(sort_col, ascending=False).head(n)
    top_ids = set(top_clusters["cluster_id"].astype(int))

    plot_points = points_proj[points_proj["cluster_id"].isin(top_ids)].copy()

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_points.plot(ax=ax, column="cluster_id", markersize=20, legend=True)

    ax.set_title(title or f"Top {n} clusters (op basis van {sort_col})")
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# =========================
# Main pipeline
# =========================

def run_return(cfg: ClusterConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Voor Streamlit: return points + clusters (geen files, geen plt.show)."""
    gdf = load_combined_csv(cfg)
    gdf = ensure_single_point_per_feature(gdf)

    # --- Globale gemiddelde bouwwerk-vraag (kW) voor toekomstige locaties ---
    # We gebruiken alleen de vraagcomponent: demand = max(0, -Max ruimte verbruik)
    if cfg.source_col in gdf.columns and "Max ruimte verbruik" in gdf.columns:
        bw_all = gdf[gdf[cfg.source_col] == "Bouwwerk"]
        if not bw_all.empty:
            vals = pd.to_numeric(bw_all["Max ruimte verbruik"], errors="coerce").dropna()
            if not vals.empty:
                demand_vals = (-vals).clip(lower=0)  # alleen vraag in kW
                cfg.global_mean_bouwwerk_demand_kw = float(demand_vals.mean())
            else:
                cfg.global_mean_bouwwerk_demand_kw = 0.0
        else:
            cfg.global_mean_bouwwerk_demand_kw = 0.0
    else:
        cfg.global_mean_bouwwerk_demand_kw = 0.0

    # --- Globaal gemiddelde bouwwerk verbruik (voor toekomstige locaties) ---
    if cfg.source_col in gdf.columns and "Max ruimte verbruik" in gdf.columns:
        bw_all = gdf[gdf[cfg.source_col] == "Bouwwerk"]
        if not bw_all.empty:
            vals = pd.to_numeric(bw_all["Max ruimte verbruik"], errors="coerce").dropna()
            if not vals.empty:
                cfg.global_mean_bouwwerk_verbruik_kw = float(vals.mean())
            else:
                cfg.global_mean_bouwwerk_verbruik_kw = 0.0
        else:
            cfg.global_mean_bouwwerk_verbruik_kw = 0.0
    else:
        cfg.global_mean_bouwwerk_verbruik_kw = 0.0

    if cfg.exclude_sources and cfg.source_col in gdf.columns:
        gdf = gdf[~gdf[cfg.source_col].isin(cfg.exclude_sources)].copy()

    gdf_proj, _ = to_projected_for_meters(gdf)
    gdf_proj = cluster_points(gdf_proj, cfg)
    gdf_proj = enforce_required_points(gdf_proj, cfg)

    points_proj, clusters_wgs84 = summarize_clusters(gdf_proj, cfg)
    clusters_wgs84 = add_normalized_scores(clusters_wgs84)
    return points_proj, clusters_wgs84


def run(cfg: ClusterConfig) -> None:
    """CLI run: doet exports en kan plots maken."""
    points_proj, clusters_wgs84 = run_return(cfg)
    export_results(points_proj, clusters_wgs84, cfg)

    if cfg.make_plot:
        fig1 = plot_clusters(points_proj, cfg)
        plt.show()

        fig2 = plot_top_n_clusters(points_proj, clusters_wgs84, n=5)
        plt.show()


if __name__ == "__main__":
    cfg = ClusterConfig(
        input_csv="combined.csv",
        algorithm="dbscan",
        eps_m=5000,
        min_samples=3,
        min_cluster_size=3,
        min_per_source={
            "Locatiespecifiek": 1,
            "Energieproject": 1
        },
        exclude_sources=["Bouwwerk", "Bovenregionaal"],
        target_n_clusters=10,
        optimization_rules=[],
        out_dir="outputs",
        make_plot=True,
    )

    run(cfg)
