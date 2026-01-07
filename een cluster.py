from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Literal, Tuple

import numpy as np
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
    source: Optional[str] = None         # None = alle sources
    column: Optional[str] = None         # None kan alleen bij count_nonnull (maar meestal kolom invullen)
    agg: Agg = "sum"
    direction: Direction = "max"
    weight: float = 1.0                  # meerdere regels combineren
    # Als je alleen Bouwwerk-punten wilt gebruiken: source="Bouwwerk"


@dataclass
class ClusterConfig:
    # Input
    input_csv: str = "/mnt/data/combined.csv"  # pas aan naar jouw pad in PyCharm
    geometry_col: str = "geometry"
    source_col: str = "source"
    assumed_input_crs_epsg: int = 28992  # jouw combined lijkt RD (x/y in meters)

    # Clustering
    algorithm: str = "dbscan"            # "dbscan" of "hdbscan"
    eps_m: float = 5000.0
    min_samples: int = 3                 # DBSCAN core points / HDBSCAN min_samples
    min_cluster_size: int = 3            # harde filter achteraf

    # Constraints
    min_per_source: Dict[str, int] = field(default_factory=dict)
    required_points: List[RequiredPoint] = field(default_factory=list)
    attach_required_max_dist_m: float = 2000.0

    # Output selectie: exact N clusters
    target_n_clusters: Optional[int] = 10

    # Nieuw: score/optimalisatie regels (gebruikt bij top-N selectie)
    # Als leeg -> default: grootste clusters (n_points)
    optimization_rules: List[OptimizationRule] = field(default_factory=list)

    # Exports
    out_dir: str = "outputs"
    out_points_csv: str = "points_with_clusters.csv"
    out_clusters_csv: str = "cluster_summary.csv"
    out_gpkg: str = "clusters.gpkg"

    make_plot: bool = False


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


def ensure_point_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centroid voor lijn/polygon zodat DBSCAN op punten kan."""
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    def to_point(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return geom
        return geom.centroid

    gdf["geometry"] = gdf.geometry.apply(to_point)
    gdf = gdf[gdf.geometry.notna()].copy()
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
# Cluster summary + filtering + scoring
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


def score_clusters(points_proj: gpd.GeoDataFrame, cluster_ids: List[int], cfg: ClusterConfig) -> pd.DataFrame:
    """
    Berekent per cluster een score op basis van cfg.optimization_rules.
    Als geen regels: score = n_points (grootste clusters).
    """
    rows = []
    for cid in cluster_ids:
        grp = points_proj[points_proj["cluster_id"] == cid].copy()
        if grp.empty:
            continue

        if not cfg.optimization_rules:
            score = float(len(grp))
            rows.append({"cluster_id": cid, "score": score})
            continue

        total_score = 0.0
        for rule in cfg.optimization_rules:
            sub = grp
            if rule.source is not None:
                if cfg.source_col not in sub.columns:
                    val = 0.0
                else:
                    sub = sub[sub[cfg.source_col] == rule.source]
                    val = 0.0 if sub.empty else _apply_agg(sub[rule.column], rule.agg)  # type: ignore[index]
            else:
                val = _apply_agg(sub[rule.column], rule.agg)  # type: ignore[index]

            # direction
            signed = val if rule.direction == "max" else -val
            total_score += float(rule.weight) * signed

        rows.append({"cluster_id": cid, "score": float(total_score)})

    return pd.DataFrame(rows)


def summarize_clusters(points_proj: gpd.GeoDataFrame, cfg: ClusterConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    points = points_proj.copy()
    clustered = points[points["cluster_id"] != -1].copy()

    if clustered.empty:
        return points, gpd.GeoDataFrame(columns=["cluster_id"], geometry=[], crs="EPSG:4326")

    rows = []
    for cid, grp in clustered.groupby("cluster_id"):
        hull = grp.unary_union.convex_hull
        centroid_proj = hull.centroid
        minx, miny, maxx, maxy = grp.total_bounds

        src_counts = grp[cfg.source_col].value_counts(dropna=False).to_dict() if cfg.source_col in grp.columns else {}

        row = {
            "cluster_id": int(cid),
            "n_points": int(len(grp)),
            "bbox_minx": float(minx),
            "bbox_miny": float(miny),
            "bbox_maxx": float(maxx),
            "bbox_maxy": float(maxy),
            "geometry": centroid_proj,
            "src_counts": src_counts,
        }
        rows.append(row)

    cluster_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=points.crs)

    # expand src counts
    all_sources = set()
    for d in cluster_gdf["src_counts"].tolist():
        all_sources.update(d.keys())
    for s in sorted(all_sources):
        cluster_gdf[f"n_{s}"] = cluster_gdf["src_counts"].apply(lambda d: int(d.get(s, 0)))
    cluster_gdf = cluster_gdf.drop(columns=["src_counts"])

    # centroid lat/lon
    cluster_wgs84 = cluster_gdf.to_crs(epsg=4326)
    cluster_wgs84["centroid_lon"] = cluster_wgs84.geometry.x
    cluster_wgs84["centroid_lat"] = cluster_wgs84.geometry.y

    # filters
    cluster_wgs84 = filter_clusters(cluster_wgs84, cfg)

    # scoring + top-N
    if cfg.target_n_clusters is not None and not cluster_wgs84.empty:
        keep_n = int(cfg.target_n_clusters)
        if len(cluster_wgs84) > keep_n:
            candidate_ids = cluster_wgs84["cluster_id"].astype(int).tolist()
            scores = score_clusters(points, candidate_ids, cfg)
            cluster_wgs84 = cluster_wgs84.merge(scores, on="cluster_id", how="left")
            cluster_wgs84 = cluster_wgs84.sort_values("score", ascending=False).head(keep_n).copy()
        else:
            # voeg score toe (handig om te zien)
            candidate_ids = cluster_wgs84["cluster_id"].astype(int).tolist()
            scores = score_clusters(points, candidate_ids, cfg)
            cluster_wgs84 = cluster_wgs84.merge(scores, on="cluster_id", how="left")

    # update points: clusters die niet geselecteerd zijn -> noise
    keep_ids = set(cluster_wgs84["cluster_id"].astype(int).tolist())
    points.loc[~points["cluster_id"].isin(keep_ids), "cluster_id"] = -1

    return points, cluster_wgs84


# =========================
# Export + plot
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

    points_proj.to_file(gpkg, layer="points", driver="GPKG")
    clusters_wgs84.to_file(gpkg, layer="cluster_centroids_wgs84", driver="GPKG")

    print(f"[OK] {points_csv}")
    print(f"[OK] {clusters_csv}")
    print(f"[OK] {gpkg}")


def plot_clusters(points_proj: gpd.GeoDataFrame, cfg: ClusterConfig) -> None:
    import matplotlib.pyplot as plt

    g = points_proj.copy()
    clustered = g[g["cluster_id"] != -1]
    noise = g[g["cluster_id"] == -1]

    fig, ax = plt.subplots(figsize=(10, 10))
    if not noise.empty:
        noise.plot(ax=ax, markersize=5, alpha=0.3, label="noise")
    if not clustered.empty:
        clustered.plot(ax=ax, column="cluster_id", markersize=10, legend=True)

    ax.set_title(f"Clusters ({cfg.algorithm}) eps={cfg.eps_m}m min_samples={cfg.min_samples}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================

def run(cfg: ClusterConfig) -> None:
    gdf = load_combined_csv(cfg)
    gdf = ensure_point_geometries(gdf)
    gdf_proj, _ = to_projected_for_meters(gdf)

    gdf_proj = cluster_points(gdf_proj, cfg)
    gdf_proj = enforce_required_points(gdf_proj, cfg)

    points_proj, clusters_wgs84 = summarize_clusters(gdf_proj, cfg)

    print(f"\nClusters na filtering/top-N: {len(clusters_wgs84)}")
    if not clusters_wgs84.empty:
        cols = ["cluster_id", "n_points", "score", "centroid_lat", "centroid_lon"]
        src_cols = [c for c in clusters_wgs84.columns if c.startswith("n_")]
        cols = [c for c in cols if c in clusters_wgs84.columns] + src_cols
        print(clusters_wgs84[cols].sort_values("score", ascending=False).head(25))

    export_results(points_proj, clusters_wgs84, cfg)
    if cfg.make_plot:
        plot_clusters(points_proj, cfg)


if __name__ == "__main__":
    # =========================
    # HIER PAS JE ALLES AAN
    # =========================
    cfg = ClusterConfig(
        input_csv="combined.csv",  # <- in PyCharm: zet dit naar jouw projectpad, bv. "data/combined.csv"

        algorithm="dbscan",
        eps_m=5000,
        min_samples=3,
        min_cluster_size=3,

        # cluster moet minimaal zoveel punten van bepaalde sources bevatten
        min_per_source={
            # "Bouwwerk": 2,
            # "Locatiespecifiek": 1,
        },

        # Required points (optioneel)
        required_points=[
            # RequiredPoint(column="EAN", value="..."),
            # RequiredPoint(x=120000.0, y=480000.0, tol_m=5.0),
        ],
        attach_required_max_dist_m=5000,

        # Exact N clusters in output:
        target_n_clusters=10,

        # === NIEUW: optimalisatie / scoring regels ===
        # Voorbeeld: maximaliseer "Max ruimte verbruik" voor Bouwwerk-punten binnen een cluster.
        # Je kunt meerdere regels combineren (met weights).
        optimization_rules=[
            OptimizationRule(
                source="Bouwwerk",
                column="Max ruimte verbruik",
                agg="sum",          # sum | mean | max | min
                direction="max",
                weight=1.0,
            ),
            # Extra voorbeeld: ook graag veel Bouwwerken?
            # OptimizationRule(source="Bouwwerk", column="Max ruimte verbruik", agg="count_nonnull", direction="max", weight=0.1),
        ],

        out_dir="outputs",
        make_plot=True,
    )

    run(cfg)
