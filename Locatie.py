import glob
from pathlib import Path
from sklearn.cluster import DBSCAN
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# === Helperfunctie ===
def merge_shapefiles(folder: str, recursive: bool = False) -> gpd.GeoDataFrame:
    """
    Lees alle shapefiles in (en optioneel onder) de map `folder` en merge ze tot één GeoDataFrame.
    """
    if recursive:
        pattern = f"{folder}/**/*.shp"
    else:
        pattern = f"{folder}/*.shp"

    shapefile_paths = glob.glob(pattern, recursive=recursive)

    if not shapefile_paths:
        raise ValueError(f"Geen shapefiles gevonden in: {folder}")

    geodataframes = [gpd.read_file(path) for path in shapefile_paths]

    merged_gdf = gpd.GeoDataFrame(
        pd.concat(geodataframes, ignore_index=True),
        crs=geodataframes[0].crs,
    )
    return merged_gdf


# === Paden naar data ===
BASE_DATA_DIR = Path("Data")
VKA_BASE_DIR = BASE_DATA_DIR / "20250827_export_VKAs"

BOVENREGIONAAL_VKA_DIR = VKA_BASE_DIR / "Bovenregionaal VKA"
LOCATIESPECIFIEK_VKA_DIR = VKA_BASE_DIR / "Locatiespecifiek VKA"

BOUWWERKEN_SHP = BASE_DATA_DIR / "Bouwwerken_netcongestie" / "Bouwwerken_netcongestie.shp"
AANSLUITINGEN_XLSX = BASE_DATA_DIR / "TUD Basislijst Bekende aansluitingen (sept 25).xlsx"


# === 1. VKA-shapefiles mergen ===
gdf_bovenregionaal = merge_shapefiles(str(BOVENREGIONAAL_VKA_DIR), recursive=False)
gdf_locatiespecifiek = merge_shapefiles(str(LOCATIESPECIFIEK_VKA_DIR), recursive=True)

# === 2. Bouwwerken + Excel combineren ===
gdf_bouwwerken = gpd.read_file(BOUWWERKEN_SHP)
df_aansluitingen = pd.read_excel(AANSLUITINGEN_XLSX, sheet_name="Gefilterde data")

# Merge op EAN
gdf_bouwwerken_merged = gdf_bouwwerken.merge(
    df_aansluitingen,
    on="EAN",
    how="left"
)

# === 3. Alles samenvoegen in één GeoDataFrame ===
combined_gdf = gpd.GeoDataFrame(
    pd.concat(
        [
            gdf_bovenregionaal.assign(source="Bovenregionaal"),
            gdf_locatiespecifiek.assign(source="Locatiespecifiek"),
            gdf_bouwwerken_merged.assign(source="Bouwwerk"),
        ],
        ignore_index=True
    ),
    crs=gdf_bovenregionaal.crs  # ga ervan uit dat alles dezelfde CRS heeft
)

combined_gdf["geometry_orig"] = combined_gdf.geometry
print(combined_gdf.columns)

METRIC_COL = "Max ruimte opwek"        # bv. "Max ruimte verbruik" of "Max ruimte opwek"

FILTER_MODE = "positive"


SORT_DESCENDING = True


EPS_METERS = 5000
MIN_SAMPLES = 2
BUFFER_RADIUS = 5000



locspec = combined_gdf[combined_gdf["source"] == "Locatiespecifiek"].copy()
bouww = combined_gdf[combined_gdf["source"] == "Bouwwerk"].copy()



locspec["geometry"] = locspec.geometry.centroid
bouww["geometry"] = bouww.geometry.centroid


mask = bouww[METRIC_COL].notna()

if FILTER_MODE == "positive":
    mask &= bouww[METRIC_COL] > 0
elif FILTER_MODE == "negative":
    mask &= bouww[METRIC_COL] < 0
elif FILTER_MODE == "all":
    pass


bouww = bouww[mask].copy()

print(f"Aantal Bouwwerken na filter ({FILTER_MODE}) op '{METRIC_COL}':", len(bouww))


locspec["buffer_5km"] = locspec.geometry.buffer(BUFFER_RADIUS)


sindex = bouww.sindex
bouww_candidates = []

for _, loc in locspec.iterrows():
    idx_list = list(sindex.intersection(loc["buffer_5km"].bounds))
    possible = bouww.iloc[idx_list]
    matches = possible[possible.intersects(loc["buffer_5km"])]
    if not matches.empty:
        bouww_candidates.append(matches)

if bouww_candidates:
    bouww_near = pd.concat(bouww_candidates, ignore_index=True).drop_duplicates()
else:
    bouww_near = bouww.iloc[0:0].copy()  # lege df met dezelfde structuur


# Combineer LocSpec + Bouwwerk binnen buffer
cluster_points = pd.concat([locspec, bouww_near], ignore_index=True)
cluster_points = gpd.GeoDataFrame(cluster_points, crs=combined_gdf.crs)

# === Voor clustering: maak numpy array van XY-coördinaten ===
coords = np.vstack([cluster_points.geometry.x, cluster_points.geometry.y]).T

# === DBSCAN Clustering ===
model = DBSCAN(eps=EPS_METERS, min_samples=MIN_SAMPLES)
cluster_points["cluster_id"] = model.fit_predict(coords)


clustered = cluster_points[cluster_points["cluster_id"] != -1].copy()


cluster_stats = []
export_rows = []


for cid, group in clustered.groupby("cluster_id"):
    n_loc = (group["source"] == "Locatiespecifiek").sum()
    n_bw = (group["source"] == "Bouwwerk").sum()

    # Alleen clusters die voldoen aan jouw voorwaarden
    if n_loc >= 1 and n_bw >= 2:
        metric_sum = group[group["source"] == "Bouwwerk"][METRIC_COL].sum()

        # Stats voor cluster_result_gdf
        cluster_stats.append({
            "cluster_id": cid,
            "n_locatiespecifiek": n_loc,
            "n_bouwwerken": n_bw,
            f"totaal_{METRIC_COL}": metric_sum,
            "geometry": group.unary_union.centroid
        })

        # Print details
        print("\n----------------------------------------------------")
        print(f"CLUSTER {cid}")
        print(f"Locatiespecifiek: {n_loc}, Bouwwerken: {n_bw}")
        print(f"Totaal {METRIC_COL}: {metric_sum}")
        print("----------------------------------------------------")
        print(
            group[
                [col for col in ["source", "EAN", METRIC_COL]
                 if col in group.columns]
            ]
        )

        # Voor CSV-export
        group_export = group.copy()
        group_export["cluster_id"] = cid
        group_export["geometry_wkt"] = group_export.geometry.apply(lambda g: g.wkt)
        export_rows.append(group_export)

# === Maak GeoDataFrame met clustercentroids ===
cluster_results_gdf = gpd.GeoDataFrame(cluster_stats, crs=combined_gdf.crs)

print("\nClusters gevonden:", len(cluster_results_gdf))
if not cluster_results_gdf.empty:
    sort_col = f"totaal_{METRIC_COL}"
    print(
        cluster_results_gdf[
            ["cluster_id", "n_locatiespecifiek", "n_bouwwerken", sort_col]
        ].sort_values(sort_col, ascending=not SORT_DESCENDING)
    )

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 12))

# Achtergrond
combined_gdf.plot(ax=ax, color="lightgrey", alpha=0.3)

# Bouwwerken in de buurt
if not bouww_near.empty:
    bouww_near.plot(ax=ax, color="blue", markersize=20, label=f"Bouwwerken binnen buffer ({METRIC_COL})")

# LocSpec
locspec.set_geometry("geometry").plot(ax=ax, color="red", markersize=50, label="Locatiespecifiek VKA")


if not cluster_results_gdf.empty:
    cluster_results_gdf.plot(ax=ax, color="yellow", markersize=200, label="Cluster centroid")

plt.legend()
plt.title(f"Clusters rond Locatiespecifieke VKA's op basis van '{METRIC_COL}'")
plt.show()

# === CSV-export ===
if export_rows:
    export_df = pd.concat(export_rows, ignore_index=True)

    # WKT van centroid-geometry
    export_df["geometry_centroid_wkt"] = export_df.geometry.apply(
        lambda g: g.wkt if g is not None else None
    )

    # WKT van originele polygon/multipolygon (als beschikbaar)
    if "geometry_orig" in export_df.columns:
        export_df["geometry_polygon_wkt"] = export_df["geometry_orig"].apply(
            lambda g: g.wkt if g is not None else None
        )

    # Alle kolommen meenemen, behalve de shapely-objects zelf
    cols_for_csv = [
        c for c in export_df.columns
        if c not in ["geometry", "geometry_orig"]  # shapely object niet direct in CSV
    ]

    csv_name = f"gevonden_clusters_{METRIC_COL.replace(' ', '_')}_{FILTER_MODE}.csv"
    export_df[cols_for_csv].to_csv(csv_name, index=False, encoding="utf-8")
    print(f"CSV opgeslagen als: {csv_name}")
else:
    print("Geen geldige clusters gevonden → CSV wordt niet gemaakt.")

