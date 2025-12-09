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








EPS_METERS = 2000   # max afstand in meters voor clustering
MIN_SAMPLES = 3     # minimaal aantal LocSpec-punten voor een cluster


locspec = combined_gdf[combined_gdf["source"] == "Locatiespecifiek"].copy()

# 2. Gebruik centroids als punten
locspec["geometry"] = locspec.geometry.centroid

# 3. DBSCAN clustering
coords = np.vstack([locspec.geometry.x, locspec.geometry.y]).T
model = DBSCAN(eps=EPS_METERS, min_samples=MIN_SAMPLES)
locspec["cluster_id"] = model.fit_predict(coords)

# 4. Filter echte clusters (geen ruis)
locspec_clusters = locspec[locspec["cluster_id"] != -1].copy()

# === CLUSTERS PRINTEN ===
print("\n=== GEVONDEN LOCATIESPECIFIEKE CLUSTERS ===")

if locspec_clusters.empty:
    print("Geen clusters gevonden.")
else:
    for cid, group in locspec_clusters.groupby("cluster_id"):
        print("\n---------------------------------------")
        print(f"CLUSTER {cid}  |  Aantal locaties: {len(group)}")
        print("---------------------------------------")
        print(group[["cluster_id", "source", "geometry"]])

# 5. Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Achtergrond (optioneel)
combined_gdf.plot(ax=ax, color="lightgrey", alpha=0.2)

# Plot LocSpecs die in een cluster zitten
locspec_clusters.plot(
    ax=ax,
    column="cluster_id",
    cmap="tab20",
    markersize=80,
    legend=True,
    label="Clusters"
)

# Plot niet-geclusterde LocSpecs (cluster_id = -1)
locspec[locspec["cluster_id"] == -1].plot(
    ax=ax,
    color="black",
    markersize=40,
    label="Geen cluster"
)

ax.set_title("Locatiespecifieke VKA clusters (binnen 2 km)")
ax.set_aspect("equal")
plt.show()