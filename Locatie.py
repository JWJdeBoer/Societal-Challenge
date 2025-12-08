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


# === 4. Data selecteren voor de punten-plot ===
# Filter op geldige geometrie en niet-lege "Max ruimte opwek"
gdf_plot = gdf_bouwwerken_merged[
    gdf_bouwwerken_merged.geometry.notna()
    & gdf_bouwwerken_merged["Max ruimte opwek"].notna()
].copy()

# Centroids van de geometrie (meestal polygonen)
centroids = gdf_plot.geometry.centroid

# X, Y en waarden
x = centroids.x.to_numpy()
y = centroids.y.to_numpy()
values = gdf_plot["Max ruimte opwek"].to_numpy()  # consistent met colorbar-label


# # === 5. Plotten ===
# fig, ax = plt.subplots(figsize=(10, 10))
#
# # Eerst de achtergrondlaag (alle VKA + bouwwerken) licht tonen
# combined_gdf.plot(ax=ax, alpha=0.3, edgecolor="grey", linewidth=0.5)
#
# # Daarboven de punten met kleur op basis van "Max ruimte opwek"
# scatter = ax.scatter(
#     x,
#     y,
#     c=values,
#     cmap="viridis",
#     s=80,               # maak groter/kleiner indien nodig
#     edgecolors="black",
#     linewidths=0.5,
# )
#
# # Colorbar
# colorbar = plt.colorbar(scatter, ax=ax)
# colorbar.set_label("Max ruimte opwek")
#
# ax.set_aspect("equal")
# ax.set_title("Max ruimte opwek per bouwwerk")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
#
# plt.tight_layout()
# plt.show()

# === 1. Gebruik alleen punten (centroids van polygonen) ===
points_gdf = combined_gdf.copy()
points_gdf["geometry"] = points_gdf.geometry.centroid

# Nut: matrix van XY-coördinaten
coords = np.vstack([points_gdf.geometry.x, points_gdf.geometry.y]).T

# === 2. Clustering met DBSCAN ===
# eps = 750 meter (pas gerust aan)
model = DBSCAN(eps=3000, min_samples=2)
points_gdf["cluster_id"] = model.fit_predict(coords)

# -1 betekent "geen cluster"
clusters = points_gdf[points_gdf["cluster_id"] != -1]

valid_clusters = []

for cid, group in clusters.groupby("cluster_id"):
    has_locatiespecifiek = (group["source"] == "Locatiespecifiek").any()
    has_bouwwerk = (group["source"] == "Bouwwerk").any()

    if has_locatiespecifiek and has_bouwwerk:
        valid_clusters.append((cid, group))

# Resultaat tonen
print(f"Aantal gevonden clusters: {len(valid_clusters)}")

for cid, grp in valid_clusters:
    print("\nCluster", cid)
    print(grp[["source", "geometry"]])

fig, ax = plt.subplots(figsize=(10, 10))

combined_gdf.plot(ax=ax, alpha=0.2, color="grey")

for cid, grp in valid_clusters:
    grp.plot(ax=ax, markersize=80, label=f"Cluster {cid}")

plt.legend()
plt.title("Gevonden clusters met Locatiespecifiek + Bouwwerk")
plt.show()
