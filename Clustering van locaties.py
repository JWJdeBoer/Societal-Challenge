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


gdf_energieprojecten = gpd.read_file("energieprojecten.gpkg", layer="projecten")

# Zorg dat energieprojecten dezelfde CRS hebben als de rest
if gdf_energieprojecten.crs != gdf_bovenregionaal.crs:
    gdf_energieprojecten = gdf_energieprojecten.to_crs(gdf_bovenregionaal.crs)
    combined_gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                gdf_bovenregionaal.assign(source="Bovenregionaal"),
                gdf_locatiespecifiek.assign(source="Locatiespecifiek"),
                gdf_bouwwerken_merged.assign(source="Bouwwerk"),
                gdf_energieprojecten.assign(source="Energieproject"),
            ],
            ignore_index=True
        ),
        crs=gdf_bovenregionaal.crs
    )

# Plot alles gekleurd per 'source'
fig, ax = plt.subplots(figsize=(10, 12))

combined_gdf.plot(
    ax=ax,
    column="source",
    categorical=True,
    legend=True,
    cmap="tab10",
    markersize=100  # voor punten; polygonen negeren dit
)

ax.set_title("Alle lagen gekleurd per bron (source)")
ax.set_axis_off()
plt.tight_layout()
plt.show()


# === Clusters: 1 Energieproject met Locatiespecifiek eromheen ===

BUFFER_RADIUS = 10000  # in meters, pas aan naar 2000 / 10000 etc.

# Werk met kopieën van de originele lagen
locspec = gdf_locatiespecifiek.copy()
energie = gdf_energieprojecten.copy()

# Zorg dat we met punten werken (centroids als het polygonen zijn)
if not (locspec.geom_type == "Point").all():
    locspec = locspec.set_geometry(locspec.geometry.centroid)

if not (energie.geom_type == "Point").all():
    energie = energie.set_geometry(energie.geometry.centroid)

# Spatial index op Locatiespecifiek
sindex = locspec.sindex

cluster_summaries = []
cluster_member_rows = []

for e_idx, e_row in energie.iterrows():
    buf = e_row.geometry.buffer(BUFFER_RADIUS)

    # Kandidaten via bounding box
    candidate_idx = list(sindex.intersection(buf.bounds))
    candidates = locspec.iloc[candidate_idx]

    # Echte intersectie: LocSpec binnen buffer
    matches = candidates[candidates.intersects(buf)]

    if matches.empty:
        continue  # geen cluster voor dit energieproject

    # Kies cluster_id (bijv. OBJECTID als die bestaat, anders index)
    cluster_id = e_row.get("OBJECTID", e_idx)

    # Opslaan samenvatting
    cluster_summaries.append({
        "cluster_id": cluster_id,
        "energie_idx": e_idx,
        "n_locatiespecifiek": len(matches),
        "geometry": e_row.geometry  # centroid energieproject
    })

    # Leden-tabel maken: energieproject + alle LocSpec in dit cluster
    energie_member = e_row.to_frame().T.copy()
    energie_member["cluster_id"] = cluster_id
    energie_member["role"] = "Energieproject"

    locspec_members = matches.copy()
    locspec_members["cluster_id"] = cluster_id
    locspec_members["role"] = "Locatiespecifiek"

    cluster_member_rows.append(pd.concat([energie_member, locspec_members], ignore_index=True))

# GeoDataFrames maken
if cluster_summaries:
    cluster_summary_gdf = gpd.GeoDataFrame(cluster_summaries, crs=energie.crs)
    cluster_members_gdf = gpd.GeoDataFrame(
        pd.concat(cluster_member_rows, ignore_index=True),
        crs=energie.crs
    )
else:
    cluster_summary_gdf = gpd.GeoDataFrame(columns=["cluster_id", "energie_idx", "n_locatiespecifiek", "geometry"], crs=energie.crs)
    cluster_members_gdf = gpd.GeoDataFrame(columns=list(energie.columns) + list(locspec.columns) + ["cluster_id", "role"], crs=energie.crs)

print("Aantal energieproject-clusters met ≥1 Locatiespecifiek:", len(cluster_summary_gdf))


fig, ax = plt.subplots(figsize=(10, 12))

combined_gdf.plot(ax=ax, color="lightgrey", alpha=0.3)

# Locatiespecifiek in clusters
cluster_members_gdf[cluster_members_gdf["role"] == "Locatiespecifiek"].plot(
    ax=ax, color="red", markersize=30, label="Locatiespecifiek (in cluster)"
)

# Energieprojecten (centra)
cluster_members_gdf[cluster_members_gdf["role"] == "Energieproject"].plot(
    ax=ax, color="blue", markersize=60, label="Energieproject"
)

plt.legend()
plt.title(f"Energieproject-clusters met Locatiespecifiek binnen {BUFFER_RADIUS/1000:.1f} km")
plt.axis("equal")
plt.show()

print("\n=== CLUSTER-INFORMATIE ===")

if cluster_summary_gdf.empty:
    print("Geen clusters gevonden.")
else:
    for cid in cluster_summary_gdf["cluster_id"]:
        print("\n--------------------------------------------------------")
        print(f"CLUSTER ID: {cid}")

        # Filter leden van dit cluster
        members = cluster_members_gdf[cluster_members_gdf["cluster_id"] == cid]

        # Energieproject in dit cluster
        energy = members[members["role"] == "Energieproject"]

        # Locatiespecifiek
        locs = members[members["role"] == "Locatiespecifiek"]

        print(f"Aantal Locatiespecifiek: {len(locs)}")

        print("\n--- Energieproject ---")
        print(energy[[col for col in energy.columns if col not in ["geometry"]]])

        print("\n--- Locatiespecifieke VKA’s ---")
        # print enkele bruikbare kolommen (pas aan als je er meer wil)
        loc_cols = [c for c in ["OBJECTID", "source", "Naam", "ID", "Type"] if c in locs.columns]
        print(locs[loc_cols])

        print("--------------------------------------------------------")


