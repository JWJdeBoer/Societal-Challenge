import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


gdf_energieprojecten = gpd.read_file("cleaned_data/energieprojecten.gpkg", layer="data")
gdf_locatiespecifiek = gpd.read_file("cleaned_data/locatiespecifiek.gpkg", layer="data")
gdf_bouwwerken = gpd.read_file("cleaned_data/bouwwerken.gpkg", layer="data")
gdf_energieprojecten = gpd.read_file("cleaned_data/energieprojecten.gpkg", layer="data")
combined_gdf = gpd.read_file("cleaned_data/combined.gpkg", layer="data")

gdf_nl = gpd.read_file("Data/Netherlands_shapefile/nl_1km.shp")  # pas pad/naam aan





# Plot alles gekleurd per 'source'
# fig, ax = plt.subplots(figsize=(10, 12))
#
# combined_gdf.plot(
#     ax=ax,
#     column="source",
#     categorical=True,
#     legend=True,
#     cmap="tab10",
#     markersize=100  # voor punten; polygonen negeren dit
# )
#
# ax.set_title("Alle lagen gekleurd per bron (source)")
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()


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
    cluster_summary_gdf = gpd.GeoDataFrame(
        columns=["cluster_id", "energie_idx", "n_locatiespecifiek", "geometry"],
        crs=energie.crs
    )
    cluster_members_gdf = gpd.GeoDataFrame(
        columns=list(energie.columns) + list(locspec.columns) + ["cluster_id", "role"],
        crs=energie.crs
    )

print("Aantal energieproject-clusters met ≥1 Locatiespecifiek:", len(cluster_summary_gdf))

# Globale plot van alle clusters
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

# =========================
# Detail: inzoomen op één cluster (bijv. 16)
# =========================

CLUSTER_TO_ZOOM = 16  # pas dit aan naar een andere cluster_id als gewenst

cluster_to_zoom = cluster_members_gdf[cluster_members_gdf["cluster_id"] == CLUSTER_TO_ZOOM]

if cluster_to_zoom.empty:
    print(f"\nGeen cluster gevonden met cluster_id = {CLUSTER_TO_ZOOM}")
else:
    print(f"\nDetailplot voor cluster_id = {CLUSTER_TO_ZOOM}")

    fig, ax = plt.subplots(figsize=(8, 10))

    # Optioneel: lichte achtergrond van alle data
    combined_gdf.plot(ax=ax, color="lightgrey", alpha=0.2)

    # Locatiespecifiek in dit cluster
    cluster_to_zoom[cluster_to_zoom["role"] == "Locatiespecifiek"].plot(
        ax=ax, color="red", markersize=40, label="Locatiespecifiek (cluster)"
    )

    # Energieproject(en) in dit cluster
    cluster_to_zoom[cluster_to_zoom["role"] == "Energieproject"].plot(
        ax=ax, color="blue", markersize=80, label="Energieproject (clustercenter)"
    )

    # Inzoomen op de bounding box van dit cluster
    minx, miny, maxx, maxy = cluster_to_zoom.total_bounds
    padding_x = (maxx - minx) * 0.1 or 100.0  # 10% marge, of een minimum
    padding_y = (maxy - miny) * 0.1 or 100.0

    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)

    ax.set_aspect("equal")
    ax.set_title(f"Detailweergave cluster {CLUSTER_TO_ZOOM}")
    ax.legend()
    plt.show()

    # 2. CRS afstemmen: neem het CRS van je cluster_members_gdf / combined_gdf
    if gdf_nl.crs != combined_gdf.crs:
        gdf_nl = gdf_nl.to_crs(combined_gdf.crs)

    fig, ax = plt.subplots(figsize=(8, 10))

    # 3. Nederland als achtergrond
    gdf_nl.plot(
        ax=ax,
        color="whitesmoke",  # vulling
        edgecolor="black",  # rand
        linewidth=0.5,
        alpha=1.0
    )

    # 4. Dan je eigen data (bijv. ingezoomd cluster)
    cluster_to_zoom[cluster_to_zoom["role"] == "Locatiespecifiek"].plot(
        ax=ax, color="red", markersize=40, label="Locatiespecifiek (cluster)"
    )

    cluster_to_zoom[cluster_to_zoom["role"] == "Energieproject"].plot(
        ax=ax, color="blue", markersize=80, label="Energieproject (clustercenter)"
    )

    # 5. Inzoomen op bounding box van het cluster (of heel NL, als je dat wilt)
    minx, miny, maxx, maxy = cluster_to_zoom.total_bounds
    padding_x = (maxx - minx) * 0.1 or 100.0
    padding_y = (maxy - miny) * 0.1 or 100.0

    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)

    ax.set_aspect("equal")
    ax.set_title(f"Detailweergave cluster {CLUSTER_TO_ZOOM} met Nederland als achtergrond")
    ax.legend()
    plt.show()

CLUSTER_TO_EXPORT = 16
subset = cluster_members_gdf[cluster_members_gdf["cluster_id"] == CLUSTER_TO_EXPORT].copy()

if not subset.empty:
    subset_wgs84 = subset.to_crs(epsg=4326)
    subset_wgs84["lon"] = subset_wgs84.geometry.x
    subset_wgs84["lat"] = subset_wgs84.geometry.y

    cols_for_csv = []
    for c in ["cluster_id", "role", "Naam", "Projectnaam", "source"]:
        if c in subset_wgs84.columns:
            cols_for_csv.append(c)
    cols_for_csv += ["lon", "lat"]

    subset_wgs84[cols_for_csv].to_csv(
        f"cluster_{CLUSTER_TO_EXPORT}_punten_google_maps.csv",
        index=False,
        encoding="utf-8"
    )
    print(f"Cluster {CLUSTER_TO_EXPORT} geëxporteerd naar Google Maps CSV.")
else:
    print(f"Geen punten gevonden voor cluster_id = {CLUSTER_TO_EXPORT}")
