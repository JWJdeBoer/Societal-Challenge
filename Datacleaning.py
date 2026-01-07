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
gdf_bouwwerken = gdf_bouwwerken.merge(
    df_aansluitingen,
    on="EAN",
    how="left"
)

gdf_energieprojecten = gpd.read_file("energieprojecten.gpkg", layer="projecten")

mapping = {
    "Beschrijvi": "Omschrijvi",
    "naam": "Naam",
    "id": "ID",
    "Opp ha": "Opp (ha)",
    "Area ha": "Opp (ha)",
    "Naam_1": "Naam",
    "Area (ha)": "Opp (ha)",
'area (ha)': 'Opp (ha)',
'opp ha': 'Opp (ha)',
'name': 'Naam',
"area(ha)": 'Opp (ha)',
    "Gebied":"Omschrijvi",
    "sourceTxt":"Omschrijvi", "nam":"Naam",
}

kolommen = ["Omschrijvi", "geometry","Naam"]

def clean_cols(df, mapping=mapping,kolommen=kolommen, geometry_col="geometry"):
    is_geo = isinstance(df, gpd.GeoDataFrame)
    crs = df.crs if is_geo else None

    cols_original = list(df.columns)
    df = df.rename(columns=mapping)

    df_merged = df.T.groupby(level=0).first().T
    df_merged = df_merged[[c for c in cols_original if c in df_merged.columns]]
    df_merged = df_merged[kolommen]

    if is_geo:
        return gpd.GeoDataFrame(df_merged, geometry=geometry_col, crs=crs)
    return df_merged

gdf_bovenregionaal, gdf_locatiespecifiek  = [clean_cols(df) for df in [gdf_bovenregionaal, gdf_locatiespecifiek]]

# Zorg dat energieprojecten dezelfde CRS hebben als de rest
if gdf_energieprojecten.crs != gdf_bovenregionaal.crs:
    gdf_energieprojecten = gdf_energieprojecten.to_crs(gdf_bovenregionaal.crs)
    combined_gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                gdf_bovenregionaal.assign(source="Bovenregionaal"),
                gdf_locatiespecifiek.assign(source="Locatiespecifiek"),
                gdf_bouwwerken.assign(source="Bouwwerk"),
                gdf_energieprojecten.assign(source="Energieproject"),
            ],
            ignore_index=True
        ),
        crs=gdf_bovenregionaal.crs
    )



gdf_bovenregionaal.to_file("cleaned_data/Bovenregionaal.gpkg", layer ="data",driver="GPKG")
gdf_locatiespecifiek.to_file("cleaned_data/Locatiespecifiek.gpkg", layer ="data",driver="GPKG")
gdf_energieprojecten.to_file("cleaned_data/Energieprojecten.gpkg", layer ="data",driver="GPKG")
gdf_bouwwerken.to_file("cleaned_data/Bouwwerken.gpkg", layer ="data",driver="GPKG")
combined_gdf.to_file("cleaned_data/combined.gpkg", layer ="data",driver="GPKG")