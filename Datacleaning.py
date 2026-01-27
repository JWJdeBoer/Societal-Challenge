import glob
from pathlib import Path
from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# === Helperfunctie ===
import glob
from pathlib import Path
import geopandas as gpd
import pandas as pd

import re



def _clean_str(x):
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None

def make_display_name(row):
    src = _clean_str(row.get("source"))
    naam_loc = _clean_str(row.get("Naam_locatie"))
    naam = _clean_str(row.get("Naam"))
    oms = _clean_str(row.get("Omschrijvi"))
    proj = _clean_str(row.get("Projectnaam"))
    ent = _clean_str(row.get("Entity"))
    adr = _clean_str(row.get("Adres"))

    # === Bovenregionaal & Locatiespecifiek ===
    # Naam_locatie + (Naam of Omschrijvi)
    if src in ("Bovenregionaal", "Locatiespecifiek"):
        suffix = naam or oms
        if naam_loc and suffix:
            return f"{naam_loc} ({suffix})"
        return naam_loc or suffix

    # === Bouwwerk ===
    # Entity (Adres)
    if src == "Bouwwerk":
        if ent and adr:
            return f"{ent} ({adr})"
        return ent or adr or naam or oms

    # === Energieproject ===
    # Projectnaam
    if src == "Energieproject":
        return proj or naam or oms or naam_loc

    # Fallback
    return naam_loc or naam or oms or proj or ent or adr

def clean_bovenregionaal_name(name: str) -> str:
    """
    Verwijdert 'YYYYMMDD_VKA_behoefte ' uit bestandsnamen.
    """
    return re.sub(r"^\d{8}_VKA_behoefte\s*", "", name)


def merge_shapefiles(
    folder: str,
    recursive: bool = False,
    folder_col: str = "Naam_locatie",
    name_from: str = "parent"   # "parent" of "file"
) -> gpd.GeoDataFrame:
    folder = Path(folder)

    pattern = str(folder / ("**/*.shp" if recursive else "*.shp"))
    shapefile_paths = glob.glob(pattern, recursive=recursive)

    if not shapefile_paths:
        raise ValueError(f"Geen shapefiles gevonden in: {folder}")

    geodataframes = []
    for path in shapefile_paths:
        p = Path(path)
        gdf = gpd.read_file(p)

        if name_from == "file":
            name = p.stem                      # bestandsnaam zonder .shp
            name = clean_bovenregionaal_name(name)   # 🔥 prefix strippen
            gdf[folder_col] = name
        else:
            gdf[folder_col] = p.parent.name

        geodataframes.append(gdf)

    return gpd.GeoDataFrame(pd.concat(geodataframes, ignore_index=True), crs=geodataframes[0].crs)






# === Paden naar data ===
BASE_DATA_DIR = Path("Data")
VKA_BASE_DIR = BASE_DATA_DIR / "20250827_export_VKAs"

BOVENREGIONAAL_VKA_DIR = VKA_BASE_DIR / "Bovenregionaal VKA"
LOCATIESPECIFIEK_VKA_DIR = VKA_BASE_DIR / "Locatiespecifiek VKA"

BOUWWERKEN_SHP = BASE_DATA_DIR / "Bouwwerken_netcongestie" / "Bouwwerken_netcongestie.shp"
AANSLUITINGEN_XLSX = BASE_DATA_DIR / "TUD Basislijst Bekende aansluitingen (sept 25).xlsx"


# === 1. VKA-shapefiles mergen ===
gdf_bovenregionaal = merge_shapefiles(
    str(BOVENREGIONAAL_VKA_DIR),
    recursive=False,
    folder_col="Naam_locatie",
    name_from="file"          # shapefile naam → opgeschoond
)

gdf_locatiespecifiek = merge_shapefiles(
    str(LOCATIESPECIFIEK_VKA_DIR),
    recursive=True,
    folder_col="Naam_locatie",
    name_from="parent"        # mapnaam
)



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

import re


# === Opwek (MWp) direct toevoegen aan gdf_energieprojecten (VOOR combined_gdf) ===
POT_COL = "Potentie"     # <-- pas aan naar jouw echte kolomnaam
UNKNOWN_DEFAULT_MWP = 13.0

MWP_PER_HA_HIGH = 1.20          # hoogste voor ha
KWH_PER_KWP_LOW = 850           # laagste opbrengst -> hoogste benodigde kWp
KWH_PER_HH_HIGH = 3300          # hoogste verbruik -> hoogste benodigde kWp

def _to_float(s):
    s = str(s).strip()
    if not s:
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def extract_values(text):
    t = str(text).lower()

    if t.strip() == "" or "wordt momenteel onderzocht" in t or "momenteel onderzocht" in t:
        return {"unknown": True}

    # range: "10-15 mwp" of "10 tot 15 mwp"
    m = re.search(r"(\d+[.,]?\d*)\s*(?:-|–|—|tot)\s*(\d+[.,]?\d*)\s*mwp", t)
    if m:
        return {"mwp_min": _to_float(m.group(1)), "mwp_max": _to_float(m.group(2)), "unknown": False}

    # single: "12 mwp" / "ca 12 mwp"
    m = re.search(r"(?:circa|ca\.?|ongeveer|~)?\s*(\d+[.,]?\d*)\s*mwp", t)
    if m:
        return {"mwp_one": _to_float(m.group(1)), "unknown": False}

    # ha
    m = re.search(r"(\d+[.,]?\d*)\s*(ha|hectare)", t)
    if m:
        return {"ha": _to_float(m.group(1)), "unknown": False}

    # huishoudens
    m = re.search(r"(\d+[.,]?\d*)\s*(huishoudens|hh)", t)
    if m:
        return {"hh": _to_float(m.group(1)), "unknown": False}

    return {"unknown": True}

def estimate_opwek(text):
    p = extract_values(text)

    # 1) gegeven range -> gemiddelde
    if "mwp_min" in p and "mwp_max" in p and p["mwp_min"] is not None and p["mwp_max"] is not None:
        return (p["mwp_min"] + p["mwp_max"]) / 2.0

    # 2) gegeven enkel mwp
    if "mwp_one" in p and p["mwp_one"] is not None:
        return p["mwp_one"]

    # 3) huishoudens -> hoogste
    if "hh" in p and p["hh"] is not None:
        kwp = (p["hh"] * KWH_PER_HH_HIGH) / KWH_PER_KWP_LOW
        return kwp / 1000.0

    # 4) ha -> hoogste
    if "ha" in p and p["ha"] is not None:
        return p["ha"] * MWP_PER_HA_HIGH

    # 5) onbekend -> 13 MWp
    return UNKNOWN_DEFAULT_MWP

if POT_COL not in gdf_energieprojecten.columns:
    raise KeyError(f"Kolom '{POT_COL}' niet gevonden. Beschikbare kolommen: {list(gdf_energieprojecten.columns)}")

# bereken opwek
gdf_energieprojecten["opwek"] = gdf_energieprojecten[POT_COL].apply(estimate_opwek)

# zorg dat er nooit pd.NA blijft hangen (PyCharm crasht daarop)
gdf_energieprojecten["opwek"] = (
    pd.to_numeric(gdf_energieprojecten["opwek"], errors="coerce")
      .fillna(13.0)          # jouw default voor onbekend
)

# afronden; gebruik int (geen pandas Int64 met pd.NA)
gdf_energieprojecten["opwek"] = gdf_energieprojecten["opwek"].round(0).astype(int)


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

kolommen = ["Omschrijvi", "geometry","Naam","Naam_locatie"]

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
print("Aantal regels in combined.csv:", len(gdf_bovenregionaal))
gdf_locatiespecifiek.to_file("cleaned_data/Locatiespecifiek.gpkg", layer ="data",driver="GPKG")

print("Aantal regels in combined.csv:", len(gdf_locatiespecifiek))
gdf_energieprojecten.to_file("cleaned_data/Energieprojecten.gpkg", layer ="data",driver="GPKG")
gdf_energieprojecten.to_csv("cleaned_data/Energieprojecten.csv", index=False)

print("Aantal regels in combined.csv:", len(gdf_energieprojecten))
gdf_bouwwerken.to_file("cleaned_data/Bouwwerken.gpkg", layer ="data",driver="GPKG")

print(gdf_bouwwerken.columns)

combined_gdf["LocatieNaam"] = combined_gdf.apply(make_display_name, axis=1)


combined_gdf.to_file("cleaned_data/combined.gpkg", layer ="data",driver="GPKG")

combined_gdf.to_csv("combined2.0.csv")