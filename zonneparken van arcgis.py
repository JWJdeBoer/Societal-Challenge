import geopandas as gpd

# 1. Vul hier de URL in van de laag (eindigt op /FeatureServer/0 of /MapServer/0)
FEATURE_URL = "https://arcg.is/1bLCiX2"

# 2. Maak een query-URL met pgeojson (ArcGIS geeft direct GeoJSON terug)
query_url = (
    f"{FEATURE_URL}/query?"
    "where=1=1&"
    "outFields=*&"
    "f=pgeojson"
)

# 3. Inlezen in GeoPandas
gdf = gpd.read_file(query_url)

print(gdf.head())
print(gdf.columns)

# 4. Opslaan als GeoPackage of Shapefile
gdf.to_file("energie_op_rijksgrond.gpkg", layer="projecten", driver="GPKG")
# of:
# gdf.to_file("energie_op_rijksgrond.shp")
