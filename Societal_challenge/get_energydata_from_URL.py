import geopandas as gpd
from urllib.parse import urlencode

base_url = "https://services.arcgis.com/kE0BiyvJHb5SwQv7/arcgis/rest/services/Energieprojecten_OER_RWS_test/FeatureServer/2/"
params = {
    'where': '1=1',
    'outFields': '*',
    'outSR': '4326',
    'f': 'geojson',
}

url = f"{base_url}/query?{urlencode(params)}"
gdf = gpd.read_file(url)
#
# gdf.to_file("energieprojecten.gpkg", layer="projecten", driver="GPKG")
