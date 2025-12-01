import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely  # vanaf shapely 2.x

gdf = gpd.read_file("Data/Bouwwerken_netcongestie/Bouwwerken_netcongestie.shp")  # of je lokale pad
df = pd.read_excel("Data/TUD Basislijst Bekende aansluitingen (sept 25).xlsx", sheet_name = "Gefilterde data")

gdf = gdf.merge(df, on="EAN", how="left")

# 1. Filter rijen met geldige geometrie én Contractcapaciteit
gdf_plot = gdf[gdf.geometry.notna() & gdf["Max ruimte opwek"].notna()].copy()

# 2. Centroids van de polygonen (geeft Points)
centroids = gdf_plot.geometry.centroid

# 3. X, Y en waarden eruit trekken
x = centroids.x.to_numpy()
y = centroids.y.to_numpy()
values = gdf_plot["Max ruimte opwek"].to_numpy()

# 4. Scatter plot met kleur op basis van Contractcapaciteit
fig, ax = plt.subplots(figsize=(10, 10))

sc = ax.scatter(
    x,
    y,
    c=values,           # kleur = Contractcapaciteit
    cmap="viridis",     # schaal: laag→hoog
    s=80,               # puntgrootte (vergroot/desnoods >100)
    edgecolors="black",
    linewidths=0.5,
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Max ruimte opwek")

ax.set_aspect("equal")
ax.set_title("Max ruimte opwek per bouwwerk")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.show()