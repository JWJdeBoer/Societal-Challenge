import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("Bouwwerken_netcongestie/Bouwwerken_netcongestie.shp")  # of je lokale pad

gdf.to_csv("test.csv")

print(gdf)

ax = gdf.plot(
    figsize=(10, 10),
    edgecolor="black",
    linewidth=10,
    facecolor="none",   # geen opvulling, alleen randen
)

ax.set_aspect("equal")  # zorgt dat x- en y-schaal gelijk zijn
plt.title("Bouwwerken – netcongestie")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()