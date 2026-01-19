import pandas as pd

df = pd.read_csv("Energieprojecten_MWp_logisch_eind_13MWp_onbekend.csv")

# Selecteer alleen de gewenste kolommen
df = df[["OBJECTID", "Projectnaam", "MWp_logisch_eind"]]

# Hernoem de kolom
df = df.rename(columns={"MWp_logisch_eind": "opwek"})
df["opwek"] = df["opwek"].round(0).astype(int)

df.to_csv("Energie_locatie_opwerk.csv",index=False)
