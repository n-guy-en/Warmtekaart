# %% 
import geopandas as gpd

# 1. Laad GeoJSON
gdf = gpd.read_file("/Users/anitavn/Documents/Warmtekaart_test/Warmtekaart 2/data/layers/warmtenet_full.geojson")

# 2. Zet CRS als het niet in het bestand staat
gdf = gdf.set_crs("EPSG:28992")

# 3. Converteer naar WGS84
gdf_4326 = gdf.to_crs("EPSG:4326")

# 4. Opslaan
gdf_4326.to_file("warmtenet_full.geojson", driver="GeoJSON")

# %%

import gzip
import shutil

input_file = "/Users/anitavn/Documents/Warmtekaart_test/Warmtekaart 2/data/layers/warmtenet_full.geojson"
output_file = "warmtenet_full.geojson.gz"

with open(input_file, "rb") as f_in:
    with gzip.open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# %%
