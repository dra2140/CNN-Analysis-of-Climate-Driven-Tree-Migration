import requests
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import os

# key tree species to focus on
target_species = [
    "Pinus strobus",      # Eastern White Pine
    "Abies balsamea",     # Balsam Fir
    "Acer rubrum",        # Red Maple
    "Tsuga canadensis",   # Eastern Hemlock
    "Betula papyrifera",   # Paper Birch
    "Betula alleghaniensis", # yellow birch

]

# define to maine loc
# min_lon, min_lat = -71.1, 43.0
# max_lon, max_lat = -66.9, 47.5

min_lon, min_lat = -70.5, 44.5
max_lon, max_lat = -67.5, 47.5

os.makedirs("species_data", exist_ok=True)

# downloading species data from GBIF
def download_gbif_data(species_name, output_file):
    print(f"Downloading data for {species_name}...")
    
    # params for GBIF API
    params = {
        "scientificName": species_name,
        "hasCoordinate": "true",
        "country": "US",
        "stateProvince": "Maine",
        "year": "2000,2025",
        "limit": 100000  
    }
    
    # GBIF species occurrence API
    url = "https://api.gbif.org/v1/occurrence/search"
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: GBIF API returned status code {response.status_code}")
        return None
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        print(f"No data found for {species_name}")
        return None
    
    occurrences = pd.DataFrame(data["results"]) 
    if "decimalLongitude" in occurrences.columns and "decimalLatitude" in occurrences.columns:
        occurrences = occurrences[["scientificName", "decimalLongitude", "decimalLatitude", 
                                 "year", "month", "day", "basisOfRecord"]]
        
        occurrences.to_csv(output_file, index=False)
        print(f"Saved {len(occurrences)} occurrences to {output_file}")
        return occurrences
    else:
        print(f"Data for {species_name} is missing coordinate information")
        return None

# downloading data for each species
all_species_data = []

for species in target_species:
    species_file = os.path.join("species_data", f"{species.replace(' ', '_')}.csv")
    
    if os.path.exists(species_file):
        print(f"Loading existing data for {species}...")
        species_data = pd.read_csv(species_file)
    else:
        species_data = download_gbif_data(species, species_file)
    
    if species_data is not None:
        species_data["species"] = species
        all_species_data.append(species_data)

# combining all species data
if all_species_data:
    combined_data = pd.concat(all_species_data, ignore_index=True)
    print(f"Combined {len(combined_data)} records for all species")

    # Filter to study area BEFORE saving
    combined_data = combined_data[
        (combined_data["decimalLongitude"] >= min_lon) & 
        (combined_data["decimalLongitude"] <= max_lon) & 
        (combined_data["decimalLatitude"] >= min_lat) & 
        (combined_data["decimalLatitude"] <= max_lat)
    ]
    print(f"Filtered to {len(combined_data)} records within study area")
    
    combined_file = os.path.join("species_data", "all_species.csv")
    combined_data.to_csv(combined_file, index=False)
    
    
    # create geoDataFrame for mapping
    geometry = [Point(xy) for xy in zip(combined_data["decimalLongitude"], combined_data["decimalLatitude"])]
    species_gdf = gpd.GeoDataFrame(combined_data, geometry=geometry, crs="EPSG:4326")
    
    # # filtering to study area (aka northern maine as of rn)
    # species_gdf = species_gdf[
    #     (species_gdf["decimalLongitude"] >= min_lon) & 
    #     (species_gdf["decimalLongitude"] <= max_lon) & 
    #     (species_gdf["decimalLatitude"] >= min_lat) & 
    #     (species_gdf["decimalLatitude"] <= max_lat)
    # ]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    from shapely.geometry import box
    maine_box = gpd.GeoDataFrame(
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs="EPSG:4326"
    )
    maine_box.boundary.plot(ax=ax, color="black", linewidth=1)
    
    # plotting species
    colors = ["red", "blue", "green", "purple", "orange"]
    
    for i, species in enumerate(target_species):
        species_points = species_gdf[species_gdf["species"] == species]
        if len(species_points) > 0:
            species_points.plot(
                ax=ax, 
                color=colors[i % len(colors)], 
                marker="o", 
                markersize=30, 
                alpha=0.5,
                label=species
            )
    
    plt.title("Tree Species Distribution in Maine")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    
    plt.savefig("species_distribution_map.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Analysis complete. Distribution map saved as 'species_distribution_map.png'")
else:
    print("No data was collected for any species")