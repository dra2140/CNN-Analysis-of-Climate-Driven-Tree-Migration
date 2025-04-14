import pandas as pd
from shapely.geometry import Polygon, Point
import numpy as np

# Define the polygon coordinates
polygon_coords = [
    (-69.25169, 47.44281),
    (-75.44628, 45.32212),
    (-78.86807, 43.00021),
    (-75.47091, 38.54031),
    (-71.10597, 40.19079),
    (-66.62251, 44.04165),
    (-65.84348, 47.51913),
    (-69.25169, 47.44281)
]

# Create Shapely polygon
study_area = Polygon(polygon_coords)

# Read the CSV file with proper settings
df = pd.read_csv('src/all_trees_gbif.csv', delimiter='\t')

# Define the target species
target_species = ['Acer rubrum', 'Abies balsamea', 'Betula alleghaniensis']

# Create a mask for points within the polygon
def is_within_polygon(row):
    try:
        point = Point(row['decimalLongitude'], row['decimalLatitude'])
        return study_area.contains(point)
    except:
        return False

# Filter the dataframe for species and location
species_mask = df['scientificName'].str.contains('|'.join(target_species), na=False)
df['in_polygon'] = df.apply(is_within_polygon, axis=1)
filtered_df = df[species_mask & df['in_polygon']]

# Save the filtered data to a new CSV file
filtered_df.to_csv('src/filtered_trees_gbif.csv', index=False)

# Print some statistics
print(f"Total number of records: {len(df)}")
print(f"Number of filtered records (species + location): {len(filtered_df)}")
print("\nRecords per species in study area:")
for species in target_species:
    count = len(filtered_df[filtered_df['scientificName'].str.contains(species, na=False)])
    print(f"{species}: {count} records")

# Print geographic extent of filtered data
print("\nGeographic extent of filtered data:")
print(f"Latitude range: {filtered_df['decimalLatitude'].min():.4f} to {filtered_df['decimalLatitude'].max():.4f}")
print(f"Longitude range: {filtered_df['decimalLongitude'].min():.4f} to {filtered_df['decimalLongitude'].max():.4f}") 