import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

def load_tree_data(csv_path):
    """Load and prepare tree observation data with temporal information."""
    df = pd.read_csv(csv_path)
    
    # Convert eventDate to datetime
    df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')
    
    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['decimalLongitude'], df['decimalLatitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Filter to our species of interest
    species = ['Acer rubrum', 'Abies balsamea', 'Betula alleghaniensis']
    gdf = gdf[gdf['verbatimScientificName'].isin(species)]
    
    # Extract year from eventDate
    gdf['year'] = gdf['eventDate'].dt.year
    
    print(f"Loaded {len(gdf)} tree observations")
    print("\nObservations by species and year:")
    print(gdf.groupby(['verbatimScientificName', 'year']).size().unstack())
    
    return gdf

def analyze_migration_patterns(tree_gdf, climate_data_dir):
    """Analyze changes in tree observation locations over time."""
    results = {}
    
    for species in tree_gdf['verbatimScientificName'].unique():
        print(f"\nAnalyzing migration patterns for {species}")
        species_data = tree_gdf[tree_gdf['verbatimScientificName'] == species]
        
        # Calculate centroid movement over time
        centroids = species_data.groupby('year')['geometry'].apply(
            lambda x: Point(x.unary_union.centroid)
        )
        
        # Calculate latitudinal and longitudinal shifts
        lat_shifts = []
        lon_shifts = []
        years = []
        
        for year, centroid in centroids.items():
            if year >= 2018:  # Only consider years with climate data
                lat_shifts.append(centroid.y)
                lon_shifts.append(centroid.x)
                years.append(year)
        
        # Calculate migration rates
        if len(years) > 1:
            lat_rate = stats.linregress(years, lat_shifts)[0]  # degrees per year
            lon_rate = stats.linregress(years, lon_shifts)[0]  # degrees per year
            
            results[species] = {
                'lat_rate': lat_rate,
                'lon_rate': lon_rate,
                'centroids': centroids,
                'years': years,
                'lat_shifts': lat_shifts,
                'lon_shifts': lon_shifts
            }
            
            print(f"Migration rate for {species}:")
            print(f"  Latitude: {lat_rate:.4f} degrees/year")
            print(f"  Longitude: {lon_rate:.4f} degrees/year")
    
    return results

def plot_migration_patterns(results, output_dir):
    """Create visualizations of migration patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot centroid movement over time
    plt.figure(figsize=(12, 6))
    for species, data in results.items():
        plt.plot(data['years'], data['lat_shifts'], 
                label=f'{species} (rate: {data["lat_rate"]:.4f}°/year)')
    
    plt.xlabel('Year')
    plt.ylabel('Latitude (degrees)')
    plt.title('Latitudinal Migration Patterns')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latitudinal_migration.png'))
    plt.close()
    
    # Plot longitudinal movement
    plt.figure(figsize=(12, 6))
    for species, data in results.items():
        plt.plot(data['years'], data['lon_shifts'],
                label=f'{species} (rate: {data["lon_rate"]:.4f}°/year)')
    
    plt.xlabel('Year')
    plt.ylabel('Longitude (degrees)')
    plt.title('Longitudinal Migration Patterns')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'longitudinal_migration.png'))
    plt.close()

def correlate_with_climate(tree_gdf, climate_data_dir, results):
    """Correlate migration patterns with climate variables."""
    climate_correlations = {}
    
    for species, migration_data in results.items():
        species_data = tree_gdf[tree_gdf['verbatimScientificName'] == species]
        climate_correlations[species] = {}
        
        for year in migration_data['years']:
            # Load climate data for the year
            prism_path = os.path.join(climate_data_dir, f'prism_climate_{year}.tif')
            terra_path = os.path.join(climate_data_dir, f'terraclimate_{year}.tif')
            
            if os.path.exists(prism_path) and os.path.exists(terra_path):
                # Extract climate data for observations in this year
                year_data = species_data[species_data['year'] == year]
                
                # Calculate correlations between climate variables and location
                climate_correlations[species][year] = {
                    'precip': extract_climate_data(year_data, prism_path),
                    'tmean': extract_climate_data(year_data, prism_path),
                    'water_deficit': extract_climate_data(year_data, terra_path)
                }
    
    return climate_correlations

def main():
    # Load tree data
    tree_gdf = load_tree_data('src/filtered_trees_gbif.csv')
    
    # Analyze migration patterns
    results = analyze_migration_patterns(tree_gdf, 'climate_data_2018_2023')
    
    # Create visualizations
    plot_migration_patterns(results, 'migration_analysis_results')
    
    # Correlate with climate data
    climate_correlations = correlate_with_climate(tree_gdf, 'climate_data_2018_2023', results)
    
    print("\nAnalysis complete! Results saved in 'migration_analysis_results' directory")

if __name__ == "__main__":
    main() 