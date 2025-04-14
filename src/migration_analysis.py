import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import center_of_mass, gaussian_filter
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
import seaborn as sns

def calculate_range_shifts(distribution_maps, species_names, start_year=2000, end_year=2021):
    """
    Calculate and visualize range shifts for each species over time
    
    Args:
        distribution_maps: Dictionary of distribution maps by year
        species_names: List of species names
        start_year, end_year: Range of years
        
    Returns:
        DataFrame with migration metrics
    """
    # Create output directory
    os.makedirs('migration_analysis', exist_ok=True)
    
    # Get reference data for georeferencing
    with rasterio.open(f"distribution_maps/species_distribution_{start_year}.tif") as src:
        transform = src.transform
        crs = src.crs
    
    # Store migration metrics
    migration_data = []
    
    # Process each species
    for s_idx, species in enumerate(species_names):
        print(f"Analyzing migration patterns for {species}...")
        
        # 1. Track center of mass over time
        centers = []
        range_areas = []
        elevations = []
        years = []
        
        # For visualization of movement trajectory
        center_points = []
        
        # Process each year
        for year in range(start_year, end_year + 1):
            # Get distribution map for this species and year
            if year in distribution_maps:
                species_map = distribution_maps[year][s_idx]
            else:
                # Load from saved file if not in memory
                with rasterio.open(f"distribution_maps/species_distribution_{year}.tif") as src:
                    species_map = src.read(s_idx+1)
            
            # Apply threshold to get binary presence/absence
            threshold = 0.5  # Adjust based on your model's performance
            presence = species_map > threshold
            
            if not np.any(presence):
                continue
                
            # Smooth the distribution map for more stable center calculations
            smoothed_map = gaussian_filter(species_map, sigma=3)
            
            # Calculate center of mass (weighted by probability)
            center_y, center_x = center_of_mass(smoothed_map)
            
            # Convert to geographic coordinates
            lon, lat = rasterio.transform.xy(transform, center_y, center_x)
            
            centers.append((lon, lat))
            center_points.append(Point(lon, lat))
            years.append(year)
            
            # Calculate range area (number of pixels above threshold * pixel area)
            pixel_area_km2 = abs(transform[0] * transform[4]) / 1000000  # approximate km²
            range_area = np.sum(presence) * pixel_area_km2
            range_areas.append(range_area)
            
            # Here you could also extract elevation data if available
            # For now we'll just use a placeholder
            elevations.append(0)
        
        if len(centers) < 2:
            print(f"Not enough data points for {species}")
            continue
        
        # Create GeoDataFrame of center points
        center_gdf = gpd.GeoDataFrame(
            {'species': species, 'year': years, 'geometry': center_points},
            crs=crs
        )
        
        # Create a LineString connecting the centers in chronological order
        trajectory = LineString(center_points)
        
        # 2. Calculate migration metrics
        
        # Total distance moved (in km)
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        
        distances = []
        directions = []
        
        for i in range(1, len(centers)):
            # Calculate distance and azimuth between consecutive points
            lon1, lat1 = centers[i-1]
            lon2, lat2 = centers[i]
            
            # Calculate forward and back azimuths, plus distance
            az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)
            
            # Convert distance to kilometers
            dist_km = dist / 1000.0
            
            distances.append(dist_km)
            directions.append(az12)
        
        # Calculate average annual movement rate
        avg_distance = np.mean(distances)
        
        # Calculate movement direction (average azimuth)
        mean_direction = np.mean(directions)
        
        # Convert azimuth to compass direction
        compass_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        compass_idx = int(round(mean_direction / 45)) % 8
        compass_direction = compass_dirs[compass_idx]
        
        # Calculate change in range area
        area_change = range_areas[-1] - range_areas[0]
        area_change_pct = (area_change / range_areas[0]) * 100 if range_areas[0] > 0 else 0
        
        # Store migration metrics
        migration_data.append({
            'species': species,
            'avg_annual_movement_km': avg_distance,
            'direction_degrees': mean_direction,
            'direction': compass_direction,
            'total_distance_km': sum(distances),
            'start_area_km2': range_areas[0],
            'end_area_km2': range_areas[-1],
            'area_change_km2': area_change,
            'area_change_percent': area_change_pct,
            'years_analyzed': len(years)
        })
        
        # 3. Visualize migration patterns
        
        # Plot center of mass trajectory
        plt.figure(figsize=(10, 8))
        
        # Create colormap for years
        norm = colors.Normalize(vmin=min(years), vmax=max(years))
        sm = ScalarMappable(norm=norm, cmap='viridis')
        
        # Plot trajectory
        for i in range(len(center_points) - 1):
            x1, y1 = center_points[i].x, center_points[i].y
            x2, y2 = center_points[i + 1].x, center_points[i + 1].y
            plt.plot([x1, x2], [y1, y2], '-', color=sm.to_rgba(years[i]), linewidth=2)
        
        # Plot points
        for i, point in enumerate(center_points):
            plt.plot(point.x, point.y, 'o', color=sm.to_rgba(years[i]), 
                    markersize=8, label=f"{years[i]}" if i % 3 == 0 else "")
        
        # Add colorbar
        cbar = plt.colorbar(sm)
        cbar.set_label('Year')
        
        plt.title(f"{species} Distribution Center Movement (2000-2021)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Only include legend for some years to avoid crowding
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best', title="Year")
        
        plt.savefig(f"migration_analysis/{species.replace(' ', '_')}_trajectory.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Plot range area change over time
        plt.figure(figsize=(10, 6))
        plt.plot(years, range_areas, 'o-', linewidth=2)
        plt.title(f"{species} Range Area Over Time")
        plt.xlabel("Year")
        plt.ylabel("Range Area (km²)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"migration_analysis/{species.replace(' ', '_')}_range_area.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Calculate and visualize latitudinal shift
        latitudes = [point.y for point in center_points]
        
        # Linear regression for latitudinal shift
        X = np.array(years).reshape(-1, 1)
        y = np.array(latitudes)
        model = LinearRegression().fit(X, y)
        lat_shift_rate = model.coef_[0]  # degrees latitude per year
        
        # Convert to approximate distance in km (1 degree latitude ≈ 111 km)
        lat_shift_km_per_year = lat_shift_rate * 111
        
        plt.figure(figsize=(10, 6))
        plt.scatter(years, latitudes)
        plt.plot(years, model.predict(X), 'r-')
        plt.title(f"{species} Latitudinal Shift: {lat_shift_km_per_year:.2f} km/year")
        plt.xlabel("Year")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"migration_analysis/{species.replace(' ', '_')}_latitude_shift.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    # Create summary DataFrame
    migration_df = pd.DataFrame(migration_data)
    
    # Save migration data
    migration_df.to_csv("migration_analysis/migration_metrics.csv", index=False)
    
    # Create summary visualization comparing all species
    plt.figure(figsize=(12, 8))
    
    for species in species_names:
        if species in migration_df['species'].values:
            data = migration_df[migration_df['species'] == species]
            plt.arrow(0, 0, 
                     data['avg_annual_movement_km'].values[0] * np.sin(np.radians(data['direction_degrees'].values[0])),
                     data['avg_annual_movement_km'].values[0] * np.cos(np.radians(data['direction_degrees'].values[0])),
                     head_width=0.05, head_length=0.1, fc='k', ec='k', label=species)
    
    plt.title("Migration Direction and Rate by Species")
    plt.xlabel("East-West Movement (km/year)")
    plt.ylabel("North-South Movement (km/year)")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    plt.savefig("migration_analysis/all_species_migration_vectors.png", 
               bbox_inches='tight', dpi=150)
    
    return migration_df

def correlate_climate_factors(migration_df, species_names, data_dir, start_year=2000, end_year=2021):
    """
    Analyze correlation between climate factors and migration patterns
    
    Args:
        migration_df: DataFrame with migration metrics
        species_names: List of species names
        data_dir: Directory containing climate data
        start_year, end_year: Range of years
    """
    # Create output directory
    os.makedirs('climate_analysis', exist_ok=True)
    
    # Load climate data
    climate_data = []
    
    for year in range(start_year, end_year + 1):
        try:
            # Try to load PRISM data
            prism_path = os.path.join(data_dir, f"prism_climate_{year}.tif")
            
            with rasterio.open(prism_path) as src:
                temp = src.read(1)  # Annual mean temperature
                precip = src.read(2)  # Annual precipitation
                
                # Calculate regional averages
                avg_temp = np.nanmean(temp)
                avg_precip = np.nanmean(precip)
                
                climate_data.append({
                    'year': year,
                    'avg_temperature': avg_temp,
                    'avg_precipitation': avg_precip
                })
        except:
            print(f"Could not load climate data for {year}")
    
    if not climate_data:
        print("No climate data available for analysis")
        return
    
    # Create climate DataFrame
    climate_df = pd.DataFrame(climate_data)
    
    # Calculate temperature and precipitation trends
    X = np.array(climate_df['year']).reshape(-1, 1)
    
    temp_model = LinearRegression().fit(X, climate_df['avg_temperature'])
    precip_model = LinearRegression().fit(X, climate_df['avg_precipitation'])
    
    temp_trend = temp_model.coef_[0]  # degrees C per year
    precip_trend = precip_model.coef_[0]  # mm per year
    
    # Plot climate trends
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(climate_df['year'], climate_df['avg_temperature'])
    plt.plot(climate_df['year'], temp_model.predict(X), 'r-')
    plt.title(f"Temperature Trend: {temp_trend:.3f}°C/year")
    plt.xlabel("Year")
    plt.ylabel("Average Temperature (°C)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.scatter(climate_df['year'], climate_df['avg_precipitation'])
    plt.plot(climate_df['year'], precip_model.predict(X), 'b-')
    plt.title(f"Precipitation Trend: {precip_trend:.1f} mm/year")
    plt.xlabel("Year")
    plt.ylabel("Annual Precipitation (mm)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("climate_analysis/climate_trends.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # For each species, compare migration metrics with climate trends
    # This is a simplified analysis - a more detailed approach would correlate 
    # spatial climate patterns with species movement
    
    # Example: Create a summary table of species sensitivity to climate
    climate_sensitivity = []
    
    for species in species_names:
        if species in migration_df['species'].values:
            species_data = migration_df[migration_df['species'] == species].iloc[0]
            
            # Classify migration direction
            direction = species_data['direction']
            
            # Northward movement typically indicates warming response
            temp_sensitive = direction in ['N', 'NE', 'NW']
            
            climate_sensitivity.append({
                'species': species,
                'migration_rate_km_per_year': species_data['avg_annual_movement_km'],
                'direction': direction,
                'area_change_percent': species_data['area_change_percent'],
                'temp_sensitive': temp_sensitive,
                'regional_warming_rate': temp_trend,
                'regional_precip_change': precip_trend
            })
    
    # Create summary DataFrame
    sensitivity_df = pd.DataFrame(climate_sensitivity)
    
    # Save climate sensitivity data
    sensitivity_df.to_csv("climate_analysis/climate_sensitivity.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot of migration rate vs. temperature sensitivity
    plt.scatter(
        sensitivity_df['migration_rate_km_per_year'],
        sensitivity_df['area_change_percent'],
        c=sensitivity_df['temp_sensitive'].map({True: 'red', False: 'blue'}),
        s=100,
        alpha=0.7
    )
    
    # Add labels
    for i, row in sensitivity_df.iterrows():
        plt.annotate(
            row['species'].split()[-1],  # Use just the species name for clarity
            (row['migration_rate_km_per_year'], row['area_change_percent']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title("Tree Species Migration Response to Climate Change")
    plt.xlabel("Migration Rate (km/year)")
    plt.ylabel("Range Area Change (%)")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Temperature Sensitive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Less Temperature Sensitive')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.savefig("climate_analysis/species_climate_response.png", bbox_inches='tight', dpi=150)
    plt.close()