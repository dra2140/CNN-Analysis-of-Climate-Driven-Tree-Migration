import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_climate_data(data_dir, start_year, end_year):
    """
    Load and process climate data from PRISM and TerraClimate files
    
    Args:
        data_dir: Directory containing climate data files
        start_year, end_year: Range of years to process
        
    Returns:
        DataFrame with annual climate variables
    """
    climate_data = []
    
    # Process each year
    for year in range(start_year, end_year + 1):
        # Try loading PRISM data (prioritize)
        prism_path = os.path.join(data_dir, f"prism_climate_{year}.tif")
        terraclimate_path = os.path.join(data_dir, f"terraclimate_{year}.tif")
        
        year_data = {"year": year}
        
        # Load PRISM data if available
        if os.path.exists(prism_path):
            try:
                with rasterio.open(prism_path) as src:
                    bands = src.count
                    for b in range(1, bands + 1):
                        band_data = src.read(b)
                        
                        # Determine variable based on band index
                        if b == 1:
                            var_name = "annual_precip"
                        elif b == 2:
                            var_name = "annual_tmean"
                        elif b == 3:
                            var_name = "annual_tmin"
                        elif b == 4:
                            var_name = "annual_tmax"
                        else:
                            var_name = f"prism_band_{b}"
                        
                        # Calculate average across region
                        valid_data = band_data[~np.isnan(band_data)]
                        if len(valid_data) > 0:
                            year_data[var_name] = np.mean(valid_data)
            except Exception as e:
                print(f"Error reading PRISM data for {year}: {e}")
        
        # Load TerraClimate data if available
        if os.path.exists(terraclimate_path):
            try:
                with rasterio.open(terraclimate_path) as src:
                    bands = src.count
                    for b in range(1, bands + 1):
                        band_data = src.read(b)
                        
                        # Determine variable based on band index
                        if b == 1:
                            var_name = "annual_pet"
                        elif b == 2:
                            var_name = "annual_aet"
                        elif b == 3:
                            var_name = "annual_water_deficit"
                        elif b == 4:
                            var_name = "annual_soil_moisture"
                        elif b == 5:
                            var_name = "annual_vpd"
                        else:
                            var_name = f"terraclimate_band_{b}"
                        
                        # Calculate average across region
                        valid_data = band_data[~np.isnan(band_data)]
                        if len(valid_data) > 0:
                            year_data[var_name] = np.mean(valid_data)
            except Exception as e:
                print(f"Error reading TerraClimate data for {year}: {e}")
        
        # Only add if we have some climate data
        if len(year_data) > 1:  # more than just year
            climate_data.append(year_data)
    
    # Create DataFrame
    if climate_data:
        climate_df = pd.DataFrame(climate_data)
        return climate_df
    else:
        print("No climate data could be loaded")
        return None

def analyze_climate_trends(climate_df):
    """
    Analyze trends in climate variables
    
    Args:
        climate_df: DataFrame with annual climate variables
        
    Returns:
        Dictionary with trend information
    """
    if climate_df is None or len(climate_df) < 2:
        return None
    
    trends = {}
    years = climate_df['year'].values.reshape(-1, 1)
    
    # Calculate trend for each climate variable
    for column in climate_df.columns:
        if column != 'year' and not climate_df[column].isna().all():
            # Fill missing values with column mean
            values = climate_df[column].fillna(climate_df[column].mean())
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(years, values)
            
            # Calculate trend (slope) and R²
            trend = model.coef_[0]
            r_squared = model.score(years, values)
            
            trends[column] = {
                'trend': trend,
                'r_squared': r_squared,
                'units_per_year': trend
            }
    
    return trends

def plot_climate_variables(climate_df, output_dir='climate_analysis'):
    """
    Create plots for climate variables
    
    Args:
        climate_df: DataFrame with annual climate variables
        output_dir: Directory to save plots
    """
    if climate_df is None or len(climate_df) < 2:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    temp_vars = [col for col in climate_df.columns if 'temp' in col or 'tmin' in col or 'tmax' in col or 'tmean' in col]
    precip_vars = [col for col in climate_df.columns if 'precip' in col or 'rain' in col]
    moisture_vars = [col for col in climate_df.columns if 'soil' in col or 'water' in col or 'deficit' in col]
    
    # Plot temperature variables
    if temp_vars:
        plt.figure(figsize=(12, 6))
        for var in temp_vars:
            plt.plot(climate_df['year'], climate_df[var], 'o-', label=var)
        plt.title('Temperature Trends')
        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'temperature_trends.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot precipitation variables
    if precip_vars:
        plt.figure(figsize=(12, 6))
        for var in precip_vars:
            plt.plot(climate_df['year'], climate_df[var], 'o-', label=var)
        plt.title('Precipitation Trends')
        plt.xlabel('Year')
        plt.ylabel('Precipitation (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'precipitation_trends.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot moisture variables
    if moisture_vars:
        plt.figure(figsize=(12, 6))
        for var in moisture_vars:
            plt.plot(climate_df['year'], climate_df[var], 'o-', label=var)
        plt.title('Soil Moisture and Water Balance Trends')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'moisture_trends.png'), dpi=150, bbox_inches='tight')
        plt.close()