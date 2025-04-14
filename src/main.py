import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tree_migration_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TreeMigration")

# Import custom modules
# This assumes you've saved the previously provided code as Python modules
from cnn_model import build_spectral_cnn, train_species_classifier
from distribution_mapping import generate_species_distributions
from migration_analysis import calculate_range_shifts, correlate_climate_factors
# Add to imports in main.py
from climate_data_loading import load_climate_data, analyze_climate_trends, plot_climate_variables

def main():
    """Main execution function for the tree migration analysis pipeline"""
    start_time = time.time()
    logger.info("Starting Tree Migration Analysis Pipeline")
    
    # Configuration parameters
    data_dir = "src/data"  # Directory containing Landsat and climate data
    species_file = "src/species_data/all_species.csv"
    start_year = 2000
    end_year = 2021
    output_dir = "results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load species data
    logger.info("Loading species data...")
    try:
        species_df = pd.read_csv(species_file)
        species_names = species_df["species"].unique()
        logger.info(f"Found {len(species_names)} tree species: {', '.join(species_names)}")
    except Exception as e:
        logger.error(f"Error loading species data: {e}")
        return
    
    # 2. Create training data for species classification
    logger.info("Creating training data for species classifier...")
    try:
        X_train, X_val, y_train, y_val, species_dict = create_training_data(
            species_df, data_dir, start_year, end_year
        )
        logger.info(f"Created training dataset with {len(X_train)} samples and {len(X_val)} validation samples")
    except Exception as e:
        logger.error(f"Error creating training data: {e}")
        return
    
    # 3. Train CNN model for species classification
    logger.info("Training species classification model...")
    try:
        model, history = train_species_classifier(X_train, y_train, X_val, y_val, list(species_dict.keys()))
        logger.info(f"Model training complete. Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Save model
        model.save(os.path.join(output_dir, "tree_species_model.h5"))
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return
    
    # 4. Generate species distribution maps for each year
    logger.info("Generating species distribution maps for all years...")
    try:
        distribution_maps = generate_species_distributions(
            model, list(species_dict.keys()), data_dir, start_year, end_year
        )
        logger.info(f"Generated distribution maps for {end_year - start_year + 1} years")
    except Exception as e:
        logger.error(f"Error generating distribution maps: {e}")
        return
    
    # 5. Analyze migration patterns
    logger.info("Analyzing migration patterns...")
    try:
        migration_df = calculate_range_shifts(
            distribution_maps, list(species_dict.keys()), start_year, end_year
        )
        logger.info("Migration analysis complete")
        
        # Print summary of findings
        for _, row in migration_df.iterrows():
            logger.info(f"{row['species']}: Moving {row['avg_annual_movement_km']:.2f} km/year in {row['direction']} direction")
    except Exception as e:
        logger.error(f"Error analyzing migration patterns: {e}")
        return
    
    # 6. Correlate with climate factors
    logger.info("Analyzing climate correlations...")
    try:
        correlate_climate_factors(migration_df, list(species_dict.keys()), data_dir, start_year, end_year)
        logger.info("Climate correlation analysis complete")
    except Exception as e:
        logger.error(f"Error analyzing climate correlations: {e}")
    
    # Calculate total runtime
    runtime = (time.time() - start_time) / 60  # minutes
    logger.info(f"Pipeline completed in {runtime:.1f} minutes")


def create_training_data(species_df, data_dir, start_year, end_year):
    """
    Create training data for species classification from occurrence points
    
    Args:
        species_df: DataFrame with species occurrence data
        data_dir: Directory containing Landsat data
        start_year, end_year: Range of years to consider
    
    Returns:
        X_train, X_val, y_train, y_val, species_dict
    """
    # Convert species points to GeoDataFrame
    geometry = [Point(xy) for xy in zip(species_df["decimalLongitude"], species_df["decimalLatitude"])]
    species_gdf = gpd.GeoDataFrame(species_df, geometry=geometry, crs="EPSG:4326")
    
    # Get unique species
    species_names = species_df["species"].unique()
    species_dict = {name: i for i, name in enumerate(species_names)}
    
    # Training data will come from coordinates where species are known
    X = []  # Spectral data
    y = []  # Species labels
    
    # We'll use data from multiple years to improve robustness
    # Using a subset of years for efficiency
    training_years = list(range(start_year, end_year+1, 3))  # Every 3rd year
    
    for year in training_years:
        logger.info(f"Processing training data from year {year}...")
        
        # Load Landsat data
        landsat_file = find_landsat_file(data_dir, year)
        if not landsat_file:
            logger.warning(f"No Landsat data found for year {year}, skipping")
            continue
            
        with rasterio.open(landsat_file) as src:
            landsat = src.read()
            transform = src.transform
        
        # For each species occurrence point
        for species_name in species_names:
            species_points = species_gdf[species_gdf["species"] == species_name]
            
            # Sample a subset of points for each species (for balance)
            max_points = 100  # Adjust based on your data
            if len(species_points) > max_points:
                species_points = species_points.sample(max_points)
            
            for idx, row in species_points.iterrows():
                point = row["geometry"]
                
                # Convert point to raster coordinates
                row_idx, col_idx = rasterio.transform.rowcol(transform, point.x, point.y)
                
                # Check if point is within image bounds
                if (0 <= row_idx < landsat.shape[1] and 0 <= col_idx < landsat.shape[2]):
                    # Extract spectral data at this point (all bands)
                    # Landsat has bands in first dimension
                    spectral_data = landsat[:, row_idx, col_idx]
                    
                    # Skip if no valid data
                    if np.all(spectral_data == 0) or np.any(np.isnan(spectral_data)):
                        continue
                    
                    X.append(spectral_data)
                    y.append(species_dict[species_name])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created dataset with {len(X)} samples across {len(species_names)} species")
    
    # Handle class imbalance if needed
    class_counts = np.bincount(y)
    logger.info(f"Class distribution: {class_counts}")
    
    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, species_dict


def find_landsat_file(data_dir, year):
    """Find Landsat file for a specific year"""
    # Match your actual file naming pattern
    filename = f"landsat_composite_{year}-0000000000-0000000000-{str(23-year).zfill(3)}.tif"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    # Fallback with simpler pattern
    alt_filepath = os.path.join(data_dir, f"landsat_composite_{year}.tif")
    if os.path.exists(alt_filepath):
        return alt_filepath
            
    return None


if __name__ == "__main__":
    main()