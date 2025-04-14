import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, box
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import re
import glob
from sklearn.model_selection import train_test_split
import time

# Define the species we're analyzing
SPECIES = ['Acer rubrum', 'Abies balsamea', 'Betula alleghaniensis']

# Define the years we have data for
YEARS = range(2018, 2024)  # 2018-2023

# Define all bands including vegetation indices
BANDS = ['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3',
         'NIR', 'NIR2', 'SWIR1', 'SWIR2', 'NDVI', 'NBR', 'NDRE']

def load_tree_data(csv_path):
    """Load and prepare tree observation data."""
    df = pd.read_csv(csv_path)
    
    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['decimalLongitude'], df['decimalLatitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Filter to our species of interest
    species = ['Acer rubrum', 'Abies balsamea', 'Betula alleghaniensis']
    gdf = gdf[gdf['verbatimScientificName'].isin(species)]
    
    print(f"Loaded {len(gdf)} tree observations")
    print("Species distribution:")
    print(gdf['verbatimScientificName'].value_counts())
    
    return gdf

def calculate_vegetation_indices(pixels):
    """Calculate vegetation indices from spectral bands."""
    # Extract bands
    blue = pixels[:, 0]
    green = pixels[:, 1]
    red = pixels[:, 2]
    rededge1 = pixels[:, 3]
    nir = pixels[:, 6]
    swir2 = pixels[:, 9]
    
    # Calculate indices
    ndvi = (nir - red) / (nir + red)
    nbr = (nir - swir2) / (nir + swir2)
    ndre = (nir - rededge1) / (nir + rededge1)
    
    # Stack with original bands
    return np.column_stack([pixels, ndvi, nbr, ndre])

def extract_spectral_signatures(tree_gdf, sentinel_dir):
    """Extract spectral signatures at known tree locations from Sentinel-2 imagery."""
    X = []
    y = []
    metadata = []

    total_observations = 0
    outside_bounds = 0
    invalid_pixels = 0
    valid_samples = 0
    file_access_errors = 0

    # Get all Sentinel-2 files
    sentinel_files = glob.glob(os.path.join(sentinel_dir, 'sentinel2_autumn_*.tif'))
    print(f"Found {len(sentinel_files)} Sentinel-2 files")

    # Create a dictionary to map coordinates to their containing files
    coord_to_file = {}
    
    # First pass: map coordinates to their containing files
    print("Mapping coordinates to files...")
    for sentinel_file in tqdm(sentinel_files, desc="Mapping coordinates"):
        if not os.path.exists(sentinel_file):
            print(f"File does not exist: {sentinel_file}")
            continue
            
        try:
            with rasterio.open(sentinel_file) as src:
                bounds = src.bounds
                # Store the file's bounds and transform
                coord_to_file[sentinel_file] = {
                    'bounds': bounds,
                    'transform': src.transform,
                    'crs': src.crs
                }
        except Exception as e:
            print(f"Error reading {sentinel_file}: {str(e)}")
            file_access_errors += 1
            continue

    print(f"Successfully mapped {len(coord_to_file)} files out of {len(sentinel_files)}")
    print(f"File access errors: {file_access_errors}")

    # Process each tree observation
    print("\nProcessing tree observations...")
    for idx, row in tqdm(tree_gdf.iterrows(), total=len(tree_gdf), desc="Processing trees"):
        total_observations += 1
        species = row['verbatimScientificName']
        lon = row['decimalLongitude']
        lat = row['decimalLatitude']
        
        # Find the containing file
        point = Point(lon, lat)
        containing_file = None
        
        for file_path, file_info in coord_to_file.items():
            bounds = file_info['bounds']
            if (bounds.left <= lon <= bounds.right and 
                bounds.bottom <= lat <= bounds.top):
                containing_file = file_path
                break
        
        if not containing_file:
            outside_bounds += 1
            continue
            
        file_info = coord_to_file[containing_file]
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                with rasterio.open(containing_file) as src:
                    # Transform coordinates to pixel coordinates
                    row_idx, col_idx = rasterio.transform.rowcol(
                        src.transform, lon, lat
                    )
                    
                    if not (0 <= row_idx < src.height and 0 <= col_idx < src.width):
                        outside_bounds += 1
                        break
                    
                    buffer_size = 15
                    valid_pixels = []
                    
                    for i in range(-buffer_size//2, buffer_size//2 + 1):
                        for j in range(-buffer_size//2, buffer_size//2 + 1):
                            r = row_idx + i
                            c = col_idx + j
                            if 0 <= r < src.height and 0 <= c < src.width:
                                pixel_value = src.read()[:, r, c]
                                if not np.all(np.isnan(pixel_value)) and np.sum(pixel_value) > 0:
                                    valid_pixels.append(pixel_value)
                    
                    if valid_pixels:
                        avg_pixel = np.mean(valid_pixels, axis=0)
                        valid_samples += 1
                        X.append(avg_pixel)
                        y.append(species)
                        metadata.append({
                            'species': species,
                            'lon': lon,
                            'lat': lat,
                            'file': os.path.basename(containing_file),
                            'valid_pixels_count': len(valid_pixels)
                        })
                        success = True
                    else:
                        invalid_pixels += 1
                        break
                        
            except Exception as e:
                print(f"Error processing {containing_file} (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    file_access_errors += 1
                time.sleep(1)  # Wait before retrying

    if len(X) == 0:
        print("No valid spectral signatures extracted!")
        print("\nSummary statistics:")
        print(f"Total observations processed: {total_observations}")
        print(f"Points outside image bounds: {outside_bounds} ({outside_bounds/total_observations*100:.1f}%)")
        print(f"Points with invalid pixel values: {invalid_pixels} ({invalid_pixels/total_observations*100:.1f}%)")
        print(f"File access errors: {file_access_errors}")
        print(f"Valid samples extracted: {valid_samples} ({valid_samples/total_observations*100:.1f}%)")
        return None, None, None

    X = np.array(X)
    y = np.array(y)
    metadata_df = pd.DataFrame(metadata)
    
    print(f"\nSummary statistics:")
    print(f"Total observations processed: {total_observations}")
    print(f"Points outside image bounds: {outside_bounds} ({outside_bounds/total_observations*100:.1f}%)")
    print(f"Points with invalid pixel values: {invalid_pixels} ({invalid_pixels/total_observations*100:.1f}%)")
    print(f"File access errors: {file_access_errors}")
    print(f"Valid samples extracted: {valid_samples} ({valid_samples/total_observations*100:.1f}%)")
    
    return X, y, metadata_df

def visualize_spectral_signatures(X, y, output_dir):
    """Visualize spectral signatures for each species."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create band names list for Sentinel-2
    band_names = ['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3',
                  'NIR', 'NIR2', 'SWIR1', 'SWIR2', 'NDVI', 'NBR', 'NDRE']
    
    # Average spectral signature per species
    plt.figure(figsize=(12, 8))
    for species in np.unique(y):
        mask = y == species
        if np.any(mask):
            avg_signature = np.mean(X[mask], axis=0)
            plt.plot(range(len(avg_signature)), avg_signature, label=species)
    
    if X.shape[1] <= len(band_names):
        plt.xticks(range(X.shape[1]), band_names[:X.shape[1]], rotation=45)
    else:
        plt.xlabel('Spectral Band')
    
    plt.ylabel('Reflectance')
    plt.title('Average Spectral Signature by Tree Species (Sentinel-2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_spectral_signatures.png'))
    plt.close()

def main():
    # Load tree data
    tree_gdf = load_tree_data('src/filtered_trees_gbif.csv')
    
    # Extract spectral signatures
    X, y, metadata_df = extract_spectral_signatures(tree_gdf, 'sentinel_data')
    
    if X is not None and y is not None:
        # Save metadata
        metadata_df.to_csv('spectral_signatures_metadata.csv', index=False)
        
        # Visualize spectral signatures
        visualize_spectral_signatures(X, y, 'visualizations')
        
        # Prepare data for model training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
        
        print("\nData processing complete!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Feature shape: {X[0].shape}")

if __name__ == "__main__":
    main() 