import numpy as np
import pandas as pd
import rasterio
import os
import re
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm 
import matplotlib.pyplot as plt

data_dir = 'src/data'  # directory containing Sentinel-2 TIFs
species_file = 'src/species_data/all_species.csv'

# Check if file exists
if not os.path.exists(species_file):
    print(f"Error: {species_file} not found!")
    print("Available files in species_data:")
    print(os.listdir('species_data'))
    exit(1)

species_df = pd.read_csv(species_file)

# Define target species
species_names = ['Pinus strobus', 'Abies balsamea', 'Acer rubrum', 'Tsuga canadensis', 'Betula papyrifera', 'Picea abies']
species_dict = {name: i for i, name in enumerate(species_names)}

# Print sample coordinates
print("Sample tree coordinates:")
for i, row in species_df.head(10).iterrows():
    print(f"  {row['species']}: Lon={row['decimalLongitude']}, Lat={row['decimalLatitude']}")

print("Records by species after filtering:")
print(species_df['species'].value_counts())

def extract_spectral_signatures():
    """Extract spectral signatures at known tree locations from Sentinel-2 imagery"""
    X = []
    y = []
    metadata = [] 

    total_observations = 0
    outside_bounds = 0
    invalid_pixels = 0
    valid_samples = 0
    
    # Look for Sentinel-2 files instead of Landsat
    sentinel_files = glob.glob(os.path.join(data_dir, 'sentinel2_composite_*.tif'))
    print(f"Found {len(sentinel_files)} Sentinel-2 files")

    for sentinel_file in sentinel_files[:3]:  # Check first 3 files
        with rasterio.open(sentinel_file) as src:
            bounds = src.bounds
            print(f"File: {os.path.basename(sentinel_file)}")
            print(f"  Bounds: {bounds} (left, bottom, right, top)")
    
    if len(sentinel_files) == 0:
        print("Available files in data directory:")
        print(os.listdir(data_dir))
        return None, None
    
    # Process each Sentinel-2 file
    for sentinel_file in tqdm(sentinel_files, desc="Processing Sentinel-2 files"):
        year_match = re.search(r'sentinel2_composite_(\d{4})', os.path.basename(sentinel_file))
        if not year_match:
            print(f"Couldn't extract year from {sentinel_file}, skipping")
            continue
        year = int(year_match.group(1))
    
        # Find all species observations regardless of year to increase sample size
        # This is the "alternate years" approach we discussed
        try:
            with rasterio.open(sentinel_file) as src:
                transform = src.transform
                crs = src.crs
                bands = src.count
                
                print(f"Processing {sentinel_file} - Year: {year}, Bands: {bands}")
                
                # Read all bands at once for efficiency
                sentinel_data = src.read()
                
                # Process all species observations
                for _, row in species_df.iterrows():
                    total_observations += 1
                    species = row['species']
                    lon = row['decimalLongitude']
                    lat = row['decimalLatitude']
                    original_year = row['year']
                    
                    # Convert coordinates to pixel indices
                    row_idx, col_idx = rasterio.transform.rowcol(transform, lon, lat)
                    
                    # Check if point is within image bounds
                    if not (0 <= row_idx < src.height and 0 <= col_idx < src.width):
                        outside_bounds += 1
                        continue
                    
                    # Use buffer approach (larger buffer for higher resolution Sentinel-2)
                    buffer_size = 15  # Increased from 10 to account for Sentinel-2's higher resolution
                    valid_pixels = []

                    for i in range(-buffer_size//2, buffer_size//2 + 1):
                        for j in range(-buffer_size//2, buffer_size//2 + 1):
                            r = row_idx + i
                            c = col_idx + j
                            if 0 <= r < src.height and 0 <= c < src.width:
                                pixel_value = sentinel_data[:, r, c]
                                # Relaxed validation criteria
                                if not np.all(np.isnan(pixel_value)) and np.sum(pixel_value) > 0:
                                    valid_pixels.append(pixel_value)

                    if valid_pixels:
                        avg_pixel = np.mean(valid_pixels, axis=0)
                        valid_samples += 1
                        X.append(avg_pixel)
                        y.append(species_dict[species])
                        metadata.append({
                            'species': species,
                            'original_year': original_year,
                            'sentinel_year': year,
                            'lon': lon,
                            'lat': lat,
                            'valid_pixels_count': len(valid_pixels)
                        })
                        print(f"  Found {len(valid_pixels)} valid pixels in buffer for {species}, using average")
                    else:
                        invalid_pixels += 1
                        print(f"  No valid pixels found in buffer for {species}")
        except Exception as e:
            print(f"Error processing {sentinel_file}: {str(e)}")
            continue
    
    if len(X) == 0:
        print("No valid spectral signatures extracted!")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    # Save metadata with more details
    pd.DataFrame(metadata).to_csv('sentinel_species_spectral_metadata.csv', index=False)

    print(f"\nSummary statistics:")
    print(f"Total observations processed: {total_observations}")
    if total_observations > 0:
        print(f"Points outside image bounds: {outside_bounds} ({outside_bounds/total_observations*100:.1f}%)")
        print(f"Points with invalid pixel values: {invalid_pixels} ({invalid_pixels/total_observations*100:.1f}%)")
        print(f"Valid samples extracted: {valid_samples} ({valid_samples/total_observations*100:.1f}%)")
    else:
        print("No observations processed")

    return X, y


X, y = extract_spectral_signatures()

if X is None or y is None:
    print("Failed to extract spectral data. Exiting.")
    exit(1)

print("\nSamples extracted per species:")
for species_name, idx in species_dict.items():
    count = np.sum(y == idx)
    print(f"  {species_name}: {count} samples")

print(f"\nShape X: {X.shape}, Shape y: {y.shape}")
print(f"Total samples extracted: {len(X)}")

# Check if we can use stratification
class_counts = np.bincount(y)
min_samples_per_class = np.min(class_counts[class_counts > 0])
print(f"Minimum samples for any class: {min_samples_per_class}")

if min_samples_per_class < 2:
    print("WARNING: Some classes have fewer than 2 samples, using regular train_test_split without stratification")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# Save data for model training with sentinel-specific names
np.save('sentinel_X_train.npy', X_train)
np.save('sentinel_X_val.npy', X_val)
np.save('sentinel_y_train.npy', y_train)
np.save('sentinel_y_val.npy', y_val)

# Basic stats
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Feature shape: {X[0].shape}")
print("Class distribution:")
for species, idx in species_dict.items():
    count = np.sum(y == idx)
    print(f"  {species}: {count} samples ({count/len(y)*100:.1f}%)")

# Create band names list for Sentinel-2
band_names = ['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 
              'NIR', 'NIR2', 'SWIR1', 'SWIR2', 'NDVI', 'NBR', 'NDRE']

# Average spectral signature per species
plt.figure(figsize=(12, 8))
for species, idx in species_dict.items():
    mask = y == idx
    if np.any(mask):
        avg_signature = np.mean(X[mask], axis=0)
        plt.plot(range(len(avg_signature)), avg_signature, label=species)

# Add band names on x-axis if they match the feature dimension
if X.shape[1] <= len(band_names):
    plt.xticks(range(X.shape[1]), band_names[:X.shape[1]], rotation=45)
else:
    plt.xlabel('Spectral Band')
    
plt.ylabel('Reflectance')
plt.title('Average Spectral Signature by Tree Species (Sentinel-2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sentinel_average_spectral_signatures.png')
plt.show()