import numpy as np
import pandas as pd
import rasterio
import os
import re
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = 'src/data'  # satellate tifs
species_file = 'src/species_data/all_species.csv'

# Check if file exists
if not os.path.exists(species_file):
    print(f"Error: {species_file} not found!")
    print("Available files in species_data:")
    print(os.listdir('species_data'))
    exit(1)

species_df = pd.read_csv(species_file)

# species_df = species_df[(species_df['year'] >= 2000) & (species_df['year'] <= 2021)]
# print(f"Filtered to {len(species_df)} observations from 2000-2021")

species_names = ['Pinus strobus', 'Abies balsamea', 'Acer rubrum', 'Tsuga canadensis', 'Betula papyrifera', 'Picea abies']
species_dict = {name: i for i, name in enumerate(species_names)}

print("Records by species after filtering:")
print(species_df['species'].value_counts())

def extract_spectral_signatures():
    """Extract spectral signatures at known tree locations from Landsat imagery"""
    X = []
    y = []
    metadata = [] 

    total_observations = 0
    outside_bounds = 0
    invalid_pixels = 0
    valid_samples = 0
    
    landsat_files = glob.glob(os.path.join(data_dir, 'landsat_composite_*.tif'))
    print(f"Found {len(landsat_files)} Landsat files")
    
    if len(landsat_files) == 0:
        print("Available files in data directory:")
        print(os.listdir(data_dir))
        return None, None
    
    # print("\nChecking Landsat file bounds:")
    # for landsat_file in landsat_files[:3]:  # Check first 3 files
    #     try:
    #         with rasterio.open(landsat_file) as src:
    #             bounds = src.bounds
    #             print(f"File: {os.path.basename(landsat_file)}")
    #             print(f"  Bounds: {bounds} (left, bottom, right, top)")
    #             print(f"  Width: {src.width}, Height: {src.height}")
    #             print(f"  CRS: {src.crs}")
    #     except Exception as e:
    #         print(f"Error reading {landsat_file}: {str(e)}")

    # # Also add this to check a few sample points
    # print("\nChecking sample coordinates:")
    # sample_species = species_df.head(5)
    # for _, row in sample_species.iterrows():
    #     print(f"Species: {row['species']}, Lon: {row['decimalLongitude']}, Lat: {row['decimalLatitude']}")


    
    for landsat_file in tqdm(landsat_files, desc="Processing Landsat files"):
        year_match = re.search(r'landsat_composite_(\d{4})', os.path.basename(landsat_file))
        if not year_match:
            print(f"couldn't extract year from {landsat_file}, skipping")
            continue
        year = int(year_match.group(1))
    
        year_species = species_df[species_df['year'] == year]
        
        if len(year_species) == 0:
            print(f"no species observations for year {year}, skipping")
            continue
            
        # Open the Landsat file
        try:
            with rasterio.open(landsat_file) as src:
                # Get metadata
                transform = src.transform
                crs = src.crs
                bands = src.count
                
                print(f"processing {landsat_file} - Year: {year}, Bands: {bands}")
                
                # Read all bands at once for efficiency
                landsat_data = src.read()
                
                # For each species observation in this year
                for _, row in year_species.iterrows():
                    total_observations += 1
                    species = row['species']
                    lon = row['decimalLongitude']
                    lat = row['decimalLatitude']
                    
                    # Convert coordinates to pixel indices
                    # print(f"converting coordinates: Lon={lon}, Lat={lat}")
                    row_idx, col_idx = rasterio.transform.rowcol(transform, lon, lat)
                    # print(f"  Converted to Row={row_idx}, Col={col_idx}")
                    # print(f"  Image dimensions: Height={src.height}, Width={src.width}")
                                    
                    # check if point is within image bounds
                    if not (0 <= row_idx < src.height and 0 <= col_idx < src.width):
                        outside_bounds += 1
                        continue
                    
                    # trying a buffer approach instead of just a single pixel since 90 % pixels were NaN's
                    buffer_size = 5  # 5x5 window around the point
                    valid_pixels = []

                    for i in range(-buffer_size//2, buffer_size//2 + 1):
                        for j in range(-buffer_size//2, buffer_size//2 + 1):
                            r = row_idx + i
                            c = col_idx + j
                            if 0 <= r < src.height and 0 <= c < src.width:
                                pixel_value = landsat_data[:, r, c]
                                if not np.any(np.isnan(pixel_value)) and np.sum(pixel_value) > 0.01: 
                                    valid_pixels.append(pixel_value)

                    if valid_pixels:
                        avg_pixel = np.mean(valid_pixels, axis=0)
                        valid_samples += 1
                        X.append(avg_pixel)
                        y.append(species_dict[species])
                        metadata.append({
                            'species': species,
                            'year': year,
                            'lon': lon,
                            'lat': lat
                        })
                        print(f"  Found {len(valid_pixels)} valid pixels in buffer, using average")
                    else:
                        invalid_pixels += 1
                        print(f"  No valid pixels found in buffer")
                        
                
        except Exception as e:
            print(f"Error processing {landsat_file}: {str(e)}")
            continue
    
    if len(X) == 0:
        print("No valid spectral signatures extracted!")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    pd.DataFrame(metadata).to_csv('species_spectral_metadata.csv', index=False)

    print(f"\nSummary statistics:")
    print(f"total observations processed: {total_observations}")
    if total_observations > 0:
        print(f"Points outside image bounds: {outside_bounds} ({outside_bounds/total_observations*100:.1f}%)")
        print(f"Points with invalid pixel values: {invalid_pixels} ({invalid_pixels/total_observations*100:.1f}%)")
        print(f"Valid samples extracted: {valid_samples} ({valid_samples/total_observations*100:.1f}%)")
    else:
        print("no observations  processed")
    pd.DataFrame(metadata).to_csv('species_spectral_metadata.csv', index=False)

    return X, y


X, y = extract_spectral_signatures()

# if X is None or y is None:
#     print("Failed to extract spectral data. Exiting.")
#     exit(1)

print("\nSamples extracted per species:")
for species_name, idx in species_dict.items():
    count = np.sum(y == idx)
    print(f"  {species_name}: {count} samples")


# problem_classes = [species_name for species_name, idx in species_dict.items() if np.sum(y == idx) == 1]
# if problem_classes:
#     print(f"\nWARNING: These species have only 1 sample: {problem_classes}")

print(f"\nshape X: {X.shape}, shape y: {y.shape}")

print(f"total samples extracted: {len(X)}")

if len(X) == 0:
    print("No valid spectral signatures extracted! Exiting.")
    exit(1)

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

# saving data for model training
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

# basic stats
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Feature shape: {X[0].shape}")
print("Class distribution:")
for species, idx in species_dict.items():
    count = np.sum(y == idx)
    print(f"  {species}: {count} samples ({count/len(y)*100:.1f}%)")

# avg spectral signature per species
plt.figure(figsize=(12, 8))
for species, idx in species_dict.items():
    mask = y == idx
    if np.any(mask):
        avg_signature = np.mean(X[mask], axis=0)
        plt.plot(range(len(avg_signature)), avg_signature, label=species)
plt.xlabel('Spectral Band')
plt.ylabel('Reflectance')
plt.title('Average Spectral Signature by Tree Species')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('average_spectral_signatures.png')
plt.show()