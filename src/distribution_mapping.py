import numpy as np
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm

def generate_species_distributions(model, species_names, data_dir, start_year=2000, end_year=2021):
    """
    Generate species distribution maps for each year in the time series
    
    Args:
        model: Trained CNN model for species classification
        species_names: List of species names
        data_dir: Directory containing Landsat data
        start_year, end_year: Range of years to process
    
    Returns:
        Dictionary of distribution maps by year and species
    """
    # Create output directory for maps
    os.makedirs('src/distribution_maps', exist_ok=True)
    
    # Store distribution maps by year
    distribution_maps = {}
    
    # Process each year
    for year in tqdm(range(start_year, end_year + 1), desc="Generating distribution maps"):
        # Load Landsat data for this year
        filename = f"landsat_composite_{year}-0000000000-0000000000-{str(23-year).zfill(3)}.tif"
        filepath = os.path.join(data_dir, filename)
        
        with rasterio.open(filepath) as src:
            data = src.read()
            meta = src.meta
            height, width = src.height, src.width
            transform = src.transform
        
        # Reshape for prediction
        # We need to extract patches from the image to feed to the model
        # This is simplified - in practice you'd process in batches
        
        num_bands = data.shape[0]
        predictions = np.zeros((len(species_names), height, width), dtype=np.float32)
        
        # Process in blocks to avoid memory issues
        block_size = 1000  # Adjust based on your memory constraints
        
        for i in range(0, height, block_size):
            i_end = min(i + block_size, height)
            for j in range(0, width, block_size):
                j_end = min(j + block_size, width)
                
                # Extract block
                block = data[:, i:i_end, j:j_end]
                
                # Reshape to have one pixel per row, bands as columns
                pixels = block.reshape(num_bands, -1).T
                
                # Skip if all pixels are zero/nodata
                if np.all(pixels == 0) or np.all(np.isnan(pixels)):
                    continue
                
                # Predict
                pred = model.predict(pixels, verbose=0)
                
                # Reshape predictions back to spatial layout
                block_pred = pred.T.reshape(len(species_names), i_end-i, j_end-j)
                
                # Store predictions
                predictions[:, i:i_end, j:j_end] = block_pred
        
        # Store distribution maps for this year
        distribution_maps[year] = predictions
        
        # Save maps as GeoTIFFs
        output_meta = meta.copy()
        output_meta.update({
            'count': len(species_names),
            'dtype': 'float32',
            'nodata': 0
        })
        
        output_path = f"src/distribution_maps/species_distribution_{year}.tif"
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(predictions)
        
        # Create visualization for each species
        for s_idx, species in enumerate(species_names):
            plt.figure(figsize=(10, 8))
            plt.imshow(predictions[s_idx], cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Probability')
            plt.title(f"{species} Distribution - {year}")
            plt.axis('off')
            plt.savefig(f"distribution_maps/{species.replace(' ', '_')}_{year}.png", 
                        bbox_inches='tight', dpi=150)
            plt.close()
    
    return distribution_maps