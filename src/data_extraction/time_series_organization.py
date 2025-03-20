import os
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd
from glob import glob
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime


# change to your downloaded Hansen data file
hansen_path = "/Users/diaadem/Downloads/CS/ML&Climate/CNN-Analysis-of-Climate-Driven-Tree-Migration/hansen_forest_data.tif"

with rasterio.open(hansen_path) as src:
    hansen_data = src.read()
    hansen_meta = src.meta
    
    treecover2000 = hansen_data[0]  
    lossyear = hansen_data[1]       
    gain = hansen_data[2]           

# forest mask (areas with >30% tree cover in 2000)
forest_mask = (treecover2000 > 30).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(treecover2000, cmap='Greens')
plt.colorbar(label='Tree Cover %')
plt.title('Tree Cover 2000')

plt.subplot(132)
plt.imshow(lossyear, cmap='hot')
plt.colorbar(label='Year of Loss (1-21 = 2001-2021)')
plt.title('Year of Forest Loss')

plt.subplot(133)
plt.imshow(gain, cmap='Blues')
plt.title('Forest Gain 2000-2021')

plt.tight_layout()
plt.show()
