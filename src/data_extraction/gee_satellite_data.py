# import ee
# import geemap
# import os
# import folium
# from folium import plugins
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# try:
#     ee.Initialize(project='ee-treemig')
# except Exception as e:
#     ee.Authenticate()
#     ee.Initialize(project='ee-treemig')

# study_area = ee.Geometry.Rectangle([-70.5, 44.5, -67.5, 47.5])  # Maine 

# # administrative boundaries for context
# states = ee.FeatureCollection("TIGER/2018/States")
# counties = ee.FeatureCollection("TIGER/2018/Counties")

# # filter to Maine
# maine = states.filter(ee.Filter.eq('NAME', 'Maine'))
# maine_counties = counties.filter(ee.Filter.eq('STATEFP', '23'))  # Maine FIPS code

# hansen = ee.Image("UMD/hansen/global_forest_change_2021_v1_9")
# treecover2000 = hansen.select('treecover2000')
# lossyear = hansen.select('lossyear')
# gain = hansen.select('gain')

# # forest mask (areas with >30% tree cover in 2000)
# forest_mask = treecover2000.gt(30)

# # define the mathcing time range to Hansen data
# start_year = 2000
# end_year = 2021

# # creating annual Landsat composites 
# def create_annual_composite(year, region):
#     """Create cloud-free annual composite for a given year and region."""
#     start_date = ee.Date.fromYMD(year, 1, 1)
#     end_date = ee.Date.fromYMD(year, 12, 31)
    
#     # filtering satellite based on year
#     if year < 2013:
#         if year < 2012:  # Landsat 5 
#             landsat = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
#                 .filterDate(start_date, end_date) \
#                 .filterBounds(region)
#         else:  #  Landsat 7
#             landsat = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
#                 .filterDate(start_date, end_date) \
#                 .filterBounds(region)
#     else:
#         # Landsat 8
#         landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
#                 .filterDate(start_date, end_date) \
#                 .filterBounds(region)
    
#     # filter out cloudy images
#     landsat = landsat.filter(ee.Filter.lt('CLOUD_COVER', 50))
    
#     # collection is empty check
#     count = landsat.size().getInfo()
#     if count == 0:
#         print(f"No Landsat images available for {year}. Skipping.")
#         return None
    
#     # scaling and prepping landsat data
#     def prepare_landsat(image):
#         # scale bands based on Landsat version
#         if year < 2013:
#             # landsat 5/7:bands are B1, B2, B3, B4, B5, B7
#             optical = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']) \
#                         .multiply(0.0000275).add(-0.2)  # scaling
#             return optical.rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
#         else:
#             # Landsat 8: bands are B2, B3, B4, B5, B6, B7
#             optical = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
#                         .multiply(0.0000275).add(-0.2)  # scaling
#             return optical.rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    
#     landsat_prep = landsat.map(prepare_landsat)
#     composite = landsat_prep.median()
    
#     ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    
#     nbr = composite.normalizedDifference(['nir', 'swir2']).rename('nbr')
    
#     # composite with vegetation indices and metadata
#     return composite.addBands([ndvi, nbr]) \
#                   .set('year', year) \
#                   .set('system:time_start', start_date.millis())

# # ----------------- GEOGRAPHIC MAPPING CODE -----------------

# Map = geemap.Map()
# Map.centerObject(study_area, 8)

# # add administrative boundaries for geographic context
# Map.addLayer(maine.style(**{'color': 'black', 'fillColor': '00000000'}), {}, 'Maine State Boundary')
# Map.addLayer(maine_counties.style(**{'color': 'gray', 'fillColor': '00000000'}), {}, 'Maine Counties')

# # add Hansen layers
# Map.addLayer(treecover2000.updateMask(treecover2000.gt(30)), 
#              {'palette': ['#96ED89', '#0A970F', '#074B06'], 'min': 30, 'max': 100}, 
#              'Forest Cover 2000')

# # color gradient for loss year visualization
# loss_palette = ['#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#55FF00', '#00FF00', '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF', '#AA00FF', '#FF00FF', '#FF00AA', 
#                 '#FF0055', '#8B008B', '#4B0082']

# Map.addLayer(lossyear.updateMask(lossyear), 
#              {'palette': loss_palette, 'min': 1, 'max': 19}, 
#              'Forest Loss Year')

# Map.addLayer(gain.updateMask(gain), 
#              {'palette': ['blue']}, 
#              'Forest Gain 2000-2012')

# # Landsat composite from 2000 (beginning)
# composite_2000 = create_annual_composite(2000, study_area)
# Map.addLayer(composite_2000, 
#              {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
#              'Landsat 2000', False)  # Initially turned off

# # Landsat composite from 2019 (end)
# composite_2019 = create_annual_composite(2019, study_area)
# Map.addLayer(composite_2019, 
#              {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
#              'Landsat 2019', False)  # Initially turned off

# # NDVI for vegetation analysis
# Map.addLayer(composite_2000.select('ndvi').updateMask(forest_mask), 
#              {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
#              'NDVI 2000', False)

# Map.addLayer(composite_2019.select('ndvi').updateMask(forest_mask), 
#              {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
#              'NDVI 2019', False)

# years = [str(year) for year in range(2001, 2020)] # time slider
# loss_images = []

# for year in years:
#     # creating a visualization for each years loss
#     year_code = int(year) - 2000  
#     this_year_loss = lossyear.eq(year_code).updateMask(lossyear.eq(year_code))
    
#     # vis params
#     vis_params = {
#         'min': 0,
#         'max': 1,
#         'palette': ['FF0000']
#     }
#     rgb_vis = this_year_loss.visualize(**vis_params)
#     loss_images.append(rgb_vis)

# # adding time series to the map
# Map.add_time_slider(
#     ee.ImageCollection.fromImages(loss_images),
#     vis_params={'min': 0, 'max': 255},
#     labels=years,
#     time_interval=1000 
# )

# Map.add_draw_control()

# # export the time series as GeoTIFFs
# for year in range(start_year, end_year + 1):
#     year_composite = create_annual_composite(year, study_area)
    
#     # Add the loss information for this year (if after 2000)
#     if year > 2000:
#         this_year_loss = lossyear.eq(year - 2000).toFloat()
#         year_composite = year_composite.addBands(this_year_loss.rename('loss_this_year'))

#     task = ee.batch.Export.image.toDrive(
#         image=year_composite.toFloat(),  
#         description=f'landsat_composite_{year}',
#         folder='tree_migration_project',
#         scale=30,
#         region=study_area,
#         maxPixels=1e13,
#         crs='EPSG:4326'  # ensure consistent geographic mapping (WGS84)
#     )
#     task.start()
#     print(f"Started export task for Landsat {year}")

# # Export the Hansen dataset for the region
# task = ee.batch.Export.image.toDrive(
#     image=hansen.select(['treecover2000', 'lossyear', 'gain']).toFloat(),
#     description='hansen_forest_data',
#     folder='tree_migration_project',
#     scale=30,
#     region=study_area,
#     maxPixels=1e13,
#     crs='EPSG:4326'  # ensure consistent geographic mapping (WGS84)
# )
# task.start()
# print("Started export task for Hansen forest data")

# Map.add_legend(title="Forest Cover and Change", builtin_legend="NLCD")
# Map.scale_control = True
# Map.layer_control = True
# Map.add_basemap("HYBRID")  
# Map.add_inspector(position="bottomright")  # newer api for coordinates
# Map.fullscreen_control = True

# # save the map aas html
# Map.save("forest_change_map.html")
# print("Map created successfully! Open forest_change_map.html to view.")


import ee
import geemap
import os
import folium
from folium import plugins
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    ee.Initialize(project='ee-treemig')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ee-treemig')

# Define the study area polygon coordinates
polygon_coords = [
    [-69.25169, 47.44281],
    [-75.44628, 45.32212],
    [-78.86807, 43.00021],
    [-75.47091, 38.54031],
    [-71.10597, 40.19079],
    [-66.62251, 44.04165],
    [-65.84348, 47.51913],
    [-69.25169, 47.44281]
]

# Create Earth Engine geometry
study_area = ee.Geometry.Polygon([polygon_coords])

# Administrative boundaries for context
states = ee.FeatureCollection("TIGER/2018/States")
counties = ee.FeatureCollection("TIGER/2018/Counties")

# Filter to Maine
maine = states.filter(ee.Filter.eq('NAME', 'Maine'))
maine_counties = counties.filter(ee.Filter.eq('STATEFP', '23'))  # Maine FIPS code

# Using the newer version of Hansen dataset
hansen = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")
treecover2000 = hansen.select('treecover2000')
lossyear = hansen.select('lossyear')
gain = hansen.select('gain')

# Forest mask (areas with >30% tree cover in 2000)
forest_mask = treecover2000.gt(30)

# Define the time range for autumn data collection
# Note: Sentinel-2 data is available from 2015 onwards
start_year = 2015
end_year = 2023  # Most recent full year

def create_autumn_composite(year, region):
    """Create cloud-free autumn composite for a given year and region using Sentinel-2."""
    # Define autumn months (September and October)
    autumn_start = ee.Date.fromYMD(year, 9, 1)  # September 1
    autumn_end = ee.Date.fromYMD(year, 10, 31)  # October 31
    
    # Get Sentinel-2 Surface Reflectance collection
    sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
              .filterDate(autumn_start, autumn_end) \
              .filterBounds(region) \
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # Accept up to 50% cloud cover
    
    # Check if collection is empty
    count = sentinel.size().getInfo()
    if count == 0:
        print(f"No Sentinel-2 images available for autumn {year}. Skipping.")
        return None
    else:
        print(f"Found {count} Sentinel-2 images for autumn {year}")
    
    # Function to mask clouds using the SCL band
    def maskClouds(image):
        # Get the SCL band
        scl = image.select('SCL')
        # Mask pixels classified as clouds, cloud shadows, or saturated/defective
        mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8)).And(scl.neq(10))
        return image.updateMask(mask)
    
    # Function to prepare Sentinel-2 data
    def prepare_sentinel(image):
        # First apply cloud masking
        image = maskClouds(image)
        
        # Select the bands we want
        # B2=blue, B3=green, B4=red, B8=NIR, B11,B12=SWIR
        # Also include red edge bands which are unique to Sentinel-2
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        optical = image.select(bands)
        
        # Apply scaling
        optical = optical.divide(10000)  # Sentinel-2 data is scaled by 10000
        
        # Rename bands for consistency
        return optical.rename(['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir2', 'swir1', 'swir2'])
    
    sentinel_prep = sentinel.map(prepare_sentinel)
    
    # Create median composite
    composite = sentinel_prep.median()
    
    # Calculate vegetation indices
    ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    nbr = composite.normalizedDifference(['nir', 'swir2']).rename('nbr')
    
    # Add red edge NDVI (unique to Sentinel-2)
    ndre = composite.normalizedDifference(['nir', 'rededge1']).rename('ndre')
    
    # Composite with vegetation indices and metadata
    return composite.addBands([ndvi, nbr, ndre]) \
                  .set('year', year) \
                  .set('season', 'autumn') \
                  .set('system:time_start', autumn_start.millis())

# ----------------- GEOGRAPHIC MAPPING CODE -----------------

Map = geemap.Map()
Map.centerObject(study_area, 6)

# Add the study area polygon
Map.addLayer(study_area, {'color': 'red'}, 'Study Area')

# Add administrative boundaries for geographic context
Map.addLayer(maine.style(**{'color': 'black', 'fillColor': '00000000'}), {}, 'Maine State Boundary')
Map.addLayer(maine_counties.style(**{'color': 'gray', 'fillColor': '00000000'}), {}, 'Maine Counties')

# Add Hansen layers
Map.addLayer(treecover2000.updateMask(treecover2000.gt(30)), 
             {'palette': ['#96ED89', '#0A970F', '#074B06'], 'min': 30, 'max': 100}, 
             'Forest Cover 2000')

# Color gradient for loss year visualization
loss_palette = ['#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#55FF00', '#00FF00', '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF', '#AA00FF', '#FF00FF', '#FF00AA', 
                '#FF0055', '#8B008B', '#4B0082']

Map.addLayer(lossyear.updateMask(lossyear), 
             {'palette': loss_palette, 'min': 1, 'max': 22}, 
             'Forest Loss Year')

Map.addLayer(gain.updateMask(gain), 
             {'palette': ['blue']}, 
             'Forest Gain 2000-2022')

# Create and add first year Sentinel-2 composite
composite_first = create_autumn_composite(start_year, study_area)
if composite_first is not None:
    Map.addLayer(composite_first, 
                {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
                f'Sentinel-2 {start_year}', False)
    
    # Add NDVI layers for vegetation analysis
    Map.addLayer(composite_first.select('ndvi').updateMask(forest_mask), 
                {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
                f'NDVI {start_year}', False)

# Create and add most recent Sentinel-2 composite
composite_last = create_autumn_composite(end_year, study_area)
if composite_last is not None:
    Map.addLayer(composite_last, 
                {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
                f'Sentinel-2 {end_year}', False)
    
    # Add NDVI layer for vegetation analysis
    Map.addLayer(composite_last.select('ndvi').updateMask(forest_mask), 
                {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
                f'NDVI {end_year}', False)
    
    # Add Red Edge NDVI layer (unique to Sentinel-2)
    Map.addLayer(composite_last.select('ndre').updateMask(forest_mask), 
                {'palette': ['brown', 'yellow', 'green'], 'min': 0.1, 'max': 0.7}, 
                f'Red Edge NDVI {end_year}', False)

# Create time slider for forest loss years
years = [str(year) for year in range(2001, end_year)]
loss_images = []

for year in years:
    year_code = int(year) - 2000  
    this_year_loss = lossyear.eq(year_code).updateMask(lossyear.eq(year_code))
    
    vis_params = {
        'min': 0,
        'max': 1,
        'palette': ['FF0000']
    }
    rgb_vis = this_year_loss.visualize(**vis_params)
    loss_images.append(rgb_vis)

Map.add_time_slider(
    ee.ImageCollection.fromImages(loss_images),
    vis_params={'min': 0, 'max': 255},
    labels=years,
    time_interval=1000 
)

Map.add_draw_control()

# Export autumn composites to Google Drive
for year in range(start_year, end_year + 1):
    year_composite = create_autumn_composite(year, study_area)
    
    if year_composite is None:
        continue
        
    # Add the loss information for this year (if after 2000)
    if year > 2000:
        this_year_loss = lossyear.eq(year - 2000).toFloat()
        year_composite = year_composite.addBands(this_year_loss.rename('loss_this_year'))

    task = ee.batch.Export.image.toDrive(
        image=year_composite.toFloat(),  
        description=f'sentinel2_autumn_{year}',
        folder='tree_migration_project',
        scale=10,  # Sentinel-2 has 10m resolution for RGB and NIR bands
        region=study_area,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    task.start()
    print(f"Started export task for autumn {year}")

# Export the Hansen dataset for the region
task = ee.batch.Export.image.toDrive(
    image=hansen.select(['treecover2000', 'lossyear', 'gain']).toFloat(),
    description='hansen_forest_data',
    folder='tree_migration_project',
    scale=30,
    region=study_area,
    maxPixels=1e13,
    crs='EPSG:4326'
)
task.start()
print("Started export task for Hansen forest data")

# Print information about tree species data collection
print("\nTree Species Data Collection Guidelines:")
print("----------------------------------------")
print("For optimal temporal alignment with Sentinel-2 autumn data:")
print("1. Collect tree species data during the same autumn period (September-October)")
print("2. Focus on the following years: 2015-2023")
print("3. For each year, try to collect data as close as possible to the Sentinel-2 image dates")
print("4. Record the exact date of each tree species observation")
print("5. Note the phenological state of the trees (e.g., leaf color, senescence stage)")
print("\nRecommended data collection dates for each year:")
for year in range(start_year, end_year + 1):
    print(f"{year}: September 1 - October 31")

Map.add_legend(title="Forest Cover and Change", builtin_legend="NLCD")
Map.scale_control = True
Map.layer_control = True
Map.add_basemap("HYBRID")
Map.add_inspector(position="bottomright")
Map.fullscreen_control = True

Map.save("sentinel_forest_map.html")
print("Map created successfully! Open sentinel_forest_map.html to view.")