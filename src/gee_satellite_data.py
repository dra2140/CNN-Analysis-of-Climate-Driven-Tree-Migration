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

study_area = ee.Geometry.Rectangle([-70.5, 44.5, -67.5, 47.5])  # Maine 

# administrative boundaries for context
states = ee.FeatureCollection("TIGER/2018/States")
counties = ee.FeatureCollection("TIGER/2018/Counties")

# filter to Maine
maine = states.filter(ee.Filter.eq('NAME', 'Maine'))
maine_counties = counties.filter(ee.Filter.eq('STATEFP', '23'))  # Maine FIPS code

hansen = ee.Image("UMD/hansen/global_forest_change_2021_v1_9")
treecover2000 = hansen.select('treecover2000')
lossyear = hansen.select('lossyear')
gain = hansen.select('gain')

# forest mask (areas with >30% tree cover in 2000)
forest_mask = treecover2000.gt(30)

# define the mathcing time range to Hansen data
start_year = 2000
end_year = 2021

# creating annual Landsat composites 
def create_annual_composite(year, region):
    """Create cloud-free annual composite for a given year and region."""
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)
    
    # filtering satellite based on year
    if year < 2013:
        if year < 2012:  # Landsat 5 
            landsat = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
                .filterDate(start_date, end_date) \
                .filterBounds(region)
        else:  #  Landsat 7
            landsat = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
                .filterDate(start_date, end_date) \
                .filterBounds(region)
    else:
        # Landsat 8
        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate(start_date, end_date) \
                .filterBounds(region)
    
    # filter out cloudy images
    landsat = landsat.filter(ee.Filter.lt('CLOUD_COVER', 50))
    
    # collection is empty check
    count = landsat.size().getInfo()
    if count == 0:
        print(f"No Landsat images available for {year}. Skipping.")
        return None
    
    # scaling and prepping landsat data
    def prepare_landsat(image):
        # scale bands based on Landsat version
        if year < 2013:
            # landsat 5/7:bands are B1, B2, B3, B4, B5, B7
            optical = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']) \
                        .multiply(0.0000275).add(-0.2)  # scaling
            return optical.rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
        else:
            # Landsat 8: bands are B2, B3, B4, B5, B6, B7
            optical = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
                        .multiply(0.0000275).add(-0.2)  # scaling
            return optical.rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    
    landsat_prep = landsat.map(prepare_landsat)
    composite = landsat_prep.median()
    
    ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    
    nbr = composite.normalizedDifference(['nir', 'swir2']).rename('nbr')
    
    # composite with vegetation indices and metadata
    return composite.addBands([ndvi, nbr]) \
                  .set('year', year) \
                  .set('system:time_start', start_date.millis())

# ----------------- GEOGRAPHIC MAPPING CODE -----------------

Map = geemap.Map()
Map.centerObject(study_area, 8)

# add administrative boundaries for geographic context
Map.addLayer(maine.style(**{'color': 'black', 'fillColor': '00000000'}), {}, 'Maine State Boundary')
Map.addLayer(maine_counties.style(**{'color': 'gray', 'fillColor': '00000000'}), {}, 'Maine Counties')

# add Hansen layers
Map.addLayer(treecover2000.updateMask(treecover2000.gt(30)), 
             {'palette': ['#96ED89', '#0A970F', '#074B06'], 'min': 30, 'max': 100}, 
             'Forest Cover 2000')

# color gradient for loss year visualization
loss_palette = ['#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#55FF00', '#00FF00', '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF', '#AA00FF', '#FF00FF', '#FF00AA', 
                '#FF0055', '#8B008B', '#4B0082']

Map.addLayer(lossyear.updateMask(lossyear), 
             {'palette': loss_palette, 'min': 1, 'max': 19}, 
             'Forest Loss Year')

Map.addLayer(gain.updateMask(gain), 
             {'palette': ['blue']}, 
             'Forest Gain 2000-2012')

# Landsat composite from 2000 (beginning)
composite_2000 = create_annual_composite(2000, study_area)
Map.addLayer(composite_2000, 
             {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
             'Landsat 2000', False)  # Initially turned off

# Landsat composite from 2019 (end)
composite_2019 = create_annual_composite(2019, study_area)
Map.addLayer(composite_2019, 
             {'bands': ['swir1', 'nir', 'red'], 'min': 0, 'max': 0.3}, 
             'Landsat 2019', False)  # Initially turned off

# NDVI for vegetation analysis
Map.addLayer(composite_2000.select('ndvi').updateMask(forest_mask), 
             {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
             'NDVI 2000', False)

Map.addLayer(composite_2019.select('ndvi').updateMask(forest_mask), 
             {'palette': ['brown', 'yellow', 'green'], 'min': 0.2, 'max': 0.9}, 
             'NDVI 2019', False)

years = [str(year) for year in range(2001, 2020)] # time slider
loss_images = []

for year in years:
    # creating a visualization for each years loss
    year_code = int(year) - 2000  
    this_year_loss = lossyear.eq(year_code).updateMask(lossyear.eq(year_code))
    
    # vis params
    vis_params = {
        'min': 0,
        'max': 1,
        'palette': ['FF0000']
    }
    rgb_vis = this_year_loss.visualize(**vis_params)
    loss_images.append(rgb_vis)

# adding time series to the map
Map.add_time_slider(
    ee.ImageCollection.fromImages(loss_images),
    vis_params={'min': 0, 'max': 255},
    labels=years,
    time_interval=1000 
)

Map.add_draw_control()

# export the time series as GeoTIFFs
for year in range(start_year, end_year + 1):
    year_composite = create_annual_composite(year, study_area)
    
    # Add the loss information for this year (if after 2000)
    if year > 2000:
        this_year_loss = lossyear.eq(year - 2000).toFloat()
        year_composite = year_composite.addBands(this_year_loss.rename('loss_this_year'))

    task = ee.batch.Export.image.toDrive(
        image=year_composite.toFloat(),  
        description=f'landsat_composite_{year}',
        folder='tree_migration_project',
        scale=30,
        region=study_area,
        maxPixels=1e13,
        crs='EPSG:4326'  # ensure consistent geographic mapping (WGS84)
    )
    task.start()
    print(f"Started export task for Landsat {year}")

# Export the Hansen dataset for the region
task = ee.batch.Export.image.toDrive(
    image=hansen.select(['treecover2000', 'lossyear', 'gain']).toFloat(),
    description='hansen_forest_data',
    folder='tree_migration_project',
    scale=30,
    region=study_area,
    maxPixels=1e13,
    crs='EPSG:4326'  # ensure consistent geographic mapping (WGS84)
)
task.start()
print("Started export task for Hansen forest data")

Map.add_legend(title="Forest Cover and Change", builtin_legend="NLCD")
Map.scale_control = True
Map.layer_control = True
Map.add_basemap("HYBRID")  
Map.add_inspector(position="bottomright")  # newer api for coordinates
Map.fullscreen_control = True

# save the map aas html
Map.save("forest_change_map.html")
print("Map created successfully! Open forest_change_map.html to view.")