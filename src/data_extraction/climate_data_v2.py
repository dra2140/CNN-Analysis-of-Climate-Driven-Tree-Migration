import ee
import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

try:
    ee.Initialize(project='ee-treemig') 
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ee-treemig')

# Define the study area polygon coordinates (same as in satellite data script)
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

# Set time range to match satellite data (2018-2023)
start_year = 2018
end_year = 2023
years = range(start_year, end_year + 1)

# Output folder for climate data
output_folder = 'climate_data_2018_2023'

print(f"Processing climate data for {start_year}-{end_year}")

def get_prism_annual_data(year):
    """Get annual climate data from PRISM for a specific year."""
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    # precipitation (mm)
    prism_precip = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
                    .filterDate(start_date, end_date) \
                    .select('ppt') \
                    .sum() \
                    .rename('annual_precip')
    
    # mean temperature (°C)
    prism_tmean = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
                   .filterDate(start_date, end_date) \
                   .select('tmean') \
                   .mean() \
                   .rename('annual_tmean')
                   
    # min temperature (°C)
    prism_tmin = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
                  .filterDate(start_date, end_date) \
                  .select('tmin') \
                  .mean() \
                  .rename('annual_tmin')
                  
    # max temperature (°C)
    prism_tmax = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
                  .filterDate(start_date, end_date) \
                  .select('tmax') \
                  .mean() \
                  .rename('annual_tmax')
    
    # Combine into a multi-band image
    prism_annual = ee.Image.cat([
        prism_precip, 
        prism_tmean, 
        prism_tmin, 
        prism_tmax
    ]).set('year', year)
    
    return prism_annual

def get_terraclimate_annual_data(year):
    """Get annual climate data from TerraClimate for a specific year."""
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    # TerraClimate data
    terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(study_area)
    
    # potential evapotranspiration (mm)
    pet = terraclimate.select('pet').sum().rename('annual_pet')
    
    # actual evapotranspiration (mm)
    aet = terraclimate.select('aet').sum().rename('annual_aet')
    
    # climatic water deficit (mm)
    def_mm = terraclimate.select('def').sum().rename('annual_water_deficit')
    
    # soil moisture (mm)
    soil = terraclimate.select('soil').mean().rename('annual_soil_moisture')
    
    # vapor pressure deficit (kPa)
    vpd = terraclimate.select('vpd').mean().rename('annual_vpd')
    
    # Combine into a multi-band image
    terraclimate_annual = ee.Image.cat([
        pet, 
        aet, 
        def_mm, 
        soil, 
        vpd
    ]).set('year', year)
    
    return terraclimate_annual

# Process and export data for each year
for year in years:
    print(f"Processing year {year}...")
    
    # PRISM data (higher resolution within US)
    prism_annual = get_prism_annual_data(year)
    
    # TerraClimate data (additional variables)
    terraclimate_annual = get_terraclimate_annual_data(year)
    
    # Export PRISM data
    task_prism = ee.batch.Export.image.toDrive(
        image=prism_annual.toFloat(),
        description=f'prism_climate_{year}',
        folder=output_folder,
        region=study_area,
        scale=800,  # PRISM's native resolution
        crs='EPSG:4326',
        maxPixels=1e13
    )
    
    # Export TerraClimate data
    task_terraclimate = ee.batch.Export.image.toDrive(
        image=terraclimate_annual.toFloat(),
        description=f'terraclimate_{year}',
        folder=output_folder,
        region=study_area,
        scale=4000,  # TerraClimate's approximate resolution
        crs='EPSG:4326',
        maxPixels=1e13
    )
    
    task_prism.start()
    task_terraclimate.start()
    
    print(f"Started export tasks for {year}")
    
    time.sleep(2)  # Avoid rate limiting

# Also get WorldClim 2.1 climate normals (1991-2020) for reference
worldclim = ee.ImageCollection("WORLDCLIM/V2/MONTHLY") \
             .filter(ee.Filter.date('1991-01-01', '2020-12-31')) \
             .mean()
worldclim_subset = worldclim.select([
    'tavg',  # avg temp
    'tmin',  # min temp
    'tmax',  # max temp
    'prec'   # precip
])

task_worldclim = ee.batch.Export.image.toDrive(
    image=worldclim_subset.toFloat(),
    description='worldclim_monthly_avg',
    folder=output_folder,
    region=study_area,
    scale=1000,  # resolution
    crs='EPSG:4326',
    maxPixels=1e13
)

task_worldclim.start()
print("Started export task for WorldClim data") 