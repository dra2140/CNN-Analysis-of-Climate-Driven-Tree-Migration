import ee
import os
import time

try:
    ee.Initialize(project='ee-treemig') 
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ee-treemig')

study_area = ee.Geometry.Rectangle([-70.5, 44.5, -67.5, 47.5]) # Maine 

# setting same time range as rest of data
start_year = 2000
end_year = 2021
years = range(start_year, end_year + 1)

# folder for climate data in google drive : https://drive.google.com/drive/folders/1fmgbmX2hNKR8DXfa2Wmc7HPwY53_3i6j?usp=drive_link
output_folder = 'climate_data'

print(f"Processing climate data for {start_year}-{end_year}")

# # getting annual climate summaries from PRISM
# def get_prism_annual_data(year):
#     """Get annual climate data from PRISM for a specific year."""
#     start_date = f'{year}-01-01'
#     end_date = f'{year}-12-31'
    
#     # precipitation (mm)
#     prism_precip = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
#                     .filterDate(start_date, end_date) \
#                     .select('ppt') \
#                     .sum() \
#                     .rename('annual_precip')
    
#     # mean temperature (°C)
#     prism_tmean = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
#                    .filterDate(start_date, end_date) \
#                    .select('tmean') \
#                    .mean() \
#                    .rename('annual_tmean')
                   
#     # min temperature (°C)
#     prism_tmin = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
#                   .filterDate(start_date, end_date) \
#                   .select('tmin') \
#                   .mean() \
#                   .rename('annual_tmin')
                  
#     # max temperature (°C)
#     prism_tmax = ee.ImageCollection('OREGONSTATE/PRISM/AN81m') \
#                   .filterDate(start_date, end_date) \
#                   .select('tmax') \
#                   .mean() \
#                   .rename('annual_tmax')
    
#     # Combine into a multi-band image
#     prism_annual = ee.Image.cat([
#         prism_precip, 
#         prism_tmean, 
#         prism_tmin, 
#         prism_tmax
#     ]).set('year', year)
    
#     return prism_annual


# def get_terraclimate_annual_data(year):
#     """Get annual climate data from TerraClimate for a specific year."""
#     start_date = f'{year}-01-01'
#     end_date = f'{year}-12-31'
    
#     # TerraClimate data
#     terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
#                     .filterDate(start_date, end_date) \
#                     .filterBounds(study_area)
    
#     # potential evapotranspiration (mm)
#     pet = terraclimate.select('pet').sum().rename('annual_pet')
    
#     # actual evapotranspiration (mm)
#     aet = terraclimate.select('aet').sum().rename('annual_aet')
    
#     # climatic water deficit (mm)
#     def_mm = terraclimate.select('def').sum().rename('annual_water_deficit')
    
#     # soil moisture (mm)
#     soil = terraclimate.select('soil').mean().rename('annual_soil_moisture')
    
#     # vapor pressure deficit (kPa)
#     vpd = terraclimate.select('vpd').mean().rename('annual_vpd')
    
#     # Combine into a multi-band image
#     terraclimate_annual = ee.Image.cat([
#         pet, 
#         aet, 
#         def_mm, 
#         soil, 
#         vpd
#     ]).set('year', year)
    
#     return terraclimate_annual

# # processing and exporting data for each year
# for year in years:
#     print(f"Processing year {year}...")
    
#     # PRISM data --> higher resolution within US
#     prism_annual = get_prism_annual_data(year)
    
#     # TerraClimate data --> for additional variables
#     terraclimate_annual = get_terraclimate_annual_data(year)
    
#     # export PRISM data (higher resolution)
#     task_prism = ee.batch.Export.image.toDrive(
#         image=prism_annual.toFloat(),  # Ensure consistent data type
#         description=f'prism_climate_{year}',
#         folder=output_folder,
#         region=study_area,
#         scale=800,  # PRISM's native resolution
#         crs='EPSG:4326',
#         maxPixels=1e13
#     )
    
#     # export TerraClimate data
#     task_terraclimate = ee.batch.Export.image.toDrive(
#         image=terraclimate_annual.toFloat(),  # Ensure consistent data type
#         description=f'terraclimate_{year}',
#         folder=output_folder,
#         region=study_area,
#         scale=4000,  # TerraClimate's approximate resolution
#         crs='EPSG:4326',
#         maxPixels=1e13
#     )
    
#     task_prism.start()
#     task_terraclimate.start()
    
#     print(f"Started export tasks for {year}")
    
#     time.sleep(2)

# WorldClim 2.1 climate normals (1991-2020) --> 30 years of the averages for reference 
worldclim = ee.ImageCollection("WORLDCLIM/V2/MONTHLY") \
             .filter(ee.Filter.date('1970-01-01', '2000-12-31')) \
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
    region=study_area,ß
    scale=1000,  # resolution
    crs='EPSG:4326',
    maxPixels=1e13
)

task_worldclim.start()