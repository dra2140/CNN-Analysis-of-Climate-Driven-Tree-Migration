import ee
import geemap

ee.Initialize()

aoi = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])  # SF area

# acessing Sentinel-2 data, filter by date and cloud cover
sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate('2020-01-01', '2020-12-31') \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# NDVI (useful for vegetation analysis)
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

sentinel = sentinel.map(addNDVI)

# Create a composite image (median values)
composite = sentinel.median()

# Visualize with geemap
Map = geemap.Map()
Map.centerObject(aoi, 10)
Map.addLayer(composite, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'RGB')
Map.addLayer(composite, {'bands': ['NDVI'], 'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDVI')
Map