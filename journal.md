# 02/24/25
- we both met up and disussed project ideas and created github 
- drafted up the abstract 



# 02/24/25 - 03/03/25: DataSet Research and Approach


#### Hybrid architecture:

    CNN
        Use CNN layers to process satellite imagery
        Extract spatial features related to vegetation patterns
        Identify current species distributions

    Temporal/Predictive Component
        Add recurrent layers (LSTM/GRU) to capture temporal patterns
        Or use transformer architecture for sequence modeling
        Incorporate climate variables as additional inputs


## Data Requirements:
* Satellite Imagery (Multi-temporal)
    * Landsat Archive: 
        Source: USGS Earth Explorer or Google Earth Engine (coded something by susing the google earth API to extract the data idk if it'll work though)

*  Forest Cover Change Data
    * Hansen Global Forest Change dataset: forest loss/gain at 30m resolution (2000-2019)
        Provides baseline for identifying areas of change
        Source: [University of Maryland/Google Earth Engine](https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.2.html)

* Tree Species Distribution Data
    *  Forest Inventory and Analysis (FIA) to figure out species presence
        Source: US Forest Service?? Global Biodiversity Info Facility?? State forestry departments for more regional data??

* Climate Data and Climarte Projections 
    * Historical climate records --> use fro the predictive component
        Source: WorldClim??
    * Climate projections for future climate scenarios under different emissions pathways
        Source: CMIP6 models via WorldClim

* Maybe aslo topographical data (soil, elevation , etc)--> idk about this though

## CHALLENGES: 
* main challenge rn is the data propcessing. We need to finalized on an area and then temporally align the satellite observations with tree looss/gain data to create consistent time series. 

## Related Works

https://www.sciencedirect.com/science/article/pii/S0924271620303488

Jeremy Siegrist (2024) - Describes proof of tree migration with climate models, didn’t find original paper that describes the models used, but worth looking into to find out methodology.
https://research.fs.usda.gov/nrs/news/featured/trees-move-scientific-effort-adapt-climate-change
https://research.fs.usda.gov/nrs/products/dataandtools/tools/climate-change-tree-atlas
Priya & Vani (2024) – CNNs for post-wildfire vegetation tracking, no focus on migration.
Jelas et al. (2024) – CNNs for deforestation detection
Zhang et al. (2024) – Analyzes treeline shifts with statistical models

These ones could be good for datasets/methodology
Nezami et al. (2020) – 3D-CNN on hyperspectral + RGB for species ID
Lang et al. (2022) – CNN-based global tree height mapping


# 03/03/25 - 03/10/25: Obtaining and Processing Tree Data and Satellite Data

## Data Collection: 
- Collected data from Northern Maine for now --> tentativley pciked this are because there specific instances where we see evidence of tree migration of the most in the northeast (Maine and Canada):
    * https://emergencemagazine.org/feature/they-carry-us-with-them/
    * https://www.bangordailynews.com/2024/06/17/homestead/homestead-environment/maine-preparing-for-future-without-iconic-pines-joam40zk0w/#:~:text=Researchers%20across%20the%20state%20have,forests%20functioning%20amid%20climate%20change
- Downloaded data with in time frame 2000-2021, as that's whats avaible in the more recently updated Hanset dataset --> may downsample the data later
    

### Lansat Satellite data:
* Configured Google Earth Engine script to extract apprpriate Landsat time series data b/n 2000-202: Landsat 5 for 2000-2011, Landsat 7 for 2012, and Landsat 8 for 2013-2021
* Filtered images with < 50% cloud cover for better image quality and created median composites for each year to minimize seasonal variations and cloud effects --> resulted in 22 annual snapshots spanning 2000-2021 and then we configured batch export tasks to save all 22 annual composites to Google Drive (30m resolution): https://drive.google.com/drive/folders/1hbkrBB_EEtaLryNrM5qSFJAWw4NotvYg?usp=sharing

* Also included key vegetation indices (like NDVI, NBR, but tbh may not factor this in as model inputs as well)--> was thinking it might be helpful for detecting subtler changes in vegetation health/density that might indicate species shifts. We also visualized our data to chekc out the quality and coverage.

### Hanset Forest change data:
* We intergrated the tree gain/loss data from the Hansen Global Forest Change dataset (2000-2021). Created a forest mask identifying regions with >30% tree cover in the baseline year to focus our analysis on relevant forested areas --> filtering out noise 
* ensured spatial adn termporal alignment by using the same study area boundaries and dates, maintained consistent 30m resolution across both datasets and added annual loss bands to each Landsat composite for direct correlation


### Tree species data:
* We specfically chose five climate-sensitive tree species within the Northeast and downloaded their distrubtion in Maine (original tried using the FIA API but didn't work out for reason, so we stuck to the GBIF (Global Biodiversity Information Facility) API for species dist data) :
        Eastern White Pine (Pinus strobus)
        Balsam Fir (Abies balsamea)
        Red Maple (Acer rubrum)
        Eastern Hemlock (Tsuga canadensis)
        Paper Birch (Betula papyrifera)
    Note: These species are specifally noted to be sensitive to climate change and represent a range of ecological niches within the northeast. 

* Got ~300 records for each species --> saved in the csv files for each tree type
* Visualized the species distubution with geopandas 


## Next Steps/Future Roadmap:
* We still are yet to find climate datasets so we can pontentiallu integrate temperature and precipitation variables as additional model inputs for migration prediction
* Start thinking about/bulding CNN model architecture for species classifcation
        training split: 2000-2015 for training and 2016-2021 for validation/testing??
        Input: Landsat spectral bands at species occurrence locations
        Output: Tree species classification (pine, maple, etc.)



