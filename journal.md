# 02/24/22
- we both met up and disussed project ideas and created github 
- drafted up the abstract 



# 02/24/22 - 03/03/22: DataSet Research and Approach


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
Lang et al. (2022) – CNN-based global tree height mappin