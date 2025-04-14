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

## Related Works (Added by Ben, just transfered it into the jounral from the doc!)

https://www.sciencedirect.com/science/article/pii/S0924271620303488

Jeremy Siegrist (2024) - Describes proof of tree migration with climate models, didn't find original paper that describes the models used, but worth looking into to find out methodology.
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
- Collected data from Northern Maine for now --> tentativley picked this are because there specific instances where we see evidence of tree migration of the most in the northeast (Maine and Canada):
    * https://emergencemagazine.org/feature/they-carry-us-with-them/
    * https://www.bangordailynews.com/2024/06/17/homestead/homestead-environment/maine-preparing-for-future-without-iconic-pines-joam40zk0w/#:~:text=Researchers%20across%20the%20state%20have,forests%20functioning%20amid%20climate%20change
- Downloaded data with in time frame 2000-2021, as that's whats avaible in the more recently updated Hanset dataset --> may downsample the data later
    

### Lansat Satellite data:
* Configured Google Earth Engine script to extract apprpriate Landsat time series data b/n 2000-2021: Landsat 5 for 2000-2011, Landsat 7 for 2012, and Landsat 8 for 2013-2021
* Filtered images with < 50% cloud cover for better image quality and created median composites for each year to minimize seasonal variations and cloud effects --> resulted in 22 annual snapshots spanning 2000-2021 and then we configured batch export tasks to save all 22 annual composites to Google Drive (30m resolution): https://drive.google.com/drive/folders/1hbkrBB_EEtaLryNrM5qSFJAWw4NotvYg?usp=sharing

* Also included key vegetation indices (like NDVI, NBR, but tbh may not need to factor these in as model inputs)--> was thinking it might be helpful for detecting subtler changes in vegetation health/density that might indicate species shifts. We also visualized our data to check out the quality and coverage.

### Hanset Forest change data:
* We intergrated the tree gain/loss data from the Hansen Global Forest Change dataset (2000-2021). Created a forest mask identifying regions with >30% tree cover in the baseline year to focus our analysis on relevant forested areas --> filtering out noise 
* Ensured spatial and termporal alignment by using the same study area boundaries and dates, maintained consistent 30m resolution across both datasets and added annual loss bands to each Landsat composite for direct correlation


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



#3/10/25 - 3/17/25:
* Exported climate data the temporally and gerogrpahically aligns with each the corresponding Lansat composites: 
    https://drive.google.com/drive/folders/1fmgbmX2hNKR8DXfa2Wmc7HPwY53_3i6j?usp=drive_link

* CNN for tree classfication: 


## NOTE: https://digitalcommons.library.umaine.edu/cgi/viewcontent.cgi?article=3612&context=etd
* very useful paper, on forcasting and modeling the spruce tree in main --> added the spruce tree to our tree species data


* this was a  slow week for me, got covid and bacterail infection so didn't make much progress at all
* started to filter and divide the data into training/test/splits for CNN model training but didn't get anything to work 



# 3/17/25 - 3/24/25
* Switched to Sential data for better resolution since I wasn't able to extract many specrtal signatures and pixels successfully from the Landsat data (experimented with different buffer sizes to ensure). Another issue I was having with the Lansata satellite, is that many of the observed tree points apparently weren't within the range of the images, which was odd as both the satellite image data and the GBIF data were exctracted usuign the same locaiton coordinates. The Sentinal-2 data is from 2018-2021, extracted about 300 images total of the trees (all 5) for training (which is really very little) implemented the CNN and a RF models-->sismialr resulst with the training accruacy in arounf 40% and test accuray aroun 25-30%. The confusion matrix shows how many of the trees are missidentified as one another, this makes sense given the training size and and fact that most of the spectral signatures for these trees are very similar to one another. 

Several issues with these first model runs for classification:
- Not enough tree observation samples to start with, apparently when using GBIF API, we are limited to retriving only 300 samples total, not per species, even though I set the limit at 10,000. --> solution: download the psecies data directly from the GBIF database, for example we can use species search to get: https://www.gbif.org/occurrence/search?q=abies%20balsamea&basis_of_record=HUMAN_OBSERVATION&country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&has_geospatial_issue=false&month=9&month=10&year=2015,2023&geometry=POLYGON((-69.25169%2047.44281,-75.44628%2045.32212,-78.86807%2043.00021,-75.47091%2038.54031,-71.10597%2040.19079,-66.62251%2044.04165,-65.84348%2047.51913,-69.25169%2047.44281))&occurrence_status=present 

- I also think we should expand our area of interest from Maine to the wider region in the northeast, so we can ideally get a greater number of samples. 

-Instead of looking at year round composite satellite images, we should perhaps for season specfic (like autumn, as trees are most ditinguable from one another, so we would pick up on more significantly differnt spectral signatures for each tree species making the classification task more straighforward). 



# 3/24/25 - #04/11/25
- Did not work on the project for this period, as I had 2 large projects from other classes due and couldn't find the time. 

# 3/30/25 and 04/08/25
- Met with Ben these days to update him and current state of the project, let him know that we need to look for more tree obseration data for improvent claddification model results. 
- If classification efforts still produce weak results after expanding the data set and isolating by season, then we can just move forward with a predictive model (LTSM) with the climate data tifs and the observed tree data throughout the past decade (despite there being many gaps in species documenation)

# 04/08/25
- Met with Ben, to touch base and he added some info into this doc on potetial way to go about the next steps (LTSM/GRU): https://docs.google.com/document/d/1I9abTn2klx-qOxLs4Ww-sLrz439jOuEzr_Qif94T3bU/edit?usp=sharing
- Ben shared that he found some new data from the USFDA goverment website (Altas data). 


# 04/11/25 - Present
- Started back up on the project, wasn't able find the atlas data that Ben showed me during our meeting, so I just went ahead and continued with the GBIF data.
- Dowloaded new Sential image tifs from 2018-2921 for Septemeber and October (peak seaons when leaves change colors) via GEE:
- Expanded the region to greater northeast (sticking to with the US)
- Downloaded total of ~7000 tree samples for Rad Maple, Balsam Fir, and Yellow Birch. Downsampled from 5 tree species to 3 species that would have distinct colors during Fall season. (still doesn't take into consideration, the tree classfication model idtentiying other similar colored trees) 
- Bottleneck that I am stuck at, and have been trying to work on, is that with the new dat aI am failign to successfully extract spectral signiatures from this new sallelite image data
- Will plan to work thoruhg classification on the side for now and proceed to imllemnting a GRU as its a simpler model architecture and then also trying out an LSTM??: 
    -  tree_migration_gru.py: 2-layer GRU architecture: the first layer with 128 units and return_sequences=True to maintain the temporal information, followed by a second layer with 64 units. Add dropout b/n layers of 0.3 to prevent overfitting. final layers --> dense layer with 32 units and ReLU activation, followed by an output layer with 3 units (one for each species) using softmax activation.
        - susign adam optimizer for training 
        - planning for the input data to be shaped as (samples, 6, features) where 6 represents our years (2018-2023) and features include the 10 sentinel2 spectral bands, climate variables, and vegetation indices (NDVI, NBR) 
