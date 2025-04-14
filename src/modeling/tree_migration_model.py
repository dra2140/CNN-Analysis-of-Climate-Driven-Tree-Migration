import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

def load_tree_data(csv_path):
    """Load and prepare tree observation data."""
    df = pd.read_csv(csv_path)
    
    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['decimalLongitude'], df['decimalLatitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Filter to our species of interest
    species = ['Acer rubrum', 'Abies balsamea', 'Betula alleghaniensis']
    gdf = gdf[gdf['verbatimScientificName'].isin(species)]
    
    print(f"Loaded {len(gdf)} tree observations")
    print("Species distribution:")
    print(gdf['verbatimScientificName'].value_counts())
    
    return gdf

def extract_climate_data(tree_points, climate_tif_path):
    """Extract climate data for tree observation points."""
    climate_data = []
    
    with rasterio.open(climate_tif_path) as src:
        for idx, row in tree_points.iterrows():
            point = row.geometry
            try:
                # Extract climate values at point location
                out_image, out_transform = mask(src, [point], crop=True)
                values = out_image[0, 0, 0]  # Get the value at the point
                climate_data.append(values)
            except:
                climate_data.append(np.nan)
    
    return np.array(climate_data)

def prepare_training_data(tree_gdf, climate_data_dir):
    """Prepare training data by combining tree observations with climate data."""
    X = []
    y = []
    
    # Process each year's climate data
    for year in range(2018, 2024):
        print(f"Processing climate data for {year}...")
        
        # Load PRISM data
        prism_path = os.path.join(climate_data_dir, f'prism_climate_{year}.tif')
        if os.path.exists(prism_path):
            precip = extract_climate_data(tree_gdf, prism_path)
            tmean = extract_climate_data(tree_gdf, prism_path)
            tmin = extract_climate_data(tree_gdf, prism_path)
            tmax = extract_climate_data(tree_gdf, prism_path)
            
            # Load TerraClimate data
            terra_path = os.path.join(climate_data_dir, f'terraclimate_{year}.tif')
            if os.path.exists(terra_path):
                pet = extract_climate_data(tree_gdf, terra_path)
                aet = extract_climate_data(tree_gdf, terra_path)
                water_deficit = extract_climate_data(tree_gdf, terra_path)
                soil_moisture = extract_climate_data(tree_gdf, terra_path)
                vpd = extract_climate_data(tree_gdf, terra_path)
                
                # Combine all climate variables
                climate_features = np.column_stack([
                    precip, tmean, tmin, tmax,
                    pet, aet, water_deficit, soil_moisture, vpd
                ])
                
                X.append(climate_features)
                y.append(tree_gdf['verbatimScientificName'].values)
    
    # Combine all years' data
    X = np.vstack(X)
    y = np.concatenate(y)
    
    return X, y

def train_model(X, y):
    """Train and evaluate the predictive model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    print("\nRandom Forest Results:")
    print(classification_report(y_test, rf.predict(X_test)))
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, xgb_model.predict(X_test)))
    
    # Plot feature importance
    plot_feature_importance(rf, "Random Forest Feature Importance")
    plot_feature_importance(xgb_model, "XGBoost Feature Importance")
    
    return rf, xgb_model

def plot_feature_importance(model, title):
    """Plot feature importance for the model."""
    feature_names = [
        'precip', 'tmean', 'tmin', 'tmax',
        'pet', 'aet', 'water_deficit', 'soil_moisture', 'vpd'
    ]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = model.feature_importances
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def main():
    # Load tree data
    tree_gdf = load_tree_data('src/filtered_trees_gbif.csv')
    
    # Prepare training data
    X, y = prepare_training_data(tree_gdf, 'climate_data_2018_2023')
    
    # Train models
    rf_model, xgb_model = train_model(X, y)
    
    # Save models
    joblib.dump(rf_model, 'tree_migration_rf_model.joblib')
    joblib.dump(xgb_model, 'tree_migration_xgb_model.joblib')
    
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main() 