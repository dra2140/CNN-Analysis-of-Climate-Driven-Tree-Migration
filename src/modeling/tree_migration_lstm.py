import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_spatial_features(tree_gdf, climate_data_dir, year):
    """Extract and prepare spatial features from climate data."""
    spatial_features = []
    
    # Load climate data for the year
    prism_path = os.path.join(climate_data_dir, f'prism_climate_{year}.tif')
    terra_path = os.path.join(climate_data_dir, f'terraclimate_{year}.tif')
    
    if os.path.exists(prism_path) and os.path.exists(terra_path):
        # Extract climate variables for each observation
        climate_vars = {
            'precip': extract_climate_data(tree_gdf, prism_path),
            'tmean': extract_climate_data(tree_gdf, prism_path),
            'tmin': extract_climate_data(tree_gdf, prism_path),
            'tmax': extract_climate_data(tree_gdf, prism_path),
            'pet': extract_climate_data(tree_gdf, terra_path),
            'aet': extract_climate_data(tree_gdf, terra_path),
            'water_deficit': extract_climate_data(tree_gdf, terra_path),
            'soil_moisture': extract_climate_data(tree_gdf, terra_path),
            'vpd': extract_climate_data(tree_gdf, terra_path)
        }
        
        # Combine all climate variables
        spatial_features = np.column_stack(list(climate_vars.values()))
    
    return spatial_features

def prepare_temporal_features(tree_gdf, climate_data_dir, years):
    """Prepare temporal sequence of climate features."""
    temporal_features = []
    
    for year in years:
        year_data = tree_gdf[tree_gdf['year'] == year]
        if len(year_data) > 0:
            spatial_features = prepare_spatial_features(year_data, climate_data_dir, year)
            temporal_features.append(spatial_features)
    
    return np.array(temporal_features)

def build_hybrid_model(spatial_input_shape, temporal_input_shape, num_classes):
    """Build a hybrid model combining spatial and temporal features."""
    # Spatial pathway
    spatial_input = Input(shape=spatial_input_shape)
    x = Dense(64, activation='relu')(spatial_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    spatial_features = BatchNormalization()(x)
    
    # Temporal pathway
    temporal_input = Input(shape=temporal_input_shape)
    y = LSTM(64, return_sequences=True)(temporal_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = LSTM(32)(y)
    temporal_features = BatchNormalization()(y)
    
    # Merge pathways
    merged = Concatenate()([spatial_features, temporal_features])
    
    # Final dense layers
    z = Dense(64, activation='relu')(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = Dense(32, activation='relu')(z)
    output = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[spatial_input, temporal_input], outputs=output)
    return model

def train_model(X_spatial, X_temporal, y, epochs=50, batch_size=32):
    """Train the hybrid model."""
    # Split data
    X_spatial_train, X_spatial_test, X_temporal_train, X_temporal_test, y_train, y_test = train_test_split(
        X_spatial, X_temporal, y, test_size=0.2, random_state=42
    )
    
    # Build and compile model
    model = build_hybrid_model(
        spatial_input_shape=X_spatial.shape[1:],
        temporal_input_shape=X_temporal.shape[1:],
        num_classes=len(np.unique(y))
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        [X_spatial_train, X_temporal_train],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_spatial_test, X_temporal_test], y_test),
        verbose=1
    )
    
    return model, history

def plot_training_history(history, output_dir):
    """Plot training history."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    # Load and prepare data
    tree_gdf = load_tree_data('src/filtered_trees_gbif.csv')
    
    # Define years for analysis
    years = range(2018, 2024)
    
    # Prepare features
    X_spatial = []
    X_temporal = []
    y = []
    
    for species in tree_gdf['verbatimScientificName'].unique():
        species_data = tree_gdf[tree_gdf['verbatimScientificName'] == species]
        
        # Get spatial features for each year
        for year in years:
            year_data = species_data[species_data['year'] == year]
            if len(year_data) > 0:
                spatial_features = prepare_spatial_features(year_data, 'climate_data_2018_2023', year)
                temporal_features = prepare_temporal_features(year_data, 'climate_data_2018_2023', years)
                
                if len(spatial_features) > 0 and len(temporal_features) > 0:
                    X_spatial.append(spatial_features)
                    X_temporal.append(temporal_features)
                    y.append(species)  # Use species as label
    
    # Convert to numpy arrays
    X_spatial = np.array(X_spatial)
    X_temporal = np.array(X_temporal)
    y = np.array(y)
    
    # Scale features
    scaler_spatial = StandardScaler()
    scaler_temporal = StandardScaler()
    
    X_spatial = scaler_spatial.fit_transform(X_spatial.reshape(-1, X_spatial.shape[-1])).reshape(X_spatial.shape)
    X_temporal = scaler_temporal.fit_transform(X_temporal.reshape(-1, X_temporal.shape[-1])).reshape(X_temporal.shape)
    
    # Train model
    model, history = train_model(X_spatial, X_temporal, y)
    
    # Plot training history
    plot_training_history(history, 'model_results')
    
    # Save model
    model.save('tree_migration_lstm_model.h5')
    
    print("\nModel training complete! Results saved in 'model_results' directory")

if __name__ == "__main__":
    main() 