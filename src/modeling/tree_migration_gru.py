import numpy as np
import pandas as pd
import rasterio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from datetime import datetime

def extract_sentinel_features(point, year, sentinel_dir):
    # find the appropriate sentinel-2 file for the year
    sentinel_file = os.path.join(sentinel_dir, f'sentinel2_autumn_{year}.tif')
    if not os.path.exists(sentinel_file):
        return None
        
    try:
        with rasterio.open(sentinel_file) as src:
            row_idx, col_idx = rasterio.transform.rowcol(
                src.transform, point['decimalLongitude'], point['decimalLatitude']
            )
            
            if not (0 <= row_idx < src.height and 0 <= col_idx < src.width):
                return None
            
            spectral_data = src.read()[:, row_idx, col_idx]
            
            # calculate vegetation indices
            ndvi = (spectral_data[7] - spectral_data[3]) / (spectral_data[7] + spectral_data[3] + 1e-10)
            nbr = (spectral_data[7] - spectral_data[11]) / (spectral_data[7] + spectral_data[11] + 1e-10)
            
            features = np.concatenate([spectral_data, [ndvi, nbr]])
            return features
            
    except Exception as e:
        print(f"error processing {sentinel_file}: {str(e)}")
        return None

def extract_climate_features(point, year, climate_dir):
    climate_file = os.path.join(climate_dir, f'climate_{year}.tif')
    if not os.path.exists(climate_file):
        return None
        
    try:
        with rasterio.open(climate_file) as src:
            row_idx, col_idx = rasterio.transform.rowcol(
                src.transform, point['decimalLongitude'], point['decimalLatitude']
            )
            
            if not (0 <= row_idx < src.height and 0 <= col_idx < src.width):
                return None
            
            climate_data = src.read()[:, row_idx, col_idx]
            return climate_data
            
    except Exception as e:
        print(f"error processing {climate_file}: {str(e)}")
        return None

def create_time_series(tree_data, sentinel_dir, climate_dir, years=range(2018, 2024)):
    sequences = []
    labels = []
    metadata = []
    
    scaler = StandardScaler()
    
    print("creating time series data...")
    for idx, point in tqdm(tree_data.iterrows(), total=len(tree_data)):
        sequence = []
        valid_sequence = True
        
        for year in years:
            sentinel_features = extract_sentinel_features(point, year, sentinel_dir)
            climate_features = extract_climate_features(point, year, climate_dir)
            
            if sentinel_features is None or climate_features is None:
                valid_sequence = False
                break
                
            year_features = np.concatenate([sentinel_features, climate_features])
            sequence.append(year_features)
        
        if valid_sequence and len(sequence) == len(years):
            sequences.append(sequence)
            labels.append(point['verbatimScientificName'])
            metadata.append({
                'species': point['verbatimScientificName'],
                'lon': point['decimalLongitude'],
                'lat': point['decimalLatitude'],
                'year': year
            })
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # scale features
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    return X, y, metadata

def create_gru_model(input_shape, num_classes):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_migration_model(X_train, y_train, X_val, y_val):
    # convert string labels to integers
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # convert to one-hot encoding
    from tensorflow.keras.utils import to_categorical
    y_train_onehot = to_categorical(y_train_encoded)
    y_val_onehot = to_categorical(y_val_encoded)
    
    model = create_gru_model(X_train.shape[1:], len(label_encoder.classes_))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history, label_encoder

def predict_migration(model, X, label_encoder):
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_species = label_encoder.inverse_transform(predicted_classes)
    return predicted_species, predictions

if __name__ == "__main__":
    import pandas as pd
    
    tree_data = pd.read_csv('src/filtered_trees_gbif.csv')
    
    sentinel_dir = 'path/to/sentinel/data' # all in the google drive 
    climate_dir = 'path/to/climate/data'
    
    X, y, metadata = create_time_series(
        tree_data,
        sentinel_dir,
        climate_dir
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model, history, label_encoder = train_migration_model(
        X_train, y_train, X_val, y_val
    )
    
    predictions, probabilities = predict_migration(model, X_val, label_encoder)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_val, predictions)) 