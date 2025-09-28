# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import os
import json # <-- Import json
from config import PROCESSED_DATA_PATH, MODEL_PATH, MODEL_COLUMNS_PATH # <-- Add MODEL_COLUMNS_PATH

def train_model():
    """Trains a model and saves it, along with the column order."""
    print("Training model...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    features = ['Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop']
    target = 'Yield'
    
    categorical_features = ['Season', 'State', 'Crop']
    X_raw = df[features]
    y = df[target]

    X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)
    
    # --- ADD THIS BLOCK TO SAVE COLUMN ORDER ---
    print("Saving model column order...")
    model_columns = X_encoded.columns.tolist()
    with open(MODEL_COLUMNS_PATH, 'w') as f:
        json.dump(model_columns, f)
    print(f"Columns saved to {MODEL_COLUMNS_PATH}")
    # -----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"Model R^2 score on test set: {score:.4f}")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
