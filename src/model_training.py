# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import os
from config import PROCESSED_DATA_PATH, MODEL_PATH

def train_model():
    """Trains a model on the new, richer crop dataset."""
    print("Training model...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Define features (X) and target (y)
    # CORRECTED: Using the new feature set from your image. 'Production' is excluded to prevent data leakage.
    features = ['Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop']
    target = 'Yield'
    
    # One-hot encode the categorical features
    # This turns text categories into numbers the model can understand
    categorical_features = ['Season', 'State', 'Crop']
    X = pd.get_dummies(df[features], columns=categorical_features, drop_first=True)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model to print score during training
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"Model R^2 score on test set: {score:.4f}")
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
