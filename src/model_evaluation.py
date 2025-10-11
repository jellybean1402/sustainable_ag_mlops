# src/model_evaluation.py

import pandas as pd
import pickle
import json
import yaml
import argparse
import os  # <-- Make sure 'os' is imported
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def evaluate_model(params_path):
    """Evaluates the model and saves the metrics."""
    with open(params_path) as f:
        params = yaml.safe_load(f)

    processed_data_path = params['data']['processed']
    model_path = params['model']['path']
    metrics_path = params['reports']['metrics']
    
    print("Evaluating model...")
    df = pd.read_csv(processed_data_path)
    
    features = params['model']['features']
    target = params['model']['target']
    categorical_features = params['model']['categorical_features']
    
    X_raw = df[features]
    y = df[target]
    
    X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)
    
    _, X_test, _, y_test = train_test_split(
        X_encoded, y, 
        test_size=params['process']['test_size'], 
        random_state=params['process']['random_state']
    )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    
    print(f"Model R^2 score: {score:.4f}")
    
    # --- THIS IS THE FIX ---
    # Ensure the reports directory exists before writing to it
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    # -----------------------
    
    with open(metrics_path, 'w') as f:
        json.dump({"r2_score": score}, f)
        
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    evaluate_model(args.params)
