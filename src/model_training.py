import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import yaml
import argparse
import os
import json

def train_model(params_path):
    """Trains a model using specified hyperparameters."""
    with open(params_path) as f:
        params = yaml.safe_load(f)

    processed_data_path = params['data']['processed']
    model_path = params['model']['path']
    model_cols_path = params['model']['columns_path']
    
    print("Training model...")
    df = pd.read_csv(processed_data_path)
    
    features = params['model']['features']
    target = params['model']['target']
    categorical_features = params['model']['categorical_features']
    
    X_raw = df[features]
    y = df[target]

    X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)
    
    print("Saving model column order...")
    model_columns = X_encoded.columns.tolist()
    with open(model_cols_path, 'w') as f:
        json.dump(model_columns, f)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=params['process']['test_size'], 
        random_state=params['process']['random_state']
    )
    
    model = RandomForestRegressor(
        n_estimators=params['train']['n_estimators'],
        max_depth=params['train']['max_depth'],
        random_state=params['train']['random_state'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    train_model(args.params)
