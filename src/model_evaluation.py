# src/model_evaluation.py

import pandas as pd
import pickle
import os
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from config import PROCESSED_DATA_PATH, MODEL_PATH

# --- Configuration ---
S3_BUCKET_NAME = "india-crop-yield-dvc-storage"
CHAMPION_MODEL_S3_KEY = "production/model.pkl"
CHAMPION_MODEL_LOCAL_PATH = "models/champion.pkl"
BASELINE_SCORE_THRESHOLD = 0.7  # For the very first model

def download_from_s3(bucket, key, local_path):
    """Downloads a file from S3, returns True on success, False on failure."""
    s3 = boto3.client('s3')
    try:
        print(f"Attempting to download champion model from s3://{bucket}/{key}")
        s3.download_file(bucket, key, local_path)
        print("Champion model downloaded successfully.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Champion model not found. This must be the first run.")
        else:
            print(f"An S3 error occurred: {e}")
        return False

def evaluate_model():
    """
    Evaluates the challenger model against the champion model.
    Fails if the challenger is not better than the champion.
    If no champion exists, fails if the challenger is not better than a baseline.
    """
    print("--- Starting Champion vs. Challenger Evaluation ---")
    
    # 1. Prepare data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    features = ['Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop']
    target = 'Yield'
    categorical_features = ['Season', 'State', 'Crop']
    X = pd.get_dummies(df[features], columns=categorical_features, drop_first=True)
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Load the newly trained challenger model
    with open(MODEL_PATH, 'rb') as f:
        challenger_model = pickle.load(f)
    challenger_preds = challenger_model.predict(X_test)
    challenger_score = r2_score(y_test, challenger_preds)
    print(f"Challenger Model Score: {challenger_score:.4f}")

    # 3. Download and evaluate the champion model (if it exists)
    champion_exists = download_from_s3(S3_BUCKET_NAME, CHAMPION_MODEL_S3_KEY, CHAMPION_MODEL_LOCAL_PATH)
    
    if champion_exists:
        with open(CHAMPION_MODEL_LOCAL_PATH, 'rb') as f:
            champion_model = pickle.load(f)
        champion_preds = champion_model.predict(X_test)
        champion_score = r2_score(y_test, champion_preds)
        print(f"Champion Model Score: {champion_score:.4f}")

        # 4. Compare models
        print("Comparing models...")
        assert challenger_score > champion_score, \
            f"Challenger score ({challenger_score:.4f}) is not better than champion score ({champion_score:.4f})!"
        print("Challenger is better! Proceeding.")
    else:
        # 4. Compare challenger to baseline if no champion exists
        print("Comparing to baseline score...")
        assert challenger_score > BASELINE_SCORE_THRESHOLD, \
            f"Challenger score ({challenger_score:.4f}) is below the baseline threshold of {BASELINE_SCORE_THRESHOLD}!"
        print("Challenger passed baseline! Proceeding.")

if __name__ == "__main__":
    evaluate_model()
