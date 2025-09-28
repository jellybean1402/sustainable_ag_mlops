# app/main.py
import os
import pickle
import json
import boto3
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from botocore.exceptions import ClientError
from .schemas import InputFeatures, PredictionOut

# --- Configuration ---
S3_BUCKET_NAME = "india-crop-yield-dvc-storage" # <-- REPLACE WITH YOUR BUCKET NAME
MODEL_S3_KEY = "production/model.pkl"
COLUMNS_S3_KEY = "production/model_columns.json" # <-- Path for the columns file

# In-memory storage for the model and columns
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("--- Loading model at startup ---")
    s3 = boto3.client('s3')
    try:
        # Download and load model
        s3.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, "model.pkl")
        with open("model.pkl", "rb") as f:
            model_cache['model'] = pickle.load(f)
        print("Model loaded successfully.")

        # Download and load columns
        s3.download_file(S3_BUCKET_NAME, COLUMNS_S3_KEY, "model_columns.json")
        with open("model_columns.json", "r") as f:
            model_cache['columns'] = json.load(f)
        print("Model columns loaded successfully.")

    except ClientError as e:
        print(f"Error loading model from S3: {e}")
        # In a real app, you might want to prevent startup if the model fails to load
        model_cache['model'] = None
        model_cache['columns'] = None
    
    yield
    
    # This code runs on shutdown
    print("--- Cleaning up model cache ---")
    model_cache.clear()

app = FastAPI(
    title="Crop Yield Prediction API",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"status": "API is running."}

@app.post("/predict", response_model=PredictionOut)
def predict_yield(features: InputFeatures):
    if not model_cache.get('model') or not model_cache.get('columns'):
        return {"error": "Model not loaded. Check server logs."}

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # One-hot encode the input data
    input_encoded = pd.get_dummies(input_df)
    
    # Reindex to match the training columns
    # This ensures consistency and handles missing categorical levels
    input_reindexed = input_encoded.reindex(columns=model_cache['columns'], fill_value=0)
    
    # Make prediction
    prediction = model_cache['model'].predict(input_reindexed)[0]
    
    return {"predicted_yield": prediction}
