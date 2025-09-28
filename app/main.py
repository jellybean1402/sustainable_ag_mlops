import os
import pickle
import json
import boto3
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from botocore.exceptions import ClientError
from fastapi.responses import FileResponse
from .schemas import InputFeatures, PredictionOut

# --- Configuration ---
S3_BUCKET_NAME = "india-crop-yield-dvc-storage" # <-- REPLACE WITH YOUR BUCKET NAME
MODEL_S3_KEY = "production/model.pkl"
COLUMNS_S3_KEY = "production/model_columns.json"

# In-memory storage for the model and columns
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading model at startup ---")
    s3 = boto3.client('s3')
    try:
        s3.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, "model.pkl")
        with open("model.pkl", "rb") as f:
            model_cache['model'] = pickle.load(f)
        print("Model loaded successfully.")

        s3.download_file(S3_BUCKET_NAME, COLUMNS_S3_KEY, "model_columns.json")
        with open("model_columns.json", "r") as f:
            model_cache['columns'] = json.load(f)
        print("Model columns loaded successfully.")

    except ClientError as e:
        print(f"Error loading model from S3: {e}")
        model_cache['model'] = None
        model_cache['columns'] = None
    
    yield
    
    print("--- Cleaning up model cache ---")
    model_cache.clear()

app = FastAPI(
    title="Crop Yield Prediction API",
    lifespan=lifespan
)

# --- NEW: Serve the frontend from the root path ("/") ---
@app.get("/", response_class=FileResponse)
def read_index():
    return "frontend/index.html"

# --- NEW: Serve the frontend from "/predict" on a GET request ---
@app.get("/predict", response_class=FileResponse)
def read_predict_form():
    return "frontend/index.html"

# --- EXISTING: Handle prediction on a POST request ---
@app.post("/predict", response_model=PredictionOut)
def predict_yield(features: InputFeatures):
    if not model_cache.get('model') or not model_cache.get('columns'):
        return {"error": "Model not loaded. Check server logs."}

    input_df = pd.DataFrame([features.dict()])
    input_encoded = pd.get_dummies(input_df)
    input_reindexed = input_encoded.reindex(columns=model_cache['columns'], fill_value=0)
    prediction = model_cache['model'].predict(input_reindexed)[0]
    
    return {"predicted_yield": prediction}
