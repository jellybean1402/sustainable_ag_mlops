# src/data_processing.py

import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import os

def process_data():
    """Cleans the Indian crop yield dataset."""
    print("Processing data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Drop rows where the target 'Yield' is missing
    df_cleaned = df.dropna(subset=['Yield'])
    
    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Data processed and saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_data()
