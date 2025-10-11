import pandas as pd
import yaml
import argparse
import os

def process_data(config_path):
    """Cleans raw data and saves it."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_data_path = config['data']['raw']
    processed_data_path = config['data']['processed']
    
    print("Processing data...")
    df = pd.read_csv(raw_data_path)
    df_cleaned = df.dropna(subset=['Yield'])
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_cleaned.to_csv(processed_data_path, index=False)
    print(f"Data processed and saved to {processed_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    # Note: We pass params.yaml but use it to find other paths.
    # A more advanced setup might pass paths directly.
    process_data(args.config)
