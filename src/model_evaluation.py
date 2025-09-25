# src/model_evaluation.py

import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from config import PROCESSED_DATA_PATH, MODEL_PATH

def evaluate_model():
    """Evaluates the model and fails if performance is below a threshold."""
    print("Evaluating model...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    features = ['State', 'District', 'Season', 'Crop', 'Area', 'Temperature', 'Precipitation']
    target = 'Yield'
    
    X = pd.get_dummies(df[features], columns=['State', 'District', 'Season', 'Crop'], drop_first=True)
    y = df[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    
    print(f"Model R^2 score: {score:.4f}")
    
    # We set a quality gate. The pipeline will fail if the model is not good enough.
    assert score > 0.5, f"Model score {score} is below the threshold of 0.5!"
    print("Model performance is acceptable.")

if __name__ == "__main__":
    evaluate_model()
