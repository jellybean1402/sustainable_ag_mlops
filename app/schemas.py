# app/schemas.py
from pydantic import BaseModel

class InputFeatures(BaseModel):
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float
    Crop: str

    class Config:
        schema_extra = {
            "example": {
                "Crop_Year": 2010,
                "Season": "Kharif",
                "State": "Maharashtra",
                "Area": 1500.0,
                "Annual_Rainfall": 1200.5,
                "Fertilizer": 180000.0,
                "Pesticide": 450.0,
                "Crop": "Rice"
            }
        }

class PredictionOut(BaseModel):
    predicted_yield: float
