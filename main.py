import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Crop Yield Prediction API",
    description="An API to predict crop yield based on the 'yield_dataset_realistic.csv' data.",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

try:
    model = joblib.load('crop_yield_prediction_model.pkl')
    print("Loaded")
except FileNotFoundError:
    print("Not loaded")
    model = None
class CropData(BaseModel):
    N: float
    P: float
    K: float
    Soil_Moisture: float
    Temperature: float
    Humidity: float
    pH: float
    FN: float
    FP: float
    FK: float
    
    class Config:
        schema_extra = {
            "example": {
                "N": 90.0,
                "P": 42.0,
                "K": 43.0,
                "Soil_Moisture": 60.0,
                "Temperature": 20.2,
                "Humidity": 80.0,
                "pH": 6.5,
                "FN": 162.0,
                "FP": 73.0,
                "FK": 29.0
            }
        }
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Yield Prediction API. Go to /docs for usage."}

@app.post("/predict/")
def predict_yield(data: CropData):
    """
    Takes in agricultural data and returns a predicted crop yield.
    """
    if model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}

    input_df = pd.DataFrame([data.dict()])
    

    training_features = [
        'N', 'P', 'K', 'Soil_Moisture', 'Temperature', 
        'Humidity', 'pH', 'FN', 'FP', 'FK'
    ]
    input_df = input_df[training_features]

    prediction = model.predict(input_df)

    return {
        "predicted_yield": round(prediction[0], 2)
    }

