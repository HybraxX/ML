import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Create the FastAPI Application ---
app = FastAPI(
    title="Crop Yield Prediction API",
    description="An API to predict crop yield based on the 'yield_dataset_realistic.csv' data.",
    version="2.0.0"
)

# --- ADD THIS SECTION FOR FRONTEND INTEGRATION ---
# This middleware allows your frontend (running in a browser) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# -------------------------------------------------

# --- 2. Load the Trained Model ---
# This loads the .pkl file created by your 'crop_yield_model_trainer.py' script.
try:
    model = joblib.load('crop_yield_prediction_model.pkl')
    print("✅ Model loaded successfully from 'crop_yield_prediction_model.pkl'")
except FileNotFoundError:
    print("❌ Error: Model file 'crop_yield_prediction_model.pkl' not found. Please run the trainer script first.")
    model = None

# --- 3. Define the Input Data Structure ---
# Pydantic model to validate the input data from the frontend.
# The field names MUST exactly match the feature names from your training script.
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
        # Provide an example for the interactive API docs
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

# --- 4. Create the Prediction Endpoint ---
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

    # Convert the incoming Pydantic object to a dictionary, then to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Ensure the column order matches the order used during model training
    training_features = [
        'N', 'P', 'K', 'Soil_Moisture', 'Temperature', 
        'Humidity', 'pH', 'FN', 'FP', 'FK'
    ]
    input_df = input_df[training_features]

    # Use the loaded model to make a prediction
    prediction = model.predict(input_df)

    # Return the prediction in a JSON response
    return {
        "predicted_yield": round(prediction[0], 2)
    }

