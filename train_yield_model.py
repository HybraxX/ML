# -*- coding: utf-8 -*-
"""
train_yield_model.py

This script implements the core machine learning model for the CipherStorm project.
It creates a synthetic dataset, trains an XGBoost model, evaluates it,
and saves the trained model to 'yield_model.pkl'. This must be run first.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def create_synthetic_data():
    """Creates and saves a synthetic dataset for demonstration."""
    print("Creating a synthetic dataset for demonstration...")
    data = {
        'temperature_celsius': np.random.uniform(15, 35, 500),
        'rainfall_mm': np.random.uniform(50, 250, 500),
        'sunlight_hours': np.random.uniform(4, 10, 500),
        'soil_ph': np.random.uniform(5.5, 7.5, 500),
        'soil_moisture': np.random.uniform(0.2, 0.6, 500),
        'nitrogen_kg_ha': np.random.uniform(40, 120, 500),
        'phosphorus_kg_ha': np.random.uniform(20, 80, 500),
        'potassium_kg_ha': np.random.uniform(30, 90, 500)
    }
    # Create a target variable that has a logical relationship with the features
    data['yield_kg_per_hectare'] = (
        1500 + data['temperature_celsius'] * 15 + data['rainfall_mm'] * 5 +
        data['sunlight_hours'] * 50 + (6.5 - abs(6.5 - data['soil_ph'])) * 200 +
        data['nitrogen_kg_ha'] * 10 + data['phosphorus_kg_ha'] * 12 +
        data['potassium_kg_ha'] * 8 + np.random.normal(0, 150, 500)
    )
    synthetic_df = pd.DataFrame(data)
    synthetic_df.to_csv('synthetic_crop_data.csv', index=False)
    print("Synthetic dataset 'synthetic_crop_data.csv' created.")
    return synthetic_df

def train_and_save_model(df):
    """Trains an XGBoost model and saves it to a file."""
    X = df.drop('yield_kg_per_hectare', axis=1)
    y = df['yield_kg_per_hectare']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        early_stopping_rounds=50 # Parameter moved here
    )

    print("\nTraining XGBoost model...")
    # The 'early_stopping_rounds' parameter is removed from the .fit() method call
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("Model training complete.")

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel Evaluation - R-squared (R²): {r2:.2f}")

    joblib.dump(model, 'yield_model.pkl')
    print(f"\nModel successfully saved as 'yield_model.pkl'")

if __name__ == '__main__':
    dataset = create_synthetic_data()
    train_and_save_model(dataset)

