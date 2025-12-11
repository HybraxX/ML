#!/usr/bin/env python3
"""
interactive_recommend.py

Interactive terminal tool to provide TOP-3 crop recommendations (best -> 3rd)
based on live/manual input of sensor values.

Behavior:
- If saved models exist (model_crop.pkl), they will be loaded.
- Otherwise the script will train models from 'crop_yield_data.csv' and save them.
- Interactive prompt asks for N, P, K, pH, Humidity, Temperature, Rainfall, SoilType.
- Outputs the top 3 recommended crops (no yield values).

Usage:
    python interactive_recommend.py

Dependencies:
    pip install pandas numpy scikit-learn joblib
"""

import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Paths
MODEL_CROP_PATH = "model_crop.pkl"
MODEL_YIELD_PATH = "model_yield.pkl"
DATA_CSV = "crop_yield_data.csv"


def train_models(df: pd.DataFrame):
    """
    Train both a crop classifier and a yield regressor (kept for compatibility).
    Returns trained models and feature lists and soil types.
    """
    print("Preparing data for training...")

    # Features (no 'District')
    numeric_features = ["N", "P", "K", "pH", "Humidity", "Temperature", "Rainfall"]
    categorical_crop = ["SoilType"]
    categorical_yield = ["SoilType", "CropName"]

    features_for_crop = numeric_features + categorical_crop
    features_for_yield = numeric_features + categorical_yield

    # Drop rows missing targets
    df_clean = df.copy()
    df_clean.replace("#VALUE!", np.nan, inplace=True)
    df_clean = df_clean.dropna(subset=["CropName", "CropYield"])

    X_crop = df_clean[features_for_crop]
    y_crop = df_clean["CropName"]

    X_yield = df_clean[features_for_yield]
    y_yield = df_clean["CropYield"]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer_crop = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    categorical_transformer_yield = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor_crop = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_crop, categorical_crop),
    ])

    preprocessor_yield = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_yield, categorical_yield),
    ])

    # Models
    model_crop = Pipeline(steps=[
        ("preprocessor", preprocessor_crop),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    model_yield = Pipeline(steps=[
        ("preprocessor", preprocessor_yield),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Train/test split and fit
    print("Splitting data and training models...")
    X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
    X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

    model_crop.fit(X_train_crop, y_train_crop)
    model_yield.fit(X_train_yield, y_train_yield)

    # Evaluate classifier accuracy (informational)
    try:
        y_pred_crop = model_crop.predict(X_test_crop)
        acc = accuracy_score(y_test_crop, y_pred_crop)
        print(f"Crop classifier accuracy (test set): {acc:.4f}")
    except Exception:
        print("Could not compute accuracy (possible issues with test split).")

    # Save models
    joblib.dump(model_crop, MODEL_CROP_PATH)
    joblib.dump(model_yield, MODEL_YIELD_PATH)
    print(f"Saved models to '{MODEL_CROP_PATH}' and '{MODEL_YIELD_PATH}'")

    soil_types = df_clean["SoilType"].dropna().unique().tolist()
    return model_crop, model_yield, numeric_features, features_for_crop, features_for_yield, soil_types


def load_models_if_exist():
    """
    Attempt to load saved models. Return (model_crop, model_yield) or (None, None).
    """
    if os.path.exists(MODEL_CROP_PATH) and os.path.exists(MODEL_YIELD_PATH):
        try:
            model_crop = joblib.load(MODEL_CROP_PATH)
            model_yield = joblib.load(MODEL_YIELD_PATH)
            return model_crop, model_yield
        except Exception as e:
            print("Failed to load existing models:", e)
            return None, None
    return None, None


def predict_top_crops(model_crop, input_data: dict, features_for_crop: List[str], top_n: int = 3) -> List[str]:
    """
    Return top_n crop names by predicted probability from model_crop.
    """
    sample = pd.DataFrame([input_data])[features_for_crop]
    probs = model_crop.predict_proba(sample)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_crops = np.array(model_crop.classes_)[top_idx].tolist()
    return top_crops


def prompt_numeric(prompt_text: str, required: bool = True, default=None):
    """
    Prompt for a numeric value. If the user types 'exit' the script exits.
    """
    while True:
        raw = input(f"{prompt_text}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
        if raw.lower() == "exit":
            print("Exiting.")
            sys.exit(0)
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Invalid input. Please enter a numeric value (or type 'exit').")


def prompt_choice(prompt_text: str, choices: List[str], default=None) -> str:
    """
    Prompt for a value from choices (case-insensitive). Accepts exact choice text as well.
    """
    if not choices:
        # free text if we don't have choices
        while True:
            raw = input(f"{prompt_text}: ").strip()
            if raw.lower() == "exit":
                sys.exit(0)
            if raw:
                return raw
    choices_lower = [c.lower() for c in choices]
    while True:
        raw = input(f"{prompt_text}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
        if raw.lower() == "exit":
            sys.exit(0)
        if raw == "" and default is not None:
            return default
        if raw.lower() in choices_lower:
            return choices[choices_lower.index(raw.lower())]
        # allow picking by index
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        print("Invalid choice. Examples:", ", ".join(choices[:10]) + (", ..." if len(choices) > 10 else ""))


def interactive_loop(model_crop, model_yield, numeric_features, features_for_crop, features_for_yield, soil_types):
    """
    Main interactive prompt loop. Prints only top-3 crops (no yields).
    """
    print("\n----- Interactive Crop Recommender (Top-3) -----")
    print("Type 'exit' at any prompt to quit.\n")
    print("Units / notes:")
    print(" - N, P, K: nutrient levels (same scale your dataset uses, e.g., mg/kg)")
    print(" - pH: soil pH (typically 3.0 - 9.0)")
    print(" - Humidity: percent (0-100) or soil moisture depending on your dataset")
    print(" - Temperature: °C")
    print(" - Rainfall: units matching dataset (e.g., mm/year)\n")

    soil_default = soil_types[0] if soil_types else None

    while True:
        try:
            N = prompt_numeric("Enter N")
            P = prompt_numeric("Enter P")
            K = prompt_numeric("Enter K")
            pH = prompt_numeric("Enter pH")
            Humidity = prompt_numeric("Enter Humidity (0-100)")
            Temperature = prompt_numeric("Enter Temperature (°C)")
            Rainfall = prompt_numeric("Enter Rainfall (e.g. mm/year)")

            if soil_types:
                print("\nAvailable SoilType values from dataset (showing up to 10):")
                for i, st in enumerate(soil_types[:10], start=1):
                    print(f" {i}. {st}")
            SoilType = prompt_choice("Enter SoilType (type exact or pick shown)", soil_types, default=soil_default)

            input_data = {
                "N": N,
                "P": P,
                "K": K,
                "pH": pH,
                "Humidity": Humidity,
                "Temperature": Temperature,
                "Rainfall": Rainfall,
                "SoilType": SoilType
            }

            top3 = predict_top_crops(model_crop, input_data, features_for_crop, top_n=3)

            print("\n--- Top 3 Recommended Crops ---")
            for i, crop in enumerate(top3, start=1):
                print(f"{i}. {crop}")
            print("-------------------------------\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as exc:
            print("Error:", exc)
            print("Please try again.\n")


def main():
    # 1) Check dataset presence
    if not os.path.exists(DATA_CSV):
        print(f"Dataset '{DATA_CSV}' not found in current directory: {os.getcwd()}")
        print("Place your 'crop_yield_data.csv' file here and re-run.")
        sys.exit(1)

    # 2) Load dataset
    df = pd.read_csv(DATA_CSV)

    # 3) Load models if present; otherwise train
    model_crop, model_yield = load_models_if_exist()
    numeric_features = ["N", "P", "K", "pH", "Humidity", "Temperature", "Rainfall"]
    features_for_crop = numeric_features + ["SoilType"]
    features_for_yield = numeric_features + ["SoilType", "CropName"]

    if model_crop is None or model_yield is None:
        print("No existing trained models found or failed to load. Training new models...")
        model_crop, model_yield, numeric_features, features_for_crop, features_for_yield, soil_types = train_models(df)
    else:
        print("Loaded saved models from disk.")
        # Use soil types from dataset (if available)
        df.replace("#VALUE!", np.nan, inplace=True)
        soil_types = df["SoilType"].dropna().unique().tolist()

    # 4) Run interactive loop
    interactive_loop(model_crop, model_yield, numeric_features, features_for_crop, features_for_yield, soil_types)


if __name__ == "__main__":
    main()
