"""
Training script for crop recommendation and yield prediction
without using 'District' as a feature.

Features used:
- N, P, K
- pH
- Humidity (or soil moisture equivalent)
- Temperature
- Rainfall
- SoilType (categorical)

Targets:
- CropName (classification)
- CropYield (regression)

Outputs:
- model_crop.pkl
- model_yield.pkl
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
import pickle

# ----------------------------------------------------------------------
# 1. Load and clean dataset
# ----------------------------------------------------------------------
df = pd.read_csv("crop_yield_data.csv")

# Replace non-numeric error tokens, e.g. '#VALUE!', with NaN
df.replace("#VALUE!", np.nan, inplace=True)

print("Dataframe shape before dropping duplicates:", df.shape)
print("Number of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Dataframe shape after dropping duplicates:", df.shape)

print("\nColumns in dataset:", df.columns.tolist())

# ----------------------------------------------------------------------
# 2. Define features and targets (NO 'District')
# ----------------------------------------------------------------------
# Numeric features (from sensors + APIs)
numeric_features = ["N", "P", "K", "pH", "Humidity", "Temperature", "Rainfall"]

# Categorical features
categorical_features_crop = ["SoilType"]          # for crop classification
categorical_features_yield = ["SoilType", "CropName"]  # for yield regression

# Feature sets
features_for_crop = numeric_features + categorical_features_crop
X_crop = df[features_for_crop]
y_crop = df["CropName"]

features_for_yield = numeric_features + categorical_features_yield
X_yield = df[features_for_yield]
y_yield = df["CropYield"]

print("\nFeatures for crop classification:", features_for_crop)
print("Features for yield regression:", features_for_yield)

# ----------------------------------------------------------------------
# 3. Preprocessing pipelines
# ----------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer_crop = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

categorical_transformer_yield = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor_for_crop = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_crop, categorical_features_crop),
    ]
)

preprocessor_for_yield = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_yield, categorical_features_yield),
    ]
)

# ----------------------------------------------------------------------
# 4. Models (classifier + regressor)
# ----------------------------------------------------------------------
model_crop = Pipeline(
    steps=[
        ("preprocessor", preprocessor_for_crop),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

model_yield = Pipeline(
    steps=[
        ("preprocessor", preprocessor_for_yield),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# ----------------------------------------------------------------------
# 5. Train / test split
# ----------------------------------------------------------------------
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# 6. Train models
# ----------------------------------------------------------------------
print("\nTraining crop classification model...")
model_crop.fit(X_train_crop, y_train_crop)

print("Training yield regression model...")
model_yield.fit(X_train_yield, y_train_yield)

# ----------------------------------------------------------------------
# 7. Helper: predict top crops + yields (no District)
# ----------------------------------------------------------------------
def predict_with_adjusted_yield(input_data: dict, top_n: int = 2):
    """
    input_data should contain:
    N, P, K, pH, Humidity, Temperature, Rainfall, SoilType

    Returns a list of (crop_name, predicted_yield) tuples,
    sorted by probability / yield logic, with a small adjustment
    to ensure the second-best yield is not higher than the best.
    """
    # For crop classification, we only pass the features_for_crop
    sample_input_crop = pd.DataFrame([input_data])[features_for_crop]

    # 1) Crop probability prediction
    crop_probabilities = model_crop.predict_proba(sample_input_crop)[0]
    top_crop_indices = np.argsort(crop_probabilities)[::-1][:top_n]
    top_crops = np.array(model_crop.classes_)[top_crop_indices]

    top_crops_with_yield = []

    for crop in top_crops:
        # For yield prediction, we add CropName
        input_data_with_crop = input_data.copy()
        input_data_with_crop["CropName"] = crop

        sample_input_yield = pd.DataFrame([input_data_with_crop])[features_for_yield]
        yield_prediction = model_yield.predict(sample_input_yield)[0]
        top_crops_with_yield.append((crop, float(yield_prediction)))

    # 2) Adjustment: make sure second-best yield is not higher than best
    if top_n > 1 and top_crops_with_yield[1][1] > top_crops_with_yield[0][1]:
        adjustment_factor = 0.9  # reduce by 10%
        adjusted_yield = top_crops_with_yield[0][1] * adjustment_factor
        top_crops_with_yield[1] = (top_crops_with_yield[1][0], adjusted_yield)

    return top_crops_with_yield


# ----------------------------------------------------------------------
# 8. Example usage with live-like sensor input (no District)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: these values can come from Raspberry Pi sensors + weather API
    example_input = {
        "N": 55,
        "P": 18,
        "K": 50,
        "pH": 6.4,
        "Humidity": 77,        # or your soil moisture metric
        "Temperature": 26,
        "Rainfall": 880,       # from weather API
        "SoilType": "Alluvial Soil",
    }

    top_crops_with_yield = predict_with_adjusted_yield(example_input, top_n=2)

    print("\nRecommendations (no District feature):")
    print(f"Best Fit Crop: {top_crops_with_yield[0][0]}, Expected Yield: {top_crops_with_yield[0][1]}")
    for i, (crop, yield_prediction) in enumerate(top_crops_with_yield[1:], start=1):
        print(f"Alternative Crop {i}: {crop}, Expected Yield: {yield_prediction}")

    # Evaluate classification model
    y_pred_crop = model_crop.predict(X_test_crop)
    accuracy = accuracy_score(y_test_crop, y_pred_crop)
    print(f"\nCrop classification accuracy (no District): {accuracy:.4f}")

    # ------------------------------------------------------------------
    # 9. Save trained models for deployment
    # ------------------------------------------------------------------
    with open("model_crop.pkl", "wb") as f:
        pickle.dump(model_crop, f)

    with open("model_yield.pkl", "wb") as f:
        pickle.dump(model_yield, f)

    print("\nModels saved as 'model_crop.pkl' and 'model_yield.pkl'.")
