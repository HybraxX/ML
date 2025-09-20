import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- 1. Load and Prepare Your New Data ---
# Load the 'yield_dataset.csv'
try:
    df = pd.read_csv('yield_dataset_realistic.csv')
    print("✅ Dataset 'yield_dataset_realistic.csv' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'yield_dataset_realistic.csv' not found. Please place it in the same directory.")
    exit()

# Display the columns to verify
print("\nDataset columns:")
print(df.columns)

# Define the features and the target based on the new dataset
# Features are all columns except 'Yield'
features = [
    'N', 'P', 'K', 'Soil_Moisture', 'Temperature', 
    'Humidity', 'pH', 'FN', 'FP', 'FK'
]
target = 'Yield'

X = df[features]
y = df[target]

# --- 2. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- 3 & 4: Choose and Train the Model ---
print("\nTraining the Random Forest model on the new dataset...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 5: Evaluate the Model's Performance ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📈 Model Performance on New Test Data:")
print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
print(f"   - R-squared (R2 Score): {r2:.2f}")

# --- 6: Save the Trained Model ---
# This will overwrite the old model file with one trained on the new data
model_filename = 'crop_yield_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"\n✅ New model saved successfully as '{model_filename}'")

