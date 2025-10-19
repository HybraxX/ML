import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

try:
    df = pd.read_csv('yield_dataset_realistic.csv')
    print("✅ Dataset 'yield_dataset_realistic.csv' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'yield_dataset_realistic.csv' not found. Please place it in the same directory.")
    exit()
print("\nDataset columns:")
print(df.columns)
features = [
    'N', 'P', 'K', 'Soil_Moisture', 'Temperature', 
    'Humidity', 'pH', 'FN', 'FP', 'FK'
]
target = 'Yield'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")
print("\nTraining the Random Forest model on the new dataset...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📈 Model Performance on New Test Data:")
print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
print(f"   - R-squared (R2 Score): {r2:.2f}")

model_filename = 'crop_yield_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"\n✅ New model saved successfully as '{model_filename}'")

