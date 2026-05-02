import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD DATASET
# ==========================================
# Make sure the filename matches your actual CSV file
try:
    df = pd.read_csv('water_sensor_data.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'water_sensor_data.csv' not found. Check your file path.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
# We drop 'id' because it leads to 'overfitting' (cheating).
# We drop 'angle' if it was constant 0 in your lab test.
# 'water_level' is our Target (y). Everything else is a Feature (X).
X = df.drop(columns=['id', 'water_level', 'angle'])
y = df['water_level']

# Split: 80% for training, 20% for the 'Environment Factors' test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. INITIALIZE & TRAIN RANDOM FOREST
# ==========================================
# n_estimators=100 builds 100 decision trees to average out sensor noise.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Save model and feature order for consistent future inference.
model_bundle = {
    'model': model,
    'feature_columns': list(X.columns)
}
joblib.dump(model_bundle, 'water_level_model.joblib')
print("✅ Saved model bundle: water_level_model.joblib")

# ==========================================
# 4. PREDICTIONS & EVALUATION
# ==========================================
predictions = model.predict(X_test)

# Calculate metrics for your comparative study
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n--- AI PERFORMANCE REPORT ---")
print(f"Average Error (MAE): {mae:.4f} units")
print(f"Severity Error (RMSE): {rmse:.4f} units")

# ==========================================
# 5. LIVE SENSOR SIMULATION (Test a single reading)
# ==========================================
# Format: [ir_value, ir_strength, us_value, acc_x, acc_y, acc_z, gyr_acc_x, gyr_acc_y, gyr_acc_z, gyr_x, gyr_y, gyr_z]
# Example: Testing a jump to 150 with some noise in the ultrasonic sensor (148.5)
noisy_sample = [[151, 4800, 148.5, 1.024, 0.152, 2.044, -0.003, 0.970, -0.145, -0.465, 0.503, -0.267]]

single_prediction = model.predict(pd.DataFrame(noisy_sample, columns=X.columns))
print(f"\n--- LIVE TEST ---")
print(f"Input: Noisy 150 level reading")
print(f"AI Predicted Level: {single_prediction[0]:.2f} units")

# ==========================================
# 6. VISUALIZING THE ROBUSTNESS (Optional)
# ==========================================
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Water Level', alpha=0.5)
plt.scatter(range(len(predictions)), predictions, color='red', label='AI Prediction', marker='x')
plt.title('Random Forest: Actual vs Predicted Water Levels')
plt.xlabel('Sample Index')
plt.ylabel('Water Level')
plt.legend()

plot_path = 'robustness_plot.png'
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
print(f"✅ Saved plot: {plot_path}")

# Optional display in local interactive sessions.
if os.getenv('SHOW_PLOT', '0') == '1':
    plt.show()
else:
    plt.close()