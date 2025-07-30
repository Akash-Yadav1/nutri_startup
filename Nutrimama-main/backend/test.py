import joblib
import pandas as pd

# Load model
model = joblib.load("model/model.pkl")

print("✅ Model loaded successfully!")

# Define column names used in training
columns = ["age", "pregnant", "meals", "water", "iron", "income"]

# Input example
input_data = [[25, 1, 2, 2.0, 1, 1]]
input_df = pd.DataFrame(input_data, columns=columns)

# Predict
prediction = model.predict(input_df)[0]

print(f"🔍 Predicted Nutritional Risk (encoded): {prediction}")
