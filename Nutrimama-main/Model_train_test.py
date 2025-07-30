import pandas as pd
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load CSV
csv_path = r"c:\Users\hp\OneDrive\Desktop\Nutrimamma\Nutrimama_langmem\synthetic_pregnancy_nutrition_data (1).csv"
df = pd.read_csv(csv_path)

print("✅ Dataset loaded. Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())

# STEP 2: Generate 'risk' column if missing
if "risk" not in df.columns:
    print("⚠️ 'risk' column not found. Creating synthetic labels...")
    def compute_risk(row):
        if row.get("Currently_Pregnant") == "Yes" and row.get("Symptoms") == "Yes":
            return "High"
        elif row.get("Supplements") == "No" or row.get("Clean_Water") == "No":
            return "Moderate"
        else:
            return "Low"
    df["risk"] = df.apply(compute_risk, axis=1)
    print("✅ 'risk' column created.")

# STEP 3: Define categorical features
cat_features = [
    "Currently_Pregnant", "Trimester", "Supplements", "Fried_Foods", "Sugary_Foods",
    "Use_Fortified_Food", "Supplement_Frequency",
    "Avoiding_Foods", "Symptoms", "Smoke_Alcohol", "Physical_Activity",
    "Clean_Water", "Food_Access_Difficulty", "Nutrition_Info_Sources", "Use_App_Hindi_Voice",
    "Prefer_Pictures", "Breastfeeding", "Child_Food", "Child_Eats_Veg", "Child_Dairy",
    "Child_Deficiency_Symptoms", "Food_Options_Available"
]


# STEP 4: Handle NaNs in categorical and numerical features
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("Unknown")

for col in df.columns:
    if col not in cat_features + ["risk"]:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())

# STEP 5: Prepare features and target
X = df.drop("risk", axis=1)
y = df["risk"]

# STEP 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# STEP 7: CatBoost Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# STEP 8: Train model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric='Accuracy',
    early_stopping_rounds=20,
    verbose=100
)
model.fit(train_pool, eval_set=test_pool)

# STEP 9: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on test set: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 10: Save model
os.makedirs("model", exist_ok=True)
model.save_model("model/model.cbm")
print("✅ Model saved to 'model/model.cbm'")
