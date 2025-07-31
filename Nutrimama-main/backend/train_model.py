import pandas as pd
import os
from catboost import CatBoostClassifier, Pool

# Load dataset
df = pd.read_csv("nutri_startup\\Nutrimama-main\\backend\\nutri_training_dataset.csv")

# Define features and target
X = df.drop("risk", axis=1)
y = df["risk"]

# Define categorical columns
cat_features = ["pregnant", "iron", "income"]

# Build CatBoost pool
train_pool = Pool(data=X, label=y, cat_features=cat_features)

# Create and train model with more rounds and early stopping
model = CatBoostClassifier(
    iterations=1000,          # 🚀 More boosting rounds
    learning_rate=0.05,       # 🔁 Lower learning rate = better generalization
    depth=6,
    cat_features=cat_features,
    eval_metric='Accuracy',
    early_stopping_rounds=20, # ⏹️ Stops if no improvement
    verbose=100               # Shows progress every 100 iterations
)

# Fit model (uses 20% of data automatically for validation)
model.fit(train_pool)

# Save the trained model
os.makedirs("model", exist_ok=True)
model.save_model("model/model.cbm")

print("✅ Better trained CatBoost model saved at model/model.cbm")
