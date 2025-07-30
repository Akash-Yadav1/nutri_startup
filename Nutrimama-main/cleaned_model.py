


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import random

# Load dataset
df = pd.read_csv('pregnancy_health_data_7000_variety.csv')
df.info()

# Handle missing values
df['Supplements'].fillna('magnesium', inplace=True)
df['Symptoms'].fillna('dizziness', inplace=True)

# Drop unwanted columns
if 'Trimester' in df.columns:
    df.drop(columns=['Trimester'], inplace=True)

# Filter pregnant women
temp = df[df['Currently_Pregnant'] == 'Yes'].copy()

# Risk Score Calculation Function
def calculate_risk_score(row):
    score = 0
    if row['Sugary_Foods'] in ['Never', 'Rarely']:
        score -= 1
    else:
        score += 1
    if row['Supplement_Frequency'] == 'Daily':
        score -= 1
    elif row['Supplement_Frequency'] == 'Never':
        score += 1
    if row['Smoke_Alcohol'] == 'Yes':
        score += 2
    if row['Physical_Activity'] == 'High':
        score -= 1
    elif row['Physical_Activity'] == 'Low':
        score += 1
    return score

# Risk Category Function with Noise
def create_multiclass_output_with_noise(row):
    score = calculate_risk_score(row)
    if random.random() < 0.15:     # Reduced from 0.35
        score += random.choice([-1, 1])
    if score <= 1:
        return 'Low'
    elif 1 < score <= 5:
        return 'Moderate'
    else:
        return 'High'
# Apply risk score calculation
temp['Risk_Score'] = temp.apply(calculate_risk_score, axis=1)

temp['Pregnancy_Risk_Level'] = temp.apply(create_multiclass_output_with_noise, axis=1)
temp['Pregnancy_Risk_Level'] = temp['Pregnancy_Risk_Level'].map({'Low':0, 'Moderate':1, 'High':2})

# Drop column no longer needed
temp.drop(columns=['Currently_Pregnant'], inplace=True)

# Train-test split
train_df, test_df = train_test_split(temp, test_size=0.2, random_state=42, stratify=temp['Pregnancy_Risk_Level'])

# Binary columns mapping
binary_cols = ['Currently_Pregnant', 'Use_Fortified_Food', 'Smoke_Alcohol', 'Clean_Water', 'Food_Access_Difficulty']
binary_cols = [col for col in binary_cols if col in train_df.columns]
train_df = train_df.copy()
test_df = test_df.copy()
for col in binary_cols:
    train_df[col] = train_df[col].map({'Yes':1, 'No':0})
    test_df[col] = test_df[col].map({'Yes':1, 'No':0})

# Target encoding for categorical columns
cat_cols = ['Supplements', 'Fried_Foods', 'Sugary_Foods', 'Supplement_Frequency', 'Avoiding_Foods', 'Symptoms', 'Physical_Activity']
target_col = 'Pregnancy_Risk_Level'

for col in cat_cols:
    if col in train_df.columns:
        train_df[f'{col}_te'] = pd.Series(index=train_df.index, dtype='float64')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(train_df):
            fold_train = train_df.iloc[train_index]
            fold_val = train_df.iloc[val_index]
            means = fold_train.groupby(col)[target_col].mean()
            train_df.loc[train_df.index[val_index], f'{col}_te'] = fold_val[col].map(means)
        train_df[f'{col}_te'].fillna(train_df[target_col].mean(), inplace=True)
        test_means = train_df.groupby(col)[target_col].mean()
        test_df[f'{col}_te'] = test_df[col].map(test_means)
        test_df[f'{col}_te'].fillna(train_df[target_col].mean(), inplace=True)

# Feature selection
features = binary_cols + [f"{col}_te" for col in cat_cols if f"{col}_te" in train_df.columns] + ['Age', 'Height_cm', 'Weight_kg', 'Water_Intake_L', 'Previous_Pregnancies']
features = [feat for feat in features if feat in train_df.columns]

X_train = train_df[features]
y_train = train_df[target_col]
X_test = test_df[features]
y_test = test_df[target_col]

# Optional: SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# CatBoost Classifier
model = CatBoostClassifier(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=20,
    early_stopping_rounds=30,
    eval_metric='MultiClass',
    verbose=100
)
# Fit the model     

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
