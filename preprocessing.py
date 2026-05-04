#!/usr/bin/env python3


import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("Crop_recommendation.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns      : {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE / TARGET SPLIT
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET   = 'label'

X = df[FEATURES]
y = df[TARGET]

# ──────────────────────────────────────────────────────────────────────────────
# 3. LABEL ENCODING  (string → integer class index)
# ──────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n{len(le.classes_)} unique crops: {list(le.classes_)}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. OUTLIER ASSESSMENT  (IQR — log only; no rows dropped)
#    The dataset is perfectly balanced (100 samples/class) and domain outliers
#    (e.g. very high N for rice) are agronomically meaningful — we keep them.
# ──────────────────────────────────────────────────────────────────────────────
print("\nIQR outlier counts (kept in data — domain valid):")
for col in FEATURES:
    Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
    IQR = Q3 - Q1
    n = ((X[col] < Q1 - 1.5*IQR) | (X[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col}: {n}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)  — stratified
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded          # keeps class balance in every split
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
print(f"\nSplit sizes → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. FEATURE SCALING  (StandardScaler)
#    Fit ONLY on training data → transform val & test to prevent data leakage.
#    StandardScaler is ideal for:
#      • KNN        — distance-based, sensitive to scale
#      • RandomForest — tree splits are scale-invariant but scaling doesn't hurt
#      • DecisionTree — scale-invariant; scaler still applied for consistency
# ──────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only

print(f"\nPost-scaling (train) mean ≈ 0: {X_train_scaled.mean(axis=0).round(3)}")
print(f"Post-scaling (train) std  ≈ 1: {X_train_scaled.std(axis=0).round(3)}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE ARTEFACTS
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)

preprocessed = {
    # Scaled arrays (ready for model training)
    "X_train": X_train_scaled,
    "X_val"  : X_val_scaled,
    "X_test" : X_test_scaled,
    "y_train": y_train,
    "y_val"  : y_val,
    "y_test" : y_test,

    # Fitted transformers (needed for inference on new data)
    "scaler"       : scaler,
    "label_encoder": le,

    # Metadata
    "feature_names": FEATURES,
    "class_names"  : list(le.classes_),
}

with open("artifacts/preprocessed_data.pkl", "wb") as f:
    pickle.dump(preprocessed, f)

print("\n✅ Saved: artifacts/preprocessed_data.pkl")
print("Keys:", list(preprocessed.keys()))
