#!/usr/bin/env python3
"""
Crop Recommendation System — Feature Engineering
==================================================
Input  : Crop_recommendation.csv  +  artifacts/preprocessed_data.pkl
Output : artifacts/feature_engineered_data.pkl
         artifacts/feature_importance_report.csv
         artifacts/feature_correlation_heatmap.png
         artifacts/feature_distributions.png
         artifacts/feature_importance_rf.png
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD RAW DATA  (feature engineering is done on raw, not scaled data)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("Crop_recommendation.csv")
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET   = 'label'

os.makedirs("artifacts", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN-DRIVEN RATIO FEATURES
#    Agronomists use nutrient ratios to assess soil balance.
# ─────────────────────────────────────────────────────────────────────────────
df['N_P_ratio']    = df['N'] / (df['P'] + 1e-6)   # Nitrogen-Phosphorus balance
df['N_K_ratio']    = df['N'] / (df['K'] + 1e-6)   # Nitrogen-Potassium balance
df['P_K_ratio']    = df['P'] / (df['K'] + 1e-6)   # Phosphorus-Potassium balance
df['NPK_sum']      = df['N'] + df['P'] + df['K']   # Total nutrient content
df['NPK_product']  = df['N'] * df['P'] * df['K']   # Nutrient interaction term

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLIMATE INTERACTION FEATURES
#    Heat index, growing degree days, moisture-temperature coupling
# ─────────────────────────────────────────────────────────────────────────────
df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    # Mimics wet-bulb / heat-stress index; relevant for tropical crops
df['rainfall_humidity']   = df['rainfall'] * df['humidity'] / 100
    # Water availability proxy: high rain + high humidity = very wet conditions
df['temp_rainfall_ratio'] = df['temperature'] / (df['rainfall'] + 1e-6)
    # Aridity indicator: high temp / low rain → dry conditions

# ─────────────────────────────────────────────────────────────────────────────
# 3. PH-NUTRIENT INTERACTION FEATURES
#    Nutrient availability is pH-dependent (solubility curves)
# ─────────────────────────────────────────────────────────────────────────────
df['ph_N_interaction'] = df['ph'] * df['N']
df['ph_P_interaction'] = df['ph'] * df['P']
df['ph_K_interaction'] = df['ph'] * df['K']

# ─────────────────────────────────────────────────────────────────────────────
# 4. POLYNOMIAL FEATURES  (key agronomic variables)
#    Capture non-linear response curves (e.g., too little or too much rain)
# ─────────────────────────────────────────────────────────────────────────────
for col in ['temperature', 'rainfall', 'ph', 'humidity']:
    df[f'{col}_sq'] = df[col] ** 2

# ─────────────────────────────────────────────────────────────────────────────
# 5. LOG TRANSFORMS  (right-skewed features: NPK_product, rainfall)
# ─────────────────────────────────────────────────────────────────────────────
df['log_NPK_product'] = np.log1p(df['NPK_product'])
df['log_rainfall']    = np.log1p(df['rainfall'])
df['log_N']           = np.log1p(df['N'])
df['log_P']           = np.log1p(df['P'])
df['log_K']           = np.log1p(df['K'])

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARISE ENGINEERED FEATURE SET
# ─────────────────────────────────────────────────────────────────────────────
original_features  = FEATURES
engineered_features = [c for c in df.columns if c not in original_features + [TARGET]]

all_features = original_features + engineered_features
print(f"Original features  : {len(original_features)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Total features     : {len(all_features)}")
print(f"\nEngineered features:\n  {engineered_features}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE  via Random Forest (quick proxy — full training later)
# ─────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature'   : all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

importances['rank'] = importances.index + 1
print("\n── Top 15 Features by RF Importance ──")
print(importances.head(15).to_string(index=False))

importances.to_csv("artifacts/feature_importance_report.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION 1 — Feature Importance Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
top20 = importances.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#2ecc71' if f in original_features else '#3498db' for f in top20['feature']]
bars = ax.barh(top20['feature'][::-1], top20['importance'][::-1], color=colors[::-1])
ax.set_xlabel("Mean Decrease in Impurity", fontsize=12)
ax.set_title("Top 20 Feature Importances (Random Forest)\n"
             "■ Original   ■ Engineered", fontsize=13, fontweight='bold')

# custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Original'),
                   Patch(facecolor='#3498db', label='Engineered')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig("artifacts/feature_importance_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_importance_rf.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALISATION 2 — Correlation Heatmap of all features
# ─────────────────────────────────────────────────────────────────────────────
corr = df[all_features].corr()
fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.4, ax=ax,
            cbar_kws={'shrink': 0.6})
ax.set_title("Feature Correlation Heatmap (all engineered features)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("artifacts/feature_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATION 3 — Distribution of Top-8 features
# ─────────────────────────────────────────────────────────────────────────────
top8 = importances['feature'].head(8).tolist()
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, top8):
    ax.hist(df[feat], bins=40, color='#3498db', edgecolor='white', alpha=0.85)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
fig.suptitle("Distributions of Top-8 Features (by RF Importance)", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("artifacts/feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SELECT FINAL FEATURE SET  (drop highly correlated engineered features)
#     Threshold: |r| > 0.97 → keep the one with higher importance
# ─────────────────────────────────────────────────────────────────────────────
corr_abs = corr.abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
drop_cols = set()
for col in upper.columns:
    if any(upper[col] > 0.97):
        # keep the one with higher importance; drop the lower-ranked one
        partners = upper.index[upper[col] > 0.97].tolist()
        for partner in partners:
            rank_col     = importances.loc[importances.feature == col,     'rank'].values[0]
            rank_partner = importances.loc[importances.feature == partner, 'rank'].values[0]
            drop_cols.add(col if rank_col > rank_partner else partner)

print(f"\nDropping {len(drop_cols)} highly-correlated redundant features: {drop_cols}")
final_features = [f for f in all_features if f not in drop_cols]
print(f"Final feature count: {len(final_features)}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SCALE & SPLIT WITH FINAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
X_final = df[final_features]
y_final = le.transform(df[TARGET])

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_final, y_final, test_size=0.30, random_state=42, stratify=y_final
)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

scaler = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_val_sc = scaler.transform(X_val)
X_te_sc  = scaler.transform(X_te)

# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
payload = {
    # Scaled arrays
    "X_train": X_tr_sc,  "X_val": X_val_sc,  "X_test": X_te_sc,
    "y_train": y_tr,     "y_val": y_val,      "y_test": y_te,
    # Transformers
    "scaler": scaler, "label_encoder": le,
    # Metadata
    "feature_names"     : final_features,
    "original_features" : original_features,
    "engineered_features": [f for f in engineered_features if f not in drop_cols],
    "class_names"       : list(le.classes_),
    "importance_df"     : importances,
}

with open("artifacts/feature_engineered_data.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\n✅ Saved: artifacts/feature_engineered_data.pkl")
print("Keys:", list(payload.keys()))
print("\nDone. All artifacts written to ./artifacts/")
