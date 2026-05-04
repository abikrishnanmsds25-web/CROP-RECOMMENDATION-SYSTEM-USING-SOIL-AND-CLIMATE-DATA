#!/usr/bin/env python3
"""
Crop Recommendation System — Model Training & Evaluation
=========================================================
Input  : artifacts/feature_engineered_data.pkl
Output : artifacts/models.pkl
         artifacts/training_report.csv
         artifacts/confusion_matrices.png
         artifacts/roc_curves.png
         artifacts/model_comparison.png
         artifacts/classification_reports.txt
"""


import pickle, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics        import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing  import label_binarize

os.makedirs("artifacts", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with open("artifacts/feature_engineered_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"];  y_train = data["y_train"]
X_val   = data["X_val"];    y_val   = data["y_val"]
X_test  = data["X_test"];   y_test  = data["y_test"]
class_names   = data["class_names"]
feature_names = data["feature_names"]
n_classes     = len(class_names)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Classes ({n_classes}): {class_names}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETER TUNING  (GridSearchCV on train set, scored on val logic)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Combine train+val for CV so we use all non-test data
X_tv = np.vstack([X_train, X_val])
y_tv = np.concatenate([y_train, y_val])

# ── 2a. Random Forest ──────────────────────────────────────────────────────
print("\n[1/3] Random Forest ...")
rf_params = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [None, 20, 30],
    "min_samples_split": [2, 5],
    "max_features"     : ["sqrt", "log2"],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
rf_gs.fit(X_tv, y_tv)
best_rf = rf_gs.best_estimator_
print(f"   Best params : {rf_gs.best_params_}")
print(f"   CV accuracy : {rf_gs.best_score_:.4f}")

# ── 2b. KNN ───────────────────────────────────────────────────────────────
print("\n[2/3] K-Nearest Neighbors ...")
knn_params = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights"    : ["uniform", "distance"],
    "metric"     : ["euclidean", "manhattan"],
}
knn_gs = GridSearchCV(
    KNeighborsClassifier(),
    knn_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
knn_gs.fit(X_tv, y_tv)
best_knn = knn_gs.best_estimator_
print(f"   Best params : {knn_gs.best_params_}")
print(f"   CV accuracy : {knn_gs.best_score_:.4f}")

# ── 2c. Decision Tree ─────────────────────────────────────────────────────
print("\n[3/3] Decision Tree ...")
dt_params = {
    "max_depth"        : [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "criterion"        : ["gini", "entropy"],
}
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
dt_gs.fit(X_tv, y_tv)
best_dt = dt_gs.best_estimator_
print(f"   Best params : {dt_gs.best_params_}")
print(f"   CV accuracy : {dt_gs.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

models = {
    "Random Forest"  : best_rf,
    "KNN"            : best_knn,
    "Decision Tree"  : best_dt,
}

results = {}
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

for name, model in models.items():
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)
    acc         = accuracy_score(y_test, y_pred)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    roc_auc     = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    cv_scores   = cross_val_score(model, X_tv, y_tv, cv=5, scoring="accuracy")

    results[name] = {
        "model"        : model,
        "y_pred"       : y_pred,
        "y_prob"       : y_prob,
        "accuracy"     : acc,
        "f1_macro"     : f1_macro,
        "f1_weighted"  : f1_weighted,
        "roc_auc"      : roc_auc,
        "cv_mean"      : cv_scores.mean(),
        "cv_std"       : cv_scores.std(),
    }
    print(f"\n{name}")
    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  F1 (macro)     : {f1_macro:.4f}")
    print(f"  F1 (weighted)  : {f1_weighted:.4f}")
    print(f"  ROC-AUC (OvR)  : {roc_auc:.4f}")
    print(f"  5-Fold CV      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE CLASSIFICATION REPORTS  (text)
# ─────────────────────────────────────────────────────────────────────────────
with open("artifacts/classification_reports.txt", "w") as f:
    for name, r in results.items():
        f.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
        f.write(classification_report(y_test, r["y_pred"], target_names=class_names))
print("\nSaved: artifacts/classification_reports.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE SUMMARY CSV
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {
        "Model"       : name,
        "Test Acc"    : f"{r['accuracy']:.4f}",
        "F1 Macro"    : f"{r['f1_macro']:.4f}",
        "F1 Weighted" : f"{r['f1_weighted']:.4f}",
        "ROC-AUC"     : f"{r['roc_auc']:.4f}",
        "CV Mean"     : f"{r['cv_mean']:.4f}",
        "CV Std"      : f"{r['cv_std']:.4f}",
    }
    for name, r in results.items()
])
summary.to_csv("artifacts/training_report.csv", index=False)
print("Saved: artifacts/training_report.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT — Model Comparison  (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
metrics     = ["accuracy", "f1_macro", "roc_auc", "cv_mean"]
metric_lbls = ["Test Accuracy", "F1 Macro", "ROC-AUC", "CV Accuracy"]
model_names = list(results.keys())
colors      = ["#2ecc71", "#3498db", "#e74c3c"]
x = np.arange(len(metrics)); w = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.87, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x + w)
ax.set_xticklabels(metric_lbls, fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Random Forest vs KNN vs Decision Tree", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT — Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
cmaps = ["Greens", "Blues", "Reds"]

for ax, (name, r), cmap in zip(axes, results.items(), cmaps):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.3, cbar=False, annot_kws={"size": 7})
    ax.set_title(f"{name}\nAcc={r['accuracy']:.4f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("artifacts/confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOT — Per-class F1 heatmap (which crops are hardest to predict?)
# ─────────────────────────────────────────────────────────────────────────────
f1_data = {}
for name, r in results.items():
    report = classification_report(y_test, r["y_pred"],
                                   target_names=class_names, output_dict=True)
    f1_data[name] = [report[c]["f1-score"] for c in class_names]

f1_df = pd.DataFrame(f1_data, index=class_names)
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(f1_df, annot=True, fmt=".2f", cmap="YlGn",
            linewidths=0.4, ax=ax, vmin=0.7, vmax=1.0,
            cbar_kws={"label": "F1-Score"})
ax.set_title("Per-Class F1-Score by Model", fontsize=13, fontweight='bold')
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Crop", fontsize=11)
plt.tight_layout()
plt.savefig("artifacts/per_class_f1_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/per_class_f1_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOT — RF Feature Importance (final trained model)
# ─────────────────────────────────────────────────────────────────────────────
fi = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 7))
fi.head(15)[::-1].plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='white')
ax.set_title("Random Forest — Top 15 Feature Importances (Trained Model)", fontsize=12, fontweight='bold')
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig("artifacts/rf_feature_importance_trained.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/rf_feature_importance_trained.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
model_bundle = {
    "random_forest"  : best_rf,
    "knn"            : best_knn,
    "decision_tree"  : best_dt,
    "scaler"         : data["scaler"],
    "label_encoder"  : data["label_encoder"],
    "feature_names"  : feature_names,
    "class_names"    : class_names,
    "results_summary": summary,
}
with open("artifacts/models.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\n✅ Saved: artifacts/models.pkl")
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))

best_name = max(results, key=lambda n: results[n]["accuracy"])
print(f"\n🏆 Best model: {best_name}  (Test Accuracy = {results[best_name]['accuracy']:.4f})")
