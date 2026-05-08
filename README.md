# Crop Recommendation System Using Machine Learning

> A supervised multi-class classification pipeline that recommends optimal crop types based on soil nutrient profiles and climatic conditions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [System Architecture](#3-system-architecture)
4. [Module Descriptions](#4-module-descriptions)
   - [4.1 Data Preprocessing](#41-data-preprocessing)
   - [4.2 Feature Engineering](#42-feature-engineering)
   - [4.3 Model Training & Evaluation](#43-model-training--evaluation)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Artifacts & Outputs](#7-artifacts--outputs)
8. [Reproducing the Experiments](#8-reproducing-the-experiments)
9. [Project Structure](#9-project-structure)
10. [Dependencies](#10-dependencies)

---

## 1. Project Overview

Precision agriculture demands data-driven decision support systems that can guide farmers in selecting the most suitable crop for a given parcel of land. This project implements a complete end-to-end machine learning pipeline for **automated crop recommendation**, leveraging soil macro-nutrient measurements and local climatic parameters as predictive features.

The pipeline addresses the following research objectives:

- To preprocess and normalise multi-variate agro-climatic data in a manner that prevents data leakage across train, validation, and test partitions.
- To systematically engineer domain-informed features — including nutrient ratios, climate interaction indices, pH-nutrient coupling terms, and polynomial expansions — that improve the discriminative capacity of baseline features.
- To compare the predictive performance of three classical supervised classifiers — **Random Forest**, **K-Nearest Neighbours (KNN)**, and **Decision Tree** — under rigorous hyperparameter optimisation via stratified cross-validation.
- To produce interpretable evaluation artefacts, including confusion matrices, per-class F1 heatmaps, ROC curves, and feature importance rankings.

---

## 2. Dataset Description

| Property | Detail |
|---|---|
| **Source** | `Crop_recommendation.csv` |
| **Total Samples** | 2,200 |
| **Classes** | 22 crop types (100 samples per class — perfectly balanced) |
| **Class Balance** | Stratified; no oversampling required |
| **Missing Values** | None |

### Input Features

| Feature | Unit | Description |
|---|---|---|
| `N` | kg/ha | Ratio of Nitrogen content in soil |
| `P` | kg/ha | Ratio of Phosphorus content in soil |
| `K` | kg/ha | Ratio of Potassium content in soil |
| `temperature` | °C | Mean ambient temperature |
| `humidity` | % | Relative humidity |
| `ph` | — | Soil pH value (0–14 scale) |
| `rainfall` | mm | Annual rainfall |

### Target Variable

| Variable | Type | Description |
|---|---|---|
| `label` | Categorical (string) | Recommended crop name |

> **Note on Outliers:** IQR-based outlier analysis is performed during preprocessing for diagnostic purposes. No records are removed, as agronomically extreme values — such as elevated nitrogen for rice cultivation — are domain-valid observations that encode meaningful signal.

---

## 3. System Architecture

The pipeline is composed of three sequential, modular scripts. Each module reads from the output of its predecessor and writes artefacts to the shared `artifacts/` directory.

```
Crop_recommendation.csv
         │
         ▼
┌─────────────────────────┐
│  preprocessing.py       │  → Label encoding, stratified split (70/15/15),
│                         │    StandardScaler fit on train only
└────────────┬────────────┘
             │  preprocessed_data.pkl
             ▼
┌─────────────────────────┐
│  feature_engineering.py │  → Domain ratio features, climate interactions,
│                         │    pH-nutrient coupling, polynomial & log transforms,
│                         │    RF importance ranking, correlation pruning
└────────────┬────────────┘
             │  feature_engineered_data.pkl
             ▼
┌─────────────────────────┐
│  model_training.py      │  → GridSearchCV hyperparameter tuning,
│                         │    test set evaluation, visualisation, model export
└────────────┬────────────┘
             │  models.pkl + plots + reports
             ▼
         artifacts/
```

---

## 4. Module Descriptions

### 4.1 Data Preprocessing

**Script:** `preprocessing.py`

This module establishes a clean, leak-free data foundation for all downstream modelling steps.

**Key operations:**

**Label Encoding** — The categorical target variable (`label`) is transformed into integer class indices using `sklearn.preprocessing.LabelEncoder`. The fitted encoder is persisted alongside the data to ensure consistent inverse mapping at inference time.

**Stratified Train/Validation/Test Split** — The dataset is partitioned in a 70/15/15 ratio using two sequential `train_test_split` calls with `stratify=y_encoded`. Stratification preserves the original class distribution (100 samples per class) across all three partitions, guarding against evaluation bias.

**Feature Scaling** — A `StandardScaler` is fit exclusively on the training partition and subsequently applied to the validation and test partitions via `transform` (not `fit_transform`). This design strictly prevents data leakage, as statistics from held-out data must never influence the scaling parameters used during training.

**Output:** `artifacts/preprocessed_data.pkl` containing scaled feature arrays, encoded target arrays, the fitted scaler, and the label encoder.

---

### 4.2 Feature Engineering

**Script:** `feature_engineering.py`

This module enriches the original seven-feature space with 22 additional domain-informed and statistically motivated features, expanding the total feature dimensionality before pruning redundant variables.

#### 4.2.1 Domain-Driven Ratio Features

Nutrient ratios are widely used in agronomy to characterise soil fertility balance:

| Feature | Formula | Agronomic Rationale |
|---|---|---|
| `N_P_ratio` | N / (P + ε) | Nitrogen–Phosphorus balance |
| `N_K_ratio` | N / (K + ε) | Nitrogen–Potassium balance |
| `P_K_ratio` | P / (K + ε) | Phosphorus–Potassium balance |
| `NPK_sum` | N + P + K | Total macro-nutrient load |
| `NPK_product` | N × P × K | Nutrient synergy interaction term |

#### 4.2.2 Climate Interaction Features

These composite indices capture non-additive relationships between temperature, humidity, and precipitation:

| Feature | Formula | Interpretation |
|---|---|---|
| `temp_humidity_index` | temp × humidity / 100 | Wet-bulb / heat-stress proxy |
| `rainfall_humidity` | rainfall × humidity / 100 | Water availability index |
| `temp_rainfall_ratio` | temp / (rainfall + ε) | Aridity indicator |

#### 4.2.3 pH–Nutrient Interaction Features

Nutrient availability in soil is highly pH-dependent due to solubility dynamics. Interaction terms between pH and each macro-nutrient (`ph_N_interaction`, `ph_P_interaction`, `ph_K_interaction`) encode these coupled effects.

#### 4.2.4 Polynomial Features

Second-degree polynomial expansions of key agronomic variables (`temperature`, `rainfall`, `ph`, `humidity`) allow the model to capture non-linear response curves — for instance, the optimal rainfall range for a given crop.

#### 4.2.5 Log Transforms

Right-skewed features — including `NPK_product`, `rainfall`, and individual macro-nutrients — are log-transformed via `log1p` to reduce skewness and improve numerical stability during optimisation.

#### 4.2.6 Feature Selection via Correlation Pruning

A Random Forest classifier is trained on the expanded feature set to compute Mean Decrease in Impurity (MDI) importance scores. Features with pairwise absolute Pearson correlation exceeding **|r| > 0.97** are considered redundant; the lower-ranked member of each correlated pair is dropped. This process retains the most informative non-redundant feature subset for model training.

**Output:** `artifacts/feature_engineered_data.pkl`, `artifacts/feature_importance_report.csv`, and three visualisation files.

---

### 4.3 Model Training & Evaluation

**Script:** `model_training.py`

This module performs systematic hyperparameter optimisation and comparative evaluation of three candidate classifiers.

#### Hyperparameter Optimisation

`GridSearchCV` with 5-fold stratified cross-validation is applied to the combined train+validation data (`X_tv`, `y_tv`). The scoring criterion is **accuracy**. Search spaces are defined as follows:

**Random Forest**
```
n_estimators      : [100, 200]
max_depth         : [None, 20, 30]
min_samples_split : [2, 5]
max_features      : ['sqrt', 'log2']
```

**K-Nearest Neighbours**
```
n_neighbors : [3, 5, 7, 9, 11]
weights     : ['uniform', 'distance']
metric      : ['euclidean', 'manhattan']
```

**Decision Tree**
```
max_depth         : [5, 10, 15, 20, None]
min_samples_split : [2, 5, 10]
criterion         : ['gini', 'entropy']
```

The best estimator from each grid search is subsequently evaluated on the held-out test set.

---

## 5. Machine Learning Models

### Random Forest Classifier

An ensemble of decision trees trained via bagging. Each tree is fit on a bootstrap sample of the training data, and predictions are aggregated by majority vote. The model is robust to overfitting and naturally provides feature importance estimates via MDI. It is well-suited to high-dimensional tabular data with complex feature interactions.

### K-Nearest Neighbours (KNN)

A non-parametric, instance-based classifier that assigns class labels by plurality vote among the *k* nearest training examples in feature space. KNN is sensitive to feature scale, making prior standardisation essential. Its performance depends heavily on the choice of *k*, distance metric, and weighting scheme.

### Decision Tree Classifier

A recursive binary partitioning model that selects splits to minimise impurity (Gini or entropy) at each node. Decision trees are interpretable and scale-invariant, but are prone to overfitting without adequate depth constraints. `max_depth` and `min_samples_split` regularisation are applied during grid search.

---

## 6. Evaluation Metrics

All three models are evaluated on the held-out **test set** using the following metrics:

| Metric | Description |
|---|---|
| **Test Accuracy** | Proportion of correctly classified samples |
| **F1-Score (Macro)** | Unweighted mean F1 across all 22 classes |
| **F1-Score (Weighted)** | Class-frequency-weighted mean F1 |
| **ROC-AUC (OvR, Macro)** | One-vs-Rest multi-class AUC averaged across classes |
| **5-Fold CV Accuracy** | Mean ± std of cross-validated accuracy on train+val data |

---

## 7. Artifacts & Outputs

All outputs are written to the `artifacts/` directory:

| File | Description |
|---|---|
| `preprocessed_data.pkl` | Scaled splits + fitted scaler + label encoder |
| `feature_engineered_data.pkl` | Engineered + pruned feature splits |
| `feature_importance_report.csv` | RF importance rank of all features |
| `models.pkl` | Serialised best models + transformers |
| `training_report.csv` | Summary table of all evaluation metrics |
| `classification_reports.txt` | Per-class precision, recall, F1 for all models |
| `model_comparison.png` | Grouped bar chart of test metrics across models |
| `confusion_matrices.png` | Side-by-side confusion matrices (test set) |
| `per_class_f1_heatmap.png` | Heatmap of per-class F1 by model |
| `rf_feature_importance_trained.png` | RF feature importances (final trained model) |
| `feature_importance_rf.png` | RF importances from feature engineering phase |
| `feature_correlation_heatmap.png` | Lower-triangle Pearson correlation heatmap |
| `feature_distributions.png` | Histograms of top-8 features by importance |

---

## 8. Reproducing the Experiments

Ensure `Crop_recommendation.csv` is present in the working directory, then execute the scripts in order:

```bash
# Step 1 — Preprocess raw data
python preprocessing.py

# Step 2 — Engineer and select features
python feature_engineering.py

# Step 3 — Train, tune, and evaluate models
python model_training.py
```

All intermediate and final artefacts will be written to `artifacts/`.

---

## 9. Project Structure

```
.
├── Crop_recommendation.csv       # Raw dataset
├── preprocessing.py              # Data cleaning, encoding, splitting, scaling
├── feature_engineering.py        # Feature creation, importance ranking, pruning
├── model_training.py             # Hyperparameter tuning, evaluation, export
├── 01_data_preprocessing_eda.ipynb  # Exploratory data analysis notebook
└── artifacts/
    ├── preprocessed_data.pkl
    ├── feature_engineered_data.pkl
    ├── feature_importance_report.csv
    ├── models.pkl
    ├── training_report.csv
    ├── classification_reports.txt
    └── *.png                     # Visualisation outputs
```

---

## 10. Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | Preprocessing, modelling, evaluation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `pickle` | Artefact serialisation |

Install all dependencies via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (make sure `artifacts/models.pkl` is committed, or use Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Point to `app.py` in your repo.
4. Done — no extra config needed.

---



- `artifacts/models.pkl` **must exist** before launching the app.
- The app auto-engineers features to match the training pipeline.
- If you added custom features during training, ensure `build_features()` in `app.py` mirrors them.


