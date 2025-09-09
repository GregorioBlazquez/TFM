# Data Models Documentation

## Overview
This document describes the **datasets, preprocessing workflows, and machine learning models** used in this project.  
It is intended for **developers** who want to reproduce, retrain, or extend the existing models.  

The pipeline covers:
- Loading and cleaning **FRONTUR** (tourist flows) and **EGATUR** (tourist expenditure) datasets.
- Feature engineering and exploratory analysis.
- Model training for:
  - **Tourist forecasting (ARIMA models)**
  - **Tourist profile clustering**
  - **Daily average expenditure regression**
- Model persistence and loading for inference in the API.
- Interpretability tools (SHAP) for expenditure predictions.

---

## Data Sources

### Raw Data
Located in `data/raw/`:
- **FRONTUR/** → Survey on inbound tourist flows (visitors, trip characteristics).
- **EGATUR/** → Survey on tourist expenditure (spending patterns, trip details).
- **Accompanying PDFs** (e.g., `FRONTUR0625.pdf`, `EGATUR0625.pdf`) with methodological notes, avaiable in `docs`.

### Processed Data
Located in `data/processed/`:
- **`num_tourists.csv`** → Aggregated monthly number of tourists by Autonomous Community (used for ARIMA).
- **`egatur_data_ready.csv`** → Cleaned dataset with selected EGATUR features for clustering and expenditure modeling.
- **`egatur_full_dataset.csv`** → Full processed EGATUR dataset with additional variables.

---

## Notebooks and Workflow
Notebooks are stored in `notebooks/` and serve as the main training scripts:
1. **01_EDA_FRONTUR.ipynb** → Exploratory analysis of FRONTUR data.
2. **02_EDA_EGATUR.ipynb** → Exploratory analysis of EGATUR data, feature selection.
3. **03_Model_ARIMA.ipynb** → Training ARIMA models per region.
4. **04_Modeling_EGATUR.ipynb** → Training clustering and expenditure regression models.

The workflow is:
- **EDA** → Validate and clean data.
- **Feature engineering** → Prepare inputs (categorical encoding, numerical scaling).
- **Model training** → Train and validate models.
- **Persistence** → Save trained models under `models/`.

---

## Models

### ARIMA Tourist Forecasting
- Folder: `models/arima/`
- Files: `arima_model_<Region>.pkl`
  - Example: `arima_model_Andalucia.pkl`, `arima_model_Spain.pkl`
- Structure: Each pickle contains:
  - Fitted ARIMA model
  - Last training period
- Usage: Predicts number of tourists for a target period beyond training data.

### Tourist Clustering
- Folder: `models/cluster/`
- Files:
  - `cluster_classifier_eda_lr.joblib` (logistic regression-based clustering)
  - `cluster_classifier_eda_nn.joblib` (neural network alternative, not used in production)
- Input features: trip purpose, accommodation, region, origin country, trip length, daily expenditure.
- Output: Cluster ID (integer label).

### Expenditure Regression
- Folder: `models/daily_avg_exp/`
- Files:
  - `expenditure_model.pkl` (default model with preprocessor and SHAP explainer)
  - `histgradientboosting_model.joblib` (alternative model for experimentation)
- Output: Predicted daily average expenditure.
- Includes **SHAP values** for interpretability.

---

## Interpretability

### SHAP (SHapley Additive exPlanations)
- Integrated with the expenditure regression model.
- Returns per-feature impact values for each prediction.
- Used in the API (`/expenditure` endpoint) to explain predictions.

### Other Techniques
- Feature importance during training.
- Partial Dependence Plots (in notebooks, not served in API).

---

## Limitations and Retraining

- **Historical dependency** → ARIMA requires recent and consistent time series; models must be retrained as new data arrives.
- **Data drift** → Tourist behaviors may change (e.g., pandemics, geopolitical events), making clustering and expenditure models less accurate over time.
- **EGATUR/FRONTUR updates** → New survey versions may introduce schema changes requiring preprocessing adjustments.

**Retraining workflow**:
1. Update raw data in `data/raw/`.
2. Re-run preprocessing notebooks.
3. Retrain models in notebooks (`03_Model_ARIMA.ipynb`, `04_Modeling_EGATUR.ipynb`).
4. Replace model files in `models/`.
5. Restart server or redeploy Docker/ACI.

---

## Usage in Codebase

Models are consumed in `api/server/model_handler.py`:
- **ARIMA models** → Preloaded from `models/arima/`.
- **Clustering model** → Loaded once from `models/cluster/cluster_classifier_eda_lr.joblib`.
- **Expenditure model** → Loaded once from `models/daily_avg_exp/expenditure_model.pkl`.

Endpoints in `api/server/main.py`:
- `/predict` → Tourist forecasting (ARIMA).
- `/historical` → Historical tourist numbers (from `data/processed/num_tourists.csv`).
- `/cluster` → Tourist profile clustering.
- `/expenditure` → Daily average expenditure with SHAP explanations.

---

## 🧭 Navigation

- [⬅️ Previous: Client](/04_client.md)
- [🏠 Main index](../README.md#documentation)
- [➡️ Next: Deployment](/06_deployment.md)