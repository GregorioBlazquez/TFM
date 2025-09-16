import os
from typing import Optional
import joblib
from datetime import datetime
from api.server.schemas import VALID_REGIONS
import pandas as pd
from typing import Optional, Dict, List, Tuple
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/frontur_data_ready.csv")

ARIMA_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models/arima")
CLUSTER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/cluster/cluster_classifier_eda_lr.joblib")
EXPENDITURE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/daily_avg_exp/expenditure_model.pkl")

# --- Preload models at import time ---
# Load all ARIMA models into a dict {region: model_data}
ARIMA_MODELS = {}
for filename in os.listdir(ARIMA_MODELS_DIR):
    if filename.endswith(".pkl"):
        region = filename.replace("arima_model_", "").replace(".pkl", "")
        ARIMA_MODELS[region] = joblib.load(os.path.join(ARIMA_MODELS_DIR, filename))

# Load cluster model once
CLUSTER_MODEL = joblib.load(CLUSTER_MODEL_PATH)

# Load expenditure model once
EXPENDITURE_MODEL = joblib.load(EXPENDITURE_MODEL_PATH)

# --- Helpers ---
def parse_period(period_str: str) -> datetime:
    try:
        return datetime.strptime(period_str, "%Y%m")
    except ValueError:
        year = int(period_str[:4])
        month = int(period_str[-2:])
        return datetime(year, month, 1)
# --- Tourist prediction (ARIMA) ---
def predict_tourists(region: str, period: str):
    if region not in ARIMA_MODELS:
        raise ValueError(f"Region '{region}' not recognized. Valid: {list(ARIMA_MODELS.keys())}")

    model_data = ARIMA_MODELS[region]
    model = model_data["model"]
    last_period = model_data["last_period"]

    target_date = parse_period(period)
    if target_date <= last_period:
        raise ValueError(f"Target date {target_date.date()} must be after last trained date {last_period.date()}.")

    steps_ahead = (target_date.year - last_period.year) * 12 + (target_date.month - last_period.month)
    forecast, conf_int = model.predict(n_periods=steps_ahead, return_conf_int=True)

    pred_value = int(forecast[-1])
    lower_ci = int(conf_int[-1, 0])
    upper_ci = int(conf_int[-1, 1])

    return pred_value, lower_ci, upper_ci, f"ARIMA-{region}"

# --- Historical data ---
TOURIST_DATA = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Period'])
TOURIST_DATA.rename(columns={"CCAA": "region", "Total": "tourists"}, inplace=True)

def get_historical_tourists(region: Optional[str] = None, period: Optional[str] = None):
    df = TOURIST_DATA.copy()

    if period:
        try:
            period_dt = pd.to_datetime(period)
            df = df[df["Period"] == period_dt]
        except Exception:
            raise ValueError(f"Invalid period format: {period}, expected YYYY-MM or YYYY-MM-DD")

    if region:
        if region != "Total":
            df = df[df["region"].str.contains(region, case=False)]
        else:
            df = df[df["region"] == "Total"]

    if df.empty:
        raise ValueError(f"No data found for region='{region}' and period='{period}'")

    output = {}
    for p, group in df.groupby("Period"):
        output[str(p.date())] = {r: float(v) for r, v in zip(group["region"], group["tourists"])}

    return output

# --- Tourist clustering ---
def predict_cluster(features: Dict) -> int:
    features = features.copy()

    # Check if daily_average_expenditure is present
    dae = features.get("daily_average_expenditure", None)

    # If not predict
    if dae is None or (isinstance(dae, float) and pd.isna(dae)):
        features_for_pred = {k: v for k, v in features.items() if k != "daily_average_expenditure"}
        pred_exp, _ = predict_expenditure(features_for_pred)
        features["daily_average_expenditure"] = float(pred_exp)

    X = pd.DataFrame([features])
    cluster = int(CLUSTER_MODEL.predict(X)[0])
    return cluster


# --- Expenditure regression with SHAP ---
def predict_expenditure(features: Dict) -> Tuple[float, List[Dict[str, float]]]:
    regressor = EXPENDITURE_MODEL["model"]
    preproc = EXPENDITURE_MODEL["preprocessor"]
    explainer = EXPENDITURE_MODEL["explainer"]

    X = pd.DataFrame([features])
    X_trans = preproc.transform(X)

    pred = float(regressor.predict(X_trans)[0])
    shap_values = explainer(X_trans)
    feature_names = preproc.get_feature_names_out()
    shap_list = [
        {"feature": fname, "impact": float(val)} 
        for fname, val in zip(feature_names, shap_values.values[0])
    ]

    return pred, shap_list