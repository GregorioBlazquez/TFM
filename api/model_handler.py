import os
from typing import Optional
import joblib
from datetime import datetime
from api.schemas import VALID_REGIONS
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/num_tourists.csv")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models/arima")

def parse_period(period_str: str) -> datetime:
    try:
        return datetime.strptime(period_str, "%Y%m")
    except ValueError:
        year = int(period_str[:4])
        month = int(period_str[-2:])
        return datetime(year, month, 1)

def predict_tourists(region: str, period: str):
    if region not in VALID_REGIONS:
        raise ValueError(f"Region '{region}' not recognized. Valid: {VALID_REGIONS}")

    model_filename = f"arima_model_{region}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found for region '{region}'.")

    model_data = joblib.load(model_path)
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

