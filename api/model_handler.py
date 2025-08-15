import os
import joblib
from datetime import datetime
from api.schemas import VALID_REGIONS

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
