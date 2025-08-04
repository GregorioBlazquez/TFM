import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/arima_model_total.pkl")
model = joblib.load(MODEL_PATH)

def predict_tourists(periodo: str) -> int:
    # Arima logic
    # (...)
    pred = model.predict(n_periods=1)[0]  # dummy prediction logic
    return int(pred)
