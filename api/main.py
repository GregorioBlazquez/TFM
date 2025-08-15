from fastapi import FastAPI
from api.schemas import PredictionInput, PredictionOutput
from api.model_handler import predict_tourists

app = FastAPI(title="Tourist Prediction API")

@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict number of tourists",
    description=(
        "Given an autonomous community and a period in format YYYYMMM, "
        "returns the estimated number of tourists using an ARIMA model."
    )
)
def predict(input_data: PredictionInput):
    """
    Predicts the number of tourists for the specified region and period.
    
    - **region**: Full name of the autonomous community, e.g. `\"Andalusia\"`.
    - **period**: Period in format `YYYYMM` or `YYYYMmm`, e.g. `\"2025M08\"`.
    """
    pred = predict_tourists(input_data.region, input_data.period)
    return PredictionOutput(
        predicted_tourists=pred,
        model="ARIMA-v1"
    )
