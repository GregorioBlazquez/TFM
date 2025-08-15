from fastapi import FastAPI, HTTPException
from api.schemas import PredictionInput, PredictionOutput
from api.model_handler import predict_tourists
from api.schemas import REGION_DESCRIPTION

app = FastAPI(title="Tourist Prediction API")

@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict number of tourists",
    description=(
        "Predicts the number of tourists for a given region and period using an ARIMA model.\n\n"
        f"**Region options:** {REGION_DESCRIPTION}\n\n"
        "**Period format:** YYYYMM or YYYY'MMM'. Example: '2025M08'."
    )
)
def predict(input_data: PredictionInput):
    """
    Predicts the number of tourists for the specified region and period.
    """
    try:
        pred, lower_ci, upper_ci, model_name = predict_tourists(input_data.region, input_data.period)
        return PredictionOutput(
            predicted_tourists=pred,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            model=model_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
