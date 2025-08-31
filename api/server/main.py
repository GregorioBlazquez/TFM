from fastapi import FastAPI, HTTPException
from api.server.schemas import (
    ArimaInput, ArimaOutput, HistoricalInput, HistoricalOutput,
    TouristFeatures, ClusterOutput, ExpenditureOutput
)
from api.server.model_handler import (
    predict_tourists, get_historical_tourists,
    predict_cluster, predict_expenditure
)
from api.server.schemas import REGION_DESCRIPTION

app = FastAPI(title="Tourist Prediction API")

# --- Tourist prediction ---
@app.post(
    "/predict",
    response_model=ArimaOutput,
    summary="Predict number of tourists",
    description=(
        "Predicts the number of tourists for a given region and period using an ARIMA model.\n\n"
        f"**Region options:** {REGION_DESCRIPTION}\n\n"
        "**Period format:** YYYYMM or YYYY'MMM'. Example: '2025M08'."
    )
)
def predict(input_data: ArimaInput):
    """
    Predicts the number of tourists for the specified region and period.
    """
    try:
        pred, lower_ci, upper_ci, model_name = predict_tourists(input_data.region, input_data.period)
        return ArimaOutput(
            predicted_tourists=pred,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            model=model_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Historical data ---
@app.post(
    "/historical",
    response_model=HistoricalOutput,
    summary="Retrieve historical tourist data",
    description=(
        "Returns historical tourist numbers for a given region and period.\n\n"
        "**Region options:** Name of the autonomous community, 'Total' for Spain, or omit to get all regions.\n\n"
        "**Period format:** YYYY-MM (e.g., '2020-09'). If omitted, all available periods are returned.\n\n"
        "You can combine region and period, or provide only one of them:\n"
        "- region only → all periods for that region\n"
        "- period only → all regions for that period\n"
        "- both → specific period and region\n"
        "- neither → full dataset"
    )
)
def historical(input_data: HistoricalInput):
    """
    Retrieves historical tourist data for the specified region and/or period.
    """
    try:
        data = get_historical_tourists(input_data.region, input_data.period)
        return HistoricalOutput(data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Tourist clustering ---
@app.post(
    "/cluster",
    response_model=ClusterOutput,
    summary="Assign tourist profile cluster",
    description="Assigns a tourist profile to a cluster based on input features."
)
def cluster(input_data: TouristFeatures):
    try:
        cluster_id = predict_cluster(input_data.dict())
        return ClusterOutput(cluster=cluster_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Expenditure regression ---
@app.post(
    "/expenditure",
    response_model=ExpenditureOutput,
    summary="Predict tourist expenditure",
    description="Predicts daily average expenditure and provides SHAP-based interpretability."
)
def expenditure(input_data: TouristFeatures):
    try:
        pred, shap_list = predict_expenditure(input_data.dict())
        return ExpenditureOutput(predicted_expenditure=pred, shap_values=shap_list)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))