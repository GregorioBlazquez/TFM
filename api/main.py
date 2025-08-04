from fastapi import FastAPI
from api.schemas import PredictionInput, PredictionOutput
from api.model_handler import predict_tourists

app = FastAPI(title="Tuorist Prediction API")

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    pred = predict_tourists(input_data.comunidad, input_data.periodo)
    return PredictionOutput(
        turistas_predichos=pred,
        modelo="ARIMA-v1"
    )
