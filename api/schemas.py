from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    region: str = Field(..., description="Name of the autonomous community. Example: 'Andalusia'")
    period: str = Field(..., description="Prediction period in format YYYYMM or YYYY'MMM'. Example: '2025M08'")

class PredictionOutput(BaseModel):
    predicted_tourists: int = Field(..., description="Estimated number of tourists")
    model: str = Field(..., description="Model used for the prediction, e.g. 'ARIMA-v1'")

