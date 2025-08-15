from pydantic import BaseModel, Field

VALID_REGIONS = [
    "Andalucia",
    "Baleares",
    "Canarias",
    "Cataluña",
    "Madrid",
    "OtherCCAA",
    "Spain",
    "Valencia"
]

REGION_DESCRIPTION = (
    "Name of the autonomous community or 'Spain' for the total. "
    f"Possible values: {', '.join([repr(r) for r in VALID_REGIONS])}."
)

class PredictionInput(BaseModel):
    region: str = Field(..., description=REGION_DESCRIPTION)
    period: str = Field(..., description="Prediction period in format YYYYMM or YYYY'MMM'. Example: '2025M08'")

class PredictionOutput(BaseModel):
    predicted_tourists: int = Field(..., description="Estimated number of tourists")
    lower_ci: int = Field(..., description="Lower bound of the 95% confidence interval")
    upper_ci: int = Field(..., description="Upper bound of the 95% confidence interval")
    model: str = Field(..., description="Model used for the prediction, e.g. 'ARIMA-Andalucia'")

