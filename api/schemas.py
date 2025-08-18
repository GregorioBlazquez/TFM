from pydantic import BaseModel, Field
from typing import Dict, Optional, List

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
    "Alternatively, use 'OtherCCAA' for the aggregated group of other regions.\n\n"
    f"**Possible values:** {', '.join([repr(r) for r in VALID_REGIONS])}."
)

class PredictionInput(BaseModel):
    region: str = Field(..., description=REGION_DESCRIPTION)
    period: str = Field(..., description="Prediction period in format YYYYMM or YYYY'MMM'. Example: '2025M08'")

class PredictionOutput(BaseModel):
    predicted_tourists: int = Field(..., description="Estimated number of tourists")
    lower_ci: int = Field(..., description="Lower bound of the 95% confidence interval")
    upper_ci: int = Field(..., description="Upper bound of the 95% confidence interval")
    model: str = Field(..., description="Model used for the prediction, e.g. 'ARIMA-Andalucia'")

class HistoricalInput(BaseModel):
    region: Optional[str] = Field(
        None, 
        description=(
            "Name of the autonomous community, 'Spain' for the total, "
            "or 'OtherCCAA' for the aggregated group of other regions. "
            "If omitted, data for all regions is returned.\n\n"
            f"**Possible values:** {', '.join([repr(r) for r in VALID_REGIONS])}."
        )
    )
    period: Optional[str] = Field(None, description="YYYY-MM format. If omitted, returns all available periods.")

class HistoricalOutput(BaseModel):
    data: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Mapping from period → region → observed tourists"
    )
