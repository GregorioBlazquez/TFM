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

# Schemas for ARIMA model input and output
class ArimaInput(BaseModel):
    region: str = Field(..., description=REGION_DESCRIPTION)
    period: str = Field(..., description="Prediction period in format YYYYMM or YYYY'MMM'. Example: '2025M08'")

class ArimaOutput(BaseModel):
    predicted_tourists: int = Field(..., description="Estimated number of tourists")
    lower_ci: int = Field(..., description="Lower bound of the 95% confidence interval")
    upper_ci: int = Field(..., description="Upper bound of the 95% confidence interval")
    model: str = Field(..., description="Model used for the prediction, e.g. 'ARIMA-Andalucia'")

# Schemas for historical data retrieval
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


# Input for cluster and expenditure model
class TouristFeatures(BaseModel):
    accommodation: str = Field(..., description="Type of accommodation ('Hotels and similar', 'Non-market', 'Other market')")
    purpose: str = Field(..., description="Trip purpose ('Leisure/holidays', 'Business', 'Other')")
    season: str = Field(..., description="Season of the year ('summer', 'spring', 'winter', 'autumn')")
    country: str = Field(..., description="Country of origin ('Occident and America', 'Russia + Rest of the world', 'United Kingdom')")
    region: str = Field(..., description="Destination region within Spain ('Valencian Community', 'Catalonia', 'Andalusia', 'Balearic Islands', 'Canary Islands', 'Madrid', 'Other')")
    trip_length: float = Field(..., description="Trip length in nights")
    daily_average_expenditure: Optional[float] = Field(
        None, description="Daily average expenditure if available (used only in clustering training)"
    )

# Schemas for clustering model output
class ClusterOutput(BaseModel):
    cluster: int = Field(..., description="Cluster label assigned to the tourist profile")

# Schemas for daily average expenditure prediction model output
class ShapValue(BaseModel):
    feature: str = Field(..., description="Name of the feature")
    impact: float = Field(..., description="Impact of the feature on the prediction")

class ExpenditureOutput(BaseModel):
    predicted_expenditure: float = Field(..., description="Predicted daily average expenditure")
    shap_values: List[ShapValue] = Field(
        ..., description="List of feature contributions for interpretability"
    )