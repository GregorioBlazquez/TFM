from pydantic import BaseModel

class PredictionInput(BaseModel):
    comunidad: str  # Ej: "Andalucía"
    periodo: str    # Ej: "2025M08"

class PredictionOutput(BaseModel):
    turistas_predichos: int
    modelo: str
