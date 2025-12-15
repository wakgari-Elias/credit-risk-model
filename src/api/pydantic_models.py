from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
