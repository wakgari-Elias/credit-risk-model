from fastapi import FastAPI
import mlflow
import pandas as pd
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

# -----------------------------
# Load model from MLflow Registry
# -----------------------------
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    df = pd.DataFrame([data.dict()])
    prob = model.predict(df)[0]
    return PredictionResponse(risk_probability=float(prob))
