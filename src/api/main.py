from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
from src.data_processing import build_pipeline

app = FastAPI()
model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
preprocessor = build_pipeline()

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """Predict risk probability for a new customer."""
    data = [features.dict().values()]
    columns = features.dict().keys()
    input_df = pd.DataFrame(data, columns=columns)
    input_processed = preprocessor.transform(input_df)
    prob = model.predict_proba(input_processed)[0][1]
    return {"risk_probability": prob}
