from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    txn_count: int
    amount_std: float
    txn_hour: int
    txn_day: int
    txn_month: int
    ProductCategory: str
    ChannelId: str


class PredictionResponse(BaseModel):
    risk_probability: float
