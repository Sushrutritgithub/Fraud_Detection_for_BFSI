from fastapi import APIRouter
import joblib
from pydantic import BaseModel
from src.preprocessing.Preprocessing1 import preprocess_input
from src.utils.explainer import generate_risk_explanation
from src.database.save_prediction import save_prediction
import uuid
import json
import numpy as np
import time

router = APIRouter()

model = joblib.load("src/models/fraud_model.pkl")
feature_order = joblib.load("src/models/model_features.pkl")


try:
    with open("metrics.json", "r") as f:
        saved_metrics = json.load(f)
except:
    saved_metrics = {}

class Transaction(BaseModel):
    amount: float
    sender_old_balance: float
    sender_new_balance: float
    receiver_old_balance: float
    receiver_new_balance: float
    is_flagged: int
    transaction_type: str  


# @router.get("/")
# def root():
#     return {"message": "Fraud Detection API Running"}


@router.post("/predict")
def predict_transaction(txn: Transaction):
    start_time = time.time()
    
    
    data = txn.dict()

   
    X = preprocess_input(data, feature_order)

    
    prob = model.predict_proba(X)[0][1]

    
    threshold = 0.10
    pred = int(prob > threshold)

    explanation = generate_risk_explanation(data, prob)

    transaction_id = str(uuid.uuid4())

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    save_prediction(
        transaction_id,
        float(prob),
        pred,
        explanation,
        round(latency_ms, 2)
    )
    # Optional: Save prediction to DB
    # from src.database.mysql_connection import save_prediction
    # save_prediction(data, prob, pred)

    return {
        "transaction_id": transaction_id,
        "fraud_probability": float(prob),
        "is_fraud": pred,
        "explanation": explanation,
        "latency_ms": round(latency_ms, 2)
    }


@router.get("/metrics")
def get_metrics():
    print("Loaded metrics:", saved_metrics)
    return saved_metrics

# uvicorn src.api.predict:app --reload