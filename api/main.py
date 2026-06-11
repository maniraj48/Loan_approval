from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "loan_pipeline.pkl"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "predictions.log"

API_KEY = os.getenv("LOAN_API_KEY", "loan123")

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0.0",
    description="Predicts loan approval from applicant, income, loan, and credit details.",
)

request_count = 0


class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float = Field(gt=0, description="Loan amount in thousands.")
    Loan_Amount_Term: float = Field(gt=0, description="Loan term in months.")
    Credit_History: Literal[0.0, 1.0]
    Property_Area: Literal["Rural", "Semiurban", "Urban"]


class PredictionResponse(BaseModel):
    prediction: Literal["Approved", "Rejected"]
    approval_probability: float
    request_count: int


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python train_model.py` first."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


def require_api_key(x_api_key: str) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanApplication, x_api_key: str = Header(...)):
    global request_count

    require_api_key(x_api_key)
    request_count += 1

    payload = data.model_dump()
    df = pd.DataFrame([payload], columns=FEATURES)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    result = "Approved" if prediction == 1 else "Rejected"

    logging.info(
        "prediction=%s approval_probability=%.4f applicant_income=%.2f loan_amount=%.2f",
        result,
        probability,
        data.ApplicantIncome,
        data.LoanAmount,
    )

    return PredictionResponse(
        prediction=result,
        approval_probability=round(probability, 4),
        request_count=request_count,
    )


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/metrics")
def metrics(x_api_key: str = Header(...)):
    require_api_key(x_api_key)

    return {
        "request_count": request_count,
    }
