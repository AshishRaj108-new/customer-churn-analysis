"""FastAPI service for one-customer churn estimates and SHAP reasons.

Run from the repository root:
    python -m uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", PROJECT_ROOT / "models" / "churn_live_model.joblib"))
THRESHOLD = 0.5
_BUNDLE: dict | None = None

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Scores one Telco customer with the selected trained model and explains the result with SHAP.",
    version="1.0.0",
)

_extra_origins = [value.strip() for value in os.environ.get("CHURN_ALLOWED_ORIGINS", "").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ashishraj108-new.github.io",
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        *_extra_origins,
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    class Config:
        extra = "forbid"
        allow_inf_nan = False


def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def get_explainer(bundle: dict):
    if "_explainer" not in bundle:
        classifier = bundle["pipeline"].named_steps["classifier"]
        if bundle["modelName"] == "Logistic Regression":
            bundle["_explainer"] = shap.LinearExplainer(classifier, bundle["background"])
        else:
            bundle["_explainer"] = shap.TreeExplainer(classifier, data=bundle["background"])
    return bundle["_explainer"]


def positive_class_values(explanation) -> np.ndarray:
    values = explanation.values if hasattr(explanation, "values") else explanation
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] > 1:
            values = values[:, :, 1]
        elif values.shape[0] > 1:
            values = values[1]
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"Unexpected SHAP output shape for one customer: {values.shape}")
    return values[0]


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        return {"status": "model_missing", "message": "Run src/train_churn_model.py to create the local model artifact."}
    return {"status": "ready", "model": bundle["modelName"]}


@app.post("/predict")
def predict(customer: CustomerFeatures) -> dict:
    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="The model file is missing. From the project root, run: python src/train_churn_model.py",
        ) from error

    features = pd.DataFrame([customer.dict()], columns=bundle["featureColumns"])
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(features)[0, 1])
    threshold = float(bundle.get("threshold", THRESHOLD))
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    shap_values = positive_class_values(get_explainer(bundle)(transformed))

    reasons = []
    for feature, indices in bundle["featureGroups"]:
        impact = float(shap_values[indices].sum())
        reasons.append({
            "feature": feature,
            "value": str(features.iloc[0][feature]),
            "direction": "raises" if impact > 0 else "lowers",
            "signedImpact": round(impact, 5),
            "impact": round(abs(impact), 5),
        })
    reasons.sort(key=lambda reason: reason["impact"], reverse=True)

    return {
        "model": bundle["modelName"],
        "probability": round(probability, 5),
        "predictedChurn": "Yes" if probability >= threshold else "No",
        "threshold": threshold,
        "reasons": reasons[:5],
        "explanationNote": "SHAP describes how these recorded features influenced this model score; it does not prove cause. SHAP impact values are model-score units, not percentage points.",
    }
