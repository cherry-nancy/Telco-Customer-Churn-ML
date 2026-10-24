"""
Shared pytest setup.

src/serving/inference.py loads the model at import time, so MODEL_DIR must be set
before any test imports it. CI has no /app/model or mlruns/, so point it at the
model bundled in the repo (the same one the Dockerfile copies into the image).
"""

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_MODEL = ROOT / "src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model"
os.environ.setdefault("MODEL_DIR", str(BUNDLED_MODEL))

FIXTURE_CSV = ROOT / "tests/fixtures/sample_customers.csv"

# The 18 fields the API's CustomerData schema accepts
API_FIELDS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges",
]


@pytest.fixture  # a fixture is a resuable test setup data that pytest builds and passes
# into your tests for you. 
def raw_df() -> pd.DataFrame:
    """20 real rows covering every category value, both Churn labels and a blank TotalCharges."""
    return pd.read_csv(FIXTURE_CSV)


@pytest.fixture # it builds and returns one sample customer as a Python dictionary.
def customer() -> dict:
    """One valid /predict payload (high-risk profile from the Gradio examples)."""
    return {
        "gender": "Female", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1, "MonthlyCharges": 85.0, "TotalCharges": 85.0,
    }

