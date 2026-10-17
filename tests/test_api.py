import os
import sys
import pytest
from fastapi.testclient import TestClient

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "backend"))

from main import app

client = TestClient(app)

valid_patient = {
    "Age": 55,
    "Gender": "Female",
    "Heart_Rate": 78,
    "Systolic_BP": 125,
    "Diastolic_BP": 82,
    "Cholesterol": 195,
    "Glucose": 98,
    "Troponin": 0.04,
    "PR_Interval": 155,
    "QRS_Duration": 92,
    "QT_Interval": 380,
    "P_Wave_Duration": 88,
    "T_Wave_Duration": 140,
    "ST_Segment_Level": 0.02,
    "RR_Interval": 820,
    "HRV_SDNN": 58,
    "HRV_RMSSD": 44,
    "R_Peak_Amplitude": 1.9,
    "S_Wave_Amplitude": -1.1
}


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_dataset_summary():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 4327
    assert data["total_columns"] == 20
    assert len(data["target_classes"]) == 4


def test_api_dataset_features():
    response = client.get("/api/dataset/features")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 19
    assert len(data["clinical_features"]) == 7
    assert len(data["ecg_features"]) == 11


def test_api_target_distribution():
    response = client.get("/api/dataset/target-distribution")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 4327
    assert "Normal" in data["distribution"]


def test_api_model_metrics():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_accuracy" in data
    assert "test_macro_f1" in data
    assert "confusion_matrix" in data


def test_api_model_comparison():
    response = client.get("/api/model/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "Random Forest" in data
    assert "Logistic Regression" in data


def test_api_predict_valid():
    response = client.post("/api/predict", json=valid_patient)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "class_probabilities" in data
    assert "prediction_entropy" in data
    # Probabilities must sum to ~1.0
    total_prob = sum(data["class_probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-3


def test_api_predict_invalid_input():
    invalid_data = valid_patient.copy()
    invalid_data["Age"] = 10 # Below 18 minimum
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422 # Pydantic validation error

    invalid_gender = valid_patient.copy()
    invalid_gender["Gender"] = "UnknownGender"
    response2 = client.post("/api/predict", json=invalid_gender)
    assert response2.status_code == 422


def test_api_explain():
    response = client.post("/api/explain", json=valid_patient)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "top_drivers" in data
    assert len(data["top_drivers"]) > 0
    assert "methodology" in data
