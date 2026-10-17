import os
import sys
import pytest
from fastapi.testclient import TestClient

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "backend"))

from main import app

client = TestClient(app)


def test_figures_endpoints():
    """Verify that all figures rendered on frontend pages exist and are served with HTTP 200."""
    required_figures = [
        "roc_curves.png",
        "model_comparison.png",
        "final_test_confusion_matrix.png",
        "calibration_curves.png",
        "robustness_curves.png",
        "shap_bar.png",
        "correlation_heatmap.png",
        "ecg_features_distribution.png"
    ]
    for fig_name in required_figures:
        response = client.get(f"/api/figures/{fig_name}")
        assert response.status_code == 200, f"Figure {fig_name} could not be served by backend."


def test_research_endpoints_contracts():
    """Verify contracts for all analytical endpoints consumed by the React app."""
    # 1. Feature fusion
    fus_res = client.get("/api/model/feature-fusion")
    assert fus_res.status_code == 200
    fus_data = fus_res.json()
    assert "clinical" in fus_data
    assert "ecg" in fus_data
    assert "combined" in fus_data

    # 2. Robustness
    rob_res = client.get("/api/model/robustness")
    assert rob_res.status_code == 200
    rob_data = rob_res.json()
    assert "mcar_rates" in rob_data
    assert "group_omissions" in rob_data

    # 3. Calibration
    cal_res = client.get("/api/model/calibration")
    assert cal_res.status_code == 200
    cal_data = cal_res.json()
    assert "Platt Sigmoid Calibrated" in cal_data
    assert "expected_calibration_error_ece" in cal_data["Platt Sigmoid Calibrated"]

    # 4. Global SHAP
    shap_res = client.get("/api/model/shap-global")
    assert shap_res.status_code == 200
    shap_data = shap_res.json()
    assert "feature_importance_ranking" in shap_data
    assert len(shap_data["feature_importance_ranking"]) > 0
