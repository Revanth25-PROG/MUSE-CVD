"""
MUSE-CVD: FastAPI Application Entrypoint
Provides REST API endpoints for cardiovascular disease risk classification,
explainability via SHAP, dataset diagnostics, and comparative model metrics.
"""

import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from schemas import (
    PatientInput, PredictionResponse, ExplanationResponse, HealthResponse,
    TargetDistributionResponse, DatasetFeaturesResponse
)
from services.model_service import ModelService
from services.dataset_service import DatasetService

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="MUSE-CVD REST API",
    description="Explainable Multimodal Machine Learning Framework for Cardiovascular Risk Stratification",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permissive in local research prototype
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService(base_dir)
dataset_service = DatasetService(base_dir)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting API and model readiness."""
    return {
        "status": "healthy",
        "app_name": "MUSE-CVD AI Research Framework",
        "version": "1.0.0",
        "model_loaded": model_service.is_ready,
        "calibrated_model_loaded": model_service.calibrated_model is not None
    }


@app.get("/api/dataset/summary")
def get_dataset_summary():
    """Returns verified dataset dimensions, missingness, and provenance summary."""
    return dataset_service.get_summary()


@app.get("/api/dataset/features", response_model=DatasetFeaturesResponse)
def get_dataset_features():
    """Returns detailed feature schema table and physiological groupings."""
    return dataset_service.get_features()


@app.get("/api/dataset/target-distribution", response_model=TargetDistributionResponse)
def get_target_distribution():
    """Returns target class stratification counts and empirical percentages."""
    return dataset_service.get_target_distribution()


@app.get("/api/model/status")
def get_model_status():
    """Returns current model deployment status and classes."""
    return model_service.get_status()


@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns final held-out test evaluation metrics and confusion matrix."""
    metrics = model_service.get_test_metrics()
    if not metrics or metrics.get("status") == "Not evaluated yet":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model evaluation metrics are not available yet. Please run evaluation script."
        )
    return metrics


@app.get("/api/model/comparison")
def get_model_comparison():
    """Returns comparative baseline benchmarks across all 6 models."""
    comparison = model_service.get_baseline_comparison()
    if not comparison or comparison.get("status") == "Not evaluated yet":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Baseline comparison results are not available yet."
        )
    return comparison


@app.get("/api/model/feature-fusion")
def get_feature_fusion():
    """Returns feature-group fusion experiment results (Clinical vs ECG vs Combined)."""
    return model_service.get_feature_fusion()


@app.get("/api/model/robustness")
def get_robustness():
    """Returns missing-feature robustness degradation curves and imputation benchmarks."""
    return model_service.get_robustness()


@app.get("/api/model/calibration")
def get_calibration():
    """Returns probability calibration metrics (Brier, ECE, Log-loss, Entropy)."""
    return model_service.get_calibration()


@app.get("/api/model/metrics-v2")
def get_model_metrics_v2():
    """Returns v2 held-out test evaluation metrics and confusion matrix."""
    return model_service.get_test_metrics_v2()


@app.get("/api/model/benchmark-v2")
def get_benchmark_v2():
    """Returns the 10-model leakage-free benchmark comparison."""
    return model_service.get_improved_benchmark()


@app.get("/api/model/calibration-v2")
def get_calibration_v2():
    """Returns v2 calibration metrics (Platt Sigmoid vs Isotonic)."""
    return model_service.get_calibration_v2()


@app.get("/api/model/feature-fusion-v2")
def get_feature_fusion_v2():
    """Returns v2 multimodal feature fusion ablation results."""
    return model_service.get_feature_fusion_v2()


@app.get("/api/model/shap-global")
def get_shap_global():
    """Returns global feature importance rankings derived from SHAP TreeExplainer."""
    return model_service.get_shap_global()


@app.post("/api/predict", response_model=PredictionResponse)
def predict_cardiovascular_state(patient: PatientInput):
    """
    Predicts cardiovascular disease risk category from demographic, clinical, and ECG features.
    Returns predicted class, calibrated probability distribution, and prediction entropy.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not loaded. Please train and evaluate model before making predictions."
        )
    try:
        res = model_service.predict(patient.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/explain", response_model=ExplanationResponse)
def explain_prediction(patient: PatientInput):
    """
    Generates local SHAP explanation for an individual patient.
    Returns positive and negative attribution values for each feature.
    """
    if not model_service.is_ready or model_service.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explanation engine is not available."
        )
    try:
        res = model_service.explain(patient.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/figures/{figure_name}")
def get_figure(figure_name: str):
    """Serves generated scientific plots and figures."""
    candidates = [
        os.path.join(base_dir, "reports", "figures"),
        os.path.join(base_dir, "..", "reports", "figures"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures")
    ]
    figures_dir = next((p for p in candidates if os.path.exists(p)), candidates[0])
    sanitized = os.path.basename(figure_name)
    file_path = os.path.join(figures_dir, sanitized)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Figure {sanitized} not found.")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8008))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
