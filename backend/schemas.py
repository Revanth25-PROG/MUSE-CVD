"""
MUSE-CVD: Pydantic Validation Schemas for FastAPI API Endpoints
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator


class PatientInput(BaseModel):
    # Demographics
    Age: int = Field(..., ge=18, le=120, description="Patient age in years (18-120)", json_schema_extra={"example": 62})
    Gender: str = Field(..., description="Biological sex ('Female' or 'Male')", json_schema_extra={"example": "Male"})

    # Clinical & Laboratory
    Heart_Rate: int = Field(..., ge=30, le=220, description="Resting heart rate in bpm (30-220)", json_schema_extra={"example": 95})
    Systolic_BP: int = Field(..., ge=60, le=250, description="Systolic blood pressure in mmHg (60-250)", json_schema_extra={"example": 135})
    Diastolic_BP: int = Field(..., ge=30, le=150, description="Diastolic blood pressure in mmHg (30-150)", json_schema_extra={"example": 85})
    Cholesterol: int = Field(..., ge=50, le=600, description="Total serum cholesterol in mg/dL (50-600)", json_schema_extra={"example": 220})
    Glucose: int = Field(..., ge=30, le=600, description="Fasting blood glucose in mg/dL (30-600)", json_schema_extra={"example": 110})
    Troponin: float = Field(..., ge=0.0, le=10.0, description="Cardiac troponin in ng/mL (0.0-10.0)", json_schema_extra={"example": 0.15})

    # ECG Measurements
    PR_Interval: int = Field(..., ge=50, le=400, description="PR interval in ms (50-400)", json_schema_extra={"example": 165})
    QRS_Duration: int = Field(..., ge=30, le=250, description="QRS duration in ms (30-250)", json_schema_extra={"example": 105})
    QT_Interval: int = Field(..., ge=200, le=700, description="QT interval in ms (200-700)", json_schema_extra={"example": 390})
    P_Wave_Duration: int = Field(..., ge=30, le=200, description="P wave duration in ms (30-200)", json_schema_extra={"example": 95})
    T_Wave_Duration: int = Field(..., ge=50, le=300, description="T wave duration in ms (50-300)", json_schema_extra={"example": 155})
    ST_Segment_Level: float = Field(..., ge=-2.0, le=2.0, description="ST segment elevation/depression in mV (-2.0 to 2.0)", json_schema_extra={"example": 0.05})
    RR_Interval: int = Field(..., ge=300, le=2000, description="RR interval in ms (300-2000)", json_schema_extra={"example": 880})
    HRV_SDNN: int = Field(..., ge=0, le=300, description="Heart rate variability SDNN in ms (0-300)", json_schema_extra={"example": 65})
    HRV_RMSSD: int = Field(..., ge=0, le=300, description="Heart rate variability RMSSD in ms (0-300)", json_schema_extra={"example": 55})
    R_Peak_Amplitude: float = Field(..., ge=-5.0, le=10.0, description="R wave peak amplitude in mV (-5.0 to 10.0)", json_schema_extra={"example": 1.85})
    S_Wave_Amplitude: float = Field(..., ge=-10.0, le=5.0, description="S wave nadir amplitude in mV (-10.0 to 5.0)", json_schema_extra={"example": -1.25})

    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        clean_v = v.strip().capitalize()
        if clean_v not in ['Male', 'Female']:
            raise ValueError("Gender must be strictly 'Male' or 'Female'")
        return clean_v


class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    prediction_entropy: float
    disclaimer: str


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    abs_impact: float
    direction: str


class ExplanationResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    top_drivers: List[FeatureContribution]
    all_contributions: List[FeatureContribution]
    methodology: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    calibrated_model_loaded: bool


class TargetDistributionResponse(BaseModel):
    total_records: int
    distribution: Dict[str, Dict[str, Any]]


class FeatureInfo(BaseModel):
    name: str
    group: str
    type: str
    range_or_categories: str
    mean: Optional[float] = None
    std: Optional[float] = None


class DatasetFeaturesResponse(BaseModel):
    total_features: int
    clinical_features: List[str]
    ecg_features: List[str]
    categorical_features: List[str]
    feature_details: List[FeatureInfo]
