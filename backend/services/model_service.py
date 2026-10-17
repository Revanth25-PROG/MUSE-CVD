"""
MUSE-CVD: Model Service Layer
Encapsulates model loading, inference, uncertainty calculation, and SHAP explainability.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional, List


class ModelService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
        # Check both local backend/models and root/models
        candidates_models = [
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "..", "models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        ]
        self.models_dir = next((p for p in candidates_models if os.path.exists(p)), candidates_models[0])

        candidates_metrics = [
            os.path.join(base_dir, "reports", "metrics"),
            os.path.join(base_dir, "..", "reports", "metrics"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "metrics")
        ]
        self.metrics_dir = next((p for p in candidates_metrics if os.path.exists(p)), candidates_metrics[0])
        
        self.model = None
        self.calibrated_model = None
        self.preprocessor = None
        self.feature_names = []
        self.classes = []
        self.inv_label_mapping = {}
        self.explainer = None
        self.metadata = {}

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized models, preprocessors, and metadata."""
        # 1. Preprocessor
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        if os.path.exists(prep_path):
            prep_artifact = joblib.load(prep_path)
            self.preprocessor = prep_artifact['preprocessor']
            self.feature_names = prep_artifact['feature_names_out']
            self.inv_label_mapping = prep_artifact['inv_label_mapping']
            self.classes = [self.inv_label_mapping[i] for i in range(len(self.inv_label_mapping))]

        # 2. Final Model
        model_path = os.path.join(self.models_dir, "final_model.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"Warning: TreeExplainer init note: {e}")

        # 3. Calibrated Model
        cal_path = os.path.join(self.models_dir, "calibrated_model.joblib")
        if os.path.exists(cal_path):
            self.calibrated_model = joblib.load(cal_path)

        # 4. Metadata
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def predict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs validation-checked model inference."""
        if not self.is_ready:
            raise RuntimeError("Validated ML model is currently unavailable on the server.")

        df = pd.DataFrame([patient_dict])
        X_trans = self.preprocessor.transform(df)

        # Prefer calibrated probabilities if available
        if self.calibrated_model:
            probs = self.calibrated_model.predict_proba(X_trans)[0]
        else:
            probs = self.model.predict_proba(X_trans)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Compute Shannon entropy (normalized by log2(K))
        entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))) / np.log2(len(self.classes)))

        prob_dict = {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}

        return {
            "model_name": "Random Forest Ensemble (Calibrated)",
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": prob_dict,
            "prediction_entropy": round(entropy, 4),
            "disclaimer": (
                "This predicted category reflects statistical patterns in the experimental research cohort. "
                "It is not a clinical diagnosis and must not replace professional clinical evaluation."
            )
        }

    def explain(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generates real-time patient-level SHAP explanation."""
        if not self.is_ready or self.explainer is None:
            raise RuntimeError("SHAP explanation engine is currently unavailable.")

        df = pd.DataFrame([patient_dict])
        X_trans = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]

        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            inst_shap = shap_vals[pred_idx][0]
        elif len(shap_vals.shape) == 3:
            inst_shap = shap_vals[0, :, pred_idx]
        else:
            inst_shap = shap_vals[0]

        contributions = []
        for i, val in enumerate(inst_shap):
            contributions.append({
                "feature": self.feature_names[i],
                "shap_value": round(float(val), 5),
                "abs_impact": round(float(abs(val)), 5),
                "direction": "positive" if val > 0 else "negative"
            })

        contributions = sorted(contributions, key=lambda x: x["abs_impact"], reverse=True)

        return {
            "predicted_class": pred_class,
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))},
            "top_drivers": contributions[:8],
            "all_contributions": contributions,
            "methodology": "TreeSHAP game-theoretic Shapley value attribution for the predicted class.",
            "disclaimer": "Feature importance reflects algorithmic weighting, not medical etiology or causation."
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.is_ready else "not_trained",
            "model_name": self.metadata.get("model_name", "Random Forest Ensemble"),
            "classes": self.classes,
            "calibrated": self.calibrated_model is not None,
            "explainer_ready": self.explainer is not None
        }

    def get_test_metrics(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "final_test_evaluation.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}

    def get_baseline_comparison(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "baseline_models.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}

    def get_feature_fusion(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "feature_fusion.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}

    def get_robustness(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "robustness_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}

    def get_calibration(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "calibration_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}

    def get_shap_global(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, "shap_global.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "Not evaluated yet"}
