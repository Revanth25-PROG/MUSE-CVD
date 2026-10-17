"""
MUSE-CVD: Dataset Service Layer
Reads and serves verified dataset metrics, schema descriptions, and distributions.
"""

import os
import json
from typing import Dict, Any, List


class DatasetService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        candidates_metrics = [
            os.path.join(base_dir, "reports", "metrics"),
            os.path.join(base_dir, "..", "reports", "metrics"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "metrics")
        ]
        self.metrics_dir = next((p for p in candidates_metrics if os.path.exists(p)), candidates_metrics[0])
        self.summary_cache = None

    def _load_summary(self) -> Dict[str, Any]:
        if self.summary_cache is None:
            path = os.path.join(self.metrics_dir, "dataset_summary.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.summary_cache = json.load(f)
            else:
                self.summary_cache = {}
        return self.summary_cache

    def get_summary(self) -> Dict[str, Any]:
        data = self._load_summary()
        if not data:
            return {"status": "Dataset not validated yet"}
        return {
            "total_records": data.get("row_count", 0),
            "total_columns": data.get("column_count", 0),
            "total_missing_values": data.get("total_missing_values", 0),
            "duplicate_records": data.get("duplicate_records", 0),
            "categorical_columns": data.get("categorical_columns", []),
            "numerical_columns": data.get("numerical_columns", []),
            "target_classes": list(data.get("target_distribution", {}).keys()),
            "dataset_provenance": data.get("dataset_provenance", "Synthetic physiological cohort")
        }

    def get_target_distribution(self) -> Dict[str, Any]:
        data = self._load_summary()
        return {
            "total_records": data.get("row_count", 0),
            "distribution": data.get("target_distribution", {})
        }

    def get_features(self) -> Dict[str, Any]:
        data = self._load_summary()
        num_stats = data.get("numerical_statistics", {})
        cat_summary = data.get("categorical_summary", {})

        feature_details: List[Dict[str, Any]] = []

        # Clinical features
        clinical_set = {'Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol', 'Glucose', 'Troponin'}
        ecg_set = {
            'PR_Interval', 'QRS_Duration', 'QT_Interval', 'P_Wave_Duration', 'T_Wave_Duration',
            'ST_Segment_Level', 'RR_Interval', 'HRV_SDNN', 'HRV_RMSSD', 'R_Peak_Amplitude', 'S_Wave_Amplitude'
        }

        for col in data.get("columns", []):
            if col == 'Target':
                continue
            if col in cat_summary:
                feature_details.append({
                    "name": col,
                    "group": "Demographic",
                    "type": "Categorical",
                    "range_or_categories": ", ".join(cat_summary[col]["unique_values"]),
                    "mean": None,
                    "std": None
                })
            elif col in num_stats:
                s = num_stats[col]
                grp = "Clinical & Lab" if col in clinical_set else "ECG Numerical"
                feature_details.append({
                    "name": col,
                    "group": grp,
                    "type": "Numerical",
                    "range_or_categories": f"{s['min']} - {s['max']}",
                    "mean": s["mean"],
                    "std": s["std"]
                })

        return {
            "total_features": len(feature_details),
            "clinical_features": list(clinical_set),
            "ecg_features": list(ecg_set),
            "categorical_features": list(cat_summary.keys()),
            "feature_details": feature_details
        }
