"""
MUSE-CVD: Explainable AI (XAI) Module using SHAP
Computes:
1. TreeExplainer on trained ensemble models
2. Global feature importance rankings
3. SHAP summary and bar visualizations
4. Local individual patient attribution (positive vs. negative feature drivers)
5. Methodological distinction between model association and clinical causation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from typing import Dict, Any, List
from preprocessing import DataPreprocessor


class ModelExplainer:
    def __init__(self, model_path: str, preprocessor_path: str, reports_dir: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")
        self.metrics_dir = os.path.join(reports_dir, "metrics")

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.model = joblib.load(model_path)
        prep_artifact = joblib.load(preprocessor_path)
        self.preprocessor = prep_artifact['preprocessor']
        self.feature_names = prep_artifact['feature_names_out']
        self.label_mapping = prep_artifact['label_mapping']
        self.inv_label_mapping = prep_artifact['inv_label_mapping']
        self.classes = [self.inv_label_mapping[i] for i in range(len(self.inv_label_mapping))]

        # Background reference data for TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(self, X_transformed: np.ndarray, max_samples: int = 300) -> Dict[str, Any]:
        """Calculates global SHAP values across a representative sample of patients."""
        sample_size = min(len(X_transformed), max_samples)
        X_sample = X_transformed[:sample_size]

        print(f"Calculating SHAP values for {sample_size} samples...")
        shap_values = self.explainer.shap_values(X_sample)

        # Handling shape: list of arrays (one per class) or 3D array [N, features, classes]
        if isinstance(shap_values, list):
            # list of [N, D] for each class
            shap_array = np.array(shap_values) # [C, N, D]
            shap_array = np.transpose(shap_array, (1, 2, 0)) # [N, D, C]
        elif len(shap_values.shape) == 3:
            shap_array = shap_values
        else:
            shap_array = shap_values

        # Mean absolute SHAP per feature across all classes
        if len(shap_array.shape) == 3:
            mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
            per_class_mean = {}
            for c_idx, c_name in enumerate(self.classes):
                per_class_mean[c_name] = np.mean(np.abs(shap_array[:, :, c_idx]), axis=0).tolist()
        else:
            mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
            per_class_mean = {self.classes[0]: mean_abs_shap.tolist()}

        # Rank features globally
        ranking_indices = np.argsort(mean_abs_shap)[::-1]
        ranked_features = []
        for idx in ranking_indices:
            feat_info = {
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 5),
                "rank": int(len(ranked_features) + 1)
            }
            ranked_features.append(feat_info)

        global_summary = {
            "num_evaluated_samples": sample_size,
            "feature_importance_ranking": ranked_features,
            "classes": self.classes,
            "interpretability_disclaimer": (
                "SHAP values reflect model-internal feature associations and marginal contributions. "
                "They describe algorithmic decision weighting rather than physiological etiology or clinical causality."
            )
        }

        # Save global metrics
        json_path = os.path.join(self.metrics_dir, "shap_global.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2)

        # Plot global bar chart
        self.plot_global_importance(ranked_features)

        # Generate summary plot
        self.plot_shap_summary(shap_values, X_sample)

        return global_summary

    def plot_global_importance(self, ranked_features: List[Dict[str, Any]]):
        """Plots top feature importances by mean absolute SHAP value."""
        top_k = min(len(ranked_features), 15)
        top_feats = ranked_features[:top_k]
        names = [f["feature"] for f in top_feats][::-1]
        vals = [f["mean_abs_shap"] for f in top_feats][::-1]

        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        bars = ax.barh(names, vals, color='#0284c7', edgecolor='#0f172a', height=0.6)

        ax.set_title("MUSE-CVD: Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Mean Absolute SHAP Impact on Output Magnitude", fontsize=10, fontweight='semibold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2, f"{w:.4f}",
                    ha='left', va='center', fontsize=8.5, color='#1e293b')

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "shap_bar.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_shap_summary(self, shap_values, X_sample: np.ndarray):
        """Generates SHAP summary plot."""
        fig = plt.figure(figsize=(11, 7), dpi=300)
        
        # If multi-class list, plot summary for the first class or multi-class bar
        try:
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                                  class_names=self.classes, show=False, max_display=12)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                                  show=False, max_display=12)

            plt.title("SHAP Multi-Class Summary Attribution", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            out_path = os.path.join(self.figures_dir, "shap_summary.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Notice: SHAP summary plot rendered fallback: {e}")
            plt.close()

    def explain_instance(self, instance_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explains an individual patient prediction.
        Returns top positive and negative driving features for the predicted class.
        """
        df_inst = pd.DataFrame([instance_dict])
        X_trans = self.preprocessor.transform(df_inst)
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

        # Sort by absolute impact
        contributions = sorted(contributions, key=lambda x: x["abs_impact"], reverse=True)

        explanation = {
            "predicted_class": pred_class,
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))},
            "top_drivers": contributions[:8],
            "all_contributions": contributions,
            "disclaimer": "Local explanations explain model weighting, not medical etiology."
        }
        return explanation


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    model_path = os.path.join(base_dir, "models", "random_forest.joblib")
    prep_path = os.path.join(base_dir, "models", "preprocessor.joblib")
    reports_dir = os.path.join(base_dir, "reports")

    preprocessor_artifact = joblib.load(prep_path)
    preprocessor = preprocessor_artifact['preprocessor']
    
    df = pd.read_csv(data_path)
    X_raw = df[[col for col in df.columns if col != 'Target']]
    X_trans = preprocessor.transform(X_raw)

    explainer = ModelExplainer(model_path, prep_path, reports_dir)
    global_res = explainer.explain_dataset(X_trans, max_samples=250)

    # Test single instance explanation
    sample_patient = X_raw.iloc[0].to_dict()
    local_explanation = explainer.explain_instance(sample_patient)
    
    sample_out = os.path.join(base_dir, "reports", "metrics", "shap_sample_explanation.json")
    with open(sample_out, "w", encoding="utf-8") as f:
        json.dump(local_explanation, f, indent=2)
    print(f"Sample individual explanation saved to: {sample_out}")


if __name__ == "__main__":
    main()
