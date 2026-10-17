"""
MUSE-CVD: Final Model Evaluation and Deployment Serialization Module
Evaluates the validation-selected final model on the held-out test set (N=650)
and packages all metadata, feature schemas, class labels, and metrics for the API and website.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from preprocessing import DataPreprocessor


class FinalEvaluator:
    def __init__(self, data_path: str, models_dir: str, reports_dir: str, random_state: int = 42):
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")
        self.metrics_dir = os.path.join(reports_dir, "metrics")
        self.random_state = random_state

        self.preprocessor_helper = DataPreprocessor(random_state=self.random_state)
        self.X_raw, self.y_raw = self.preprocessor_helper.prepare_data(self.data_path)
        (
            self.X_train_df, self.X_val_df, self.X_test_df,
            self.y_train_s, self.y_val_s, self.y_test_s
        ) = self.preprocessor_helper.split_data(self.X_raw, self.y_raw)

        self.classes = sorted(self.y_train_s.unique())
        self.n_classes = len(self.classes)
        self.y_test_bin = label_binarize(self.y_test_s, classes=self.classes)

        # Load preprocessor and model
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        prep_artifact = joblib.load(prep_path)
        self.preprocessor = prep_artifact['preprocessor']
        self.feature_names = prep_artifact['feature_names_out']

        model_path = os.path.join(self.models_dir, "random_forest.joblib")
        self.model = joblib.load(model_path)

        cal_model_path = os.path.join(self.models_dir, "calibrated_model.joblib")
        self.calibrated_model = joblib.load(cal_model_path) if os.path.exists(cal_model_path) else None

        self.X_test = self.preprocessor.transform(self.X_test_df)

    def evaluate_test_set(self) -> Dict[str, Any]:
        """Performs final held-out test evaluation."""
        print("\n=================== FINAL HELD-OUT TEST EVALUATION ===================")
        y_pred = self.model.predict(self.X_test)
        y_probs = self.model.predict_proba(self.X_test)

        acc = float(accuracy_score(self.y_test_s, y_pred))
        prec = float(precision_score(self.y_test_s, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(self.y_test_s, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(self.y_test_s, y_pred, average='macro', zero_division=0))
        roc_auc = float(roc_auc_score(self.y_test_bin, y_probs, multi_class='ovr', average='macro'))

        report = classification_report(self.y_test_s, y_pred, output_dict=True, zero_division=0)
        per_class = {}
        for cls in self.classes:
            if cls in report:
                per_class[cls] = {
                    "precision": round(report[cls]['precision'], 4),
                    "recall": round(report[cls]['recall'], 4),
                    "f1_score": round(report[cls]['f1-score'], 4),
                    "support": int(report[cls]['support'])
                }

        cm = confusion_matrix(self.y_test_s, y_pred, labels=self.classes)

        # Also evaluate calibrated model if available
        cal_metrics = {}
        if self.calibrated_model:
            cal_probs = self.calibrated_model.predict_proba(self.X_test)
            cal_pred = self.calibrated_model.predict(self.X_test)
            cal_metrics = {
                "accuracy": round(float(accuracy_score(self.y_test_s, cal_pred)), 4),
                "macro_f1": round(float(f1_score(self.y_test_s, cal_pred, average='macro', zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(self.y_test_bin, cal_probs, multi_class='ovr', average='macro')), 4)
            }

        final_results = {
            "model_architecture": "Random Forest Classifier (Ensemble)",
            "selected_via": "Stratified Validation Macro-F1 & Calibration Integrity",
            "test_sample_size": len(self.y_test_s),
            "test_accuracy": round(acc, 4),
            "test_macro_precision": round(prec, 4),
            "test_macro_recall": round(rec, 4),
            "test_macro_f1": round(f1, 4),
            "test_roc_auc_ovr": round(roc_auc, 4),
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes,
            "calibrated_model_test_metrics": cal_metrics
        }

        print(f"Test Accuracy:          {acc*100:.2f}%")
        print(f"Test Macro Precision:   {prec:.4f}")
        print(f"Test Macro Recall:      {rec:.4f}")
        print(f"Test Macro F1:          {f1:.4f}")
        print(f"Test ROC-AUC (OvR):     {roc_auc:.4f}")

        # Save metrics
        json_path = os.path.join(self.metrics_dir, "final_test_evaluation.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

        # Save metadata and final model package
        metadata = {
            "model_name": "MUSE-CVD Random Forest Classifier",
            "version": "1.0.0",
            "status": "Trained & Validated",
            "feature_schema": {
                "clinical_features": self.preprocessor_helper.clinical_features,
                "ecg_features": self.preprocessor_helper.ecg_features,
                "categorical_features": self.preprocessor_helper.categorical_features,
                "all_features": self.preprocessor_helper.all_feature_cols,
                "transformed_feature_names": self.feature_names
            },
            "classes": self.classes,
            "label_mapping": self.preprocessor_helper.label_mapping,
            "inv_label_mapping": self.preprocessor_helper.inv_label_mapping,
            "metrics": final_results
        }
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        final_bundle_path = os.path.join(self.models_dir, "final_model.joblib")
        joblib.dump(self.model, final_bundle_path)
        print(f"Saved final deployment model to: {final_bundle_path}")
        print(f"Saved model metadata to: {meta_path}")

        # Plot Test Confusion Matrix & ROC Curves
        self.plot_confusion_matrix(cm)
        self.plot_roc_curves(y_probs)

        return final_results

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Generates publication-ready confusion matrix for the test set."""
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=self.classes, yticklabels=self.classes)
        ax.set_title("MUSE-CVD: Held-Out Test Confusion Matrix (N=650)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight='semibold')
        ax.set_ylabel("True Ground Truth Class", fontsize=10, fontweight='semibold')
        ax.tick_params(axis='x', rotation=15, labelsize=8.5)
        ax.tick_params(axis='y', rotation=0, labelsize=8.5)
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "final_test_confusion_matrix.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_roc_curves(self, y_probs: np.ndarray):
        """Generates multi-class ROC curves on held-out test data."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        colors = ['#1e3a8a', '#2563eb', '#0284c7', '#38bdf8']

        for i, cls in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(self.y_test_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Line (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='semibold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='semibold')
        ax.set_title('MUSE-CVD: Multiclass ROC Curves (Test Subset)', fontsize=12, fontweight='bold', pad=12)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "roc_curves.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    evaluator = FinalEvaluator(data_path, models_dir, reports_dir, random_state=42)
    evaluator.evaluate_test_set()


if __name__ == "__main__":
    main()
