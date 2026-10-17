"""
MUSE-CVD: Probability Calibration and Uncertainty Quantification Module
Evaluates uncalibrated vs. calibrated probability outputs using:
- Multiclass Brier Score
- Expected Calibration Error (ECE) with equal-width binning
- CalibratedClassifierCV (Isotonic and Sigmoid / Platt scaling)
- Shannon Entropy-based Prediction Uncertainty Quantification
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import label_binarize

from preprocessing import DataPreprocessor


def compute_multiclass_brier(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes the multi-class Brier score: (1/N) * sum_{i,k} (p_{ik} - y_{ik})^2."""
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def compute_ece(y_true_indices: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across maximum confidence predictions.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_prob)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


class ProbabilityCalibrator:
    def __init__(self, data_path: str, models_dir: str, reports_dir: str, random_state: int = 42):
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")
        self.metrics_dir = os.path.join(reports_dir, "metrics")
        self.random_state = random_state

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.preprocessor_helper = DataPreprocessor(random_state=self.random_state)
        self.X_raw, self.y_raw = self.preprocessor_helper.prepare_data(self.data_path)
        (
            self.X_train_df, self.X_val_df, self.X_test_df,
            self.y_train_s, self.y_val_s, self.y_test_s
        ) = self.preprocessor_helper.split_data(self.X_raw, self.y_raw)

        self.preprocessor = self.preprocessor_helper.fit_preprocessor(self.X_train_df, feature_subset='all')
        self.X_train = self.preprocessor.transform(self.X_train_df)
        self.X_val = self.preprocessor.transform(self.X_val_df)
        self.X_test = self.preprocessor.transform(self.X_test_df)

        self.classes = sorted(self.y_train_s.unique())
        self.n_classes = len(self.classes)
        self.y_val_bin = label_binarize(self.y_val_s, classes=self.classes)
        self.y_val_idx = np.array([self.preprocessor_helper.label_mapping[y] for y in self.y_val_s])

    def run_calibration(self) -> Dict[str, Any]:
        """Trains uncalibrated and calibrated classifiers and measures Brier, ECE, Log-loss, and Entropy."""
        print("\n=================== PROBABILITY CALIBRATION ANALYSIS ===================")
        
        # 1. Base uncalibrated model (Random Forest)
        base_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1)
        base_rf.fit(self.X_train, self.y_train_s)
        uncal_probs = base_rf.predict_proba(self.X_val)

        # 2. Sigmoid Calibrated Model (Platt scaling via internal CV on train)
        cal_sigmoid = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1),
                                             method='sigmoid', cv=3)
        cal_sigmoid.fit(self.X_train, self.y_train_s)
        sigmoid_probs = cal_sigmoid.predict_proba(self.X_val)

        # 3. Isotonic Calibrated Model
        cal_isotonic = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1),
                                              method='isotonic', cv=3)
        cal_isotonic.fit(self.X_train, self.y_train_s)
        isotonic_probs = cal_isotonic.predict_proba(self.X_val)

        models_dict = {
            "Uncalibrated Random Forest": uncal_probs,
            "Platt Sigmoid Calibrated": sigmoid_probs,
            "Isotonic Calibrated": isotonic_probs
        }

        results = {}
        for name, probs in models_dict.items():
            brier = compute_multiclass_brier(self.y_val_bin, probs)
            ece = compute_ece(self.y_val_idx, probs, n_bins=10)
            ll = float(log_loss(self.y_val_s, probs))
            preds = [self.classes[i] for i in np.argmax(probs, axis=1)]
            acc = float(accuracy_score(self.y_val_s, preds))
            
            # Shannon entropy of probability distributions (normalized to [0, 1])
            # H_max = log2(4) = 2.0
            entropies = -np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)), axis=1) / 2.0
            avg_entropy = float(np.mean(entropies))

            results[name] = {
                "accuracy": round(acc, 4),
                "multiclass_brier_score": round(brier, 4),
                "expected_calibration_error_ece": round(ece, 4),
                "log_loss": round(ll, 4),
                "mean_prediction_entropy": round(avg_entropy, 4)
            }
            print(f"{name:30s} | Acc: {acc*100:.2f}% | Brier: {brier:.4f} | ECE: {ece:.4f} | LogLoss: {ll:.4f} | Entropy: {avg_entropy:.4f}")

        # Save calibrated model artifact
        cal_model_path = os.path.join(self.models_dir, "calibrated_model.joblib")
        joblib.dump(cal_sigmoid, cal_model_path)
        print(f"Saved calibrated model to: {cal_model_path}")

        # Save metrics
        json_path = os.path.join(self.metrics_dir, "calibration_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Plot calibration curves
        self.plot_calibration_curves(models_dict)

        return results

    def plot_calibration_curves(self, models_dict: Dict[str, np.ndarray]):
        """Plots one-vs-rest reliability diagrams across the 4 classes."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
        axes = axes.flatten()

        colors = {
            "Uncalibrated Random Forest": "#ef4444",
            "Platt Sigmoid Calibrated": "#2563eb",
            "Isotonic Calibrated": "#16a34a"
        }

        for idx, cls_name in enumerate(self.classes):
            ax = axes[idx]
            y_binary = (self.y_val_idx == idx).astype(int)

            # Diagonal perfect calibration
            ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=1.2)

            for m_name, probs in models_dict.items():
                prob_pos = probs[:, idx]
                fraction_of_positives, mean_predicted_value = calibration_curve(y_binary, prob_pos, n_bins=8, strategy='uniform')
                ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=m_name, color=colors[m_name], linewidth=1.8, markersize=5)

            ax.set_title(f"Class: {cls_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Mean Predicted Probability", fontsize=9.5)
            ax.set_ylabel("Empirical Fraction of Positives", fontsize=9.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.05, 1.05)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)

        fig.suptitle("Reliability Calibration Curves (One-vs-Rest Validation Predictions)", fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "calibration_curves.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    calibrator = ProbabilityCalibrator(data_path, models_dir, reports_dir, random_state=42)
    calibrator.run_calibration()


if __name__ == "__main__":
    main()
