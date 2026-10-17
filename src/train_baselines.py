"""
MUSE-CVD: Machine Learning Baseline Training & Comparative Benchmarking Module
Trains and systematically evaluates:
1. Dummy Classifier (Most Frequent & Stratified)
2. Multinomial Logistic Regression
3. Decision Tree Classifier
4. Random Forest Classifier
5. Extra Trees Classifier
6. Gradient Boosting Classifier (and XGBoost Classifier)

Calculates comprehensive metrics (Accuracy, Macro Precision, Recall, F1, Per-class metrics, ROC-AUC, Confusion Matrix)
and saves results in JSON and CSV formats for frontend and API integration.
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

from typing import Dict, Any, List
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from preprocessing import DataPreprocessor


class BaselineModelTrainer:
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

        self.preprocessor = DataPreprocessor(random_state=self.random_state)
        self.X_raw, self.y_raw = self.preprocessor.prepare_data(self.data_path)
        (
            self.X_train_df, self.X_val_df, self.X_test_df,
            self.y_train_s, self.y_val_s, self.y_test_s
        ) = self.preprocessor.split_data(self.X_raw, self.y_raw)

        # Fit preprocessor strictly on train
        self.preprocessor.fit_preprocessor(self.X_train_df, feature_subset='all')
        self.X_train = self.preprocessor.transform(self.X_train_df)
        self.X_val = self.preprocessor.transform(self.X_val_df)
        self.X_test = self.preprocessor.transform(self.X_test_df)

        self.classes = sorted(self.y_train_s.unique())
        self.n_classes = len(self.classes)

        # Numerical labels for XGBoost
        self.y_train_num = np.array([self.preprocessor.label_mapping[y] for y in self.y_train_s])
        self.y_val_num = np.array([self.preprocessor.label_mapping[y] for y in self.y_val_s])
        self.y_test_num = np.array([self.preprocessor.label_mapping[y] for y in self.y_test_s])

        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}

    def init_models(self) -> Dict[str, Any]:
        """Initializes all specified baseline classifiers with reproducible random states."""
        self.models = {
            "Dummy Classifier": DummyClassifier(strategy="stratified", random_state=self.random_state),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=8, random_state=self.random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=8, min_samples_split=8, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=self.random_state),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='mlogloss', random_state=self.random_state)
        }
        return self.models

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Trains all models and evaluates on the validation split."""
        if not self.models:
            self.init_models()

        summary_table = []
        detailed_results = {}

        y_val_bin = label_binarize(self.y_val_s, classes=self.classes)

        print("\n=================== BASELINE MODEL BENCHMARKING (VAL SET) ===================")
        for name, model in self.models.items():
            print(f"Training: {name}...")
            if name == "XGBoost":
                model.fit(self.X_train, self.y_train_num)
                val_preds_num = model.predict(self.X_val)
                val_preds = [self.preprocessor.inv_label_mapping[p] for p in val_preds_num]
                val_probs = model.predict_proba(self.X_val)
            else:
                model.fit(self.X_train, self.y_train_s)
                val_preds = model.predict(self.X_val)
                if hasattr(model, "predict_proba"):
                    val_probs = model.predict_proba(self.X_val)
                else:
                    val_probs = np.zeros((len(self.X_val), self.n_classes))

            # Metric computations
            acc = float(accuracy_score(self.y_val_s, val_preds))
            prec_macro = float(precision_score(self.y_val_s, val_preds, average='macro', zero_division=0))
            rec_macro = float(recall_score(self.y_val_s, val_preds, average='macro', zero_division=0))
            f1_macro = float(f1_score(self.y_val_s, val_preds, average='macro', zero_division=0))

            # Multi-class ROC-AUC (OvR)
            try:
                roc_auc = float(roc_auc_score(y_val_bin, val_probs, multi_class='ovr', average='macro'))
            except Exception:
                roc_auc = 0.50

            # Per-class metrics
            report = classification_report(self.y_val_s, val_preds, output_dict=True, zero_division=0)
            per_class = {}
            for cls_name in self.classes:
                if cls_name in report:
                    per_class[cls_name] = {
                        "precision": round(report[cls_name]['precision'], 4),
                        "recall": round(report[cls_name]['recall'], 4),
                        "f1_score": round(report[cls_name]['f1-score'], 4),
                        "support": int(report[cls_name]['support'])
                    }

            # Confusion matrix
            cm = confusion_matrix(self.y_val_s, val_preds, labels=self.classes)

            # Record
            model_metrics = {
                "model_name": name,
                "accuracy": round(acc, 4),
                "macro_precision": round(prec_macro, 4),
                "macro_recall": round(rec_macro, 4),
                "macro_f1": round(f1_macro, 4),
                "roc_auc_ovr": round(roc_auc, 4),
                "per_class_metrics": per_class,
                "confusion_matrix": cm.tolist(),
                "classes": self.classes
            }
            detailed_results[name] = model_metrics
            summary_table.append({
                "Model": name,
                "Accuracy": f"{acc*100:.2f}%",
                "Macro Precision": f"{prec_macro:.4f}",
                "Macro Recall": f"{rec_macro:.4f}",
                "Macro F1": f"{f1_macro:.4f}",
                "ROC-AUC (OvR)": f"{roc_auc:.4f}"
            })

            # Save individual model artifact
            model_save_path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, model_save_path)

        # Print summary
        summary_df = pd.DataFrame(summary_table)
        print("\nValidation Summary Table:")
        print(summary_df.to_string(index=False))

        # Save CSV and JSON
        csv_path = os.path.join(self.metrics_dir, "baseline_models.csv")
        summary_df.to_csv(csv_path, index=False)
        json_path = os.path.join(self.metrics_dir, "baseline_models.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)

        self.evaluation_results = detailed_results

        # Plot comparison chart and confusion matrices
        self.plot_model_comparison(summary_df)
        self.plot_confusion_matrices()

        return detailed_results

    def plot_model_comparison(self, summary_df: pd.DataFrame):
        """Generates a high-quality model comparison bar chart."""
        df_plot = summary_df.copy()
        df_plot['Acc_Val'] = df_plot['Accuracy'].str.rstrip('%').astype(float)
        df_plot['F1_Val'] = df_plot['Macro F1'].astype(float) * 100

        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        x = np.arange(len(df_plot))
        width = 0.35

        rects1 = ax.bar(x - width/2, df_plot['Acc_Val'], width, label='Accuracy (%)', color='#2563eb', edgecolor='black')
        rects2 = ax.bar(x + width/2, df_plot['F1_Val'], width, label='Macro F1 (%)', color='#0284c7', edgecolor='black')

        ax.set_ylabel('Score (%)', fontsize=11, fontweight='semibold')
        ax.set_title('Baseline Model Performance Comparison (Validation Subset)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['Model'], rotation=15, ha='right', fontsize=9.5)
        ax.set_ylim(0, 45)
        ax.axhline(25.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Naive Chance Level (25%)')
        ax.legend(loc='upper right')

        # Annotate
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "model_comparison.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_confusion_matrices(self):
        """Plots confusion matrices for key representative models."""
        key_models = ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
        axes = axes.flatten()

        for idx, name in enumerate(key_models):
            if name not in self.evaluation_results:
                continue
            ax = axes[idx]
            cm = np.array(self.evaluation_results[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=self.classes, yticklabels=self.classes)
            ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Predicted Class", fontsize=10)
            ax.set_ylabel("True Class", fontsize=10)
            ax.tick_params(axis='x', rotation=15, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

        fig.suptitle("Validation Confusion Matrices Across Core Baselines", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "confusion_matrices.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    trainer = BaselineModelTrainer(data_path, models_dir, reports_dir, random_state=42)
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
