"""
MUSE-CVD: Feature-Group Fusion Experiment Module
Compares:
1. Clinical & Laboratory Subset (Age, Gender, HR, BP, Cholesterol, Glucose, Troponin)
2. ECG-Derived Numerical Subset (PR, QRS, QT, P/T wave, ST segment, RR, HRV, R/S amplitudes)
3. Combined Feature-Group Fusion (All Features)

Evaluates whether early feature-level concatenation yields empirical synergies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from preprocessing import DataPreprocessor


class FeatureGroupFusionExperiment:
    def __init__(self, data_path: str, reports_dir: str, random_state: int = 42):
        self.data_path = data_path
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")
        self.metrics_dir = os.path.join(reports_dir, "metrics")
        self.random_state = random_state

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.preprocessor_helper = DataPreprocessor(random_state=self.random_state)
        self.X_raw, self.y_raw = self.preprocessor_helper.prepare_data(self.data_path)
        (
            self.X_train_df, self.X_val_df, self.X_test_df,
            self.y_train_s, self.y_val_s, self.y_test_s
        ) = self.preprocessor_helper.split_data(self.X_raw, self.y_raw)

        self.classes = sorted(self.y_train_s.unique())
        self.y_val_bin = label_binarize(self.y_val_s, classes=self.classes)

        # Numerical labels for XGBoost
        self.y_train_num = np.array([self.preprocessor_helper.label_mapping[y] for y in self.y_train_s])
        self.y_val_num = np.array([self.preprocessor_helper.label_mapping[y] for y in self.y_val_s])

        self.clinical_cols = self.preprocessor_helper.clinical_features
        self.cat_cols = self.preprocessor_helper.categorical_features
        self.ecg_cols = self.preprocessor_helper.ecg_features

    def build_transformer(self, group: str) -> ColumnTransformer:
        """Constructs dedicated ColumnTransformer for the specified feature group."""
        if group == 'clinical':
            transformers = [
                ('num', StandardScaler(), self.clinical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
            ]
        elif group == 'ecg':
            transformers = [
                ('num', StandardScaler(), self.ecg_cols)
            ]
        elif group == 'combined':
            transformers = [
                ('num', StandardScaler(), self.clinical_cols + self.ecg_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
            ]
        else:
            raise ValueError(f"Unknown group: {group}")

        ct = ColumnTransformer(transformers=transformers, remainder='drop')
        return ct

    def run_experiments(self) -> Dict[str, Any]:
        """Runs comparative evaluations across feature groups and classifiers."""
        groups = ['clinical', 'ecg', 'combined']
        group_names = {
            'clinical': 'Clinical & Laboratory Only (8 feats)',
            'ecg': 'ECG-Derived Numerical Only (11 feats)',
            'combined': 'Combined Early Fusion (19 feats)'
        }

        classifier_factories = {
            "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1),
            "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1),
            "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=4, eval_metric='mlogloss', random_state=self.random_state)
        }

        results = {}
        tabular_rows = []

        print("\n=================== FEATURE GROUP FUSION EXPERIMENTS ===================")
        for grp in groups:
            results[grp] = {
                "name": group_names[grp],
                "models": {}
            }
            # Preprocess for this group
            transformer = self.build_transformer(grp)
            X_tr = transformer.fit_transform(self.X_train_df)
            X_va = transformer.transform(self.X_val_df)

            for clf_name, clf_fn in classifier_factories.items():
                clf = clf_fn()
                if clf_name == "XGBoost":
                    clf.fit(X_tr, self.y_train_num)
                    preds_num = clf.predict(X_va)
                    preds = [self.preprocessor_helper.inv_label_mapping[p] for p in preds_num]
                    probs = clf.predict_proba(X_va)
                else:
                    clf.fit(X_tr, self.y_train_s)
                    preds = clf.predict(X_va)
                    probs = clf.predict_proba(X_va)

                acc = float(accuracy_score(self.y_val_s, preds))
                f1_macro = float(f1_score(self.y_val_s, preds, average='macro', zero_division=0))
                try:
                    auc = float(roc_auc_score(self.y_val_bin, probs, multi_class='ovr', average='macro'))
                except Exception:
                    auc = 0.50

                results[grp]["models"][clf_name] = {
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1_macro, 4),
                    "roc_auc": round(auc, 4)
                }

                tabular_rows.append({
                    "Feature Group": group_names[grp],
                    "Model": clf_name,
                    "Accuracy": f"{acc*100:.2f}%",
                    "Macro F1": f"{f1_macro:.4f}",
                    "ROC-AUC": f"{auc:.4f}"
                })

        df_table = pd.DataFrame(tabular_rows)
        print(df_table.to_string(index=False))

        # Save results
        json_path = os.path.join(self.metrics_dir, "feature_fusion.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        csv_path = os.path.join(self.metrics_dir, "feature_fusion.csv")
        df_table.to_csv(csv_path, index=False)

        # Plot comparison
        self.plot_fusion_results(df_table)

        return results

    def plot_fusion_results(self, df_table: pd.DataFrame):
        """Generates a grouped visualization comparing fusion strategies."""
        df_plot = df_table.copy()
        df_plot['Acc_Val'] = df_plot['Accuracy'].str.rstrip('%').astype(float)

        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        sns.barplot(data=df_plot, x='Model', y='Acc_Val', hue='Feature Group', palette=['#38bdf8', '#0284c7', '#1e3a8a'], ax=ax, edgecolor='black')

        ax.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight='semibold')
        ax.set_title("Feature-Group Fusion Analysis: Clinical vs. ECG vs. Combined", fontsize=13, fontweight='bold', pad=15)
        ax.set_ylim(0, 40)
        ax.axhline(25.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Naive Chance Baseline (25%)')
        ax.legend(title='Input Modality/Group', loc='upper right', fontsize=8.5)

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=7.5, xytext=(0, 2),
                            textcoords='offset points')

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "feature_fusion_comparison.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    reports_dir = os.path.join(base_dir, "reports")

    experiment = FeatureGroupFusionExperiment(data_path, reports_dir, random_state=42)
    experiment.run_experiments()


if __name__ == "__main__":
    main()
