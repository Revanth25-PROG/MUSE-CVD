"""
MUSE-CVD: Missing-Feature Robustness Analysis Module
Evaluates model behavior under controlled Missing-Completely-At-Random (MCAR)
scenarios (0%, 10%, 20%, 30% missingness), feature-group omission (clinical vs ECG missing),
and tests fallback imputation strategies (Mean vs Median).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from preprocessing import DataPreprocessor


class RobustnessExperiment:
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

        self.num_cols = self.preprocessor_helper.clinical_features + self.preprocessor_helper.ecg_features
        self.cat_cols = self.preprocessor_helper.categorical_features
        self.classes = sorted(self.y_train_s.unique())

    def run_missingness_rates(self) -> Dict[str, Any]:
        """Tests controlled missingness rates: 0%, 10%, 20%, 30% using Mean and Median imputation."""
        rates = [0.0, 0.10, 0.20, 0.30]
        strategies = ['mean', 'median']
        results = {"mcar_rates": {}, "group_omissions": {}}

        print("\n=================== MISSING-FEATURE ROBUSTNESS EXPERIMENTS ===================")
        
        # Fit baseline model on clean training set
        ct_clean = ColumnTransformer([
            ('num', StandardScaler(), self.num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
        ])
        X_tr_clean = ct_clean.fit_transform(self.X_train_df)
        model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state, n_jobs=-1)
        model.fit(X_tr_clean, self.y_train_s)

        for rate in rates:
            results["mcar_rates"][str(int(rate*100))] = {}
            for strat in strategies:
                np.random.seed(self.random_state)
                X_val_corrupted = self.X_val_df.copy()

                if rate > 0.0:
                    # Inject NaNs randomly into numerical columns
                    for col in self.num_cols:
                        mask = np.random.rand(len(X_val_corrupted)) < rate
                        X_val_corrupted.loc[mask, col] = np.nan

                # Impute using statistics learned from training set
                imputer = SimpleImputer(strategy=strat)
                imputer.fit(self.X_train_df[self.num_cols])
                X_val_imputed_num = imputer.transform(X_val_corrupted[self.num_cols])
                
                # Reconstruct DF for transformer
                X_val_imputed_df = pd.DataFrame(X_val_imputed_num, columns=self.num_cols, index=self.X_val_df.index)
                X_val_imputed_df[self.cat_cols] = self.X_val_df[self.cat_cols].values

                # Transform with fitted scaler
                X_val_trans = ct_clean.transform(X_val_imputed_df)
                preds = model.predict(X_val_trans)

                acc = float(accuracy_score(self.y_val_s, preds))
                rec = float(recall_score(self.y_val_s, preds, average='macro', zero_division=0))
                f1 = float(f1_score(self.y_val_s, preds, average='macro', zero_division=0))

                results["mcar_rates"][str(int(rate*100))][strat] = {
                    "accuracy": round(acc, 4),
                    "macro_recall": round(rec, 4),
                    "macro_f1": round(f1, 4)
                }
                print(f"MCAR {int(rate*100)}% ({strat} impute): Acc={acc*100:.2f}%, Recall={rec:.4f}, F1={f1:.4f}")

        # Group Omission Scenarios (100% of ECG missing vs 100% of Clinical missing)
        group_scenarios = [
            ("All_ECG_Missing", self.preprocessor_helper.ecg_features),
            ("All_Clinical_Missing", self.preprocessor_helper.clinical_features)
        ]

        # Fit imputer on training set (leakage-free!)
        train_imputer = SimpleImputer(strategy='median')
        train_imputer.fit(self.X_train_df[self.num_cols])

        for scenario_name, omitted_cols in group_scenarios:
            X_val_dropped = self.X_val_df.copy()
            for col in omitted_cols:
                X_val_dropped[col] = np.nan

            X_val_imp_num = train_imputer.transform(X_val_dropped[self.num_cols])
            X_val_imp_df = pd.DataFrame(X_val_imp_num, columns=self.num_cols, index=self.X_val_df.index)
            X_val_imp_df[self.cat_cols] = self.X_val_df[self.cat_cols].values

            X_val_trans = ct_clean.transform(X_val_imp_df)
            preds = model.predict(X_val_trans)

            acc = float(accuracy_score(self.y_val_s, preds))
            rec = float(recall_score(self.y_val_s, preds, average='macro', zero_division=0))
            f1 = float(f1_score(self.y_val_s, preds, average='macro', zero_division=0))

            results["group_omissions"][scenario_name] = {
                "accuracy": round(acc, 4),
                "macro_recall": round(rec, 4),
                "macro_f1": round(f1, 4)
            }
            print(f"Group Omission [{scenario_name}]: Acc={acc*100:.2f}%, Recall={rec:.4f}, F1={f1:.4f}")

        # Save metrics
        out_json = os.path.join(self.metrics_dir, "robustness_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Plot robustness curves
        self.plot_robustness_curves(results)
        return results

    def plot_robustness_curves(self, results: Dict[str, Any]):
        """Plots accuracy and macro F1 degradation under varying missingness rates."""
        rates = [0, 10, 20, 30]
        acc_mean = [results["mcar_rates"][str(r)]["mean"]["accuracy"] * 100 for r in rates]
        f1_mean = [results["mcar_rates"][str(r)]["mean"]["macro_f1"] * 100 for r in rates]
        acc_median = [results["mcar_rates"][str(r)]["median"]["accuracy"] * 100 for r in rates]
        f1_median = [results["mcar_rates"][str(r)]["median"]["macro_f1"] * 100 for r in rates]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Accuracy degradation
        ax1.plot(rates, acc_mean, 'o-', color='#2563eb', linewidth=2, label='Mean Imputation')
        ax1.plot(rates, acc_median, 's--', color='#0284c7', linewidth=2, label='Median Imputation')
        ax1.axhline(25.0, color='#ef4444', linestyle=':', label='Chance Level (25%)')
        ax1.set_title("Validation Accuracy vs. Missing Rate (MCAR)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Missing Data Rate (%)", fontsize=10)
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.set_ylim(20, 35)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Macro F1 degradation
        ax2.plot(rates, f1_mean, 'o-', color='#16a34a', linewidth=2, label='Mean Imputation')
        ax2.plot(rates, f1_median, 's--', color='#059669', linewidth=2, label='Median Imputation')
        ax2.set_title("Macro F1-Score vs. Missing Rate (MCAR)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Missing Data Rate (%)", fontsize=10)
        ax2.set_ylabel("Macro F1-Score (%)", fontsize=10)
        ax2.set_ylim(20, 35)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        fig.suptitle("MUSE-CVD: Missing-Feature Robustness and Fallback Imputation Benchmark", fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "robustness_curves.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    reports_dir = os.path.join(base_dir, "reports")

    robustness = RobustnessExperiment(data_path, reports_dir, random_state=42)
    robustness.run_missingness_rates()


if __name__ == "__main__":
    main()
