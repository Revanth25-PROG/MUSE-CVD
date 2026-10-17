"""
MUSE-CVD: Exploratory Data Analysis (EDA) Module
Generates high-resolution scientific visualizations and statistical summaries
covering target distribution, feature groupings, correlations, and class distributions.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


class CardiovascularEDA:
    def __init__(self, data_path: str, figures_dir: str, metrics_dir: str):
        self.data_path = data_path
        self.figures_dir = figures_dir
        self.metrics_dir = metrics_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.df = pd.read_csv(data_path)

        # Domain grouping
        self.clinical_cols = [
            'Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol', 'Glucose', 'Troponin'
        ]
        self.ecg_cols = [
            'PR_Interval', 'QRS_Duration', 'QT_Interval', 'P_Wave_Duration', 'T_Wave_Duration',
            'ST_Segment_Level', 'RR_Interval', 'HRV_SDNN', 'HRV_RMSSD', 'R_Peak_Amplitude', 'S_Wave_Amplitude'
        ]

    def plot_target_distribution(self):
        """Generates target class distribution bar chart with count and percentage annotations."""
        counts = self.df['Target'].value_counts()
        total = len(self.df)
        
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        colors = ['#1e3a8a', '#2563eb', '#38bdf8', '#0284c7']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='#0f172a', linewidth=1.2, width=0.55)

        for bar in bars:
            h = bar.get_height()
            pct = (h / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., h + 20, f'{h}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e293b')

        ax.set_title("MUSE-CVD: Target Class Distribution (N=4,327)", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Record Count", fontsize=11, fontweight='semibold')
        ax.set_ylim(0, max(counts.values) + 160)
        ax.tick_params(axis='x', rotation=10, labelsize=10)
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "target_distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_clinical_histograms(self):
        """Plots clinical and laboratory feature histograms."""
        n_features = len(self.clinical_cols)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
        axes = axes.flatten()

        for idx, col in enumerate(self.clinical_cols):
            ax = axes[idx]
            sns.histplot(self.df[col], kde=True, ax=ax, color='#2563eb', bins=25, edgecolor='black', alpha=0.6)
            ax.set_title(f"{col}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)

        # Gender categorical chart in the 8th subplot
        ax_last = axes[-1]
        gender_counts = self.df['Gender'].value_counts()
        ax_last.bar(gender_counts.index, gender_counts.values, color=['#ec4899', '#3b82f6'], edgecolor='black', width=0.5)
        for idx, val in enumerate(gender_counts.values):
            ax_last.text(idx, val + 30, f"{val} ({val/len(self.df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=9)
        ax_last.set_title("Gender Distribution", fontsize=11, fontweight='bold')
        ax_last.set_ylabel("Count", fontsize=9)
        ax_last.set_ylim(0, max(gender_counts.values) + 200)

        fig.suptitle("Clinical and Demographic Feature Distributions", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "clinical_features_distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_ecg_histograms(self):
        """Plots ECG-derived numerical measurements distributions."""
        fig, axes = plt.subplots(3, 4, figsize=(18, 11), dpi=300)
        axes = axes.flatten()

        for idx, col in enumerate(self.ecg_cols):
            ax = axes[idx]
            sns.histplot(self.df[col], kde=True, ax=ax, color='#0284c7', bins=25, edgecolor='black', alpha=0.6)
            ax.set_title(f"{col}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)

        # Hide extra unused subplot
        axes[-1].axis('off')

        fig.suptitle("ECG-Derived Morphological and Interval Distributions", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "ecg_features_distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_correlation_heatmap(self):
        """Plots full correlation matrix heatmap."""
        num_cols = self.clinical_cols + self.ecg_cols
        corr = self.df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="Blues", annot=True, fmt=".2f",
                    linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 7.5})

        ax.set_title("Pairwise Feature Correlation Matrix (Pearson r)", fontsize=15, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "correlation_heatmap.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_class_feature_boxplots(self):
        """Plots boxplots of representative clinical and ECG features across target classes."""
        selected_features = [
            ('Troponin', 'Cardiac Troponin (ng/mL)'),
            ('Heart_Rate', 'Heart Rate (bpm)'),
            ('Systolic_BP', 'Systolic BP (mmHg)'),
            ('ST_Segment_Level', 'ST Segment Elevation (mV)'),
            ('QT_Interval', 'QT Interval (ms)'),
            ('HRV_SDNN', 'HRV SDNN (ms)')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
        axes = axes.flatten()

        for idx, (col, title) in enumerate(selected_features):
            ax = axes[idx]
            sns.boxplot(x='Target', y=col, data=self.df, ax=ax, palette="Set2")
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=15, labelsize=8.5)

        fig.suptitle("Class-Wise Feature Comparisons (Conditioned on Cardiovascular State)", fontsize=15, fontweight='bold', y=0.99)
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "class_feature_boxplots.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_feature_relationships(self):
        """Plots 2D scatter plots of physiologically coupled parameters."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
        
        # 1. Systolic vs Diastolic BP
        sns.scatterplot(x='Diastolic_BP', y='Systolic_BP', hue='Target', data=self.df, ax=axes[0], alpha=0.5, palette='tab10', s=20)
        axes[0].set_title("Systolic vs. Diastolic BP", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Diastolic BP (mmHg)")
        axes[0].set_ylabel("Systolic BP (mmHg)")

        # 2. QT vs RR Interval
        sns.scatterplot(x='RR_Interval', y='QT_Interval', hue='Target', data=self.df, ax=axes[1], alpha=0.5, palette='tab10', s=20)
        axes[1].set_title("QT Interval vs. RR Interval", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("RR Interval (ms)")
        axes[1].set_ylabel("QT Interval (ms)")

        # 3. Troponin vs ST Segment Level
        sns.scatterplot(x='ST_Segment_Level', y='Troponin', hue='Target', data=self.df, ax=axes[2], alpha=0.5, palette='tab10', s=20)
        axes[2].set_title("Troponin vs. ST Segment Level", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("ST Segment Level (mV)")
        axes[2].set_ylabel("Troponin (ng/mL)")

        for ax in axes:
            ax.legend(title='Target', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        fig.suptitle("Physiological Relationship Bivariate Scatters", fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, "feature_relationships.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def generate_eda_summary(self):
        """Saves JSON statistical metrics of outliers, interquartile ranges, and correlations."""
        summary = {
            "total_records": len(self.df),
            "feature_counts": {
                "clinical_features": len(self.clinical_cols),
                "ecg_features": len(self.ecg_cols),
                "demographic_categorical": 1,
                "target_classes": int(self.df['Target'].nunique())
            },
            "class_distribution": self.df['Target'].value_counts().to_dict(),
            "outlier_summary": {},
            "correlation_summary": {}
        }

        # Outlier calculation via 1.5 * IQR rule
        num_cols = self.clinical_cols + self.ecg_cols
        for col in num_cols:
            q25 = self.df[col].quantile(0.25)
            q75 = self.df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = int(((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum())
            summary["outlier_summary"][col] = {
                "iqr": round(float(iqr), 3),
                "outlier_count": outliers,
                "outlier_percentage": round(float(outliers / len(self.df) * 100), 2)
            }

        # Correlation extrema
        corr_matrix = self.df[num_cols].corr()
        np_corr = corr_matrix.values.copy()
        np.fill_diagonal(np_corr, 0)
        max_idx = np.unravel_index(np.argmax(np.abs(np_corr)), np_corr.shape)
        summary["correlation_summary"] = {
            "max_absolute_correlation": round(float(np.max(np.abs(np_corr))), 4),
            "mean_absolute_correlation": round(float(np.mean(np.abs(np_corr))), 4),
            "strongest_pair": [num_cols[max_idx[0]], num_cols[max_idx[1]]]
        }

        output_path = os.path.join(self.metrics_dir, "eda_summary.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {output_path}")

    def run_all(self):
        """Runs the entire EDA suite."""
        print("Generating Target Distribution...")
        self.plot_target_distribution()
        print("Generating Clinical Histograms...")
        self.plot_clinical_histograms()
        print("Generating ECG Histograms...")
        self.plot_ecg_histograms()
        print("Generating Correlation Heatmap...")
        self.plot_correlation_heatmap()
        print("Generating Class Feature Boxplots...")
        self.plot_class_feature_boxplots()
        print("Generating Bivariate Scatters...")
        self.plot_feature_relationships()
        print("Generating EDA Statistical Summary...")
        self.generate_eda_summary()
        print("EDA pipeline execution complete.")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    metrics_dir = os.path.join(base_dir, "reports", "metrics")
    
    eda = CardiovascularEDA(data_path, figures_dir, metrics_dir)
    eda.run_all()


if __name__ == "__main__":
    main()
