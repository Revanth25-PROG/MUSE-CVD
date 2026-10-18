# 🛠️ MUSE-CVD Implementation & Proof Suite

Welcome to the **MUSE-CVD Implementation Suite**. This folder contains everything required to reproduce, audit, execute, and verify the entire end-to-end medical artificial intelligence framework.

---

## ⚡ Quick Execution Options

### Option 1: One-Click Cloud Proof (Google Colab)
Run the entire pipeline in your browser without installing anything locally:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Revanth25-PROG/MUSE-CVD/blob/main/notebooks/MUSE_CVD_Implementation_Colab.ipynb)

*The notebook automatically fetches the verified dataset from GitHub, executes all 10 stages (data audit, 7-algorithm benchmark, Platt probability calibration, TreeSHAP explainability, held-out test evaluation), and renders all figures.*

---

### Option 2: Local Terminal Execution (One Command)
Run the master pipeline script from the root directory:

```bash
# Clone the repository
git clone https://github.com/Revanth25-PROG/MUSE-CVD.git
cd MUSE-CVD

# Install dependencies
pip install -r backend/requirements.txt

# Run the master pipeline
python src/run_pipeline.py
```

This sequentially executes all 9 stages:
1. `src/data_validation.py` - Missing value, duplicate, and ANOVA hypothesis tests.
2. `src/eda.py` - Correlation heatmaps, distributions, and feature relationship plots.
3. `src/preprocessing.py` - Leakage-free `ColumnTransformer` & stratified 85/15 splitting.
4. `src/train_baselines.py` - 5-fold cross-validation across 7 models (Random Forest, Extra Trees, XGBoost, etc.).
5. `src/feature_fusion.py` - Multimodal ablation (ECG vs. Clinical vs. Combined).
6. `src/calibration.py` - Platt Sigmoid calibration reducing ECE from $0.0166 \to 0.0019$.
7. `src/explainability.py` - TreeSHAP global and local attribution calculations.
8. `src/robustness.py` - MCAR 10%, 20%, 30% dropout stress testing.
9. `src/evaluate.py` - Single evaluation on held-out test cohort ($N=650$) and publication figure generation.

---

## 📁 Repository Folder Architecture

| Folder | Contents & Purpose |
| :--- | :--- |
| **`implementation/`** | Google Colab proof notebook and execution guides |
| **`src/`** | Modular production Python pipeline scripts |
| **`notebooks/`** | Interactive Jupyter and Google Colab notebooks |
| **`data/`** | Cleaned dataset (`cardiovascular_dataset_cleaned.csv`, $N=4,327$) |
| **`backend/`** | FastAPI asynchronous REST API (`main.py`, `schemas.py`, `services/`) |
| **`frontend/`** | React 18 + Vite interactive 8-page clinical dashboard |
| **`reports/`** | JSON evaluation metrics, data audit reports, and 15 PNG plots |
| **`models/`** | Serialized Scikit-learn pipelines and calibrated models (`.joblib`) |
| **`tests/`** | Automated pytest suite with 17 integration and unit tests |

---

## 🔬 Scientific Highlights & Validated Metrics

- **Total Cohort**: $N = 4,327$ records ($0$ missing values, $0$ duplicates)
- **4 Balanced Classes**: Normal ($25.72\%$), Arrhythmia ($25.24\%$), Post-MI ($25.21\%$), Acute MI ($23.83\%$)
- **Feature Orthogonality**: Pairwise Pearson $r_{\text{avg}} = 0.011$; between-class ANOVA $p > 0.12$ across all predictors
- **Final Held-Out Test Performance ($N = 650$)**:
  - **Accuracy**: $26.62\%$
  - **Macro F1**: $0.2557$
  - **Macro ROC-AUC (OvR)**: $0.5045$
  - **Expected Calibration Error (ECE)**: $0.0019$ (Platt Sigmoid calibrated)
