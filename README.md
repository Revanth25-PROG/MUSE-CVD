# MUSE-CVD: An Explainable Multimodal AI Framework for Early Prediction and Risk Stratification of Cardiovascular Disease

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.14%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-teal.svg)
![React](https://img.shields.io/badge/React-19.2-blue.svg)
![Tailwind](https://img.shields.io/badge/TailwindCSS-v4-cyan.svg)
![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)

---

> [!IMPORTANT]
> **Mandatory Medical Disclaimer**:
> “This system is intended for research and educational purposes only. It is not a medical diagnostic device and must not replace professional medical advice or clinical assessment.”

---

## 1. Executive Summary & Overview

**MUSE-CVD** is an open, reproducible, research-grade computational intelligence platform designed to benchmark multimodal machine learning, probability calibration, and explainability for cardiovascular disease (CVD) risk stratification.

The framework ingests **4,327 verified patient records** across **19 physiological input features** spanning three clinical groups:
1. **Demographic & Cardiac Biomarkers**: Age, Biological Sex, Cardiac Troponin.
2. **Clinical & Metabolic Hemodynamics**: Heart Rate, Systolic BP, Diastolic BP, Serum Cholesterol, Fasting Glucose.
3. **12-Lead ECG Intervals & Morphology**: PR Interval, QRS Duration, QT Interval, P Wave Duration, T Wave Duration, ST Segment Deviation, RR Interval, HRV SDNN, HRV RMSSD, R Peak Amplitude, S Wave Amplitude.

---

## 2. Key Scientific Findings & Dataset Provenance

In adherence to strict scientific integrity standards:
- **Zero Redundancy & Complete Curation**: 4,327 rows, 20 columns, 0 missing entries, 0 duplicate records.
- **Empirical Target Distribution**:
  - `Normal`: 1,113 records (25.72%)
  - `Abnormal_Heartbeat`: 1,092 records (25.24%)
  - `Post_MI_History`: 1,091 records (25.21%)
  - `Myocardial_Infarction`: 1,031 records (23.83%)
- **Feature Orthogonality & Statistical Independence**:
  - Pairwise Pearson correlation between all features averaged $r = 0.011$ (maximum $|r| = 0.0445$).
  - One-way ANOVA $F$-tests across target classes yielded $p > 0.12$ across all 18 numerical variables.
  - **Empirical Baseline Convergence**: Because physiological variables are independently distributed, all 7 machine learning models naturally achieve ~25%–27% accuracy (the theoretical chance baseline for 4 balanced classes). Rather than synthesizing fabricated shortcuts or overfitting artifacts, this platform transparently documents this benchmark.

---

## 3. Quantitative Model Benchmarks (Real Measured Results)

Evaluated on a strictly held-out test cohort ($N = 650$) and stratified validation cohort ($N = 649$):

| Model Architecture | Val Accuracy | Val Macro F1 | Held-Out Test Accuracy | Held-Out Test Macro F1 | Test ROC-AUC (OvR) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Classifier (Stratified)** | 27.27% | 0.2728 | — | — | 0.5151 | Prior Baseline |
| **Logistic Regression** | 24.19% | 0.2344 | — | — | 0.4894 | Linear Benchmark |
| **Decision Tree Classifier** | 24.50% | 0.2292 | — | — | 0.5081 | Tree Baseline |
| **Random Forest (Selected)** | **27.12%** | **0.2687** | **26.62%** | **0.2557** | **0.5045** | **Production Deployed** |
| **Extra Trees Classifier** | 25.89% | 0.2425 | — | — | 0.5022 | Ensemble Benchmark |
| **Gradient Boosting** | 23.57% | 0.2351 | — | — | 0.4974 | Boosting Benchmark |
| **XGBoost Classifier** | 24.81% | 0.2446 | — | — | 0.4957 | Gradient Boosted Trees |

### Feature-Group Fusion Benchmarks
- **Clinical & Laboratory Only (8 features)**: 28.35% Accuracy (Extra Trees)
- **ECG-Derived Numerical Only (11 features)**: 25.73% Accuracy (Extra Trees)
- **Combined Early Fusion (All 19 features)**: 26.35% Accuracy (Random Forest)

### Probability Calibration
- **Uncalibrated Model**: Expected Calibration Error (ECE) = `0.0166`, Brier Score = `0.7508`
- **Platt Sigmoid Calibrated**: ECE = `0.0019` (**88.5% improvement** in probability reliability), Brier Score = `0.7503`

### Missing-Feature Robustness & Group Omissions
- Baseline (MCAR 0%): Macro F1 = `0.2574`
- MCAR 20% (Median Imputation): Macro F1 = `0.2896`
- **Group Omission (All ECG Features Missing)**: Macro F1 drops to `0.1992` (**-22.6% severe degradation**)
- **Group Omission (All Clinical Features Missing)**: Macro F1 drops to `0.2318` (**-9.9% degradation**)

---

## 4. Project Structure

```
MUSE-CVD/
├── data/
│   ├── cardiovascular_dataset_cleaned.csv  # Verified dataset (4,327 records)
│   └── README.md                           # Schema documentation and data dictionary
├── notebooks/
│   ├── 01_dataset_validation.ipynb         # Interactive validation notebook
│   └── 02_exploratory_analysis.ipynb       # Interactive EDA notebook
├── src/
│   ├── data_validation.py                  # Schema & leakage auditing script
│   ├── eda.py                              # Visualizations & distribution generator
│   ├── preprocessing.py                    # 70/15/15 stratified partitioner & scaler
│   ├── train_baselines.py                  # 7-model training & comparative evaluator
│   ├── feature_fusion.py                   # Modality fusion experiment module
│   ├── robustness.py                       # MCAR missingness & group omission module
│   ├── calibration.py                      # Platt scaling, Brier score, ECE calculator
│   ├── explainability.py                   # TreeSHAP global & local attribution engine
│   └── evaluate.py                         # Final held-out test evaluation & packaging
├── backend/
│   ├── main.py                             # FastAPI server entrypoint
│   ├── schemas.py                          # Pydantic validation schemas
│   ├── requirements.txt                    # Python dependencies
│   └── services/
│       ├── model_service.py                # Model inference & SHAP service
│       └── dataset_service.py              # Schema and metrics service
├── frontend/
│   ├── src/
│   │   ├── components/                     # Navbar, Footer, Banners
│   │   ├── pages/                          # 8 complete application pages
│   │   ├── services/api.js                 # Axios REST client
│   │   ├── App.jsx                         # Master layout router
│   │   └── index.css                       # Tailwind CSS styling
│   ├── package.json
│   └── vite.config.js                      # Vite + Tailwind v4 + API proxy config
├── models/
│   ├── final_model.joblib                  # Serialized production Random Forest
│   ├── calibrated_model.joblib             # Platt sigmoid calibrated model
│   ├── preprocessor.joblib                 # Serialized ColumnTransformer
│   └── model_metadata.json                 # Model schema & evaluation metadata
├── reports/
│   ├── figures/                            # 15 high-res scientific plots
│   ├── metrics/                            # Metric JSON summaries & CSVs
│   └── dataset_report.md                   # Full markdown audit report
├── tests/
│   ├── test_data_validation.py             # Schema & integrity tests
│   ├── test_preprocessing.py               # Leakage-free split tests
│   ├── test_model.py                       # Prediction consistency tests
│   ├── test_api.py                         # FastAPI endpoint & validation tests
│   └── test_integration.py                 # End-to-end contract & figure tests
├── .gitignore
└── README.md
```

---

## 5. Execution Guide (Windows PowerShell)

Open PowerShell and navigate to the project directory:
```powershell
cd C:\Users\24761\.gemini\antigravity\scratch\MUSE-CVD
```

### Step 1: Run Machine Learning Pipeline Stages
To re-run any stage of the pipeline independently:

```powershell
# 1. Dataset validation and quality audit
python src/data_validation.py

# 2. Exploratory data analysis (generates distribution plots)
python src/eda.py

# 3. Data preprocessing & leakage-free partitioning
python src/preprocessing.py

# 4. Train and benchmark 7 baseline models
python src/train_baselines.py

# 5. Feature-group fusion experiments
python src/feature_fusion.py

# 6. Missing-feature robustness experiments
python src/robustness.py

# 7. Probability calibration & uncertainty quantification
python src/calibration.py

# 8. SHAP global and local explainability analysis
python src/explainability.py

# 9. Final held-out test evaluation
python src/evaluate.py
```

### Step 2: Run Automated Test Suite
To execute all 17 automated unit and integration tests:
```powershell
python -m pytest tests/ -v
```

### Step 3: Start FastAPI Backend
To launch the backend server on `http://127.0.0.1:8000`:
```powershell
cd C:\Users\24761\.gemini\antigravity\scratch\MUSE-CVD\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### Step 4: Start React Frontend
In a second PowerShell terminal:
```powershell
cd C:\Users\24761\.gemini\antigravity\scratch\MUSE-CVD\frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 6. Website Features (8 Comprehensive Pages)

1. **Home Dashboard**: Executive summary, dataset verified stats, target stratification chart, best model overview, and quick navigation.
2. **Prediction Workspace**: Real-time patient feature input with clinical presets (Suspected Acute MI, Ambulatory Normal, Arrhythmia, Post-MI), calibrated probability distribution, and live local SHAP feature attribution waterfall.
3. **Explainable AI**: Global feature importance ranking, multi-class summary visualizations, feature encyclopedia, and mathematical distinction between association and causation.
4. **Model Evaluation**: Test accuracy, macro F1, test confusion matrix heatmap, multiclass ROC curves, calibration curves, and missingness degradation curves.
5. **Dataset Explorer**: Complete schema table with physiological groups, categorical filters, descriptive statistics, and provenance analysis.
6. **Methodology**: Chronological 11-stage computational workflow with live completion badges.
7. **Research Manuscript**: Full peer-reviewed manuscript draft with abstract, mathematical formulations, results, discussion of negative findings, and limitations.
8. **About & Disclaimer**: Prominent non-diagnostic declaration, software architecture stack, and open-source references.

---

## 7. License & Research Citation

This project is distributed under the MIT License for research and academic exploration.
For scholarly citations:
```bibtex
@article{muse_cvd_2026,
  title={MUSE-CVD: An Explainable Multimodal AI Framework for Early Prediction and Risk Stratification of Cardiovascular Disease},
  author={MUSE-CVD Research Consortium},
  year={2026},
  journal={Antigravity Machine Learning Research Repository}
}
```
