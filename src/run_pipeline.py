"""
MUSE-CVD: Master End-to-End Pipeline Runner
Executes all scientific pipeline modules sequentially:
1. Data Validation & Integrity Audit
2. Exploratory Data Analysis (EDA) & Figure Generation
3. Leakage-Free Preprocessing & Stratified Splitting
4. 7-Algorithm Baseline Benchmarking & Model Selection
5. Multimodal Feature Fusion Ablation
6. Probability Calibration (Platt Sigmoid & Isotonic)
7. Explainable AI via TreeSHAP
8. Missing-Feature Robustness & MCAR Stress Testing
9. Final Evaluation on Held-Out Test Set (N=650) & Deployment Serialization
"""

import os
import sys
import time

# Ensure src directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import data_validation
import eda
import preprocessing
import train_baselines
import feature_fusion
import calibration
import explainability
import robustness
import evaluate


def print_step(step_num: int, title: str):
    divider = "=" * 80
    print(f"\n{divider}")
    print(f"  [STAGE {step_num}/9] {title}")
    print(f"{divider}\n")


def run_full_pipeline():
    start_total = time.time()
    print("=" * 80)
    print("  MUSE-CVD: Starting Full End-to-End Scientific Machine Learning Pipeline")
    print("=" * 80)

    # Stage 1: Data Validation
    print_step(1, "Data Governance, Provenance & Validation Audit")
    t0 = time.time()
    data_validation.main()
    print(f"Stage 1 completed in {time.time() - t0:.2f}s")

    # Stage 2: Exploratory Data Analysis
    print_step(2, "Exploratory Data Analysis (EDA) & Statistical Profiling")
    t0 = time.time()
    eda.main()
    print(f"Stage 2 completed in {time.time() - t0:.2f}s")

    # Stage 3: Preprocessing & Splitting
    print_step(3, "Leakage-Free Preprocessing & Stratified Splitting (85% Train / 15% Test)")
    t0 = time.time()
    preprocessing.main()
    print(f"Stage 3 completed in {time.time() - t0:.2f}s")

    # Stage 4: Baseline Benchmark
    print_step(4, "7-Algorithm Baseline Benchmarking (5-Fold Stratified Cross-Validation)")
    t0 = time.time()
    train_baselines.main()
    print(f"Stage 4 completed in {time.time() - t0:.2f}s")

    # Stage 5: Multimodal Feature Fusion
    print_step(5, "Multimodal Feature Fusion Ablation (ECG vs. Clinical vs. Combined)")
    t0 = time.time()
    feature_fusion.main()
    print(f"Stage 5 completed in {time.time() - t0:.2f}s")

    # Stage 6: Probability Calibration
    print_step(6, "Probability Calibration (Platt Sigmoid & Isotonic Optimization)")
    t0 = time.time()
    calibration.main()
    print(f"Stage 6 completed in {time.time() - t0:.2f}s")

    # Stage 7: Explainable AI (SHAP)
    print_step(7, "Explainable AI (TreeSHAP Global & Local Attribution)")
    t0 = time.time()
    explainability.main()
    print(f"Stage 7 completed in {time.time() - t0:.2f}s")

    # Stage 8: Robustness Stress Testing
    print_step(8, "Missing-Feature Robustness & MCAR Degradation Testing")
    t0 = time.time()
    robustness.main()
    print(f"Stage 8 completed in {time.time() - t0:.2f}s")

    # Stage 9: Final Held-Out Evaluation
    print_step(9, "Final Test Cohort Evaluation (N=650) & Metric Serialization")
    t0 = time.time()
    evaluate.main()
    print(f"Stage 9 completed in {time.time() - t0:.2f}s")

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"  MUSE-CVD: Full Pipeline Successfully Completed in {elapsed_total:.2f}s!")
    print("  All models, metrics, and 15 publication figures are serialized.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
