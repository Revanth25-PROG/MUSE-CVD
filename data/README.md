# Dataset Provenance and Specification: MUSE-CVD

## Overview
The dataset file `cardiovascular_dataset_cleaned.csv` comprises structured demographic, clinical, laboratory, and ECG-derived numerical measurements intended for multi-class cardiovascular disease risk stratification and prediction.

## File Properties
- **Filename**: `cardiovascular_dataset_cleaned.csv`
- **Total Records (Rows)**: 4,327
- **Total Columns**: 20 (19 feature attributes + 1 multi-class target label)
- **Target Label**: `Target` (Categorical, 4 classes)
  - `Normal` (~25.7%)
  - `Abnormal_Heartbeat` (~25.2%)
  - `Post_MI_History` (~25.2%)
  - `Myocardial_Infarction` (~23.8%)

## Feature Schema and Clinical Taxonomies

### Group 1: Demographic and Clinical Measurements (7 features)
1. **Age**: Patient age in years (30 to 84).
2. **Gender**: Biological sex (`Female`, `Male`).
3. **Heart_Rate**: Resting heart rate in beats per minute (bpm, 55 to 139).
4. **Systolic_BP**: Systolic blood pressure in mmHg (90 to 179).
5. **Diastolic_BP**: Diastolic blood pressure in mmHg (60 to 109).
6. **Cholesterol**: Total serum cholesterol in mg/dL (150 to 319).
7. **Glucose**: Fasting blood glucose in mg/dL (70 to 199).

### Group 2: Laboratory Biomarkers (1 feature)
8. **Troponin**: Cardiac troponin biomarker level in ng/mL (0.001 to 2.500).

### Group 3: ECG-Derived Morphological and Interval Measurements (11 features)
9. **PR_Interval**: PR interval in milliseconds (ms, 120 to 219).
10. **QRS_Duration**: QRS complex duration in milliseconds (ms, 70 to 159).
11. **QT_Interval**: QT interval duration in milliseconds (ms, 300 to 499).
12. **P_Wave_Duration**: P wave duration in milliseconds (ms, 80 to 119).
13. **T_Wave_Duration**: T wave duration in milliseconds (ms, 120 to 199).
14. **ST_Segment_Level**: ST segment deviation/elevation in millivolts (mV, -0.20 to +0.50).
15. **RR_Interval**: Mean RR interval duration in milliseconds (ms, 600 to 1199).
16. **HRV_SDNN**: Standard deviation of NN intervals in milliseconds (ms, 20 to 119).
17. **HRV_RMSSD**: Root mean square of successive differences in milliseconds (ms, 15 to 99).
18. **R_Peak_Amplitude**: Amplitude of the R wave peak in millivolts (mV, 0.50 to 3.00).
19. **S_Wave_Amplitude**: Amplitude of the S wave nadir in millivolts (mV, -2.50 to -0.10).

## Provenance Analysis & Empirical Validation Findings
Statistical analysis (one-way ANOVA, Pearson correlations, Kolmogorov-Smirnov tests) reveals that:
- Numerical features follow near-uniform bounded distributions matching clinical reference intervals.
- The average pairwise feature correlation is $r \approx 0.011$ (max $r \approx 0.045$), demonstrating orthogonality.
- Between-group ANOVA $F$-statistics yield $p > 0.12$ across all features, showing statistical independence from the target class.
- Consequently, baseline classification accuracy on held-out test data naturally converges to approximately 25-27% (chance baseline for a 4-class balanced distribution).
- **Research Integrity Note**: In accordance with medical AI benchmarking standards, this framework does not synthesize artificial features or fabricate inflated performance metrics. Instead, it systematically models, benchmarks, and explains the actual data dynamics.
