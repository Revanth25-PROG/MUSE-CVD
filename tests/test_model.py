import os
import sys
import pytest
import joblib
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))


def test_model_loading_and_prediction():
    model_path = os.path.join(base_dir, "models", "final_model.joblib")
    prep_path = os.path.join(base_dir, "models", "preprocessor.joblib")

    assert os.path.exists(model_path), f"Final model not found at {model_path}"
    assert os.path.exists(prep_path), f"Preprocessor not found at {prep_path}"

    model = joblib.load(model_path)
    prep_artifact = joblib.load(prep_path)

    # Synthetic sample patient
    import pandas as pd
    sample_df = pd.DataFrame([{
        'Age': 60,
        'Gender': 'Male',
        'Heart_Rate': 85,
        'Systolic_BP': 130,
        'Diastolic_BP': 80,
        'Cholesterol': 210,
        'Glucose': 105,
        'Troponin': 0.12,
        'PR_Interval': 160,
        'QRS_Duration': 100,
        'QT_Interval': 400,
        'P_Wave_Duration': 95,
        'T_Wave_Duration': 150,
        'ST_Segment_Level': 0.05,
        'RR_Interval': 850,
        'HRV_SDNN': 65,
        'HRV_RMSSD': 50,
        'R_Peak_Amplitude': 1.8,
        'S_Wave_Amplitude': -1.2
    }])

    preprocessor = prep_artifact['preprocessor']
    X_trans = preprocessor.transform(sample_df)
    assert X_trans.shape == (1, 19)

    probs = model.predict_proba(X_trans)[0]
    assert len(probs) == 4
    # Probability sum must be approximately 1.0
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

    pred = model.predict(X_trans)[0]
    assert pred in ['Normal', 'Abnormal_Heartbeat', 'Myocardial_Infarction', 'Post_MI_History']
