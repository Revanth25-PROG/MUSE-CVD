import os
import sys
import pytest
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from data_validation import DatasetValidator


def test_dataset_exists_and_loads():
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    assert os.path.exists(data_path), f"Dataset file does not exist at {data_path}"
    validator = DatasetValidator(data_path)
    df = validator.load_data()
    assert not df.empty
    assert df.shape[0] == 4327
    assert df.shape[1] == 20


def test_dataset_validation_metrics():
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    validator = DatasetValidator(data_path)
    res = validator.validate_schema()

    assert res["row_count"] == 4327
    assert res["column_count"] == 20
    assert res["total_missing_values"] == 0
    assert res["duplicate_records"] == 0
    assert len(res["target_distribution"]) == 4
    assert "Normal" in res["target_distribution"]
    assert "Abnormal_Heartbeat" in res["target_distribution"]
    assert "Myocardial_Infarction" in res["target_distribution"]
    assert "Post_MI_History" in res["target_distribution"]


def test_no_target_leakage_flags():
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    validator = DatasetValidator(data_path)
    res = validator.validate_schema()
    # Ensure no leakage flags exist
    assert len(res["potential_target_leakage"]) == 0
