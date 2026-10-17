import os
import sys
import pytest
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from preprocessing import DataPreprocessor


def test_stratified_split_proportions():
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    preprocessor = DataPreprocessor(random_state=42)
    X, y = preprocessor.prepare_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    total_len = len(X)
    assert len(X_train) == 3028
    assert len(X_val) == 649
    assert len(X_test) == 650

    # Test set must be completely non-overlapping with train and val
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_preprocessing_transform_dimensions():
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    preprocessor = DataPreprocessor(random_state=42)
    X, y = preprocessor.prepare_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    preprocessor.fit_preprocessor(X_train, feature_subset='all')
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 18 numerical + 1 one-hot gender = 19 transformed features
    assert X_train_trans.shape[1] == 19
    assert X_test_trans.shape[1] == 19
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
