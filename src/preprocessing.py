"""
MUSE-CVD: Preprocessing and Data Partitioning Module
Provides leakage-free train/validation/test splitting, categorical encoding,
feature scaling, and feature-group partitioning.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_col = 'Target'
        
        # Clinical & Laboratory features
        self.clinical_features = [
            'Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
            'Cholesterol', 'Glucose', 'Troponin'
        ]
        
        # Categorical features
        self.categorical_features = ['Gender']
        
        # ECG-derived morphological and interval features
        self.ecg_features = [
            'PR_Interval', 'QRS_Duration', 'QT_Interval', 'P_Wave_Duration',
            'T_Wave_Duration', 'ST_Segment_Level', 'RR_Interval', 'HRV_SDNN',
            'HRV_RMSSD', 'R_Peak_Amplitude', 'S_Wave_Amplitude'
        ]
        
        self.all_feature_cols = self.categorical_features + self.clinical_features + self.ecg_features
        self.preprocessor: ColumnTransformer = None
        self.label_mapping: Dict[str, int] = {}
        self.inv_label_mapping: Dict[int, str] = {}

    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads and separates features and target."""
        df = pd.read_csv(data_path)
        X = df[self.all_feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Establish deterministic label mapping
        unique_classes = sorted(y.unique().tolist())
        self.label_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.inv_label_mapping = {idx: cls for idx, cls in enumerate(unique_classes)}
        
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, train_size: float = 0.70, val_size: float = 0.15, test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Stratified three-way train/validation/test split.
        Uses train_size=0.70, val_size=0.15, test_size=0.15.
        """
        # First split: train vs (val + test)
        temp_size = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: val vs test (equal split of the remaining 30%)
        relative_test_size = test_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_preprocessor(self, X_train: pd.DataFrame, feature_subset: str = 'all') -> ColumnTransformer:
        """
        Fits a ColumnTransformer strictly on the training set.
        feature_subset options: 'all', 'clinical', 'ecg'
        """
        if feature_subset == 'clinical':
            num_cols = self.clinical_features
            cat_cols = self.categorical_features
        elif feature_subset == 'ecg':
            num_cols = self.ecg_features
            cat_cols = []
        else: # 'all'
            num_cols = self.clinical_features + self.ecg_features
            cat_cols = self.categorical_features

        transformers = []
        if num_cols:
            transformers.append(('num', StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        self.preprocessor.fit(X_train)
        return self.preprocessor

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X)

    def get_feature_names(self) -> List[str]:
        """Returns transformed feature column names."""
        if self.preprocessor is None:
            return []
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                cats = trans.get_feature_names_out(cols)
                feature_names.extend(list(cats))
        return feature_names

    def save_pipeline(self, output_path: str):
        """Persists the preprocessor and label mappings."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        artifact = {
            'preprocessor': self.preprocessor,
            'label_mapping': self.label_mapping,
            'inv_label_mapping': self.inv_label_mapping,
            'clinical_features': self.clinical_features,
            'ecg_features': self.ecg_features,
            'categorical_features': self.categorical_features,
            'feature_names_out': self.get_feature_names()
        }
        joblib.dump(artifact, output_path)

    @classmethod
    def load_pipeline(cls, input_path: str) -> Dict:
        """Loads the persisted preprocessor and metadata."""
        return joblib.load(input_path)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cardiovascular_dataset_cleaned.csv")
    models_dir = os.path.join(base_dir, "models")
    
    preprocessor = DataPreprocessor(random_state=42)
    X, y = preprocessor.prepare_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    preprocessor.fit_preprocessor(X_train, feature_subset='all')
    pipeline_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save_pipeline(pipeline_path)
    
    print(f"Data partitioning complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Preprocessor saved to: {pipeline_path}")
    print(f"Features: {preprocessor.get_feature_names()}")


if __name__ == "__main__":
    main()
