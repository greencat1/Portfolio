# tests/test_transformers.py
"""
Tests for data cleaning and transformation.
"""

import pandas as pd
import numpy as np
from app.scripts.transformers import TotalChargesCleaner, FeatureEngineer, CategoricalEncoder


def test_cleaner_converts_bad_data():
    """Should convert text to numbers and fix zeros."""
    df = pd.DataFrame({
        'TotalCharges': ['100', 'abc', np.nan],
        'tenure': [10, 5, 0]
    })
    
    cleaner = TotalChargesCleaner()
    result = cleaner.transform(df)
    
    # tenure=0 should have TotalCharges=0
    assert result.loc[2, 'TotalCharges'] == 0


# app/tests/test_transformers.py
def test_feature_engineer_creates_new_columns():
    """
    Should create avg_monthly_spend and tenure_group columns.
    
    The FeatureEngineer needs all expected columns to work properly.
    """
    # Create DataFrame with ALL required columns
    df = pd.DataFrame({
        'tenure': [12],
        'TotalCharges': [600],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],        # Required for has_addons calculation
        'DeviceProtection': ['No'],    # Required for has_addons calculation
        'TechSupport': ['No'],         # Required for has_addons calculation
        'StreamingTV': ['No'],         # Required for has_addons calculation
        'StreamingMovies': ['No'],     # Required for has_addons calculation
        'Contract': ['Month-to-month'],
        'PaymentMethod': ['Electronic check']
    })
    
    engineer = FeatureEngineer()
    result = engineer.transform(df)
    
    # Check that new columns were created
    assert 'avg_monthly_spend' in result.columns
    assert 'tenure_group' in result.columns
    assert 'has_internet' in result.columns
    assert 'has_addons' in result.columns
    assert 'risky_contract' in result.columns
    assert 'auto_payment' in result.columns
    
    # Verify calculation
    assert result.loc[0, 'avg_monthly_spend'] == 50.0  # 600 / 12 = 50


# app/tests/test_transformers.py
def test_encoder_removes_text():
    """
    Should convert all text columns to numbers.
    
    The encoder needs all expected categorical columns to work.
    """
    # Create DataFrame with ALL expected categorical columns
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'MultipleLines': ['No', 'Yes'],               # Required for one-hot encoding
        'InternetService': ['DSL', 'Fiber optic'],    # Required for one-hot encoding
        'PaymentMethod': ['Electronic check', 'Mailed check']  # Required for one-hot encoding
    })
    
    encoder = CategoricalEncoder()
    result = encoder.transform(df)
    
    # After encoding, no 'object' (text) columns should remain
    # All columns should be numeric (int, float, or uint8)
    assert all(result.dtypes != 'object')
    
    # Verify we have numeric columns only
    for col in result.columns:
        assert result[col].dtype in ['int64', 'int32', 'float64', 'uint8']