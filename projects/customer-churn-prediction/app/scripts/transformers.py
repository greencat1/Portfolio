import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import os 
from catboost import CatBoostClassifier

import pickle
import warnings
from app.utils.logger import logger

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# =====================================================
# 3.1 TotalCharges Cleaner
# =====================================================

class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean the TotalCharges column:
    - Convert to numeric (handling errors)
    - Apply domain logic: if tenure == 0 → TotalCharges = 0
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert TotalCharges to numeric
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        
        # Apply domain logic: if tenure = 0 → TotalCharges = 0
        X.loc[X['tenure'] == 0, 'TotalCharges'] = 0
        
        
        return X


# =====================================================
# 3.2 Feature Engineering
# =====================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new features:
    - avg_monthly_spend
    - tenure_group
    - has_internet
    - has_addons
    - risky_contract
    - auto_payment
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Average monthly spend
        X['avg_monthly_spend'] = np.where(
            X['tenure'] > 0,
            X['TotalCharges'] / X['tenure'],
            0
        )
        
        # Tenure group (customer lifecycle)
        bins = [0, 12, 36, float('inf')]
        labels = ['new', 'mid', 'loyal']
        X['tenure_group'] = pd.cut(X['tenure'], bins=bins, labels=labels, right=False)
        
        # Has internet
        X['has_internet'] = X['InternetService'].apply(lambda x: 0 if x == 'No' else 1)
        
        # Has addons (internet additional services)
        addon_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        X['has_addons'] = X[addon_cols].apply(
            lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0})
        ).sum(axis=1)
        
        # Risky contract (Month-to-month)
        X['risky_contract'] = X['Contract'].apply(
            lambda x: 1 if x == 'Month-to-month' else 0
        )
        
        # Auto payment
        auto_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
        X['auto_payment'] = X['PaymentMethod'].apply(
            lambda x: 1 if x in auto_methods else 0
        )
        
     
        return X


# =====================================================
# 3.3 Categorical Encoder
# =====================================================

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features:
    - Binary features (Yes/No, Female/Male) → 0/1
    - Ordinal features: tenure_group and Contract
    - One-hot encoding for nominal features: MultipleLines, InternetService, PaymentMethod
    - Converts boolean columns to int
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Binary encoding
        binary_cols = [
            'gender', 'Partner', 'Dependents', 
            'PhoneService', 'PaperlessBilling',
        ]
        
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map({
                    'Yes': 1, 
                    'No': 0, 
                    'Female': 1, 
                    'Male': 0
                })
        
        # 2. Ordinal encoding for tenure_group
        tenure_mapping = {'new': 0, 'mid': 1, 'loyal': 2}
        if 'tenure_group' in X.columns:
            X['tenure_group'] = X['tenure_group'].map(tenure_mapping)
            X['tenure_group'] = X['tenure_group'].astype(int)
        
        # 3. Ordinal encoding for Contract
        contract_map = {
            'Month-to-month': 0,
            'One year': 1,
            'Two year': 2
        }
        if 'Contract' in X.columns:
            X['Contract'] = X['Contract'].map(contract_map)
        
        # 4. One-hot encoding for multi-category features
        multi_cols = ['MultipleLines', 'InternetService', 'PaymentMethod']
        # Apply one-hot encoding
        X = pd.get_dummies(X, columns=multi_cols, drop_first=True)
        
        # 5. Convert boolean columns to int
        bool_cols = X.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)
        
        # 6. Ensure only numeric columns remain
        X = X.select_dtypes(exclude='object')
        
        return X


# =====================================================
# 3.4 Drop Redundant Columns
# =====================================================

class DropRedundant(BaseEstimator, TransformerMixin):
    """
    Drops redundant and unnecessary columns:
    - customerID (identifier)
    - TotalCharges (correlated with tenure)
    - MonthlyCharges (duplicated by avg_monthly_spend)
    - num_services (duplicated by has_addons)
    - Contract (duplicated by risky_contract)
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        cols_to_drop = [
            'customerID',        # unique identifier
            'TotalCharges',      # highly correlated with tenure
            'MonthlyCharges',    # duplicated by avg_monthly_spend
            'num_services',      # duplicated by has_addons
            'Contract'           # duplicated by risky_contract
        ]
        
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
       
        
        return X

# =====================================================
# 4. Scaling Numerical Features
# =====================================================

class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using StandardScaler:
    - tenure
    - avg_monthly_spend
    - has_addons
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols = ['tenure', 'avg_monthly_spend', 'has_addons']
    
    def fit(self, X, y=None):
        # Fit scaler only on the specified columns
        cols_to_scale = [col for col in self.scale_cols if col in X.columns]
        if cols_to_scale:
            self.scaler.fit(X[cols_to_scale])
        return self
    
    def transform(self, X):
        X = X.copy()
        cols_to_scale = [col for col in self.scale_cols if col in X.columns]
        
        if cols_to_scale:
            X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
   
        return X


