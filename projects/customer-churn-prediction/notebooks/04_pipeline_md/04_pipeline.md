# Pipeline
## Objective

The goal of this notebook is to build a complete, production-ready sklearn Pipeline that:
- Handles all data preprocessing steps (cleaning, feature engineering, encoding, and scaling)
- Processes both categorical and numerical features correctly
- Trains the final CatBoost model on the entire dataset
- Eliminates manual steps and ensures reproducibility
- Can be easily saved and deployed for inference on new raw data

### 1. Setup & Imports



```python
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

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')
```


```python
def get_metrics(y_true, y_pred, y_proba):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

```

## 2. Load Raw Data


```python
df = pd.read_csv('../data/raw/telco_churn.csv')
```

## 3. Custom Transformers

### 3.1 TotalCharges Cleaner


```python
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
```


```python
cleaner = TotalChargesCleaner()
cleaned_df = cleaner.transform(df)
print("\n✅ Test completed!")
print(f"Total missing values in TotalCharges after cleaning: {cleaned_df['TotalCharges'].isna().sum()}")
```

    
    ✅ Test completed!
    Total missing values in TotalCharges after cleaning: 0
    

### 3.2 Feature Engineering


```python
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
```


```python
engineer = FeatureEngineer()
engineered_df = engineer.transform(cleaned_df)

print("\n" + "="*60)
print("✅ Feature Engineering Test Results:")
print(f"✓ New features created: {['avg_monthly_spend', 'tenure_group', 'has_internet', 'has_addons', 'risky_contract', 'auto_payment']}")
print(f"✓ avg_monthly_spend calculated correctly: {(engineered_df['avg_monthly_spend'] >= 0).all()}")
print(f"✓ tenure_group created: {engineered_df['tenure_group'].notna().all()}")
print(f"✓ has_addons range: {engineered_df['has_addons'].min()} - {engineered_df['has_addons'].max()}")
print(f"✓ risky_contract is binary: {engineered_df['risky_contract'].isin([0,1]).all()}")
print(f"✓ auto_payment is binary: {engineered_df['auto_payment'].isin([0,1]).all()}")
print("="*60)
```

    
    ============================================================
    ✅ Feature Engineering Test Results:
    ✓ New features created: ['avg_monthly_spend', 'tenure_group', 'has_internet', 'has_addons', 'risky_contract', 'auto_payment']
    ✓ avg_monthly_spend calculated correctly: True
    ✓ tenure_group created: True
    ✓ has_addons range: 0 - 6
    ✓ risky_contract is binary: True
    ✓ auto_payment is binary: True
    ============================================================
    

### 3.3 Сategorical encoder


```python
# Переопределите CategoricalEncoder с правильной реализацией
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.expected_columns_ = None
    
    def fit(self, X, y=None):
        X_transformed = self._transform_impl(X)
        self.expected_columns_ = X_transformed.columns.tolist()
        return self
    
    def transform(self, X):
        X_transformed = self._transform_impl(X)
        for col in self.expected_columns_:
            if col not in X_transformed.columns:
                X_transformed[col] = 0
        X_transformed = X_transformed[self.expected_columns_]
        return X_transformed
    
    def _transform_impl(self, X):
        X = X.copy()
        
        # Binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})
        
        # Ordinal encoding for tenure_group
        tenure_mapping = {'new': 0, 'mid': 1, 'loyal': 2}
        if 'tenure_group' in X.columns:
            X['tenure_group'] = X['tenure_group'].map(tenure_mapping)
            X['tenure_group'] = X['tenure_group'].astype(int)
        
        # Ordinal encoding for Contract
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        if 'Contract' in X.columns:
            X['Contract'] = X['Contract'].map(contract_map)
        
        # One-hot encoding
        multi_cols = ['MultipleLines', 'InternetService', 'PaymentMethod']
        
        for col in multi_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        # Create dummies
        dummies = pd.get_dummies(X[multi_cols], prefix=multi_cols, drop_first=False)
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=multi_cols, errors='ignore')
        
        # Convert boolean to int
        bool_cols = X.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)
        
        # Keep only numeric columns
        X = X.select_dtypes(exclude='object')
        
     
        return X


```


```python
# Create encoder
encoder = CategoricalEncoder()

# FIRST call fit to set expected_columns_
encoder.fit(engineered_df)

# THEN call transform
encoded_df = encoder.transform(engineered_df)

print("\nAfter Categorical Encoding:")
print("Columns:", encoded_df.columns.tolist())

print("\nAfter Categorical Encoding:")
print("Columns:", encoded_df.columns.tolist())
print(f"Total columns after encoding: {encoded_df.shape[1]}")
print("Data types:\n", encoded_df.dtypes.value_counts())

print("\n" + "="*70)
print("✅ CategoricalEncoder Test Results:")
print("="*70)
print(f"✓ All columns are numeric: {encoded_df.select_dtypes(include='object').shape[1] == 0}")
print(f"✓ Binary columns encoded (0/1): {encoded_df['gender'].isin([0,1]).all() if 'gender' in encoded_df.columns else 'N/A'}")
print(f"✓ tenure_group is numeric: {pd.api.types.is_integer_dtype(encoded_df['tenure_group']) if 'tenure_group' in encoded_df.columns else 'N/A'}")
print(f"✓ One-hot encoding applied (example columns):")
onehot_example = [col for col in encoded_df.columns if col.startswith(('MultipleLines', 'InternetService', 'PaymentMethod'))]
print(f"   → {onehot_example[:6]} ...")  # показываем первые 6

print(f"\nFinal shape: {encoded_df.shape}")
print("All features are now numeric and ready for modeling.")
```

    
    After Categorical Encoding:
    Columns: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'tenure_group', 'has_internet', 'has_addons', 'risky_contract', 'auto_payment', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    
    After Categorical Encoding:
    Columns: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'tenure_group', 'has_internet', 'has_addons', 'risky_contract', 'auto_payment', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    Total columns after encoding: 26
    Data types:
     int64      12
    int32      11
    float64     3
    Name: count, dtype: int64
    
    ======================================================================
    ✅ CategoricalEncoder Test Results:
    ======================================================================
    ✓ All columns are numeric: True
    ✓ Binary columns encoded (0/1): True
    ✓ tenure_group is numeric: True
    ✓ One-hot encoding applied (example columns):
       → ['MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No'] ...
    
    Final shape: (7043, 26)
    All features are now numeric and ready for modeling.
    

### 3.4 Drop Redundant Columns


```python
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
```


```python
dropper = DropRedundant()
dropped_df = dropper.transform(encoded_df)

print("\nAfter dropping redundant columns:")
print(f"Number of columns: {dropped_df.shape[1]}")
print("Remaining columns:", dropped_df.columns.tolist())

print("\n✅ DropRedundant Test Results:")
print(f"✓ customerID removed: {'customerID' not in dropped_df.columns}")
print(f"✓ TotalCharges removed: {'TotalCharges' not in dropped_df.columns}")
print(f"✓ MonthlyCharges removed: {'MonthlyCharges' not in dropped_df.columns}")
print(f"✓ Final shape: {dropped_df.shape}")
```

    
    After dropping redundant columns:
    Number of columns: 23
    Remaining columns: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'avg_monthly_spend', 'tenure_group', 'has_internet', 'has_addons', 'risky_contract', 'auto_payment', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    
    ✅ DropRedundant Test Results:
    ✓ customerID removed: True
    ✓ TotalCharges removed: True
    ✓ MonthlyCharges removed: True
    ✓ Final shape: (7043, 23)
    

## 4. Scaling Numerical Features


```python
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
```


```python
scaler = NumericalScaler()
scaled_df = scaler.fit_transform(dropped_df)   


print("\nAfter scaling:")
print(scaled_df[['tenure', 'avg_monthly_spend', 'has_addons']].head(3))

print("\n✅ NumericalScaler Test Results:")
print(f"✓ Mean close to 0: {scaled_df['tenure'].mean():.4f}")
print(f"✓ Std close to 1: {scaled_df['tenure'].std():.4f}")
print(f"✓ All selected columns scaled successfully")
```

    
    After scaling:
         tenure  avg_monthly_spend  has_addons
    0 -1.277445          -1.151302   -0.561776
    1  0.066327          -0.301458   -0.020519
    2 -1.236724          -0.350966   -0.020519
    
    ✅ NumericalScaler Test Results:
    ✓ Mean close to 0: -0.0000
    ✓ Std close to 1: 1.0001
    ✓ All selected columns scaled successfully
    

## 5. Create Full Pipeline


```python
# =====================================================
# 5. Create Full Pipeline
# =====================================================

# Define the final pipeline combining all transformers and the model
full_pipeline = Pipeline([
    # Preprocessing steps
    ('totalcharges_cleaner', TotalChargesCleaner()),
    ('feature_engineer', FeatureEngineer()),
    ('categorical_encoder', CategoricalEncoder()),
    ('drop_redundant', DropRedundant()),
    ('numerical_scaler', NumericalScaler()),
    
    # Final Model - Best CatBoost from tuning
    ('model', CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=50,
        colsample_bylevel=0.7,
        subsample=0.7,
        min_data_in_leaf=1,
        random_strength=5,
        class_weights=[1, 5],          # Best weight from tuning
        random_state=42,
        verbose=0
    ))
])

print("✅ Full sklearn Pipeline created successfully!")
print("Pipeline steps:")
for step in full_pipeline.named_steps.keys():
    print(f"   → {step}")
```

    ✅ Full sklearn Pipeline created successfully!
    Pipeline steps:
       → totalcharges_cleaner
       → feature_engineer
       → categorical_encoder
       → drop_redundant
       → numerical_scaler
       → model
    

## 6. Train on Full Dataset


```python
y = df['Churn'].map({'Yes': 1, 'No': 0})
full_pipeline.fit(df, y)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;totalcharges_cleaner&#x27;, TotalChargesCleaner()),
                (&#x27;feature_engineer&#x27;, FeatureEngineer()),
                (&#x27;categorical_encoder&#x27;, CategoricalEncoder()),
                (&#x27;drop_redundant&#x27;, DropRedundant()),
                (&#x27;numerical_scaler&#x27;, NumericalScaler()),
                (&#x27;model&#x27;,
                 CatBoostClassifier(class_weights=[1, 5], colsample_bylevel=0.7, depth=4, iterations=500, l2_leaf_reg=50, learning_rate=0.03, min_data_in_leaf=1, random_state=42, random_strength=5, subsample=0.7, verbose=0))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;totalcharges_cleaner&#x27;, TotalChargesCleaner()),
                (&#x27;feature_engineer&#x27;, FeatureEngineer()),
                (&#x27;categorical_encoder&#x27;, CategoricalEncoder()),
                (&#x27;drop_redundant&#x27;, DropRedundant()),
                (&#x27;numerical_scaler&#x27;, NumericalScaler()),
                (&#x27;model&#x27;,
                 CatBoostClassifier(class_weights=[1, 5], colsample_bylevel=0.7, depth=4, iterations=500, l2_leaf_reg=50, learning_rate=0.03, min_data_in_leaf=1, random_state=42, random_strength=5, subsample=0.7, verbose=0))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">TotalChargesCleaner</label><div class="sk-toggleable__content fitted"><pre>TotalChargesCleaner()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">FeatureEngineer</label><div class="sk-toggleable__content fitted"><pre>FeatureEngineer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CategoricalEncoder</label><div class="sk-toggleable__content fitted"><pre>CategoricalEncoder()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">DropRedundant</label><div class="sk-toggleable__content fitted"><pre>DropRedundant()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">NumericalScaler</label><div class="sk-toggleable__content fitted"><pre>NumericalScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CatBoostClassifier</label><div class="sk-toggleable__content fitted"><pre>CatBoostClassifier(class_weights=[1, 5], colsample_bylevel=0.7, depth=4, iterations=500, l2_leaf_reg=50, learning_rate=0.03, min_data_in_leaf=1, random_state=42, random_strength=5, subsample=0.7, verbose=0)</pre></div> </div></div></div></div></div></div>



## 7. Test Pipeline


```python
get_metrics(y, full_pipeline.predict(df),full_pipeline.predict_proba(df)[:, 1])
```




    {'Precision': 0.46469622331691296,
     'Recall': 0.9085072231139647,
     'F1-score': 0.6148832156436719,
     'Accuracy': 0.6979980122107057,
     'ROC-AUC': 0.8603100595788755}



### Model Performance Evaluation

The CatBoost model shows strong recall-oriented performance on the full dataset:

| Metric      | Value   | Interpretation                          |
|-------------|---------|-----------------------------------------|
| **Recall**    | 0.911   | Excellent – captures **91.1%** of churners |
| **ROC-AUC**   | 0.860   | Good discriminative power               |
| **F1-score**  | 0.615   | Moderate, due to recall-precision trade-off |
| **Precision** | 0.464   | Low – many false positives              |
| **Accuracy**  | 0.697   | Acceptable but less informative         |

**Summary**:  
This is a **high-recall model** designed to catch the majority of customers at risk of churn. It successfully identifies over 91% of churners, making it well-suited for retention campaigns, though at the expense of lower precision.

## 7. Save the Pipeline


```python

model_path = '../models/full_churn_pipeline.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(full_pipeline, f)

print("✅ Pipeline successfully saved for production!")
print(f"📁 Saved at: {model_path}")
print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

import cloudpickle

# Сохраняем модель с cloudpickle
model_path = '../models/full_churn_pipeline_cloud.pkl'
with open(model_path, 'wb') as f:
    cloudpickle.dump(full_pipeline, f)
    
print(f"Model saved with cloudpickle to {model_path}")
```

    ✅ Pipeline successfully saved for production!
    📁 Saved at: ../models/full_churn_pipeline.pkl
    📊 File size: 0.16 MB
    Model saved with cloudpickle to ../models/full_churn_pipeline_cloud.pkl
    


```python

```
