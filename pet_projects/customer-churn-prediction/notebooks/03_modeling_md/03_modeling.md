## Modeling

### Objective

The goal of this stage is to:

- Train and evaluate machine learning models for churn prediction  
- Establish a baseline model for comparison  
- Handle class imbalance using different strategies  
- Compare multiple models and select the best-performing one  
- Tune hyperparameters to improve model performance  
- Evaluate models using appropriate metrics (precision, recall, F1-score, ROC-AUC)  
- Analyze model performance and interpret key drivers of churn  

## 1. Data Loading & Setup
We begin by importing necessary libraries and loading the dataset.


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
```


```python
with open('../data/processed/train_test_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
```

### Evaluation Metrics

In churn prediction, recall is the most critical metric, as it measures the model's ability to correctly identify customers who are likely to leave.

Missing a churned customer (false negative) results in lost revenue, making recall more important than precision.

Precision and F1-score are also considered to balance the trade-off between correctly identifying churn and avoiding unnecessary retention actions.

ROC-AUC is used as an overall measure of model performance.

## 2. Baseline Models

Establish simple baselines for comparison:

- Dummy Classifier (most frequent class)  

These models provide reference performance and help validate the usefulness of more complex models.



```python
# Train
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

# Predict
y_pred = dummy.predict(X_test)
y_proba = dummy.predict_proba(X_test)[:, 1]

# Metrics
precision = precision_score(y_test, y_pred,zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
accu = accuracy_score(y_test, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accu:.4f}")
```

    Precision: 0.0000
    Recall: 0.0000
    F1-score: 0.0000
    ROC-AUC: 0.5000
    Accuracy: 0.7346
    

The Dummy Classifier achieves an accuracy of 73.46%, which may appear relatively high at first glance. However, this result is misleading due to class imbalance.

The model predicts only the majority class (non-churn), resulting in:

- Recall = 0 → no churned customers are detected  
- Precision = 0 → no positive predictions are made  
- F1-score = 0 → no balance between precision and recall  
- ROC-AUC = 0.5 → performance equivalent to random guessing  

Despite the high accuracy, the model completely fails to identify churned customers, making it ineffective for this task. This highlights the limitation of accuracy as a metric in imbalanced classification problems.

## 3. Evaluation Metrics

Define evaluation metrics:

- Recall (primary metric for churn detection)  
- Precision  
- F1-score  
- ROC-AUC
- Accuracy

Recall is prioritized due to the high cost of missing churned customers.



```python
def get_metrics(y_true, y_pred, y_proba):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, back_metrics=False):
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities (без проверок)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_metrics = get_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = get_metrics(y_test, y_test_pred, y_test_proba)
    

    if back_metrics == True:
        
        return train_metrics,test_metrics

    
    else:

        # Print
        print("=== Train Metrics ===")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        
        print("\n=== Test Metrics ===")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
```


```python
evaluate_model(dummy, X_train, y_train, X_test, y_test)
```

    === Train Metrics ===
    Precision: 0.0000
    Recall: 0.0000
    F1-score: 0.0000
    Accuracy: 0.7346
    ROC-AUC: 0.5000
    
    === Test Metrics ===
    Precision: 0.0000
    Recall: 0.0000
    F1-score: 0.0000
    Accuracy: 0.7346
    ROC-AUC: 0.5000
    

## 4. Handling Class Imbalance

Evaluate different strategies:

- Baseline (no balancing)  
- Class weighting  
- SMOTE (oversampling)  

Compare their impact on model performance.

### 4.1 Baseline


```python
model_baseline = LogisticRegression(max_iter=1000, random_state=42)

model_baseline.fit(X_train, y_train)

print("=== Baseline ===")
evaluate_model(model_baseline, X_train, y_train, X_test, y_test)
```

    === Baseline ===
    === Train Metrics ===
    Precision: 0.6658
    Recall: 0.5452
    F1-score: 0.5995
    Accuracy: 0.8067
    ROC-AUC: 0.8431
    
    === Test Metrics ===
    Precision: 0.6326
    Recall: 0.5294
    F1-score: 0.5764
    Accuracy: 0.7935
    ROC-AUC: 0.8369
    

### 4.2 Class Weight


```python
model_weighted = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

model_weighted.fit(X_train, y_train)

print("\n=== Class Weight ===")
evaluate_model(model_weighted, X_train, y_train, X_test, y_test)
```

    
    === Class Weight ===
    === Train Metrics ===
    Precision: 0.5199
    Recall: 0.7880
    F1-score: 0.6264
    Accuracy: 0.7506
    ROC-AUC: 0.8431
    
    === Test Metrics ===
    Precision: 0.5080
    Recall: 0.7674
    F1-score: 0.6113
    Accuracy: 0.7410
    ROC-AUC: 0.8363
    

### 4.3 SMOTE


```python
smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```


```python
model_smote = LogisticRegression(max_iter=1000, random_state=42)

model_smote.fit(X_train_sm, y_train_sm)

print("\n=== SMOTE ===")
evaluate_model(model_smote, X_train_sm, y_train_sm, X_test, y_test)
```

    
    === SMOTE ===
    === Train Metrics ===
    Precision: 0.7543
    Recall: 0.7758
    F1-score: 0.7649
    Accuracy: 0.7615
    ROC-AUC: 0.8477
    
    === Test Metrics ===
    Precision: 0.5110
    Recall: 0.7460
    F1-score: 0.6065
    Accuracy: 0.7431
    ROC-AUC: 0.8283
    

=== SMOTE ===
=== Train Metrics ===
Precision: 0.7543
Recall: 0.7758
F1-score: 0.7649
Accuracy: 0.7615
ROC-AUC: 0.8477

=== Test Metrics ===
Precision: 0.5110
Recall: 0.7460
F1-score: 0.6065
Accuracy: 0.7431
ROC-AUC: 0.8283

## Class Imbalance Handling – Results & Insights

The comparison of different imbalance handling strategies reveals clear trade-offs between recall, precision, and overall model stability.

### Baseline (no balancing)
- Provides the highest **accuracy (~0.79)** and relatively strong **precision (~0.63)**  
- However, **recall is low (~0.53)**, meaning many churned customers are missed  

👉 The model is conservative and biased toward the majority class.

---

### Class Weight
- **Recall significantly improves (~0.77)** → many more churned customers are detected  
- **Precision drops (~0.51)** → more false positives  
- Accuracy decreases (~0.74), but this is expected  

👉 This approach provides a strong balance between detecting churn and maintaining reasonable precision.

---

### SMOTE
- Achieves high **recall (~0.75)**, similar to class weighting  
- Slightly **lower precision (~0.51)** and **lower ROC-AUC (~0.83)**  
- Shows signs of mild overfitting (train metrics noticeably higher than test)  

👉 SMOTE improves recall but introduces synthetic data, which may add noise and reduce generalization.

---

## Final Conclusion

- **Baseline** is not suitable due to poor recall  
- **Class Weight** offers the best trade-off between recall and model stability  
- **SMOTE** performs similarly in recall but is slightly less stable and more prone to overfitting  

👉 **Class weighting is selected as the preferred strategy**, as it improves churn detection without introducing synthetic noise.

## 5. Model Training

Train a diverse set of models:

### Linear Models
- Logistic Regression  

### Tree-Based Models
- Decision Tree  
- Random Forest  


### Probabilistic Models
- Naive Bayes  

### Distance-Based Models
- K-Nearest Neighbors
  
### Boosting Models
- Gradient Boosting
  
### Advanced Boosting Models 
- XGBoost  
- LightGBM
- Catboost

### 5.1 Linear Models
- Logistic Regression 


```python
log_reg_weighted = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

log_reg_weighted.fit(X_train, y_train)

print("\n=== Logistic Regression (Class Weight) ===")
evaluate_model(log_reg_weighted, X_train, y_train, X_test, y_test)
```

    
    === Logistic Regression (Class Weight) ===
    === Train Metrics ===
    Precision: 0.5199
    Recall: 0.7880
    F1-score: 0.6264
    Accuracy: 0.7506
    ROC-AUC: 0.8431
    
    === Test Metrics ===
    Precision: 0.5080
    Recall: 0.7674
    F1-score: 0.6113
    Accuracy: 0.7410
    ROC-AUC: 0.8363
    

#### Logistic Regression – Conclusion

The class-weighted Logistic Regression significantly improves recall (~0.77), making it effective at identifying churned customers.  

While precision decreases (~0.51), the model achieves a good balance between recall and overall stability.  
Train and test metrics are close, indicating low overfitting.

Overall, this model provides a strong and reliable baseline aligned with the business goal of maximizing churn detection.

### 5.2 Tree-Based Models
- Decision Tree  
- Random Forest  

#### 5.2.1 Decision Tree 


```python
# baseline
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

print("=== Decision Tree (Baseline) ===")
evaluate_model(dt, X_train, y_train, X_test, y_test)
```

    === Decision Tree (Baseline) ===
    === Train Metrics ===
    Precision: 0.9993
    Recall: 0.9926
    F1-score: 0.9960
    Accuracy: 0.9979
    ROC-AUC: 1.0000
    
    === Test Metrics ===
    Precision: 0.4808
    Recall: 0.5027
    F1-score: 0.4915
    Accuracy: 0.7239
    ROC-AUC: 0.6528
    


```python
dt_reg = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

dt_reg.fit(X_train, y_train)

print("\n=== Decision Tree (Regularized) ===")
evaluate_model(dt_reg, X_train, y_train, X_test, y_test)
```

    
    === Decision Tree (Regularized) ===
    === Train Metrics ===
    Precision: 0.5293
    Recall: 0.8033
    F1-score: 0.6382
    Accuracy: 0.7583
    ROC-AUC: 0.8487
    
    === Test Metrics ===
    Precision: 0.5286
    Recall: 0.7914
    F1-score: 0.6338
    Accuracy: 0.7573
    ROC-AUC: 0.8363
    

##### Decision Tree – Conclusion

The baseline Decision Tree severely overfits:

- Near-perfect performance on the training set  
- Significant drop on the test set (Recall ~0.50, ROC-AUC ~0.65)  

This indicates that the model memorizes the training data and fails to generalize.

After applying regularization:

- Test performance improves significantly (Recall ~0.79, ROC-AUC ~0.84)  
- Train and test metrics become consistent  
- The model generalizes much better  

Overall, the regularized Decision Tree performs strongly and achieves high recall, making it suitable for churn detection.

#### 5.2.2 Random Forest 
Random Forest is an ensemble model that combines multiple Decision Trees to improve generalization and reduce overfitting.  
It averages predictions from many trees, making it more stable and robust than a single Decision Tree.

We evaluate both a baseline model and a tuned version.


```python
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

print("=== Random Forest (Baseline) ===")
evaluate_model(rf, X_train, y_train, X_test, y_test)
```

    === Random Forest (Baseline) ===
    === Train Metrics ===
    Precision: 0.9967
    Recall: 0.9953
    F1-score: 0.9960
    Accuracy: 0.9979
    ROC-AUC: 0.9999
    
    === Test Metrics ===
    Precision: 0.5811
    Recall: 0.4599
    F1-score: 0.5134
    Accuracy: 0.7686
    ROC-AUC: 0.8174
    


```python
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

rf_tuned.fit(X_train, y_train)

print("\n=== Random Forest (Tuned) ===")
evaluate_model(rf_tuned, X_train, y_train, X_test, y_test)
```

    
    === Random Forest (Tuned) ===
    === Train Metrics ===
    Precision: 0.5588
    Recall: 0.8268
    F1-score: 0.6668
    Accuracy: 0.7808
    ROC-AUC: 0.8784
    
    === Test Metrics ===
    Precision: 0.5252
    Recall: 0.7807
    F1-score: 0.6280
    Accuracy: 0.7544
    ROC-AUC: 0.8413
    

##### Random Forest – Conclusion

The baseline Random Forest shows clear overfitting:

- Near-perfect performance on the training set  
- Noticeable drop in recall (0.46) and ROC-AUC (0.82) on the test set  

This indicates that even ensemble models can overfit without proper constraints.

After tuning:

- Recall significantly improves (~0.78), making the model effective for churn detection  
- Train and test metrics become much closer, indicating better generalization  
- ROC-AUC improves (~0.84), showing stronger overall discrimination  

Overall, the tuned Random Forest provides a strong balance between recall, precision, and stability, outperforming the baseline and becoming one of the best models so far.

### 5.2 Probabilistic Models
- Naive Bayes


Naive Bayes is a probabilistic model based on Bayes' theorem, assuming independence between features.  
It is simple, fast, and often serves as a strong baseline despite its simplifying assumptions.

We use Gaussian Naive Bayes, which is suitable for continuous features.


```python
nb = GaussianNB()

nb.fit(X_train, y_train)

print("=== Naive Bayes ===")
evaluate_model(nb, X_train, y_train, X_test, y_test)
```

    === Naive Bayes ===
    === Train Metrics ===
    Precision: 0.5111
    Recall: 0.7692
    F1-score: 0.6142
    Accuracy: 0.7435
    ROC-AUC: 0.8176
    
    === Test Metrics ===
    Precision: 0.5108
    Recall: 0.7567
    F1-score: 0.6099
    Accuracy: 0.7431
    ROC-AUC: 0.8045
    

#### Naive Bayes – Conclusion

Naive Bayes demonstrates stable and consistent performance:

- Train and test metrics are very close, indicating no overfitting  
- Recall is relatively high (~0.75), making it effective for churn detection  
- Precision remains moderate (~0.51), similar to other models  

However, overall performance (ROC-AUC ~0.80) is slightly lower compared to more complex models.

In conclusion, Naive Bayes provides a strong and reliable baseline, but is limited by its assumption of feature independence and is generally outperformed by ensemble methods.

### 5.3 Distance-Based Models
- K-Nearest Neighbors


```python
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)
print("=== K-Nearest Neighbors ===")
evaluate_model(knn, X_train, y_train, X_test, y_test)
```

    === K-Nearest Neighbors ===
    === Train Metrics ===
    Precision: 0.7283
    Recall: 0.6274
    F1-score: 0.6741
    Accuracy: 0.8390
    ROC-AUC: 0.8987
    
    === Test Metrics ===
    Precision: 0.5757
    Recall: 0.5187
    F1-score: 0.5457
    Accuracy: 0.7708
    ROC-AUC: 0.7809
    

K-Nearest Neighbors shows high performance on the training set (F1 = 0.674, ROC-AUC = 0.899) but drops on the test set (F1 = 0.546, ROC-AUC = 0.781), indicating overfitting. The model memorizes training data well but generalizes poorly. Improvements could include tuning k, scaling features, or trying other algorithms.

### 5.4 Boosting Models
- Gradient Boosting
- XGBoost  
- LightGBM 

#### 5.4.1 Gradient Boosting


```python
gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

print("=== Gradient Boosting (Baseline) ===")
evaluate_model(gb, X_train, y_train, X_test, y_test)
```

    === Gradient Boosting (Baseline) ===
    === Train Metrics ===
    Precision: 0.7152
    Recall: 0.5525
    F1-score: 0.6234
    Accuracy: 0.8229
    ROC-AUC: 0.8743
    
    === Test Metrics ===
    Precision: 0.6608
    Recall: 0.5053
    F1-score: 0.5727
    Accuracy: 0.7999
    ROC-AUC: 0.8421
    


```python
weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

gb_tuned = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=2,
    random_state=42,
    subsample=1.0,
)


gb_tuned.fit(X_train, y_train,sample_weight=weights)


print("\n=== Gradient Boosting (Tuned) ===")
evaluate_model(gb_tuned, X_train, y_train, X_test, y_test)
```

    
    === Gradient Boosting (Tuned) ===
    === Train Metrics ===
    Precision: 0.5234
    Recall: 0.8227
    F1-score: 0.6398
    Accuracy: 0.7542
    ROC-AUC: 0.8584
    
    === Test Metrics ===
    Precision: 0.5077
    Recall: 0.7968
    F1-score: 0.6202
    Accuracy: 0.7410
    ROC-AUC: 0.8437
    

##### Gradient Boosting – Conclusion

The baseline Gradient Boosting model shows moderate performance:

- Train set: Recall ~0.55, ROC-AUC ~0.87  
- Test set: Recall ~0.51, ROC-AUC ~0.84  

This indicates that the baseline model misses a significant number of positive cases and has room for improvement in detecting the minority class.

After tuning with `sample_weight`, a low learning rate, and more estimators:

- Test recall improves significantly to ~0.80, ROC-AUC ~0.84  
- Train recall increases to 0.82, with a lower precision (0.52), reflecting more false positives  
- F1-score and ROC-AUC indicate better overall balance between classes  

Overall, the tuned Gradient Boosting model effectively prioritizes recall, making it suitable for tasks where detecting positive cases is critical, even at the expense of some precision.

#### 5.4.2 XGBoost  


```python
xgb_baseline = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train baseline
xgb_baseline.fit(X_train, y_train)

print("\n=== XGBoost (Baseline) ===")
evaluate_model(xgb_baseline, X_train, y_train, X_test, y_test)
```

    
    === XGBoost (Baseline) ===
    === Train Metrics ===
    Precision: 0.7050
    Recall: 0.5452
    F1-score: 0.6149
    Accuracy: 0.8188
    ROC-AUC: 0.8688
    
    === Test Metrics ===
    Precision: 0.6678
    Recall: 0.5107
    F1-score: 0.5788
    Accuracy: 0.8027
    ROC-AUC: 0.8425
    

    C:\Users\isazo\anaconda3\Lib\site-packages\xgboost\training.py:200: UserWarning: [03:35:11] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    


```python
weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

xgb_tuned = xgb.XGBClassifier(
    n_estimators=400,                  # more trees for careful learning
    learning_rate=0.03,                # lower learning rate for stable recall
    max_depth=5,                        # deeper trees to capture complex patterns
    subsample=0.8,                      # row sampling to reduce overfitting
    colsample_bytree=0.8,               # feature sampling for diverse trees
    scale_pos_weight=sum(y_train==0)/sum(y_train==1),  # balance classes
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train tuned model
xgb_tuned.fit(X_train, y_train, sample_weight=weights)

print("\n=== XGBoost (Tuned) ===")
evaluate_model(xgb_tuned, X_train, y_train, X_test, y_test)
```

    C:\Users\isazo\anaconda3\Lib\site-packages\xgboost\training.py:200: UserWarning: [03:35:11] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    

    
    === XGBoost (Tuned) ===
    === Train Metrics ===
    Precision: 0.5048
    Recall: 0.9846
    F1-score: 0.6674
    Accuracy: 0.7396
    ROC-AUC: 0.9147
    
    === Test Metrics ===
    Precision: 0.4611
    Recall: 0.8877
    F1-score: 0.6069
    Accuracy: 0.6948
    ROC-AUC: 0.8404
    

##### XGBoost – Conclusion

The baseline XGBoost model shows moderate performance:

- Train set: Recall ~0.55, ROC-AUC ~0.87  
- Test set: Recall ~0.51, ROC-AUC ~0.84  

This indicates that the baseline model misses nearly half of the positive cases and does not prioritize the minority class.

After tuning with `sample_weight`, `scale_pos_weight`, a low learning rate, and more trees:

- Train recall increases dramatically to ~0.98, with precision dropping to ~0.50  
- Test recall improves significantly to ~0.89, F1-score ~0.61, ROC-AUC ~0.84  
- Precision decreases (~0.46), reflecting more false positives, but this is acceptable since the focus is on maximizing recall

Overall, the tuned XGBoost model is highly effective at detecting positive cases, making it suitable for tasks where **recall is critical**, even at the cost of precision.

#### 5.4.3 LightGBM  


```python
lgb_baseline = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42,
    verbose=-1
)

# Train baseline
lgb_baseline.fit(X_train, y_train)

print("\n=== LightGBM (Baseline) ===")
evaluate_model(lgb_baseline, X_train, y_train, X_test, y_test)
```

    
    === LightGBM (Baseline) ===
    === Train Metrics ===
    Precision: 0.7897
    Recall: 0.6656
    F1-score: 0.7223
    Accuracy: 0.8642
    ROC-AUC: 0.9378
    
    === Test Metrics ===
    Precision: 0.6265
    Recall: 0.5428
    F1-score: 0.5817
    Accuracy: 0.7928
    ROC-AUC: 0.8370
    


```python
weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

lgb_tuned = lgb.LGBMClassifier(
    n_estimators=400,                # more trees for stable learning
    learning_rate=0.03,              # lower learning rate for stable recall
    max_depth=5,                     # deeper trees for complex patterns
    subsample=0.8,                   # row sampling to reduce overfitting
    colsample_bytree=0.8,            # feature sampling for diverse trees
    class_weight='balanced',         # directly balances the classes
    random_state=42,
    verbose=-1
)

# Train tuned model
lgb_tuned.fit(X_train, y_train, sample_weight=weights)

print("\n=== LightGBM (Tuned) ===")
evaluate_model(lgb_tuned, X_train, y_train, X_test, y_test)
```

    
    === LightGBM (Tuned) ===
    === Train Metrics ===
    Precision: 0.4943
    Recall: 0.9846
    F1-score: 0.6582
    Accuracy: 0.7286
    ROC-AUC: 0.9111
    
    === Test Metrics ===
    Precision: 0.4512
    Recall: 0.8904
    F1-score: 0.5989
    Accuracy: 0.6835
    ROC-AUC: 0.8363
    

##### LightGBM – Conclusion

The baseline LightGBM shows solid performance:

- Training metrics are high (Recall ~0.67, ROC-AUC ~0.94)  
- Test metrics drop noticeably (Recall ~0.54, ROC-AUC ~0.84)  

This indicates moderate overfitting, with the model failing to fully generalize to unseen data.

After tuning for higher recall:

- Recall increases substantially on the test set (~0.89), achieving the goal of capturing more positive cases  
- Precision decreases (~0.45), as expected when prioritizing recall  
- F1-score improves slightly (~0.60)  
- Training and test ROC-AUC remain strong (~0.91 train, ~0.84 test)  

Overall, the tuned LightGBM effectively maximizes recall for the minority class, making it suitable for applications where catching positives is critical, while maintaining reasonable overall performance.

#### 5.4.4 Catboost  


```python
# Baseline CatBoost
cat_base = CatBoostClassifier(
    iterations=500,         # number of trees
    learning_rate=0.1,      # standard LR
    depth=6,                # default tree depth
    loss_function='Logloss',
    verbose=0,              # suppress warnings
    random_state=42
)

# Train baseline
cat_base.fit(X_train, y_train, sample_weight=weights)

print("\n=== CatBoost (Baseline) ===")
evaluate_model(cat_base, X_train, y_train, X_test, y_test,)

```

    
    === CatBoost (Baseline) ===
    === Train Metrics ===
    Precision: 0.7337
    Recall: 0.9732
    F1-score: 0.8367
    Accuracy: 0.8992
    ROC-AUC: 0.9709
    
    === Test Metrics ===
    Precision: 0.5269
    Recall: 0.6818
    F1-score: 0.5944
    Accuracy: 0.7530
    ROC-AUC: 0.8260
    


```python
%%time
# Tuned CatBoost for high recall
cat_tuned = CatBoostClassifier(
    iterations=2000,           # more trees for stable learning
    learning_rate=0.01,        # lower learning rate for smoother learning
    depth=3,                   # tree depth
    l2_leaf_reg=100,             # L2 regularization to reduce overfitting
    subsample=0.8,             # row sampling to reduce overfitting
    colsample_bylevel=0.8,     # feature sampling for diverse trees
    loss_function='Logloss',
    random_state=42,
    class_weights=[1, 2],      # give more weight to positive class (churn)
    grow_policy='Lossguide',    # alternative tree growth policy
    verbose=0
)

# Train tuned model 
cat_tuned.fit(
    X_train, y_train,
    sample_weight=weights,
   
)

# Evaluate metrics
print("\n=== CatBoost (Tuned High Recall) ===")
evaluate_model(cat_tuned, X_train, y_train, X_test, y_test)
```

    
    === CatBoost (Tuned High Recall) ===
    === Train Metrics ===
    Precision: 0.4632
    Recall: 0.9164
    F1-score: 0.6153
    Accuracy: 0.6960
    ROC-AUC: 0.8666
    
    === Test Metrics ===
    Precision: 0.4489
    Recall: 0.8930
    F1-score: 0.5975
    Accuracy: 0.6806
    ROC-AUC: 0.8435
    CPU times: total: 56.8 s
    Wall time: 5.06 s
    

##### CatBoost – Conclusion

**Baseline:**
- Overfits training data (Recall 0.973, ROC-AUC 0.971)  
- Test performance drops (Recall 0.682, F1 0.594, ROC-AUC 0.826)  

**Tuned (High Recall):**
- High test recall (0.893) for churn detection  
- Precision drops (0.449) — expected trade-off  
- F1 and ROC-AUC improve slightly (F1 0.598, ROC-AUC 0.844)  
- Regularization and lower depth reduce overfitting  

**Insight:**  
Tuned CatBoost is strong for catching churn, prioritizing recall over precision.

## 6. Model Evaluation

Evaluate each model using selected metrics.

Analyze:

- Recall (churn detection ability)  
- Precision (cost control)  
- F1-score (balance)  
- ROC-AUC (overall performance)  



```python
import pandas as pd

# List all your trained models with display names
models = [
    ("Logistic Regression", log_reg_weighted),
    ("Decision Tree", dt_reg),
    ("Random Forest", rf_tuned),
    ("Naive Bayes", nb),
    ("K-Nearest Neighbors", knn),
    ("Gradient Boosting", gb_tuned),
    ("XGBoost", xgb_tuned),
    ("LightGBM", lgb_tuned),
    ("CatBoost", cat_tuned)
]

# Collect test metrics automatically
summary = []

for name, model in models:
    # Evaluate model and get metrics back
    _, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test, back_metrics=True)
    
    summary.append({
        "Model": name,
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "F1-score": test_metrics["F1-score"],
        "Accuracy": test_metrics["Accuracy"],
        "ROC-AUC": test_metrics["ROC-AUC"]
    })

df_summary = pd.DataFrame(summary)

# Generate Markdown table for reporting
markdown = "### Model Comparison (Test Metrics)\n\n"
markdown += "| Model | Recall | Precision | F1-score | Accuracy | ROC-AUC |\n"
markdown += "|-------|--------|-----------|----------|---------|---------|\n"

for idx, row in df_summary.iterrows():
    markdown += f"| {row['Model']} | {row['Recall']:.4f} | {row['Precision']:.4f} | {row['F1-score']:.4f} | {row['Accuracy']:.4f} | {row['ROC-AUC']:.4f} |\n"

print(markdown)
```

    ### Model Comparison (Test Metrics)
    
    | Model | Recall | Precision | F1-score | Accuracy | ROC-AUC |
    |-------|--------|-----------|----------|---------|---------|
    | Logistic Regression | 0.7674 | 0.5080 | 0.6113 | 0.7410 | 0.8363 |
    | Decision Tree | 0.7914 | 0.5286 | 0.6338 | 0.7573 | 0.8363 |
    | Random Forest | 0.7807 | 0.5252 | 0.6280 | 0.7544 | 0.8413 |
    | Naive Bayes | 0.7567 | 0.5108 | 0.6099 | 0.7431 | 0.8045 |
    | K-Nearest Neighbors | 0.5187 | 0.5757 | 0.5457 | 0.7708 | 0.7809 |
    | Gradient Boosting | 0.7968 | 0.5077 | 0.6202 | 0.7410 | 0.8437 |
    | XGBoost | 0.8877 | 0.4611 | 0.6069 | 0.6948 | 0.8404 |
    | LightGBM | 0.8904 | 0.4512 | 0.5989 | 0.6835 | 0.8363 |
    | CatBoost | 0.8930 | 0.4489 | 0.5975 | 0.6806 | 0.8435 |
    
    

1. **Recall (Churn Detection):**
   - Boosting models (XGBoost, LightGBM, CatBoost) have the highest recall (>0.88), meaning they detect most churn cases.
   - KNN has the lowest recall (0.518), missing many churn cases.

2. **Precision (Cost Control):**
   - Most tree-based models sacrifice precision for recall. CatBoost and LightGBM have very low precision (~0.45), meaning more false positives.
   - KNN has the highest precision (0.576) but very low recall.

3. **F1-score (Balance):**
   - Gradient Boosting, Random Forest, and Decision Tree have slightly higher F1-scores (~0.63), balancing recall and precision better than advanced boosting models.
   - CatBoost, LightGBM, and XGBoost have mid F1 (~0.60), emphasizing recall over precision.

4. **Accuracy:**
   - Logistic Regression, Decision Tree, and Random Forest achieve higher accuracy (~0.74–0.76) due to balanced performance.
   - Boosting models tuned for recall drop in accuracy (~0.68–0.69).

5. **ROC-AUC (Overall Performance):**
   - XGBoost, CatBoost, Gradient Boosting, and Random Forest perform very well (~0.84), showing strong discriminative ability.
   - KNN and Naive Bayes lag behind in ROC-AUC.

### Conclusion

- **Best for detecting churn (high recall):** CatBoost, LightGBM, XGBoost  
- **Best balanced models (F1 & ROC-AUC):** Random Forest, Gradient Boosting  
- **Cost-sensitive approach (precision focus):** KNN or Logistic Regression if reducing false positives is critical  
- **Overall strategy:** Use boosting models for recall-heavy tasks, possibly combine ensemble to improve precision.

## 7. Hyperparameter Tuning

Optimize the catboost model using:

- GridSearchCV 

Focus on improving recall and overall model performance.


```python
%%time
# base model
cat = CatBoostClassifier(
    loss_function='Logloss',
    verbose=0,
    random_state=42
)

# stratified CV
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# param grid
param_grid = {
    # core model complexity
    'depth': [4, 6, 8],

    # learning dynamics
    'learning_rate': [0.03, 0.05, 0.1],

    # number of trees
    'iterations': [500, 1000],

    # regularization
    'l2_leaf_reg': [3, 10, 50],

    # randomness (оставляем самое важное)
    'subsample': [0.7, 1.0],
    'colsample_bylevel': [0.7, 1.0],

    # ключевая фишка CatBoost
    'random_strength': [0, 5],

    # контроль переобучения
    'min_data_in_leaf': [1, 10]
}

# different weight configs to test
weights_list = [
    {0: 1, 1: 2},
    {0: 1, 1: 3},
    {0: 1, 1: 5}
]

best_result = None

for w in weights_list:
    print(f"\nTesting weights: {w}")
    
    # create sample weights
    sample_weights = y_train.map(w)
    
    # GridSearch
    grid_search = GridSearchCV(
        estimator=cat,
        param_grid=param_grid,
        scoring='recall',
        cv=skf,
        n_jobs=-1,
        verbose=2
    )
    
    # train
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("Best recall:", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)
    
    # save best result
    if best_result is None or grid_search.best_score_ > best_result['score']:
        best_result = {
            'weights': w,
            'score': grid_search.best_score_,
            'params': grid_search.best_params_
        }

print("\nFINAL BEST RESULT:")
print(best_result)
```

    
    Testing weights: {0: 1, 1: 2}
    Fitting 3 folds for each of 864 candidates, totalling 2592 fits
    Best recall: 0.7297634090134754
    Best params: {'colsample_bylevel': 0.7, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 50, 'learning_rate': 0.03, 'min_data_in_leaf': 1, 'random_strength': 5, 'subsample': 0.7}
    
    Testing weights: {0: 1, 1: 3}
    Fitting 3 folds for each of 864 candidates, totalling 2592 fits
    Best recall: 0.8247405118134529
    Best params: {'colsample_bylevel': 0.7, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 50, 'learning_rate': 0.03, 'min_data_in_leaf': 1, 'random_strength': 5, 'subsample': 1.0}
    
    Testing weights: {0: 1, 1: 5}
    Fitting 3 folds for each of 864 candidates, totalling 2592 fits
    Best recall: 0.8936333175051575
    Best params: {'colsample_bylevel': 0.7, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 50, 'learning_rate': 0.03, 'min_data_in_leaf': 1, 'random_strength': 5, 'subsample': 0.7}
    
    FINAL BEST RESULT:
    {'weights': {0: 1, 1: 5}, 'score': 0.8936333175051575, 'params': {'colsample_bylevel': 0.7, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 50, 'learning_rate': 0.03, 'min_data_in_leaf': 1, 'random_strength': 5, 'subsample': 0.7}}
    CPU times: total: 2min 35s
    Wall time: 1h 56min 49s
    


```python
# Create tuned CatBoost model with best parameters
best_params = grid_search.best_params_
cat_tuned_best = CatBoostClassifier(
    iterations=best_result['params']['iterations'],
    learning_rate=best_result['params']['learning_rate'],
    depth=best_result['params']['depth'],
    l2_leaf_reg=best_result['params']['l2_leaf_reg'],
    subsample=best_result['params']['subsample'],
    colsample_bylevel=best_result['params']['colsample_bylevel'],
    loss_function='Logloss',
    random_state=42,
    class_weights=best_result['weights'],      # more weight for churn
    grow_policy='Lossguide',
    verbose=0
)
# Train tuned model 
cat_tuned_best.fit(
    X_train, y_train,
    sample_weight=weights,
   
)
```




    CatBoostClassifier(class_weights={0: 1, 1: 5}, colsample_bylevel=0.7, depth=4, grow_policy='Lossguide', iterations=500, l2_leaf_reg=50, learning_rate=0.03, loss_function='Logloss', random_state=42, subsample=0.7, verbose=0)




```python
# Evaluate metrics
print("\n=== CatBoost (Tuned High Recall with GridSearchCV) ===")
evaluate_model(cat_tuned_best, X_train, y_train, X_test, y_test)
```

    
    === CatBoost (Tuned High Recall with GridSearchCV) ===
    === Train Metrics ===
    Precision: 0.3950
    Recall: 0.9819
    F1-score: 0.5634
    Accuracy: 0.5962
    ROC-AUC: 0.8702
    
    === Test Metrics ===
    Precision: 0.3887
    Recall: 0.9572
    F1-score: 0.5529
    Accuracy: 0.5891
    ROC-AUC: 0.8435
    


```python
thresholds = np.arange(0.1, 0.9, 0.05)
y_proba = cat_tuned_best.predict_proba(X_test)[:, 1]
for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    
    print(f"Threshold: {t:.2f}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print()
```

    Threshold: 0.10
    Precision: 0.2918622848200313
    Recall: 0.9973262032085561
    
    Threshold: 0.15
    Precision: 0.31083333333333335
    Recall: 0.9973262032085561
    
    Threshold: 0.20
    Precision: 0.32006920415224915
    Recall: 0.9893048128342246
    
    Threshold: 0.25
    Precision: 0.3297587131367292
    Recall: 0.9866310160427807
    
    Threshold: 0.30
    Precision: 0.33854645814167433
    Recall: 0.983957219251337
    
    Threshold: 0.35
    Precision: 0.3495238095238095
    Recall: 0.9812834224598931
    
    Threshold: 0.40
    Precision: 0.36381709741550694
    Recall: 0.9786096256684492
    
    Threshold: 0.45
    Precision: 0.37409326424870465
    Recall: 0.9652406417112299
    
    Threshold: 0.50
    Precision: 0.38870792616720956
    Recall: 0.9572192513368984
    
    Threshold: 0.55
    Precision: 0.40293453724604966
    Recall: 0.9545454545454546
    
    Threshold: 0.60
    Precision: 0.4132231404958678
    Recall: 0.9358288770053476
    
    Threshold: 0.65
    Precision: 0.43196004993757803
    Recall: 0.9251336898395722
    
    Threshold: 0.70
    Precision: 0.4499332443257677
    Recall: 0.9010695187165776
    
    Threshold: 0.75
    Precision: 0.4693295292439372
    Recall: 0.8796791443850267
    
    Threshold: 0.80
    Precision: 0.492845786963434
    Recall: 0.8288770053475936
    
    Threshold: 0.85
    Precision: 0.5253623188405797
    Recall: 0.7754010695187166
    
    

#### CatBoost – Final Results & Insights

##### Hyperparameter Tuning

GridSearchCV was used to optimize CatBoost with a focus on maximizing recall.

- Increasing class weight for the minority class significantly improved recall  
- The best configuration used weights `{0: 1, 1: 5}`, strongly prioritizing churn detection  
- Optimal parameters indicate a **simple but highly regularized model**:
  - shallow trees (`depth=4`)
  - strong regularization (`l2_leaf_reg=50`)
  - low learning rate (`learning_rate=0.03`)
  - added randomness (`random_strength=5`)

This suggests that **controlled, regularized models generalize better than overly complex ones**.

---

### Model Performance

- **Recall (Test): ~0.96** → excellent detection of churned customers  
- **Precision: ~0.39** → relatively high number of false positives  
- **ROC-AUC: ~0.84** → strong ability to rank customers by churn risk  
- **Accuracy: ~0.59** → lower due to focus on minority class  

Train and test metrics are close, indicating **good generalization and no overfitting**.

👉 The model is intentionally optimized for recall, aligning with the business objective.

---

### Threshold Analysis

Adjusting the decision threshold reveals a clear precision–recall trade-off:

- **Low thresholds (0.1–0.3):**
  - Recall ≈ 0.99  
  - Precision ≈ 0.29–0.33  
  - Almost all churn cases are detected, but with many false positives  

- **Medium thresholds (0.4–0.6):**
  - Balanced trade-off  
  - Recall ≈ 0.93–0.98  
  - Precision ≈ 0.36–0.41  

- **High thresholds (0.7–0.85):**
  - Precision improves (≈ 0.45–0.52)  
  - Recall decreases (≈ 0.77–0.90)  

---

### Final Threshold Selection

A threshold of **0.5** is selected as the final decision boundary.

At this threshold:

- Precision: ~0.39  
- Recall: ~0.96  

This provides a strong balance between:

- Detecting the majority of churned customers  
- Keeping the number of false positives at a manageable level  

Lower thresholds lead to excessive false positives, while higher thresholds risk missing too many churn cases.

👉 Therefore, **threshold = 0.5** is chosen as a practical and balanced solution.

---

### Final Conclusion

- The model achieves **very high recall**, making it highly effective for churn detection  
- Precision is lower, but acceptable given the business goal  
- Threshold tuning allows flexible adjustment depending on strategy  

👉 Overall, CatBoost is the **best-performing model**, providing strong recall, stable generalization, and controllable decision behavior.

## 9. Conclusion

### What Was Done

1. **Data Loading** — train/test split from preprocessed data
2. **Baseline** — Dummy Classifier (Recall=0, ROC-AUC=0.5) — proved accuracy is useless with imbalance
3. **Metrics** — priority: Recall > F1 > ROC-AUC > Precision
4. **Imbalance Handling** — Baseline vs Class Weight vs SMOTE. **Class Weight** won (Recall 0.77 vs 0.53, more stable than SMOTE)
5. **Model Training** — 9 models from Logistic Regression to boosting (XGBoost, LightGBM, CatBoost)
6. **Comparison** — boosting models achieved Recall >0.88, others ~0.76-0.79
7. **CatBoost Tuning** — GridSearchCV optimizing recall, best weights {0:1, 1:5}, params: depth=4, lr=0.03, l2=50
8. **Threshold Analysis** — selected threshold = 0.5

---

### Final Metrics (CatBoost)

| Metric | Value |
|--------|-------|
| **Recall** | **0.957** (96% of churners detected) |
| Precision | 0.389 |
| F1-score | 0.553 |
| ROC-AUC | 0.844 |

---

### Best Model: **CatBoost**

**Why:** highest recall, no overfitting, flexible threshold tuning.

---

### Key Churn Drivers
- Tenure (new customers)
- Month-to-month contract
- Electronic check payment
- High monthly charges
- Missing add-on services

---

### Business Recommendations
- Focus retention on new customers (first 6-12 months)
- Incentivize annual contracts
- Promote autopay
- Proactively engage high-probability customers

---

### Verdict
✅ Model ready for deployment.


```python

```
