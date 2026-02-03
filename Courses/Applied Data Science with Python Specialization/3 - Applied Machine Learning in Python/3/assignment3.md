---

_You are currently looking at **version 0.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the Jupyter Notebook FAQ course resource._

---


```python
import numpy as np
import pandas as pd
```

### Question 1
Import the data from `assets/fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?

*This function should return a float between 0 and 1.* 


```python
def answer_one():
    # YOUR CODE HERE
    df = pd.read_csv('assets/fraud_data.csv')
    return df['Class'].mean()
    raise NotImplementedError()

```


```python

```


```python
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('assets/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

### Question 2

Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?

*This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*


```python
def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train, y_train)
    
    return (accuracy_score(y_test, model.predict(X_test)), recall_score(y_test, model.predict(X_test)))

    # YOUR CODE HERE
    raise NotImplementedError()
```


```python

```

### Question 3

Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?

*This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*


```python
def answer_three():
    from sklearn.metrics import recall_score, precision_score,accuracy_score
    from sklearn.svm import SVC
    
    model = SVC()
    model.fit(X_train, y_train)
    
    return (accuracy_score(y_test, model.predict(X_test)), recall_score(y_test, model.predict(X_test)), precision_score(y_test, model.predict(X_test)))

    
    # YOUR CODE HERE
    raise NotImplementedError()
```


```python

```

### Question 4

Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.

*This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*


```python
def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    model = SVC(C= 1e9,gamma= 1e-07)
    model.fit(X_train, y_train)
        # Get decision scores
    decision_scores = model.decision_function(X_test)

    # Apply custom threshold
    y_pred = (decision_scores >= -220).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # YOUR CODE HERE
    return cm
    raise NotImplementedError()
```


```python

```

### Question 5

Train a logisitic regression classifier with default parameters using X_train and y_train. This classifier should use the parameter solver='liblinear'.

For the logisitic regression classifier, compute the scores using decision_function() or with predict_proba(), then create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).

Looking at the precision recall curve, what is the recall when the precision is `0.75`?

Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?

Note: When getting the ROC curve and finding the records where the FPR entry is closest to 0.16, take the corresponding TPRs. As there are two such records where the FPR is close to 0.16, take the higher TPR of these two records.

*This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*


```python
def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    import numpy as np

    lr = LogisticRegression(solver='liblinear', random_state=0)
    lr.fit(X_train, y_train)

    y_scores = lr.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    recall_at_075 = recall[(np.abs(precision - 0.75)).argmin()]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    diff = np.abs(fpr - 0.16)
    tpr_at_016 = tpr[np.where(diff == diff.min())].max()

    return (recall_at_075, tpr_at_016)
```


```python

```

### Question 6

Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation. (Suggest to use `solver='liblinear'`, more explanation [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))

`'penalty': ['l1', 'l2']`

`'C':[0.01, 0.1, 1, 10]`

From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.

|      	| `l1` 	| `l2` 	|
|:----:	|----	|----	|
| **`0.01`** 	|    ?	|   ? 	|
| **`0.1`**  	|    ?	|   ? 	|
| **`1`**    	|    ?	|   ? 	|
| **`10`**   	|    ?	|   ? 	|

<br>

*This function should return a 4 by 2 numpy array with 8 floats.* 

*Note: do not return a DataFrame, just the values denoted by `?` in a numpy array.*


```python
def answer_six():
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    }

    lr = LogisticRegression(solver='liblinear')

    grid = GridSearchCV(
        lr,
        param_grid=param_grid,
        scoring='recall',
        cv=3
    )

    grid.fit(X_train, y_train)

    # mean test scores in the order GridSearchCV evaluates params
    mean_scores = grid.cv_results_['mean_test_score']

    # reshape to (4 C values, 2 penalties)
    return mean_scores.reshape(4, 2)

```


```python

```


```python
# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    %matplotlib notebook
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())
```


```python

```


```python

```
