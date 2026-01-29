# Assignment 3

In this assignment you will explore text message data and create models to predict if a message is spam or not. 


```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('assets/spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FreeMsg Hey there darling it's been 3 week's n...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Even my brother is not like to speak with me. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>As per your request 'Melle Melle (Oru Minnamin...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WINNER!! As a valued network customer you have...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Had your mobile 11 months or more? U R entitle...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
```

### Question 1
What percentage of the documents in `spam_data` are spam?

*This function should return a float, the percent value (i.e. $ratio * 100$).*


```python
def answer_one():
    
    return (spam_data['target']==1).mean()*100
answer_one()
```




    13.406317300789663




```python

```

### Question 2

Fit the training data `X_train` using a Count Vectorizer with default parameters.

What is the longest token in the vocabulary?

*This function should return a string.*


```python
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    
    
    return max(vectorizer.get_feature_names_out(), key=len)
answer_two()
```




    'com1win150ppmx3age16subscription'




```python

```

### Question 3

Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.

Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train_trans, X_test_trans = vectorizer.transform(X_train), vectorizer.transform(X_test)
    
    model = MultinomialNB(alpha=0.1)
    
    model.fit(X_train_trans, y_train)
    
    return roc_auc_score(y_test, model.predict_proba(X_test_trans)[:, 1])
    
    

answer_three()
```




    0.991545422134696




```python

```

### Question 4

Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters. The transformed data will be a compressed sparse row matrix where the number of rows is the number of documents in `X_train`, the number of columns is the number of features found by the vectorizer in each document, and each value in the sparse matrix is the tf-idf value. First find the **max** tf-idf value for every feature.

What 20 features have the smallest tf-idf and what 20 have the largest tf-idf among the **max** tf-idf values?

Put these features in two series where each series is sorted by tf-idf value. The index of the series should be the feature name, and the data should be the tf-idf.

The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. Any entries with identical tf-ids should appear in lexigraphically increasing order by their feature name in boh series. For example, if the features "a", "b", "c" had the tf-idfs 1.0, 0.5, 1.0 in the series with the largest tf-idfs, then they should occur in the returned result in the order "a", "c", "b" with values 1.0, 1.0, 0.5.

*This function should return a tuple of two series
`(smallest tf-idfs series, largest tf-idfs series)`.*


```python
from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():

    vectorizer = TfidfVectorizer()
    
    X_train_trans = vectorizer.fit_transform(X_train)
    
    feature_names = np.array(vectorizer.get_feature_names_out())

    # max tf-idf value for each feature (column-wise max)
    max_tfidf = X_train_trans.max(axis=0).toarray().ravel()

    tfidf_series = pd.Series(max_tfidf, index=feature_names)

   
    smallest = (
        tfidf_series
        .sort_values(kind="mergesort")
        .head(20)
    )

    largest = (
        tfidf_series
        .sort_values(ascending=False, kind="mergesort")
        .head(20)
    )

    return smallest, largest
    
    

answer_four()
```




    (aaniye          0.074475
     athletic        0.074475
     chef            0.074475
     companion       0.074475
     courageous      0.074475
     dependable      0.074475
     determined      0.074475
     exterminator    0.074475
     healer          0.074475
     listener        0.074475
     organizer       0.074475
     pest            0.074475
     psychiatrist    0.074475
     psychologist    0.074475
     pudunga         0.074475
     stylist         0.074475
     sympathetic     0.074475
     venaam          0.074475
     afternoons      0.091250
     approaching     0.091250
     dtype: float64,
     146tf150p    1.000000
     645          1.000000
     anything     1.000000
     anytime      1.000000
     beerage      1.000000
     done         1.000000
     er           1.000000
     havent       1.000000
     home         1.000000
     lei          1.000000
     nite         1.000000
     ok           1.000000
     okie         1.000000
     thank        1.000000
     thanx        1.000000
     too          1.000000
     where        1.000000
     yup          1.000000
     tick         0.980166
     blank        0.932702
     dtype: float64)




```python

```

### Question 5

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.

Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
def answer_five():

    vectorizer = TfidfVectorizer(min_df=3)
    vectorizer.fit(X_train)
    X_train_trans, X_test_trans = vectorizer.transform(X_train), vectorizer.transform(X_test)
    
    model = MultinomialNB(alpha=0.1)
    
    model.fit(X_train_trans, y_train)
    
    return roc_auc_score(y_test, model.predict_proba(X_test_trans)[:, 1])
    

answer_five()
```




    0.9954968337775665




```python

```

### Question 6

What is the average length of documents (number of characters) for not spam and spam documents?

*This function should return a tuple (average length not spam, average length spam).*


```python
def answer_six():

    length = spam_data['text'].str.len()
    return length[spam_data['target']==0].mean(), length[spam_data['target']==1].mean()

answer_six()
```




    (71.02362694300518, 138.8661311914324)




```python

```

<br>
<br>
The following function has been provided to help you combine new features into the training data:


```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
```

### Question 7

Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.

Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.

*Hint: Since probability is set to false, use the model's `decision_function` on the test data when calculating the target scores to use in roc_auc_score*

*This function should return the AUC score as a float.*


```python
from sklearn.svm import SVC

def answer_seven():

    vectorizer = TfidfVectorizer(min_df=5)
    vectorizer.fit(X_train)
    X_train_trans, X_test_trans = vectorizer.transform(X_train), vectorizer.transform(X_test)
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train.str.len()), add_feature(X_test_trans, X_test.str.len())
    
    model = SVC(C=10000)
    
    model.fit(X_train_trans, y_train)

    return roc_auc_score(y_test, model.decision_function(X_test_trans))
    

answer_seven()
```




    0.9963202213809143




```python

```

### Question 8

What is the average number of digits per document for not spam and spam documents?

*Hint: Use `\d` for digit class*

*This function should return a tuple (average # digits not spam, average # digits spam).*


```python
def answer_eight():

    dig = spam_data['text'].str.findall('\d').agg(len)
    return dig[spam_data['target']==0].mean(), dig[spam_data['target']==1].mean()

answer_eight()
```




    (0.2992746113989637, 15.759036144578314)




```python

```

### Question 9

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* **number of digits per document**

fit a Logistic Regression model with regularization `C=100` and `max_iter=1000`. Then compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.linear_model import LogisticRegression

def answer_nine():

    vectorizer = TfidfVectorizer(min_df=5,ngram_range=(1,3))
    vectorizer.fit(X_train)
    X_train_trans, X_test_trans = vectorizer.transform(X_train), vectorizer.transform(X_test)
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train.str.len()), add_feature(X_test_trans, X_test.str.len())
    
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train.str.findall('\d').agg(len)), add_feature(X_test_trans, X_test.str.findall('\d').agg(len))
    
    model = LogisticRegression(C=100, max_iter=1000)
    
    model.fit(X_train_trans, y_train)

    return roc_auc_score(y_test, model.predict_proba(X_test_trans)[:, 1])
    

answer_nine()
```




    0.9972921582941445




```python

```

### Question 10

What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?

*Hint: Use `\w` and `\W` character classes*

*This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*


```python
def answer_ten():

    dig = spam_data['text'].str.findall('\W').agg(len)
    return dig[spam_data['target']==0].mean(), dig[spam_data['target']==1].mean()

answer_ten()
```




    (17.29181347150259, 29.041499330655956)




```python

```

### Question 11

Fit and transform the **first 2000 rows** of training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**

To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* number of digits per document
* **number of non-word characters (anything other than a letter, digit or underscore.)**

fit a Logistic Regression model with regularization C=100 and max_iter=1000. Then compute the area under the curve (AUC) score using the transformed test data.

Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.

The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
['length_of_doc', 'digit_count', 'non_word_char_count']

*This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*


```python
def answer_eleven():
    
    vectorizer = TfidfVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb')
    vectorizer.fit(X_train[:2000])
    X_train_trans, X_test_trans = vectorizer.transform(X_train[:2000]), vectorizer.transform(X_test)
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train[:2000].str.len()), add_feature(X_test_trans, X_test.str.len())
    
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train[:2000].str.findall('\d').agg(len)), add_feature(X_test_trans, X_test.str.findall('\d').agg(len))
    
    X_train_trans, X_test_trans = add_feature(X_train_trans, X_train[:2000].str.findall('\W').agg(len)), add_feature(X_test_trans, X_test.str.findall('\W').agg(len))
    
    
    model = LogisticRegression(C=100, max_iter=1000)
    
    model.fit(X_train_trans, y_train[:2000])

     # --- 6. AUC ---
    y_prob = model.predict_proba(X_test_trans)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # --- 7. Coefficients ---
    feature_names = (
        list(vectorizer.get_feature_names_out()) +
        ['length_of_doc', 'digit_count', 'non_word_char_count']
    )

    coefs = pd.Series(model.coef_[0], index=feature_names)

    smallest = coefs.sort_values().head(10)
    largest = coefs.sort_values(ascending=False).head(10)

    return auc, list(smallest.items()), list(largest.items())
    
answer_eleven()
```




    (0.997945775257627,
     [('n ', -2.8126539595609343),
      ('..', -2.33764721772591),
      (' i', -2.2281492460796133),
      ('at', -2.106013467415433),
      ('i ', -2.0836482113233945),
      ('he', -2.040737642577471),
      ('if ', -2.0223428472561027),
      (' m', -1.9670827202950725),
      ('if', -1.8496531914968286),
      (' lo', -1.8017249954089538)],
     [('xt', 3.1147457214053706),
      (' st', 3.037886065878604),
      ('xt ', 3.0374294479733166),
      ('co', 2.9515662484053924),
      ('ww', 2.8938307617182093),
      ('eply ', 2.659575114832077),
      ('der', 2.5924340740059524),
      ('s ', 2.5838524259724633),
      ('ne', 2.5437910646398167),
      ('ply ', 2.420118212425012)])




```python

```
