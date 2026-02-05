# Assignment 3: Dealing with Vector and Matrix Real-World Data (Part IV)

## Patterns in Matrix Data
In the last part of the assignment, let's explore the patterns in a matrix as a whole instead of in individual vectors. As discussed in class, one type of such patterns are *eigenvectors*, which can be obtained through *Singular Value Decomposition* (SVD):

$$X=U\Sigma V^T.$$

$X$ is an $n \times p$ matrix, $U\cdot U^T = I$, $V\cdot V^T = I$, and $\Sigma$ is an $n\times p$ diagonal matrix with non-zero singular values.

Let's first walk through the example in the lecture.


```python
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
```


```python
doc_word_df = pd.DataFrame([[1, 1, 1, 0, 0], 
                            [2, 2, 2, 0, 0],
                            [1, 1, 1, 0, 0],
                            [5, 5, 5, 0, 0],
                            [0, 0, 0, 2, 2],
                            [0, 0, 0, 3, 3],
                            [0, 0, 0, 1, 1]])
doc_word_df.columns = ['data', 'information', 'retrieval', 'brain', 'lung']
doc_word_df.index = ['Data Science 1', 'Data Science 2', 'Data Science 3, ', 'Data Science 4', 
                     'Medical Report 1', 'Medical Report 1', 'Medical Report 1']
doc_word_df

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
      <th>data</th>
      <th>information</th>
      <th>retrieval</th>
      <th>brain</th>
      <th>lung</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Science 1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Data Science 2</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Data Science 3,</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Data Science 4</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Medical Report 1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Medical Report 1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Medical Report 1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Several Python packages, such as NumPy, SciPy, and scikit-learn, all provide APIs for SVD. In this assignment, we will use `TruncatedSVD` API from scikit-learn, as it allows us to compute only the largest $k$ singular values in $\Sigma$ and the corresponding $k$ columns (rows) of $U$ ($V^T$), which is more efficient and practical in real-world applications. You can learn more about the API from its [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

For this example, we explicitly specify $k=2$ (so we are only interested in the first 2 components), and we assign a constant to `random_state` to ensure that the result is reproducible. 


```python
doc_word_svd = TruncatedSVD(n_components=2, random_state=0)
doc_word_svd.fit(doc_word_df)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>TruncatedSVD(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">TruncatedSVD</label><div class="sk-toggleable__content"><pre>TruncatedSVD(random_state=0)</pre></div></div></div></div></div>



Now the object `doc_word_svd` stores all the necessary results of the decomposed matrix. You can view the singular values (the diagonal values in $\Sigma$) in the `singular_values_` attribute and the row vectors of $V^T$ in the `components_` attribute.


```python
sigma_diag = doc_word_svd.singular_values_
v_t = doc_word_svd.components_
print(sigma_diag)
print(v_t)
```

    [9.643651 5.291503]
    [[ 0.57735   0.57735   0.57735  -0.       -0.      ]
     [ 0.        0.        0.        0.707107  0.707107]]


You may wonder where the $U$ matrix is. Although $U$ is not explicitly stored in the `TruncatedSVD` object. it can be recovered with $X$, $\Sigma$, and $V$, based on the following formula:

$$ X^* = U\Sigma = XV.$$

Thus, $U$ can be written as

$$ U = X^* \Sigma^{-1} = XV\Sigma^{-1}. $$

We can verify this with the following code:


```python
x = doc_word_df.values
v = v_t.T
sigma_inverse = np.linalg.inv(np.diag(sigma_diag))

u = x.dot(v).dot(sigma_inverse)
print(u)
```

    [[ 0.179605  0.      ]
     [ 0.359211  0.      ]
     [ 0.179605  0.      ]
     [ 0.898027  0.      ]
     [-0.        0.534522]
     [-0.        0.801784]
     [-0.        0.267261]]


Now let's try to interpret $U$, $\Sigma$, and $V^T$ by thinking about the following questions (not graded):

* Which codes the Data Science / Medical concept of each document?
* Which codes the strength of each concept?
* Which codes the word representation of each concept?

One important application of SVD is to transform the original matrix $X$ in to a new matrix $X^*$. This transformation projects the $p$-dimensional row vectors in the original matrix into $k$-dimensional vectors in the new vector space. $X^*$ encodes most of the information in the original $X$ matrix with fewer dimensions.

Mathamatically, the transformation is made by post-multiplying $X$ with the $V$ matrix ($ X^* = XV$). The `TruncatedSVD` API has implemented such transformation as a `transform` function. You may verify this with the following code:


```python
print(doc_word_svd.transform(doc_word_df))
```

    [[ 1.732051  0.      ]
     [ 3.464102  0.      ]
     [ 1.732051  0.      ]
     [ 8.660254  0.      ]
     [-0.        2.828427]
     [-0.        4.242641]
     [-0.        1.414214]]



```python
print(doc_word_df.values.dot(v))
```

    [[ 1.732051  0.      ]
     [ 3.464102  0.      ]
     [ 1.732051  0.      ]
     [ 8.660254  0.      ]
     [-0.        2.828427]
     [-0.        4.242641]
     [-0.        1.414214]]


Now that you are familiar with the SVD operations. Let's apply it to the Montréal restaurant dataset. Run the following code block to load the dataset prepared in Part I of this assignment.


```python
business_df = pd.read_csv('assets/montreal_business.csv')
business_df.set_index('business_id', inplace=True)

review_df = pd.read_csv('assets/montreal_user.csv')
rating_df = review_df.pivot_table(index=['business_id'], columns=['user_id'], values='stars')
rating_df.fillna(0, inplace=True)

missing_business_id = set(business_df.index) - set(rating_df.index)
business_df.drop(missing_business_id, inplace=True)
```

### Exercise 5. (15 pts.)
Complete the following `transformed_rating` function, which transforms the original $n\times p$ rating matrix into a new $n \times k$ matrix using the `TruncatedSVD` API. The function should return the $n \times k$ matrix.

**Note:**
1. Please set the `random_state` to 0 so that the results do not change over different runs.
2. $k$ can take values between 1 and $p$, not necessarily the 2 used in the document-word matrix example.
3. You may either use the `transform` function in the `TruncatedSVD` object or use the dot product. Both are OK.
4. We convert `original_matrix` to a sparse matrix for you in order to save time in the grading process. You do not need to make any accomodation for this in your solution.




```python
def svd_transformed_rating(original_matrix, k, random_state=0):

    transformed_matrix = None
    original_matrix = csr_matrix(original_matrix)
    doc_word_svd = TruncatedSVD(n_components=k, random_state=random_state)
    doc_word_svd.fit(original_matrix)
    
    transformed_matrix = doc_word_svd.transform(original_matrix)

    return transformed_matrix

```

With the `svd_transformed_rating` function, we can calculate the transformed matrix through the following command.


```python
rating_matrix_d2 = svd_transformed_rating(rating_df, 2)
# You can uncomment the following line to view the result
# rating_matrix_d2

```


```python
# This code block tests if the `svd_transformed_rating` function is implemented correctly.
# We hide some tests, so passing all the displayed assertions does not guarantee full points.

rating_matrix_d2 = svd_transformed_rating(rating_df, 2)
# d2_groundtruth is calculated via the following command:
# d2_groundtruth = svd_transformed_rating(rating_df, 4)[:4]
d2_groundtruth = np.array([[26.830835356   , -6.960033590895],
                           [ 0.240134782174,  0.031972398386],
                           [ 2.971341010574, -2.141467087847],
                           [ 0.997482022276, -0.599895804555],
                           [ 0.470445852682, -0.482343299048]])
assert np.allclose(rating_matrix_d2[:5], d2_groundtruth)

rating_matrix_d5 = svd_transformed_rating(rating_df, 5)
d5_groundtruth = np.array([[26.828890919511, -6.950258673617,  1.515366172479, -3.295739836372,  4.157186247066],
                           [ 0.24012936551 ,  0.031884190966, -0.018132185754, -0.166903622477, -0.028071803934],
                           [ 2.9713046863  , -2.141818808721, -1.232252381363, -0.691324426529,  0.11961284882 ],
                           [ 0.997484724453, -0.59970688578 , -0.107441873278,  0.047329773205,  0.160751420569],
                           [ 0.470427947552, -0.482204394535,  0.062480138124, -0.238280610506,  0.055425249737]])
assert np.allclose(rating_matrix_d5[:5], d5_groundtruth)

rating_matrix_transformed = svd_transformed_rating(rating_df, 100)
assert rating_matrix_transformed.shape == (2770,100), "The transformed matrix is of the wrong dimension."
for i in range(100):
    for j in range(i):
        assert abs(rating_matrix_transformed.T[i].dot(rating_matrix_transformed.T[j])) < 1e-8, \
            "The column vectors in X* should be orthogonal."

```

A quick note on the above test cell: Ideally, the first $k'$ columns should stay the same for any $k$ as long as $k > k'$. However, you may notice small differences in the test cell (e.g. 26.830835356 vs. 26.828890919511). This is because the `TruncatedSVD` API uses a randomized SVD solver to speed up the calculation, so the output may lose minor precision.

As you can see, we even increased $k$ to 100, which is much larger than $k=2$ used in the document-word example. However, this is still much smaller than the 11,937 dimensions in the original rating matrix, and we shall soon see that these 100 dimensions have preserved most of the information in the original matrix.

### Exercise 6. (5 pts).
This exercise is to help you examine the power of SVD. That is, we want to see how the 100-dimension vectors preserve much information from the 11,937-dimensional vectors. Indeed, we shall see that the similarity between vectors is preserved after the SVD transformation. To show that, we are going to combine several techniques we have learned so far, including vector similarity, patterns in matrix data, and itemset similarity.

For your first task, complete the `find_max_dot_prod_restaurants` and the `find_max_dot_prod_restaurants_transformed` function. Each function returns the `business_id` of the top-$n$ restaurants that have the largest dot product with Modavie. The `find_max_dot_prod_restaurants` function calculates the dot product based on the raw vectors (11,937 dimensions). The  `find_max_dot_prod_restaurants_transformed` function calculates with the transformed vectors (100 dimensions). You may copy and paste `find_max_dot_prod_restaurants` from Part III of this assignment.

We have provided all the test cases to help you verify your results.

**Hints** 

- For finding the top modavie dot products, you may find your solution to part 3 of this assignment useful. 


```python
# This cell helps you prepare a new rating dataframe using the transformed vectors.
rating_matrix_transformed = svd_transformed_rating(rating_df, 100)
rating_df_transformed = pd.DataFrame(rating_matrix_transformed, index=rating_df.index)
rating_df_transformed.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
    </tr>
    <tr>
      <th>business_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1xuC540Nycht_iWFeJ-dw</th>
      <td>26.829882</td>
      <td>-6.949084</td>
      <td>1.637419</td>
      <td>-3.495390</td>
      <td>4.253959</td>
      <td>-1.578638</td>
      <td>-4.723778</td>
      <td>39.660973</td>
      <td>-18.228902</td>
      <td>23.421889</td>
      <td>...</td>
      <td>0.767882</td>
      <td>1.533552</td>
      <td>1.182358</td>
      <td>-0.950001</td>
      <td>0.017549</td>
      <td>0.262306</td>
      <td>-1.707314</td>
      <td>-1.132702</td>
      <td>-0.558816</td>
      <td>0.669832</td>
    </tr>
    <tr>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <td>0.240134</td>
      <td>0.031948</td>
      <td>-0.017022</td>
      <td>-0.169980</td>
      <td>-0.026894</td>
      <td>-0.095219</td>
      <td>-0.032490</td>
      <td>0.103308</td>
      <td>-0.170128</td>
      <td>-0.030196</td>
      <td>...</td>
      <td>0.225307</td>
      <td>0.001314</td>
      <td>-0.318106</td>
      <td>-0.128878</td>
      <td>0.100352</td>
      <td>0.165589</td>
      <td>0.204660</td>
      <td>0.075091</td>
      <td>0.270025</td>
      <td>-0.273088</td>
    </tr>
    <tr>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <td>2.971325</td>
      <td>-2.141960</td>
      <td>-1.232110</td>
      <td>-0.698304</td>
      <td>0.111484</td>
      <td>-0.469199</td>
      <td>0.181528</td>
      <td>-1.522476</td>
      <td>0.333825</td>
      <td>0.420363</td>
      <td>...</td>
      <td>1.406269</td>
      <td>1.265186</td>
      <td>-1.808670</td>
      <td>-0.257573</td>
      <td>0.417470</td>
      <td>-0.545357</td>
      <td>1.153764</td>
      <td>0.511001</td>
      <td>-0.184996</td>
      <td>-1.439649</td>
    </tr>
    <tr>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <td>0.997471</td>
      <td>-0.599784</td>
      <td>-0.108910</td>
      <td>0.053457</td>
      <td>0.161304</td>
      <td>0.202928</td>
      <td>-0.815409</td>
      <td>0.270971</td>
      <td>0.649032</td>
      <td>0.253430</td>
      <td>...</td>
      <td>0.384626</td>
      <td>-1.344930</td>
      <td>0.325934</td>
      <td>-0.000308</td>
      <td>0.328897</td>
      <td>0.074603</td>
      <td>0.192419</td>
      <td>0.501441</td>
      <td>-0.059467</td>
      <td>-0.565910</td>
    </tr>
    <tr>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <td>0.470432</td>
      <td>-0.481998</td>
      <td>0.060954</td>
      <td>-0.237180</td>
      <td>0.047426</td>
      <td>0.081098</td>
      <td>-0.075756</td>
      <td>-0.257721</td>
      <td>0.141577</td>
      <td>0.436627</td>
      <td>...</td>
      <td>-0.100854</td>
      <td>-0.031623</td>
      <td>-0.198573</td>
      <td>0.596779</td>
      <td>-0.659795</td>
      <td>0.022381</td>
      <td>-0.060685</td>
      <td>0.787100</td>
      <td>-0.165884</td>
      <td>0.032865</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>




```python
def find_max_dot_prod_restaurants_old(top_n,modavie_id, rating_df,business_df,modavie_vector):
    modavie_vector = rating_df.loc[modavie_id]
    
    business_df['modavie_dot_prod'] = rating_df.apply(lambda x:  np.dot(x,modavie_vector), axis=1)
    return business_df.sort_values('modavie_dot_prod', ascending=False).drop(modavie_id).head(top_n)
```


```python
def find_max_dot_prod_restaurants(top_n):
    modavie_id = business_df[business_df.name.str.contains("Modavie")].index[0]
    modavie_vector = rating_df.loc[modavie_id]
    
    return find_max_dot_prod_restaurants_old(top_n = top_n, modavie_id = modavie_id, rating_df = rating_df, 
                                        business_df = business_df, modavie_vector = modavie_vector) 

def find_max_dot_prod_restaurants_transformed(top_n):
    modavie_id = business_df[business_df.name.str.contains("Modavie")].index[0]
    modavie_vector_transformed = rating_df_transformed.loc[modavie_id]

    return find_max_dot_prod_restaurants_old(top_n = top_n, modavie_id = modavie_id, rating_df = rating_df_transformed, 
                                        business_df = business_df, modavie_vector = modavie_vector_transformed) 
```


```python
# This code block help you verify if `find_max_dot_prod_restaurants_transformed` is implemented correctly.
answer = find_max_dot_prod_restaurants_transformed(10)
assert answer.iloc[0]['name'] == "Schwartz's"
assert answer.iloc[1]['name'] == "La Banquise"
assert answer.iloc[2]['name'] == "Olive & Gourmando"
assert answer.iloc[3]['name'] == "Maison Christian Faure"
assert answer.iloc[4]['name'] == "Reuben's Deli & Steakhouse"

assert answer.iloc[5]['name'] == "Kazu"
assert answer.iloc[6]['name'] == "Belon"
assert answer.iloc[7]['name'] == "Au Pied de Cochon"
assert answer.iloc[8]['name'] == "L'Avenue"
assert answer.iloc[9]['name'] == "Poutineville"
```

### Exercise 7. (5 pts).

Finally, complete the `jaccard_similarity_before_after_svd_transform` function, which calculates the Jaccard similarity between the top-$n$ most similar restaurants (of course, this is an itemset!) before and after the SVD transformation.

** Hints **

- Refer back to your solution for assignment 2 part 3 if you've forgotten how to accomplish this.


```python
def jaccard_similarity_before_after_svd_transform(top_n):
    max_sim_restaurants = find_max_dot_prod_restaurants(top_n)
    max_sim_restaurants_transformed = find_max_dot_prod_restaurants_transformed(top_n)
    # construct the set of the names of the most similar restaurants
    business_id_before = set(max_sim_restaurants.name)
    business_id_after = set(max_sim_restaurants_transformed.name)

    jaccard_similarity = len(business_id_before&business_id_after)/len(business_id_before|business_id_after)
    
    return jaccard_similarity
```


```python
# This code block checks `find_max_dot_prod_restaurants`, `find_max_dot_prod_restaurants_transformed`, 
# and 'jaccard_similarity_before_after_svd_transform'. You should get the correct answer if all three 
# functions are implemented correctly.
# We hide some tests, so passing all the displayed assertions does not guarantee full points.

assert abs(jaccard_similarity_before_after_svd_transform(5) - 1) < 1e-8
assert (abs(jaccard_similarity_before_after_svd_transform(10) - 9 / 11) < 1e-8) or (abs(jaccard_similarity_before_after_svd_transform(10) - 2 / 3) < 1e-8)
assert abs(jaccard_similarity_before_after_svd_transform(2770) - 1) < 1e-8

```

As you can see, you can perform many tasks on the transformed vectors just as what you can do on the raw vectors. Since the transformed vectors have far fewer dimensions, the efficiency is much higher with the transformed vectors. Beyond efficiency, dimension reduction has many other benefits for dealing with matrix data.  You will learn these in the downstream machine learning classes, but let's stop here for now.  Matrix analysis has lots of applications in recommender systems.  We will revisit some of these techniques when that topic comes up.  

This concludes this assignment.


```python

```
