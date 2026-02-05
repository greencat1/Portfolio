# Assignment 3: Mining Vectors and Matrices (Part I)

Welcome to Assignment 3! In this assigment, you will be playing with vectors and matrices. Similar to itemsets, or other representations of data, we care about *similarity* and *patterns*, the two basic outputs of data mining and building blocks of advanced functionalities. Through this assignment, you will get hands-on experience to derive both similarity measurements and patterns with various vector and matrix operations.

We have curated another real-world dataset for you -- restaurant ratings on Yelp. The original data set can be found [here](https://www.yelp.com/dataset). To make this assignment more trackable, we have preprocessed the data and only kept the restaurants in Montréal, Canada. As with all real-world problems, be sure to sanity-check the quality of the data and be ready to handle "unexpected" scenarios of the "Wild Thing."

In this assignment, you will: 
* Represent the dataset as a matrix and check the row and column vectors.
* Implement various vector similarity/distance measures. 
* Find the top similar vectors to a given vector.
* Compute SVD transformation of the Matrix/Vectors and analyze the vectors with reduced dimensionality. 

In Part I of the assignment, we will load the dataset, transform it into vector/matrix format, and perform necessary sanity checks. You only need to code very little in this part. Please run all the code blocks in order and read the descriptions carefully. 

First, let's import the packages and dependencies that will be used later.


```python
import pandas as pd
import numpy as np
```

## Data Preprocessing
Let's start by loading the data and preview the first few lines. We will use two data files. The montreal_business.csv file stores the attributes of the restaurants, and the montreal_rating.csv file stores the user ratings of the restaurants. Both businesses and users are assigned randomized string identifiers (business_id and user_id), as a common way to preserve the privacy of customers.


```python
business_df = pd.read_csv('assets/montreal_business.csv')
business_df.head()
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
      <th>business_id</th>
      <th>name</th>
      <th>categories</th>
      <th>stars</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1xuC540Nycht_iWFeJ-dw</td>
      <td>Romados</td>
      <td>Restaurants, Portuguese, Bakeries, Food, Chick...</td>
      <td>4.5</td>
      <td>H2W 1C8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-7bRnaHp7OHz8KW-THqP4w</td>
      <td>Buffalo Bill Wings</td>
      <td>Restaurants, Chicken Wings</td>
      <td>1.5</td>
      <td>H4C 1G7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-92cC6-X87HQ1DE1UHOx3w</td>
      <td>Junior</td>
      <td>Pan Asian, Restaurants, Asian Fusion, Filipino</td>
      <td>3.5</td>
      <td>H3J 1M8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-AgfhwHOYrsPKt-_xV_Ipg</td>
      <td>Soupesoup</td>
      <td>Sandwiches, Restaurants, Cafes, Soup</td>
      <td>4.0</td>
      <td>H2W 1G8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-FDkvLmwaBrtVgYFqEWeWA</td>
      <td>Café Juicy Lotus</td>
      <td>Vegetarian, Food, Salad, Restaurants, Organic ...</td>
      <td>3.5</td>
      <td>H4A 1C9</td>
    </tr>
  </tbody>
</table>
</div>




```python
business_df.set_index('business_id', inplace=True)
business_df.head()
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
      <th>name</th>
      <th>categories</th>
      <th>stars</th>
      <th>postal_code</th>
    </tr>
    <tr>
      <th>business_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1xuC540Nycht_iWFeJ-dw</th>
      <td>Romados</td>
      <td>Restaurants, Portuguese, Bakeries, Food, Chick...</td>
      <td>4.5</td>
      <td>H2W 1C8</td>
    </tr>
    <tr>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <td>Buffalo Bill Wings</td>
      <td>Restaurants, Chicken Wings</td>
      <td>1.5</td>
      <td>H4C 1G7</td>
    </tr>
    <tr>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <td>Junior</td>
      <td>Pan Asian, Restaurants, Asian Fusion, Filipino</td>
      <td>3.5</td>
      <td>H3J 1M8</td>
    </tr>
    <tr>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <td>Soupesoup</td>
      <td>Sandwiches, Restaurants, Cafes, Soup</td>
      <td>4.0</td>
      <td>H2W 1G8</td>
    </tr>
    <tr>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <td>Café Juicy Lotus</td>
      <td>Vegetarian, Food, Salad, Restaurants, Organic ...</td>
      <td>3.5</td>
      <td>H4A 1C9</td>
    </tr>
  </tbody>
</table>
</div>



Every row is a restaurant, and the "stars" field stores the aggregated rating of all its customers.  Other fields should be self-explanatory. 


```python
review_df = pd.read_csv('assets/montreal_user.csv')
review_df.head()
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
      <th>business_id</th>
      <th>user_id</th>
      <th>stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i_oghUPqLzzJtxC6Zm-D2A</td>
      <td>-8nmj3B-tfY_vFiimtBOsw</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>i_oghUPqLzzJtxC6Zm-D2A</td>
      <td>11V_zUrpadPKvPQpQ-Ctiw</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i_oghUPqLzzJtxC6Zm-D2A</td>
      <td>3AqRjAvCdBiM6s0f58A5eA</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>i_oghUPqLzzJtxC6Zm-D2A</td>
      <td>LN4ajhAiIshKkXJQhuh6uA</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i_oghUPqLzzJtxC6Zm-D2A</td>
      <td>T6_H4brErsgxmUEvv9VLOg</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Every row in this user-rating data file corresponds to a restaurant-user pair, and the "stars" column stores the individual (unaggregated) score that users gave to that restaurant. 

### Constructing restaurant-user matrix:

The first thing is to transform the user-restaurant ratings into a matrix. Each row vector represent the ratings of one restaurant and each dimension of the row vector (aka. a column) represents a user. Therefore, the $j$-th dimension of the $i$-th row vector represents the rating of user $j$ on the restaurant $i$. We can perform pivoting on the `review_df` dataframe to generate such a matrix, and then view the first few row vectors.


```python
rating_df = review_df.pivot_table(index=['business_id'], columns=['user_id'], values='stars')
rating_df.head(5)
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
      <th>user_id</th>
      <th>--BumyUHiO_7YsHurb9Hkw</th>
      <th>--NIc98RMssgy0mSZL3vpA</th>
      <th>-0zUpn_6kTWXHWzB85u7yA</th>
      <th>-1MF2tosrw2WcCxeVNk81Q</th>
      <th>-1wbglcr6x1qrUbqP1YAIA</th>
      <th>-3gey5u7j0qj5SZNrOHIQg</th>
      <th>-3s52C4zL_DHRK0ULG6qtg</th>
      <th>-3uRUjmr1uEn-lLrhwRMCg</th>
      <th>-4pYtFJbZYzKqmiOjGr4ag</th>
      <th>-4tyD87vW73HG0mQlXk0uw</th>
      <th>...</th>
      <th>zwOiJvD09N1dNf-ts7wscw</th>
      <th>zwsVQwyxkz-s4rvBCaPyZw</th>
      <th>zyLIp7DRFqeIE1Grrh3Zng</th>
      <th>zyYWUdaodH0h1jCZAvFRPg</th>
      <th>zyaQNsI73cLOKttvGnj4bw</th>
      <th>zyg4-MFtfPWmwucVazSjfw</th>
      <th>zzKsgIF472IyDpJxmHDIpw</th>
      <th>zzgb9L3NBT9V8BGw_DcVYw</th>
      <th>zznMmC74DNBcBG62gj3guA</th>
      <th>zzo--VpSQh8PpsGVeMC1dQ</th>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11937 columns</p>
</div>



However, directly using this matrix in production might not be a good practice. The matrix can be huge in size and yet very very sparse, as not every user has rated every restaurant. In other words, most of the cells of the matrix have missing values (NaN).

Let's briefly examine how sparse the matrix is.


```python
n_entry = review_df.shape[0]
n_business = review_df.business_id.unique().size
n_user = review_df.user_id.unique().size
print("total entry:", n_entry)
print("# business:", n_business)
print("# user:", n_user)
print(f"density:{n_entry / (n_business * n_user):.4f}")
```

    total entry: 71741
    # business: 2779
    # user: 11938
    density:0.0022


Before any analysis, we also need to perform sanity checks. For matrix data, the first thing we always do is the check the dimensionality. 

### Exercise 1. Check the dimensionality (5 pts)
Complete the `row_col_count` function below to return the numbers of rows and columns of the rating matrix.

Hint: the ```.shape``` attribute of a dataframe returns the number of rows and columns in a tuple.


```python
def row_col_count(rating_df):

    n_row = 0
    n_col = 0

    
    
    return rating_df.shape
```


```python
n_row, n_col = row_col_count(rating_df)
assert n_row == 2770
assert n_col == 11937
```


```python
rating_df.shape
```




    (2770, 11937)



Comparing the numbers with the earlier cells, we have 2779 businesses in review_df, but we only have 2770 rows in rating_df. What happened?  Let's further examine the difference.


```python
set(business_df.index) - set(rating_df.index)
```




    {'3mr1-uwrYFwFDQO9WNEVqw',
     '7RvCrO6n2jkC8IQvfdm04g',
     'O7uzVmRB8GDMU8XaX5l7Eg',
     'RW4wwNBB-TDJcN2vCsLMRw',
     'VTb9w9PowA76vhBVgrwsIw',
     'ZByHOUsGw7q5QC-EsJfmIg',
     'eC3LV3CpXPoB4kpMDIbCVQ',
     'nK4kWsScyDBHdcxtfszzcQ',
     'xFxlyQPT27xc2dsqyzQKUg'}



We see that difference of 9, (2779 - 2770), comes solely from 9 businesses missing in the `rating_df`. Do they exist in `review_df`?


```python
missing_business_id = set(business_df.index) - set(rating_df.index)
review_df[review_df.business_id.isin(missing_business_id)]
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
      <th>business_id</th>
      <th>user_id</th>
      <th>stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8943</th>
      <td>O7uzVmRB8GDMU8XaX5l7Eg</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>xFxlyQPT27xc2dsqyzQKUg</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43765</th>
      <td>VTb9w9PowA76vhBVgrwsIw</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44263</th>
      <td>eC3LV3CpXPoB4kpMDIbCVQ</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45411</th>
      <td>3mr1-uwrYFwFDQO9WNEVqw</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46355</th>
      <td>ZByHOUsGw7q5QC-EsJfmIg</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>57748</th>
      <td>7RvCrO6n2jkC8IQvfdm04g</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>61904</th>
      <td>RW4wwNBB-TDJcN2vCsLMRw</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70186</th>
      <td>nK4kWsScyDBHdcxtfszzcQ</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It is now clear that the 9 missing businesses do not have any valid entries in the `review_df`, so they do not have corresponding rows in `rating_df` after pivoting. To be consistent, we could create 9 additional rows in `rating_df` with all NaN values. Or, we can simply remove the businesses from `business_df` to make the two dataframes coherent.


```python
business_df.drop(missing_business_id, inplace=True)
```

There are several ways of dealing with missing values in a matrix (the NaN cells in the pivoted dataframe). For now, let's simply fill the NaN cells with 0. This may not be a good practice in reality. Not only will we be dealing with a much larger (denser) dataset, which requires a lot more computing resources, but more importantly, we are implicitly making an assumption that the users are rating those restaurants with 0 star. Do they really dislike those restaurant or have they not been to them?  Can you think of a more reasonable method to fill in NAs?


```python
rating_df.fillna(0, inplace=True)
rating_df.head(5)
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
      <th>user_id</th>
      <th>--BumyUHiO_7YsHurb9Hkw</th>
      <th>--NIc98RMssgy0mSZL3vpA</th>
      <th>-0zUpn_6kTWXHWzB85u7yA</th>
      <th>-1MF2tosrw2WcCxeVNk81Q</th>
      <th>-1wbglcr6x1qrUbqP1YAIA</th>
      <th>-3gey5u7j0qj5SZNrOHIQg</th>
      <th>-3s52C4zL_DHRK0ULG6qtg</th>
      <th>-3uRUjmr1uEn-lLrhwRMCg</th>
      <th>-4pYtFJbZYzKqmiOjGr4ag</th>
      <th>-4tyD87vW73HG0mQlXk0uw</th>
      <th>...</th>
      <th>zwOiJvD09N1dNf-ts7wscw</th>
      <th>zwsVQwyxkz-s4rvBCaPyZw</th>
      <th>zyLIp7DRFqeIE1Grrh3Zng</th>
      <th>zyYWUdaodH0h1jCZAvFRPg</th>
      <th>zyaQNsI73cLOKttvGnj4bw</th>
      <th>zyg4-MFtfPWmwucVazSjfw</th>
      <th>zzKsgIF472IyDpJxmHDIpw</th>
      <th>zzgb9L3NBT9V8BGw_DcVYw</th>
      <th>zznMmC74DNBcBG62gj3guA</th>
      <th>zzo--VpSQh8PpsGVeMC1dQ</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-7bRnaHp7OHz8KW-THqP4w</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-92cC6-X87HQ1DE1UHOx3w</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-AgfhwHOYrsPKt-_xV_Ipg</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-FDkvLmwaBrtVgYFqEWeWA</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11937 columns</p>
</div>



Now we should obtain a restaurant x user matrix fully filled with ratings. This concludes the first part of the assignment.


```python

```
