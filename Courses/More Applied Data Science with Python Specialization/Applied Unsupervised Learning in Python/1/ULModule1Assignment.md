```python
# First import some necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
up, down = True, False
np.set_printoptions(precision=3)
pd.set_option("expand_frame_repr", True)
pd.set_option("colheader_justify", "right")
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 75)
pd.set_option("display.precision", 4)
```

All import statements are provided for you- please do not import other libraries in your own code.

<a id='Topic0'></a>

# Assignment 1: Getting started with unsupervised learning

For the Module 1 programming assignment, you will focus on dimensionality reduction with PCA, using a music dataset of Spotify tracks.

Datasets often have a massive number of different features for each data point. In the case of the Spotify tracks dataset you will use, you will have features for each song, such as popularity, duration, and "loudness." Applications that utilize data like this - in this case, recommendation systems in music streaming apps - often need to reduce the number of features/dimensions to the data to make comparing each data point, or song in this case, computationally efficient. 

In this assignment, you will use PCA dimensionality reduction on a dataset of music tracks and look into some valuable tasks you can perform with the resulting PCA object. 

* First, you will create a PCA object on the song dataset and examine key attributes of the resulting object. 

* Second, with the PCA object you created in the first step, you will create a scree plot, which can help determine the number of principal components needed to account for an adequate amount of variance in the data.

* Third, you will use a biplot to visualize both the dataset and its features (variables), which will allow you to get insight into relationships between song features.

* Fourth, you will estimate a probability density over songs using kernel density estimation. This will allow you to glean interesting facts about songs and their relationships. In this step, you will compute the likelihood of songs in the dataset to see how ‘typical’ or ‘atypical’ they are in the dataset.


```python
# Here, we read in the dataset and drop any tracks which might be missing values for a feature.

tracks_dataframe = pd.read_csv("./assets/tracks.csv")
tracks_dataframe = tracks_dataframe.dropna()

# Take a look at what's in the dataset!
tracks_dataframe.head()
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
      <th>Unnamed: 0</th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>-6.746</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>1.0100e-06</td>
      <td>0.3580</td>
      <td>0.715</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>-17.235</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>5.5600e-06</td>
      <td>0.1010</td>
      <td>0.267</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>-9.734</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.0000e+00</td>
      <td>0.1170</td>
      <td>0.120</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Soundtrack)</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>-18.515</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>7.0700e-05</td>
      <td>0.1320</td>
      <td>0.143</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>-9.681</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.0000e+00</td>
      <td>0.0829</td>
      <td>0.167</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
  </tbody>
</table>
</div>




```python
# As you can see, Each track has features such as length, loudness, and speechiness. 
# We don't need every feature, so we're going to make a dataframe with only the features we want. 
feature_names = ["popularity","duration_ms","danceability","energy","key","loudness","speechiness","acousticness","liveness","tempo"]

# Here, we select only these features to simplify our dataset. Take a look at the new data!
tracks_dataframe = tracks_dataframe[feature_names]
tracks_dataframe.head()
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
      <th>popularity</th>
      <th>duration_ms</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>liveness</th>
      <th>tempo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73</td>
      <td>230666</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>-6.746</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.3580</td>
      <td>87.917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>149610</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>-17.235</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.1010</td>
      <td>77.489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57</td>
      <td>210826</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>-9.734</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.1170</td>
      <td>76.332</td>
    </tr>
    <tr>
      <th>3</th>
      <td>71</td>
      <td>201933</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>-18.515</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.1320</td>
      <td>181.740</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82</td>
      <td>198853</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>-9.681</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.0829</td>
      <td>119.949</td>
    </tr>
  </tbody>
</table>
</div>



# Part 1: Using Principal Component Analysis (PCA)

In this part, we'll walk you through the process of creating and using a PCA object, and examine some key attributes of the resulting object. 

# Task 1a. Normalizing the dataset
(10 points)

The first step when preparing a dataset for PCA is to normalize the dataset. Conceptually, remember that what normalization does is to rescale the values of each feature (i.e. variable) so that the feature values have a mean of zero and a standard deviation of 1.

Using the `tracks_dataframe` dataset object provided above, complete the function below to return a numpy ndarray containing the normalized contents of the features in `tracks_dataframe`.

Hints:
 - Helpful Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
 - The two tools you need here are default StandardScaler and `.fit_transform()`
 


```python
task_id = "1a"
```


```python


def task_1a_solution():
    scaler = StandardScaler()
    
    normalized_data = scaler.fit_transform(tracks_dataframe)
    
    return normalized_data


```


```python
# Use this cell to explore your solution.

task_1a_solution()
```




    array([[ 1.783,  0.025,  0.629, ..., -0.85 ,  0.759, -1.142],
           [ 0.976, -0.731, -0.846, ...,  1.832, -0.591, -1.49 ],
           [ 1.065, -0.16 , -0.742, ..., -0.315, -0.507, -1.528],
           ...,
           [-0.504,  0.405,  0.358, ...,  1.66 , -0.681,  0.341],
           [ 0.348,  0.521,  0.116, ...,  0.199,  0.296,  0.461],
           [-0.504,  0.129, -0.235, ...,  1.101, -0.653, -1.433]])




```python
print(f"Task {task_id} - AG tests")
stu_ans = task_1a_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, np.ndarray
), f"Task {task_id}: Your function should return an np.ndarray. "

assert (
    stu_ans.shape[1] == 10
), f"Task {task_id}: Your returned ndarray has an incorrect number of columns. "

assert np.isclose(
    np.mean(stu_ans, axis=0), 0.0
).all(), (
    f"Task {task_id}: The columns of your returned ndarray should have a zero mean. "
)

assert np.isclose(
    np.std(stu_ans, axis=0), 1.0
).all(), (
    f"Task {task_id}: The columns of your returned ndarray should have a unit std. "
)


del stu_ans
```

    Task 1a - AG tests
    Task 1a - your answer:
    [[ 1.783  0.025  0.629 ... -0.85   0.759 -1.142]
     [ 0.976 -0.731 -0.846 ...  1.832 -0.591 -1.49 ]
     [ 1.065 -0.16  -0.742 ... -0.315 -0.507 -1.528]
     ...
     [-0.504  0.405  0.358 ...  1.66  -0.681  0.341]
     [ 0.348  0.521  0.116 ...  0.199  0.296  0.461]
     [-0.504  0.129 -0.235 ...  1.101 -0.653 -1.433]]


# Task 1b: Creating the PCA object
(10 points)

Now, create a PCA object using the normalized data you produced in task 1a. 

For the arguments to PCA, be sure to use `n_components=6` and `random_state = 42`.

Helpful Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


```python
task_id = '1b'

```


```python

def task_1b_solution():
    normalized_data = task_1a_solution()
    pca = PCA(n_components=6, random_state=42)
    pca.fit(normalized_data)
    return pca
```


```python
# Use this cell to explore your solution.

task_1b_solution()
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=6, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=6, random_state=42)</pre></div></div></div></div></div>




```python
#here are some tests which may help troubleshoot your answer.


print(f"Task {task_id} - AG tests")
stu_ans = task_1b_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")


assert isinstance(
    stu_ans, PCA
), f"Task {task_id}: Your function should return a PCA object. "

assert (
    stu_ans.n_components == 6
), f"Task {task_id}: Your PCA object should be set to produce 6 components. "
assert stu_ans.components_.shape == (
    6,
    10,
), f"Task {task_id}: Your PCA fit on the grouped data should have produced 10 columns (one for each of the review_ features)."


del stu_ans
```

    Task 1b - AG tests
    Task 1b - your answer:
    PCA(n_components=6, random_state=42)


<br>
Great! We now have our PCA object based on the song dataset. Now we're going to explore some neat things we can do with it. 

# Task 1c: Finding representative features for each principal component
(20 Points)

We know that each feature has a different weight in each principal component. 

Complete the function below to return, for each of the six principal components, the 3 features that have the highest absolute value weight for that principal component. This tells us which features "influence" that principal component the most.

- At a high level, this is what the function should do:
    - Get loading vectors for the principal components from the PCA object's .components_ attribute.
    - For each loading vector, 
    -    Take the biggest 3 absolute value feature weights and create a list of the corresponding 3 feature names
    -    Append this list of feature names to the final list of results.

    
- Hints: 
   - np.argsort() and np.abs() are useful for obtaining the indices of the highest feature weights in a loading.
   - Recall that you can use index into the `feature_names` list we created above to reference feature names. For example, if the third loading in the list has the highest absolute value weight, you can get the feature name with `feature_names[2]`.
   - You should append each list of 3 feature names to the end of the final "results" list. The list should be sorted from least to most weight (the last feature name in each list should have the highest weight) 
     



```python
task_id = '1c'


```


```python


def task_1c_solution():
    result = []
    pcaObject = task_1b_solution()
    attributes = pcaObject.components_

    for i in attributes:
        
        result.append([feature_names[k] for k in np.argsort(np.abs(i))[-3:]])

    
    return (result)
```


```python
# Use this cell to explore your solution.

task_1c_solution()
```




    [['acousticness', 'loudness', 'energy'],
     ['danceability', 'speechiness', 'liveness'],
     ['speechiness', 'duration_ms', 'danceability'],
     ['duration_ms', 'popularity', 'key'],
     ['liveness', 'key', 'popularity'],
     ['key', 'duration_ms', 'tempo']]




```python

print(f"Task {task_id} - AG tests")
stu_ans = task_1c_solution()


print(f"Task {task_id} - your answer:\n{stu_ans}")


assert isinstance(
    stu_ans, list
), f"Task {task_id}: Your function should return a list. "

assert (
    len(stu_ans) == 6
), f"Task {task_id}: Your function should return an list of length 6. "


# Some hidden tests

del stu_ans
```

    Task 1c - AG tests
    Task 1c - your answer:
    [['acousticness', 'loudness', 'energy'], ['danceability', 'speechiness', 'liveness'], ['speechiness', 'duration_ms', 'danceability'], ['duration_ms', 'popularity', 'key'], ['liveness', 'key', 'popularity'], ['key', 'duration_ms', 'tempo']]


Take a look at the first 3 principal components. Think about how each component represents a certain aspect of a song. 

# Task 1d: Estimating total variance associated with principal components
(15 points)

We know that each Principal component accounts for a portion of the dataset's variance.

Complete the below function to return a value representing how much total variance the first five (5) principal components account for, using the PCA object you created in task 1b.

Hint: The explained_variance_ratio_ attribute of the PCA object is useful.


```python
task_id = '1d'

```


```python
def task_1d_solution():
    tracksPCAObject = task_1b_solution()
    cumulative_ratio = sum(tracksPCAObject.explained_variance_ratio_[:5])
    
    return cumulative_ratio
```


```python
# Use this cell to explore your solution.

task_1d_solution()
```




    0.6988450870224975




```python

print(f"Task {task_id} - AG tests")
stu_ans = task_1d_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, float
), f"Task {task_id}: Your function should return a float. "



del stu_ans
```

    Task 1d - AG tests
    Task 1d - your answer:
    0.6988450870224975


# Task 2: Analyzing a scree plot
(5 points)

With the PCA object you created in task 1b, we can create a scree plot shown below, which can help us determine the number of principal components we need to select to account for an adequate amount of variance in the data.
We've also added a vertical line at eigenvalue 1.0 which will help you in this question.


```python

#Notice how we use normal explained_variance. You just used explained_variance_ratio_ in the previous part!

df = pd.DataFrame({'eigenvalue': np.sqrt(task_1b_solution().explained_variance_), 
                   'PC': ['01', '02', '03', '04', '05', '06']})

plt.plot(df['PC'], df['eigenvalue'], marker='o', color='c', linestyle='-')

plt.axhline(y=1.0, color='r', linestyle='--', label='Eigenvalue = 1.0')

plt.xlabel('PC')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')

plt.legend()

plt.show()

```


    
![png](output_27_0.png)
    



```python
task_id = '2'
```

Based on the Kaiser heuristic, how many principal components should we use to get a good representation of the dataset?

The graph may be a bit too close to tell: try printing out the eigenvalues stored in `df`.

Put your answer as the return value in the function below:



```python

def task_2_solution():
    solution = sum(df['eigenvalue']>=1)

    
    
    return solution
```


```python

print(f"Task {task_id} - AG tests")
stu_ans = task_2_solution()

assert isinstance(
    stu_ans, int
), f"Task {task_id}: Your function should return a int. "

```

    Task 2 - AG tests


# Part 3: Analyzing a biplot

Often, biplots are used to visualize PCA components. The code below uses the PCA object you created in part 1 to draw a biplot containing a scatterplot of the data and vectors corresponding to four of the dataset variables (song features). Run the cell below to create the graph. For each vector, the name of the corresponding variable is plotted near the end and slightly to the left of the end of the vector. In this part, you will answer questions about the biplot.


```python


def biplot(score, coeff, maxdim, pcax, pcay, labels=None):
    """Routine to generate a high-quality biplot"""

    pca1 = pcax - 1
    pca2 = pcay - 1
    xs = score[:, pca1]
    ys = score[:, pca2]
    n = min(coeff.shape[0], maxdim)
    scalex = 2.0 / (xs.max() - xs.min())
    scaley = 2.0 / (ys.max() - ys.min())
    text_scale_factor = 1

    plt.figure(figsize=(10, 9))
    plt.scatter(xs * scalex, ys * scaley, s=1)
    xspecific = score[[20362, 103573, 76020], pca1]
    yspecific = score[[20362, 103573, 76020], pca2]
    ##xspecific = score[31:34, pca1]
    ##yspecific = score[31:34, pca2]
    plt.scatter(xspecific[0] * scalex, yspecific[0] * scaley, color='Yellow', marker='x', s=360)
    plt.scatter(xspecific[1] * scalex, yspecific[1] * scaley, color='Purple', marker='+', s=360)
    plt.scatter(xspecific[2] * scalex, yspecific[2] * scaley, color='Pink', marker='o', s=360)


    for i in range(n):
        plt.arrow(0, 0, coeff[i, pca1], coeff[i, pca2], color="r", alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, pca1] * text_scale_factor,
                coeff[i, pca2] * text_scale_factor,
                "Var" + str(i + 1),
                color="r",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, pca1] * text_scale_factor - 0.2,
                coeff[i, pca2],
                labels[i],
                color="r",
                ha="center",
                va="center",
                size="xx-large"
            )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    plt.tight_layout()
    return


def make_biplot():
    X_spotify_normalized = task_1a_solution()
    pca = task_1b_solution()
    X_spotify_pca = pca.transform(X_spotify_normalized)
    
    biplot(
        X_spotify_pca, np.transpose(pca.components_[0:2, :]), 4, 1, 2, labels=feature_names
    )
    print(len(X_spotify_pca))

    return

make_biplot()
```

    113999



    
![png](output_33_1.png)
    


# Task 3a: Biplot interpretation for variables
(10 points)

For the first task involving the biplot, answer the following questions: 


- What are the names of the two most correlated variables?
- What are the names of the two least correlated variables?

Populate the provided `most_correlated` and `least_correlated` lists with your solution. Order does not matter.


```python
task_id = '3a'
```


```python


def task_3a_solution():
    most_correlated = ["popularity", "danceability"]
    least_correlated = ["energy", "duration_ms"] 
    
  
    return most_correlated, least_correlated


```


```python

print(f"Task {task_id} - AG tests")
stu_ans = task_3a_solution()

assert isinstance(
    stu_ans, tuple
), f"Task {task_id}: Your function should return a tuple. "

assert len(stu_ans) == 2, f"Task {task_id}: Your function should return a tuple of length 2."

for i, item in enumerate(stu_ans):
    assert isinstance(item, list), f"Task {task_id}: Element {i} of the tuple should be a list."
    assert len(item) == 2, f"Task {task_id}: List {i} should contain exactly two elements."
    for j, val in enumerate(item):
        assert isinstance(val, str), f"Task {task_id}: Element {j} in list {i} should be a string."


del stu_ans

```

    Task 3a - AG tests


# Task 3b: Biplot interpretation for data points
(10 points)

You may have noticed that the biplot also shows three specially marked data points represented by a pink circle ("Pink"), a yellow X ("Yellow"), and a blue cross ("Blue"). Of these specially marked data points, one refers the popular pop song "Paris" by The Chainsmokers, another refers to a popular Spanish pop rock song, and a third refers to an old orchestral song. Thinking about the likely features of pop songs vs orchestral music, which of these specially marked points denotes the orchestral song? 

Submit your answer as the string 'Pink', 'Yellow', or 'Purple'.


```python
task_id = '3b'
```


```python

def task_3b_solution():
    ans = 'Pink'

    
    return ans
```


```python
# Use this cell to explore your solution.

task_3b_solution()
```




    'Pink'




```python

print(f"Task {task_id} - AG tests")
stu_ans = task_3b_solution()

assert isinstance(
    stu_ans, str
), f"Task {task_id}: Your function should return a str. "


del stu_ans
```

    Task 3b - AG tests


# Part 4: Kernel density estimation

Almost done! In the last part of this assignment, we will be using PCA components plotted in coordinate space with a kernel density object, which will allow us to glean insights about how rare or common specific types of songs are in the dataset. In this part, we will be computing how `typical` the orchestral song is compared to the other songs in the dataset. 

The code below is a helper function which will give us the first 2 principal components in coordinate format.



```python
def get_PCA_coordinates():

    X_spotify_normalized = task_1a_solution()
    pca = task_1b_solution()
    X_spotify_pca = pca.transform(X_spotify_normalized)
    xs = X_spotify_pca[:, 0]
    ys = X_spotify_pca[:, 1]
    coordinates = np.column_stack((xs, ys))
    return coordinates
```

# Task 4a: Creating the kernel density object
(10 points)

The first step is to create the kernel density object. Create one using the coordinates from our helper function. 
Use 'gaussian' for the kernel argument in the KernelDensity object.

See the following documentation to see how to create the kernel density object: 
https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KernelDensity.html

Hint: We are going to be computing a density function over the coordinate points, so use `coordinates` as the input data used when fitting the KernelDensity object.


```python
task_id = '4a'


```


```python
def task_4a_solution():
    coordinates = get_PCA_coordinates()
    kde = KernelDensity(kernel='gaussian')
    kde.fit(coordinates)

    

    return kde
```


```python
# Use this cell to explore your solution.

task_4a_solution()
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KernelDensity()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">KernelDensity</label><div class="sk-toggleable__content"><pre>KernelDensity()</pre></div></div></div></div></div>




```python

print(f"Task {task_id} - AG tests")
stu_ans = task_4a_solution()

assert isinstance(
    stu_ans, KernelDensity
), f"Task {task_id}: Your function should return a KernelDensity object. "


del stu_ans
```

    Task 4a - AG tests


# Task 4b: Computing individual song likelihood
(10 points)

Now that we have the Kernel Density estimator object, we can calculate the likelihood of specific songs! Songs with high likelihood values are more typical in this dataset, while low likelihood songs are more unusual.  Return the *likelihood* of the orchestral song in the above biplot. 

- Hints:
  - The orchestral song is the 76020th element in the `coordinates` array.
  - kde.score_samples(in the kde documentation) and `np.exp` are useful here.  Remember that `score_samples` returns the log-likelihood of the input point(s).

For fun, you could also compute the likelihood of the other two songs visualized in Task 3b, which have indexes of 20362 and 103573 respectively, to see if they are more or less 'typical' than the orchestral example.


```python
task_id = '4b'

```


```python
def task_4b_solution():
    coordinates = get_PCA_coordinates()
    kde = task_4a_solution()

    probability = float(np.exp(kde.score_samples(coordinates[76020-1].reshape(1, -1))))
    
    
    return probability
```


```python
# Use this cell to explore your solution.

task_4b_solution()
```




    0.02194474188350147




```python
print(f"Task {task_id} - AG tests")
stu_ans = task_4b_solution()


print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, float
), f"Task {task_id}: Your function should return a float "


del stu_ans
```

    Task 4b - AG tests
    Task 4b - your answer:
    0.02194474188350147


# Bonus example: Gaussian mixture models for density estimation
Here is a Gaussian mixture model based on the first 2 principal components. Where do you think this orchestral song might be in this plot: close to the center or farther out?


```python
def GMM():
    X_spotify_normalized = task_1a_solution()
    pca = task_1b_solution()
    X_spotify_pca = pca.transform(X_spotify_normalized)
    xs = X_spotify_pca[:, 0]
    ys = X_spotify_pca[:, 1]
    coordinates = np.column_stack((xs, ys))
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(coordinates)
    x = np.linspace(-10.0, 10.0)
    y = np.linspace(-10.0,10.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)
    plt.figure(figsize=(8, 6))
    plt.title("Negative log-likelihood predicted by a GMM")

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], 10)

    plt.tight_layout()

GMM()

```


    
![png](output_56_0.png)
    



```python

```
