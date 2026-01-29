# Assignment 3 - Building a Custom Visualization

In this assignment you must choose __one__ of the options presented below and submit a visual as well as your source code for peer grading. The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and you should be able to earn full grades for the assignment.

Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). [Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. ([video](https://www.youtube.com/watch?v=BI7GAs-va-Q))

In this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, and the yerr parameter of barcharts).

<br>
<img src="assets/Assignment3Fig1.png" alt="Figure 1" style="width: 400px;"/>
<h4 style="text-align: center;" markdown="1">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1 from (Ferreira et al, 2014).</h4>

<br>

A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.


<br>
<img src="assets/Assignment3Fig2c.png" alt="Figure 1" style="width: 400px;"/>
<h4 style="text-align: center;" markdown="1">  Figure 2c from (Ferreira et al. 2014). Note that the colorbar legend at the bottom as well as the arrows are not required in the assignment descriptions below.</h4>

<br>
<br>

**Easiest option:** Implement the bar coloring as described above - a color scale with at least three colors, (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.

**Harder option:** Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).

**Even Harder option:** Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. The bar colors should change with respect to what value the user has selected.

**Hardest option:** Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, see the paper for more details).

---

*Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.*


```python
# Use the following data for this assignment:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df
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
      <th>3640</th>
      <th>3641</th>
      <th>3642</th>
      <th>3643</th>
      <th>3644</th>
      <th>3645</th>
      <th>3646</th>
      <th>3647</th>
      <th>3648</th>
      <th>3649</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1992</th>
      <td>-8941.531897</td>
      <td>127788.667612</td>
      <td>-71887.743011</td>
      <td>-79146.060869</td>
      <td>425156.114501</td>
      <td>310681.166595</td>
      <td>50581.575349</td>
      <td>88349.230566</td>
      <td>185804.513522</td>
      <td>281286.947277</td>
      <td>...</td>
      <td>171938.760289</td>
      <td>150650.759924</td>
      <td>203663.976475</td>
      <td>-377877.158072</td>
      <td>-197214.093861</td>
      <td>24185.008589</td>
      <td>-56826.729535</td>
      <td>-67319.766489</td>
      <td>113377.299342</td>
      <td>-4494.878538</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>-51896.094813</td>
      <td>198350.518755</td>
      <td>-123518.252821</td>
      <td>-129916.759685</td>
      <td>216119.147314</td>
      <td>49845.883728</td>
      <td>149135.648505</td>
      <td>62807.672113</td>
      <td>23365.577348</td>
      <td>-109686.264981</td>
      <td>...</td>
      <td>-44566.520071</td>
      <td>101032.122475</td>
      <td>117648.199945</td>
      <td>160475.622607</td>
      <td>-13759.888342</td>
      <td>-37333.493572</td>
      <td>103019.841174</td>
      <td>179746.127403</td>
      <td>13455.493990</td>
      <td>34442.898855</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>152336.932066</td>
      <td>192947.128056</td>
      <td>389950.263156</td>
      <td>-93006.152024</td>
      <td>100818.575896</td>
      <td>5529.230706</td>
      <td>-32989.370488</td>
      <td>223942.967178</td>
      <td>-66721.580898</td>
      <td>47826.269111</td>
      <td>...</td>
      <td>165085.806360</td>
      <td>74735.174090</td>
      <td>107329.726875</td>
      <td>199250.734156</td>
      <td>-36792.202754</td>
      <td>-71861.846997</td>
      <td>26375.113219</td>
      <td>-29328.078384</td>
      <td>65858.761714</td>
      <td>-91542.001049</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>-69708.439062</td>
      <td>-13289.977022</td>
      <td>-30178.390991</td>
      <td>55052.181256</td>
      <td>152883.621657</td>
      <td>12930.835194</td>
      <td>63700.461932</td>
      <td>64148.489835</td>
      <td>-29316.268556</td>
      <td>59645.677367</td>
      <td>...</td>
      <td>-13901.388118</td>
      <td>50173.686673</td>
      <td>53965.990717</td>
      <td>4128.990173</td>
      <td>72202.595138</td>
      <td>39937.199964</td>
      <td>139472.114293</td>
      <td>59386.186379</td>
      <td>73362.229590</td>
      <td>28705.082908</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 3650 columns</p>
</div>




```python
#Your Code Here
```


```python
def my_plot(th):
    my_df = df.T
    means = my_df.mean()
    std = my_df.std()
    n = my_df.shape[0]
    sem = std / np.sqrt(n)     
    ci95 = 1.96 * sem     
    lower_whisker = means - ci95
    upper_whisker = means + ci95
    
    colors = []
    
    for i,j in zip(lower_whisker, upper_whisker):
        
        if th>j:
            
            colors.append('blue')
        
        elif th<i:
            
            colors.append('red')
            
        else:
            
            colors.append('white')
    
    
    plt.figure(figsize=(7,5))
    plt.bar([str(i) for i in my_df.columns], my_df.mean(), color=colors,width=1,edgecolor='black')
    plt.errorbar(
    np.arange(len(means)),
    means,
    yerr=ci95,   # можно просто 1 число — верх = низ
    fmt='none',
    color='black',
    capsize=10)
    plt.axhline(th,color='grey')
    plt.xlabel('Year')
    plt.ylabel('Mean Votes')
    plt.title('Average Votes per Year')
    
```


```python

my_plot(41000)
```


    
![png](output_4_0.png)
    



```python

```
