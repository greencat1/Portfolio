# Assignment 2: Dealing with Itemset Real-World Data (Part III)


## Itemset Similarity
Recall from Module 1 that pattern and similarity are two basic outputs of data mining. So far, we have been playing with patterns - frequent itemsets and association rules can all be seen as "patterns". In the last part, let's work on itemset similarities.

### Exercise 4. Jaccard Similarity (20 pts)
Jaccard similarity is a simple but powerful measurement of itemset similarity, defined as follows:
$$\text{Jaccard\_similarity(A, B)} = \frac{|A\cap B|}{|A\cup B|}$$

#### Exercise 4(a) (10 pts)
Complete the following function to calculate the Jaccard similarity between two sets and return the value as a float. You may assume that at least one of the sets is not empty.


```python
def jaccard_similarity(set_a, set_b):

    return len(set_a&set_b)/len(set_a|set_b)

```


```python
# This code block tests whether the `jaccard_similarity` function work as expected.
# We hide some tests, so passing all the displayed assertions does not gurantee the bonus points.

assert abs(jaccard_similarity(set(['🍇', '🍈', '🍉']), set(['🍊', '🍋', '🍌'])) - 0) < 1e-8
assert abs(jaccard_similarity(set(['🍇', '🍈', '🍎']), set(['🍊', '🍋', '🍎'])) - 0.2) < 1e-8
assert abs(jaccard_similarity(set(['🍇', '🍒', '🍎']), set(['🍊', '🍒', '🍎'])) - 0.5) < 1e-8
assert abs(jaccard_similarity(set(['🍓', '🍒', '🍎']), set(['🍒', '🍎', '🍓'])) - 1) < 1e-8

```

#### Exercise 4(b) (10 pts)
A few questions to help you better understand the properties of Jaccard similarity:
1. What is the range of Jaccard similarity?
2. When is the max/min Jaccard similarity achieved?

Complete the following function to indicate the minimum and maximum value a Jaccard similarity score can achieve. (5 pts each).


```python
# Please change min_value and max_value to your answers

min_value = 0
max_value = 1

```


```python
# This code block evaluate if min_value is assigned with the correct value
```


```python
# This code block evaluate if max_value is assigned with the correct value
```

With the Jaccard similarity calculated, we can now calculate the Jaccard similarity between any given Tweet with all other Tweets and find the Tweets that are most similar in terms of the set of food/drink emojis used. (Not graded.)


```python
import pandas as pd
import numpy as np

tweets_df = pd.read_csv("assets/food_drink_emoji_tweets.txt", sep="\t", header=None)
tweets_df.columns = ['text']

emoji_list = "🍇🍈🍉🍊🍋🍌🍍🥭🍎🍏🍐🍑🍒🍓🥝🍅🥥🥑🍆🥔🥕🌽🌶🥒🥬🥦🍄🥜🌰🍞🥐🥖🥨🥯🥞🧀🍖🍗🥩🥓🍔🍟🍕🌭🥪🌮🌯🥙🥚🍳🥘🍲🥣🥗🍿🧂🥫🍱🍘🍙🍚🍛🍜🍝🍠🍢🍣🍤🍥🥮🍡🥟🥠🥡🦀🦞🦐🦑🍦🍧🍨🍩🍪🎂🍰🧁🥧🍫🍬🍭🍮🍯🍼🥛☕🍵🍶🍾🍷🍸🍹🍺🍻🥂🥃"
emoji_set = set(emoji_list)

def extract_uniq_emojis(text):
    return 

tweets_df['emojis'] = tweets_df.text.apply(lambda text:np.unique([chr for chr in text if chr in emoji_set]))

tweets_df['jaccard'] = tweets_df.emojis.apply(lambda x:jaccard_similarity(set(tweets_df.loc[0].emojis), set(x)))
tweets_df.sort_values('jaccard',ascending=False).head(n=10)
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
      <th>emojis</th>
      <th>jaccard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RT @CalorieFixess: 🍗🌯🍔🍒 400 Calories https://t...</td>
      <td>[🌯, 🍒, 🍔, 🍗]</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>6800</th>
      <td>RT @levelscafeabuja: Chow! 🤩💦🍗🌯🍔 #LevelsCafeAb...</td>
      <td>[🌯, 🍔, 🍗]</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>9158</th>
      <td>RT @AStateRedWolves: ✅ Countertops: Installed ...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>777</th>
      <td>@SunnyAnderson @rosannascotto I don’t think KF...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>6226</th>
      <td>RT @yooojax: 3. Free Food 🍗🍔 will ready by 2PM...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>7428</th>
      <td>Kicking off the weekend with a cheeky BBQ? Her...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>7877</th>
      <td>@tafarireid07 Did you say bbq? 🔥🍔🍗🚙</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>5328</th>
      <td>RT @MAPSTTU: @EtaUpAlphas is starting the seme...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>5334</th>
      <td>RT @thatssochioma: You don’t want to miss this...</td>
      <td>[🍔, 🍗]</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>7788</th>
      <td>I’m hungry for chicken 🍗 wings or burrito 🌯</td>
      <td>[🌯, 🍗]</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>



**🍾🍾 Congratulations 🎉🎉, you have now finished up all the exercises in this assignment!**
