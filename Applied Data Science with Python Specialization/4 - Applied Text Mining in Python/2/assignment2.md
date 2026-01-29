# Assignment 2 - Introduction to NLTK

In part 1 of this assignment you will use nltk to explore the <a href='http://www.cs.cmu.edu/~ark/personas/'>CMU Movie Summary Corpus</a>. All data is released under a <a href='https://creativecommons.org/licenses/by-sa/3.0/us/legalcode'>Creative Commons Attribution-ShareAlike License</a>. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

## Part 1 - Analyzing Plots Summary Text


```python
import nltk
import pandas as pd
import numpy as np

nltk.data.path.append("assets/")

# If you would like to work with the raw text you can use 'plots_raw'
with open('assets/plots.txt', 'rt', encoding="utf8") as f:
    plots_raw = f.read()

# If you would like to work with the plot summaries in nltk.Text format you can use 'text1'.
plots_tokens = nltk.word_tokenize(plots_raw)
text1 = nltk.Text(plots_tokens)
```

### Example 1

How many tokens (words and punctuation symbols) are in text1?

*This function should return an integer.*


```python
def example_one():
    
    return len(nltk.word_tokenize(plots_raw)) # or alternatively len(text1)

example_one()
```




    374441



### Example 2

How many unique tokens (unique words and punctuation) does text1 have?

*This function should return an integer.*


```python
def example_two():
    
    return len(set(nltk.word_tokenize(plots_raw))) # or alternatively len(set(text1))

example_two()
```




    25933



### Example 3

After lemmatizing the verbs, how many unique tokens does text1 have?

*This function should return an integer.*


```python
from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()
```




    21760



### Question 1

What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)

*This function should return a float.*


```python
def answer_one():

    # YOUR CODE HERE

    return example_two()/example_one()

answer_one()
```




    0.06925790712021386




```python

```

### Question 2

What percentage of tokens is 'love'or 'Love'?

*This function should return a float.*


```python
def answer_two():

    # YOUR CODE HERE
    tokens = nltk.word_tokenize(plots_raw)
    love = 0
    
    for i in tokens:
        
        if (i == 'love') or (i == 'Love'):
            
            love = love + 1 
            
    return 100*(love/len(tokens))

tokens = answer_two()
```


```python
tokens
```




    0.12391805384559917



### Question 3

What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

*This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*


```python
def answer_three():

    
    tokens = nltk.word_tokenize(plots_raw)
    unique = list(set(tokens))
    my_dict = {i:0 for i in unique}
    
    for i in tokens:
        
        my_dict[i] = my_dict[i] + 1
    
    
    top20 = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return top20
    
answer_three()
```




    [(',', 19420),
     ('the', 18698),
     ('.', 16624),
     ('to', 12149),
     ('and', 11400),
     ('a', 8979),
     ('of', 6510),
     ('is', 5699),
     ('in', 5109),
     ('his', 4693),
     ("'s", 3682),
     ('her', 3674),
     ('he', 3556),
     ('that', 3517),
     ('with', 3293),
     ('him', 2570),
     ('for', 2433),
     ('by', 2321),
     ('The', 2234),
     ('on', 1925)]




```python

```

### Question 4

What tokens have a length of greater than 5 and frequency of more than 200?

*This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*


```python
def answer_four():

    tokens = nltk.word_tokenize(plots_raw)
    unique = list(set(tokens))
    my_dict = {i:0 for i in unique}
    
    for i in tokens:
        
        my_dict[i] = my_dict[i] + 1
        
    result = []
    
    for i in list(my_dict.keys()):
        
        if (len(i)>5) and (my_dict[i]>200):
            
            result.append(i)
    result = sorted(result)
    
    return result

answer_four()
```




    ['However',
     'Meanwhile',
     'another',
     'because',
     'becomes',
     'before',
     'begins',
     'daughter',
     'decides',
     'escape',
     'family',
     'father',
     'friend',
     'friends',
     'himself',
     'killed',
     'leaves',
     'mother',
     'people',
     'police',
     'returns',
     'school',
     'through']




```python

```

### Question 5

Find the longest token in text1 and that token's length.

*This function should return a tuple `(longest_word, length)`.*


```python
def answer_five():

    tokens = nltk.word_tokenize(plots_raw)
    
    lenth = 0
    token = ''
    
    for i in tokens:
        
        if len(i)>lenth:
            
            lenth = len(i)
            token = i
            
    return (token, lenth)

answer_five()
```




    ('live-for-today-for-tomorrow-we-die', 34)




```python

```

### Question 6

What unique words have a frequency of more than 2000? What is their frequency?

"Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."

*This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*


```python
def answer_six():

    tokens = nltk.word_tokenize(plots_raw)
    unique = list(set(tokens))
    my_dict = {i:0 for i in unique}
    
    for i in tokens:
        
        my_dict[i] = my_dict[i] + 1
    
    result = {}
    for i in list(my_dict.keys()):
        
        if (i.isalpha() == True) and (my_dict[i]>2000):
            
            result[i] = my_dict[i]
    
    
    return [(i[1], i[0]) for i in sorted(result.items(), key=lambda x: x[1], reverse=True)]
answer_six()
```




    [(18698, 'the'),
     (12149, 'to'),
     (11400, 'and'),
     (8979, 'a'),
     (6510, 'of'),
     (5699, 'is'),
     (5109, 'in'),
     (4693, 'his'),
     (3674, 'her'),
     (3556, 'he'),
     (3517, 'that'),
     (3293, 'with'),
     (2570, 'him'),
     (2433, 'for'),
     (2321, 'by'),
     (2234, 'The')]




```python

```

### Question 7

`text1` is in `nltk.Text` format that has been constructed using tokens output by `nltk.word_tokenize(plots_raw)`. 

Now, use `nltk.sent_tokenize` on the tokens in `text1` by joining them using whitespace to output a sentence-tokenized copy of `text1`. Report the average number of whitespace separated tokens per sentence in the sentence-tokenized copy of `text1`.

*This function should return a float.*


```python
def answer_seven():

    return len(nltk.word_tokenize(plots_raw))/len(nltk.sent_tokenize(plots_raw))
        
answer_seven()
```




    22.31737990225295




```python

```

### Question 8

What are the 5 most frequent parts of speech in `text1`? What is their frequency?

*This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*


```python
from collections import Counter
def answer_eight():
    tokens = nltk.word_tokenize(plots_raw)
   # POS tagging
    tagged = nltk.pos_tag(tokens)
    
    # Count POS tags
    pos_counts = Counter(tag for _, tag in tagged)
    
    # Return top 5 most frequent POS tags
    return pos_counts.most_common(5)
answer_eight()
```




    [('NN', 51452), ('IN', 39225), ('NNP', 38361), ('DT', 34471), ('VBZ', 23799)]




```python

```

## Part 2 - Spelling Recommender

For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

*Each of the three different recommenders will use a different distance measure (outlined below).

Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.


```python
from nltk.corpus import words

correct_spellings = words.words()
```

### Question 9

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**

Refer to:
- [NLTK Jaccard distance](https://www.nltk.org/api/nltk.metrics.distance.html?highlight=jaccard_distance#nltk.metrics.distance.jaccard_distance)
- [NLTK ngrams](https://www.nltk.org/api/nltk.util.html?highlight=ngrams#nltk.util.ngrams)

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
```


```python
def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    dictionary = pd.Series(correct_spellings)
    results = []

    for token in entries:
        candidates = dictionary[dictionary.str.startswith(token[0])]

        best_word = None
        best_score = float("inf")

        token_ngrams = set(ngrams(token, 3))

        for candidate in candidates:
            candidate_ngrams = set(ngrams(candidate, 3))
            score = jaccard_distance(token_ngrams, candidate_ngrams)

            if score < best_score:
                best_score = score
                best_word = candidate

        results.append(best_word)

    return results


answer_nine()
```




    ['corpulent', 'indecence', 'validate']




```python

```

### Question 10

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**

Refer to:
- [NLTK Jaccard distance](https://www.nltk.org/api/nltk.metrics.distance.html?highlight=jaccard_distance#nltk.metrics.distance.jaccard_distance)
- [NLTK ngrams](https://www.nltk.org/api/nltk.util.html?highlight=ngrams#nltk.util.ngrams)

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    dictionary = pd.Series(correct_spellings)
    results = []

    for token in entries:
        candidates = dictionary[dictionary.str.startswith(token[0])]

        best_word = None
        best_score = float("inf")

        token_ngrams = set(ngrams(token, 3))

        for candidate in candidates:
            candidate_ngrams = set(ngrams(candidate, 4))
            score = jaccard_distance(token_ngrams, candidate_ngrams)

            if score < best_score:
                best_score = score
                best_word = candidate

        results.append(best_word)

    return results

    
answer_ten()
```




    ['c', 'i', 'v']




```python

```

### Question 11

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**

Refer to:
- [NLTK edit distance](https://www.nltk.org/api/nltk.metrics.distance.html?highlight=edit_distance#nltk.metrics.distance.edit_distance)

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):


    import pandas as pd
    from nltk.metrics.distance import edit_distance

    dictionary = pd.Series(correct_spellings)
    results = []

    for word in entries:
        first_char = word[0]

        candidates = dictionary[dictionary.str.startswith(first_char)]

        best_match = None
        best_score = float('inf')

        for candidate in candidates:
            score = edit_distance(word, candidate)
            if score < best_score:
                best_score = score
                best_match = candidate

        results.append(best_match)

    return results

answer_eleven()
```




    ['corpulent', 'intendence', 'validate']




```python

```
