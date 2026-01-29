# Assignment 1 - Creating and Manipulating Graphs

Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night. These choices are stored in the file `assets/Employee_Movie_Choices.txt`.

A second file, `assets/Employee_Relationships.txt`, has data on the relationships between different coworkers. 

The relationship score has value of `-100` (Enemies) to `+100` (Best Friends). A value of zero means the two employees haven't interacted or are indifferent.

Both files are tab delimited.


```python
import networkx as nx
import pandas as pd
import numpy as np


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #%matplotlib notebook
    import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, width=weights);
    else:
        nx.draw_networkx(G, pos,);
```

### Question 1

Using NetworkX, load in the bipartite graph from `assets/Employee_Movie_Choices.txt` and return that graph.

*This function should return a bipartite networkx graph with 19 nodes and 24 edges*


```python
def answer_one():

    
    with open('assets/Employee_Movie_Choices.txt', 'r') as f:
        res = f.readlines()
        res = res[1:]
        res = [i[:-1].split('\t') for i in res]

    G1 = nx.Graph()
    G1.add_edges_from(res)
    return G1
    
    
    
 
```


```python
assert type(answer_one()) == nx.Graph , "Your return type should be a Graph object"


```

### Question 2

Using the graph from the previous question, add nodes attributes named `'type'` where movies have the value `'movie'` and employees have the value `'employee'` and return that graph.

*This function should return a bipartite networkx graph with node attributes `{'type': 'movie'}` or `{'type': 'employee'}`*


```python
def answer_two():
    G1 = answer_one()
    
    for i in movies:
        
        G1.nodes[i]['type'] = 'movie'
        
    for i in employees:
        
        
        G1.nodes[i]['type'] = 'employee'
        
    return G1
```


```python
assert type(answer_two()) == nx.Graph , "Your return type should be a Graph object"

```

### Question 3

Find a weighted projection of the graph from `answer_two` which tells us how many movies different pairs of employees have in common.

*This function should return a weighted projected graph.*


```python
def answer_three():
    
    G1 = answer_two()
    return nx.algorithms.bipartite.weighted_projected_graph(G1, employees)
    
```


```python
G = answer_three()
plot_graph(G, weight_name="weight")
G.edges(data = True)
```




    EdgeDataView([('Claude', 'Georgia', {'weight': 3}), ('Claude', 'Andy', {'weight': 1}), ('Joan', 'Lee', {'weight': 3}), ('Joan', 'Andy', {'weight': 1}), ('Vincent', 'Pablo', {'weight': 1}), ('Vincent', 'Frida', {'weight': 2}), ('Georgia', 'Andy', {'weight': 1}), ('Pablo', 'Andy', {'weight': 1}), ('Pablo', 'Frida', {'weight': 2}), ('Andy', 'Lee', {'weight': 1}), ('Andy', 'Frida', {'weight': 1})])




    
![png](output_10_1.png)
    



```python
assert type(answer_three()) == nx.Graph , "Your return type should be a Graph object"

```


```python

```

### Question 4

Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.

Find the pearson correlation between employee relationship scores and the number of movies they have in common. If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.

*This function should return a float.*


```python
def answer_four():
    
    proj = answer_three()
    with open('assets/Employee_Relationships.txt', 'r') as f:
            res = f.readlines()

            res = [i[:-1].split('\t') for i in res]
            res = {(i[0],i[1]):i[2]for i in res}

    relationships = list([int(i) for i in res.values()])
    films_same = []
    for i in res.keys():

        try:

            films_same.append((proj[i[0]][i[1]]['weight']))

        except:

            films_same.append(0)
            
    return np.corrcoef(relationships, films_same)[0,1]
    
```


```python
ans_four = answer_four()
```


```python

```
