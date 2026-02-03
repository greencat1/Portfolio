# Assignment 2 - Network Connectivity

In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient. We will also store the timestamp of each email. 


```python
import networkx as nx

#!head assets/email_network.txt
```

### Question 1

Using networkx, load up the directed multigraph from `assets/email_network.txt`. Make sure the node names are strings.

*This function should return a directed multigraph networkx graph.*


```python

```


```python
def answer_one():
    
    with open('assets/email_network.txt', 'r') as  f:
    
        lines = [i[:-1].split('\t')for i in f.readlines()[1:]]
        
    G = nx.MultiDiGraph()
    for i in lines:
        
        G.add_node(i[0])
        G.add_node(i[1])
    
        G.add_weighted_edges_from([(i[0], i[1], int(i[2]))])
    return G
```


```python
ans_one = answer_one()
```


```python

```

### Question 2

How many employees are represented in the network?

How many `sender`->`recipient` pairs of employees are there in the network such that `sender` sent at least one email to `recipient`? Note that even if a `sender` sent multiple messages to a `recipient`, they should only be counted once. You should **not** exclude cases where an employee sent emails to themselves from this [email] count. 

*This function should return a tuple with two integers (#employees, # `sender`->`recipient` pairs).*


```python

```


```python
def answer_two():
    G = answer_one()
    sender_recipt = {}
    for i in G.edges:
        sender_recipt[(i[0],i[1])] = 1
        
    return len(G.nodes()), len(sender_recipt.keys()) 
    
    
```


```python
ans_two = answer_two()

```


```python
ans_two
```




    (167, 5784)



### Question 3

* Part 1. Assume that information in this company can only be exchanged through email.

    When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the reciever, but not viceversa. 

    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?


* Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 

    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?


*This function should return a tuple of bools (part1, part2).*


```python
def answer_three():
    G = answer_one()
    
    return nx.is_strongly_connected(G), nx.is_connected(G.to_undirected(G))
```


```python
ans_three = answer_three()

```

### Question 4

How many nodes are in the largest weakly connected component of the graph?

*This function should return an int.*


```python
def answer_four():
    G = answer_one()
    
    comp = nx.weakly_connected_components(G)
    return len(max(comp, key=len))
```


```python
ans_four = answer_four()
```

### Question 5

How many nodes are in the largest strongly connected component?

*This function should return an int*


```python
def answer_five():
    G = answer_one()
    
    comp = nx.strongly_connected_components(G)
    return len(max(comp, key=len))
```


```python
ans_five = answer_five()

```

### Question 6

Using the NetworkX functions `strongly_connected_components` and `subgraph`, find the subgraph of nodes in the largest strongly connected component. 
Call this graph G_sc.

*This function should return a networkx MultiDiGraph named G_sc.*


```python
def answer_six():
    G = answer_one()
    
    res = max(nx.strongly_connected_components(G), key=len)
    
    
    
    return G.subgraph(res)
```


```python
ans_six = answer_six()
assert type(ans_six) == nx.MultiDiGraph , "Your return type should be a MultiDiGraph object"

```

### Question 7

What is the average distance between nodes in G_sc?

*This function should return a float.*


```python
def answer_seven():
    G = answer_six()
    
    return nx.average_shortest_path_length(G)
    
```


```python
ans_seven = answer_seven()

```

### Question 8

What is the largest possible distance between two employees in G_sc?

*This function should return an int.*


```python
def answer_eight():
    G = answer_six()
    
    return nx.diameter(G)
```


```python
ans_eight = answer_eight()

```

### Question 9

What is the set of nodes in G_sc with eccentricity equal to the diameter?

*This function should return a set of the node(s).*


```python
def answer_nine():
    G = answer_six()
    return set(nx.periphery(G))
```


```python
ans_nine = answer_nine()
assert type(ans_nine) == set, "Student answer must return a set"
```

### Question 10

What is the set of node(s) in G_sc with eccentricity equal to the radius?

*This function should return a set of the node(s).*


```python
def answer_ten():
    G = answer_six()
    return set(nx.center(G))
```


```python
ans_ten = answer_ten()
assert type(ans_ten) == set, "Student answer must return a set"
```

### Question 11

Which node in G_sc has the most shortest paths to other nodes whose distance equal the diameter of G_sc?


For the node with the most such shortest paths, how many of these paths are there?


*This function should return a tuple (name of node, number of paths).*


```python
def answer_eleven():
    
    G_sc = answer_six()
    
    graph_diameter = nx.diameter(G_sc)
    boundary_nodes = nx.periphery(G_sc)

    best_node = None
    best_score = -1

    for v in boundary_nodes:
        distances = nx.shortest_path_length(G_sc, source=v)
        
        far_count = sum(1 for dist in distances.values() if dist == graph_diameter)

        if far_count > best_score:
            best_score = far_count
            best_node = v

    return best_node, best_score
```


```python
ans_eleven = answer_eleven()
assert type(ans_eleven) == tuple, "Student answer must be a tuple"
```

### Question 12

Suppose you want to prevent communication flow from the node that you found in question 11 to node 10. What is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or 10)? 

*This function should return an integer.*


```python
def answer_twelve():
    G_sc = answer_six()
    
    central_node = next(iter(nx.center(G_sc)))
    

    target_node, _ = answer_eleven()
    

    cut_set = nx.minimum_node_cut(G_sc, central_node, target_node)
    
    return len(cut_set)
```


```python
ans_twelve = answer_twelve()

```

### Question 13

Convert the graph G_sc into an undirected graph by removing the direction of the edges of G_sc. Call the new graph G_un. 


*This function should return a networkx Graph.*


```python
def answer_thirteen():
    G = answer_six()
    return nx.Graph(G.to_undirected())
```


```python
ans_thirteen = answer_thirteen()
assert type(ans_thirteen) == nx.Graph , "Your return type should be a Graph object"

```

### Question 14

What is the transitivity and average clustering coefficient of graph G_un?

*This function should return a tuple (transitivity, avg clustering).*     
*Note: DO NOT round up your answer.*


```python
def answer_fourteen():
    G = answer_thirteen()
    return nx.transitivity(G), nx.average_clustering(G)
```


```python
ans_fourteen = answer_fourteen()
assert type(ans_fourteen) == tuple, "Student answer must be a tuple"

```


```python

```
