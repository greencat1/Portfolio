# Module 1 Assignment

In this module, you learned about topics such as homophily, structural  holes, and K-core decomposition. In this assignment, you will use NetworkX's library of functions relating to these topics. 



```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

```

# Part 1: Homophily, Assortative mixing

At football headquarters, we have a network representing the American football season of the year 2001, where nodes are teams and we form an undirected edge between teams that have played against each other at least once during the season. We also have additional about each team as node attributes, including the team ID, team name, number of wins, number of losses, and the conference number. Below is an example of a node in the dataset:

```
node [
    id 1
    label "FloridaState"
    wins 11
    losses 2
    conference "Atlantic Coast"
]
```

This node represents the Florida State team wth ID 1, 11 wins, 2 losses, and conference ID 0. 

In American Football, conferences are groups of teams that are in a league together and play against each other to win the league title. It is not uncommon however, for teams to play a few teams that are outside of their conference. 

# Task 1a:

Somehow, a disgruntled fan has hacked into our system and created a second corrupted graph with randomized edges. We do not know which is which, and we need your help to pick which one is the correct network.

Given that teams in conferences together tend to play each other, we know we can use one of the assortativity functions we learned about this week to deduce which graph is the uncorrupted one. It is your job to figure out which assortativity function and which feature of the nodes are relevant, return the proper coefficient, and decide which is the uncorrupted graph.

The function below should return a tuple containing the coefficient for football1.gml and football2.gml in that order.


```python
G_football1 = nx.read_gml('assets/football1.gml', label='id')

G_football2 = nx.read_gml('assets/football2.gml', label='id')
```


```python
task_id = '1'

## Utility Code will go here
```


```python
def task_1a_solution():
    g1_coeff, g2_coeff = nx.attribute_assortativity_coefficient(G_football1,'conference'), nx.attribute_assortativity_coefficient(G_football2,'conference') 
    
    
    return g1_coeff, g2_coeff
```


```python
# Use this cell to explore your solution.

task_1a_solution()
```




    (0.6275381679111909, 0.009602406049995278)




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_1a_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, tuple
), f"Task {task_id}: Your function should return a tuple."

assert len(stu_ans) == 2, f"Task {task_id}: Your returned tuple should contain exactly two elements."

assert all(
    isinstance(num, float) for num in stu_ans
), f"Task {task_id}: Both elements in the returned tuple should be floats."

assert all(
    -1.0 <= num <= 1.0 for num in stu_ans
), f"Task {task_id}: Assortativity coefficients should be between -1 and 1."

```

    Task 1 - AG tests
    Task 1 - your answer:
    (0.6275381679111909, 0.009602406049995278)


# Deciding Real_G_football

Now, given the assortativity coefficient, you should be able to deduce which network is the real one. Assign the real one to Real_G_football below.


```python
Real_G_football = G_football1


```

# Task 1b:

We want to see if football teams in our network tend to play other teams who play a similar number of teams. That is, teams that play many (few) teams play against teams that also play many (few) teams. Think about which assortativity coefficient would help us accomplish this, and return it in the function below as a float.




```python
task_id = '1b'

## Utility Code will go here

```


```python
def task_1b_solution():
    sol = nx.degree_assortativity_coefficient(Real_G_football)

    
    

    return sol
```


```python
# Use this cell to explore your solution.

task_1b_solution()
```




    0.16244224957444287




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_1b_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, float
), f"Task {task_id}: Your function should return a float."

assert (
    -1.0 <= stu_ans <= 1.0
), f"Task {task_id}: The degree assortativity coefficient should be between -1 and 1."

```

    Task 1b - AG tests
    Task 1b - your answer:
    0.16244224957444287


# Task 1c: 

As a final task for this task, we want to know if teams tend to play against teams with similar win counts. Think about which coefficient score would help us measure this, and then return that score in the function below as a float.


```python
task_id = '1c'

## Utility Code will go here


```


```python
def task_1c_solution():
    sol = nx.numeric_assortativity_coefficient(Real_G_football, "wins")
    
    

    return sol
```


```python
# Use this cell to explore your solution.

task_1c_solution()
```




    -0.049806582644503085




```python

print(f"Task {task_id} - AG tests")

stu_ans = task_1c_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, float
), f"Task {task_id}: Your function should return a float."

assert (
    -1.0 <= stu_ans <= 1.0
), f"Task {task_id}: The numeric assortativity coefficient should be between -1 and 1."

```

    Task 1c - AG tests
    Task 1c - your answer:
    -0.049806582644503085


# Part 2: Structural holes

# Task 2a: 

In American football, teams usually play within their conference. However, some teams occasionally play against teams from other conferences. These teams can be viewed as brokers spanning structural holes. 

Your task here is to find the brokers in our network and return a dictionary containing only the brokers, where the key is a string representing the name of the college, and the value is a float representing the constraint coefficient. 

For this task, we will consider a team to be a broker if they have a constraint coefficient of less than 0.15.



```python
task_id = '2a'

## Utility Code will go here


```


```python
def task_2a_solution():
    threshold = 0.15
    constraints = nx.constraint(Real_G_football)
    brokers = {Real_G_football.nodes[i]['label']:k for i,k in constraints.items() if k <threshold}
    # YOUR CODE HERE
    

    return brokers
```


```python
# Use this cell to explore your solution.

task_2a_solution()
```




    {'PennState': 0.14832231723803696,
     'LouisianaTech': 0.1257241289154168,
     'MiddleTennesseeState': 0.14153753443526168,
     'NewMexicoState': 0.14415067487631977,
     'Navy': 0.12577556177856705,
     'NotreDame': 0.11802929748271598}




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_2a_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, dict
), f"Task {task_id}: Your function should return a dictionary."

assert all(
    isinstance(k, str) for k in stu_ans.keys()
), f"Task {task_id}: The dictionary keys should be strings representing node labels."

assert all(
    isinstance(v, float) for v in stu_ans.values()
), f"Task {task_id}: The dictionary values should be floats representing constraint scores."

assert all(
    v < 0.15 for v in stu_ans.values()
), f"Task {task_id}: All constraint values should be below 0.15."


```

    Task 2a - AG tests
    Task 2a - your answer:
    {'PennState': 0.14832231723803696, 'LouisianaTech': 0.1257241289154168, 'MiddleTennesseeState': 0.14153753443526168, 'NewMexicoState': 0.14415067487631977, 'Navy': 0.12577556177856705, 'NotreDame': 0.11802929748271598}


# Task 2b

Some fans are criticizing teams who don't play out of their conference, and say that they don't play against other teams so they win more. To ascertain the validity of this statement, we would like you to compute the following: 


- The average win rates of the bottom 10 brokers (`bottom_10_broker_avg_wr` in the function below).
- The average win rates of the top 10 brokers (`top_10_broker_avg_wr` in the function below).
- The correlation coefficient between the win rates of all teams and their constraint coefficients  (`correlation_coefficient` in the function below).

Complete the function below so the appropriate values are returned. The return format has already been defined for you below.

Hints:

-  `sorted_constraints_list` contains a list of tuples, where the first value is a node id, and the second is the constraint coefficient associated with that node id.

- Loop through `sorted_constraints_list`'s nodes to create a list containing the win rate of each node. Since `sorted_constraints_list` is sorted, you'll see the nodes in order from smallest constraint coefficient to largest. 

- Now that you have a list containing the win rates of each node, you can take the first 10 and last 10 items in the list you made to calculate the average win rates of the top and bottom 10 brokers. 

- You can also use the win rate list and a list containing `sorted_constraints_list`'s values to calculate the correlation coefficient.

- A lower constraint coefficient means the team is more of a broker. 

- For this question, win rate will equal the number of wins divided by the number of wins plus losses. 


```python
task_id = '2b'

## Utility Code will go here


```


```python
def task_2b_solution():
    constraints = nx.constraint(Real_G_football)
    sorted_constraints_list = (sorted(constraints.items(), key=lambda item: item[1]))

    
    wins = [Real_G_football.nodes[i[0]]['wins'] for i in sorted_constraints_list]
    
    top_10_broker_avg_wr = np.mean(wins[:10])
    bottom_10_broker_avg_wr = np.mean(wins[-10:])
    correlation_coefficient = np.corrcoef(wins, [i[1] for i in sorted_constraints_list])[0,1]
    return top_10_broker_avg_wr, bottom_10_broker_avg_wr, correlation_coefficient
```


```python
# Use this cell to explore your solution.

task_2b_solution()
```




    (5.5, 6.3, 0.0750729112210732)




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_2b_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(stu_ans, tuple), f"Task {task_id}: Your function should return a tuple."
assert len(stu_ans) == 3, f"Task {task_id}: Your returned tuple should have three elements."

top_10_broker_avg_wr, bottom_10_broker_avg_wr, correlation_coefficient = stu_ans

assert isinstance(
    top_10_broker_avg_wr, float
), f"Task {task_id}: The first element of the returned tuple should be a float."
assert isinstance(
    bottom_10_broker_avg_wr, float
), f"Task {task_id}: The second element of the returned tuple should be a float."

assert isinstance(
    correlation_coefficient, float
), f"Task {task_id}: The third element of the returned tuple should be a float."
assert -1 <= correlation_coefficient <= 1, f"Task {task_id}: The correlation coefficient should be between -1 and 1."

```

    Task 2b - AG tests
    Task 2b - your answer:
    (5.5, 6.3, 0.0750729112210732)


# Task 2b Supplemental Question:

Do the top brokers tend to win more? Return the answer as a true or false boolean value in the function below.


```python
task_id = '2bs'

def task_2b_supplemental_solution():
    sol = False
    
    

    return sol
```


```python
print(f"Task {task_id} - AG tests")

stu_ans = task_2b_supplemental_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, bool
), f"Task {task_id}: Your function should return a bool."

```

    Task 2bs - AG tests
    Task 2bs - your answer:
    False


# Task 2c:


Due to budget cuts, the NCAA must remove three teams from the league. To minimize the impact on the diversity of matchups, the organization aims to cut the teams that contribute the least unique connectivity within the game network.

In network terms, a team’s effective size reflects the extent to which its opponents provide unique rather than redundant connections. A team with a low effective size primarily competes against teams that already play against each other frequently, meaning its removal would have minimal impact on the variety of matchups in the league.

To determine which teams should be removed, identify the three teams with the lowest effective size scores. Return the names of these teams as a list of strings, ordered from the lowest to highest effective size value.


```python
task_id = '2c'

## Utility Code will go here


```


```python
def task_2c_solution():
    sol = sorted(nx.effective_size(Real_G_football).items(), key = lambda x: x[1])

    sol = [Real_G_football.nodes[i[0]]['label'] for i in sol[:3]]
   
    return sol
```


```python
task_2c_solution()
```




    ['WakeForest', 'Virginia', 'Clemson']




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_2c_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, list
), f"Task {task_id}: Your function should return a list."

assert len(stu_ans) == 3, f"Task {task_id}: Your list should contain exactly 3 elements."

assert all(
    isinstance(team, str) for team in stu_ans
), f"Task {task_id}: All elements in the list should be strings (team labels)."


```

    Task 2c - AG tests
    Task 2c - your answer:
    ['WakeForest', 'Virginia', 'Clemson']


# Part 3: Dolphins

# Task 3a:

Dolphins are social creatures. We have a network of friendships stored in the file dolphins.gml. However, due to experimentation by the evil Dr. Romero, the dolphins become very judgemental and start kicking dolphins who don't have a certain number of friends from their pods. On day 0, the dolphins kick out every dolphin that doesn't have at least 0 friends, so every dolphin survives. On day 1, the dolphins kick out every dolphin that doesn't have at least 1 friend. On day two, they kick out every dolphin who doesn't have at least two friends.  This goes on until all dolphins will be kicked out if they increase the required number of friends to stay in the pod. 

Note: If a dolphin doesn't have enough friends after its friends are kicked out on a day, that dolphin is also kicked out on that same day. For example, let's say on day two, dolphin A is friends with only dolphins B and C, and dolphins B and C are friends with only dolphin A. When the dolphins are kicked out, dolphins B and C will be kicked out because they only have one friend, dolphin A. But then, dolphin A will have no friends left, so dolphin A will be kicked from the group as well. 

You need to make and return a list of lists, where the first list contains the names of the dolphins that survive the first day, and the second list contains the names of those who survive the second day, and so on. You should only keep adding lists while there are still survivors (you should not have an empty list at the end of the list)

You should also return a list containing the percentage of dolphins that have survived each day while there are still survivors (you should not end the list with 0.0). When calculating the survival percentage list, use the total number of dolphins in the dataset as the source of comparison. 

Ex. If we started with 10 dolphins and 7 dolphins survived the first day and 5 were left on the second, our list would be [1.0, 0.7, 0.5].

Hint: Which algorithm from this module's lectures would help you accomplish this easily?



```python
G_dolphins = nx.read_gml("assets/dolphins.gml")
```


```python
task_id = '3a'

## Utility Code will go here

```


```python
def task_3a_solution():
    daily_survival_list = []
    daily_survival_percentage = []

    number_of_dolphins = 1
    day = 1
    total_number_of_dolphins = len(list(G_dolphins.nodes))
    while number_of_dolphins != 0:

        
        survivors = nx.k_core(G_dolphins, k=day)
        day = day + 1
        daily_survival_list.append(list(survivors.nodes))
        number_of_dolphins = len(daily_survival_list[-1])
        daily_survival_percentage.append(number_of_dolphins/
                                        total_number_of_dolphins)

    
    daily_survival_list = daily_survival_list[:-1]
    daily_survival_percentage = daily_survival_percentage[:-1]
    return daily_survival_list, daily_survival_percentage
```


```python
task_3a_solution()
```




    ([['Beak',
       'Beescratch',
       'Bumper',
       'CCL',
       'Cross',
       'DN16',
       'DN21',
       'DN63',
       'Double',
       'Feather',
       'Fish',
       'Five',
       'Fork',
       'Gallatin',
       'Grin',
       'Haecksel',
       'Hook',
       'Jet',
       'Jonah',
       'Knit',
       'Kringel',
       'MN105',
       'MN23',
       'MN60',
       'MN83',
       'Mus',
       'Notch',
       'Number1',
       'Oscar',
       'Patchback',
       'PL',
       'Quasi',
       'Ripplefluke',
       'Scabs',
       'Shmuddel',
       'SMN5',
       'SN100',
       'SN4',
       'SN63',
       'SN89',
       'SN9',
       'SN90',
       'SN96',
       'Stripes',
       'Thumper',
       'Topless',
       'TR120',
       'TR77',
       'TR82',
       'TR88',
       'TR99',
       'Trigger',
       'TSN103',
       'TSN83',
       'Upbang',
       'Vau',
       'Wave',
       'Web',
       'Whitetip',
       'Zap',
       'Zig',
       'Zipfel'],
      ['Beak',
       'Beescratch',
       'Bumper',
       'CCL',
       'DN16',
       'DN21',
       'DN63',
       'Double',
       'Feather',
       'Fish',
       'Gallatin',
       'Grin',
       'Haecksel',
       'Hook',
       'Jet',
       'Jonah',
       'Knit',
       'Kringel',
       'MN105',
       'MN60',
       'MN83',
       'Mus',
       'Notch',
       'Number1',
       'Oscar',
       'Patchback',
       'PL',
       'Ripplefluke',
       'Scabs',
       'Shmuddel',
       'SN100',
       'SN4',
       'SN63',
       'SN89',
       'SN9',
       'SN90',
       'SN96',
       'Stripes',
       'Thumper',
       'Topless',
       'TR120',
       'TR77',
       'TR88',
       'TR99',
       'Trigger',
       'TSN103',
       'TSN83',
       'Upbang',
       'Vau',
       'Wave',
       'Web',
       'Zap',
       'Zipfel'],
      ['Beak',
       'Beescratch',
       'Bumper',
       'CCL',
       'DN16',
       'DN21',
       'DN63',
       'Double',
       'Feather',
       'Fish',
       'Gallatin',
       'Grin',
       'Haecksel',
       'Hook',
       'Jet',
       'Jonah',
       'Knit',
       'Kringel',
       'MN105',
       'MN60',
       'MN83',
       'Mus',
       'Notch',
       'Number1',
       'Oscar',
       'Patchback',
       'PL',
       'Scabs',
       'Shmuddel',
       'SN100',
       'SN4',
       'SN63',
       'SN9',
       'SN90',
       'SN96',
       'Stripes',
       'Thumper',
       'Topless',
       'TR77',
       'TR99',
       'Trigger',
       'TSN103',
       'Upbang',
       'Web',
       'Zap'],
      ['Beak',
       'Beescratch',
       'DN21',
       'DN63',
       'Double',
       'Feather',
       'Fish',
       'Gallatin',
       'Grin',
       'Haecksel',
       'Hook',
       'Jet',
       'Jonah',
       'Knit',
       'Kringel',
       'MN105',
       'MN83',
       'Oscar',
       'Patchback',
       'PL',
       'Scabs',
       'SN100',
       'SN4',
       'SN63',
       'SN9',
       'SN90',
       'SN96',
       'Stripes',
       'Topless',
       'TR77',
       'TR99',
       'Trigger',
       'TSN103',
       'Upbang',
       'Web',
       'Zap']],
     [1.0, 0.8548387096774194, 0.7258064516129032, 0.5806451612903226])




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_3a_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(stu_ans, tuple), f"Task {task_id}: Your function should return a tuple."
assert len(stu_ans) == 2, f"Task {task_id}: Your returned tuple should have two elements."

daily_survival_list, daily_survival_percentage = stu_ans

assert isinstance(
    daily_survival_list, list
), f"Task {task_id}: The first element of the returned tuple should be a list."
assert all(
    isinstance(day_list, list) for day_list in daily_survival_list
), f"Task {task_id}: Each element in `daily_survival_list` should be a list."

assert isinstance(
    daily_survival_percentage, list
), f"Task {task_id}: The second element of the returned tuple should be a list."
assert all(
    isinstance(p, float) for p in daily_survival_percentage
), f"Task {task_id}: Each element in `daily_survival_percentage` should be a float."

assert all(
    0 <= p <= 1 for p in daily_survival_percentage
), f"Task {task_id}: Values in `daily_survival_percentage` should be between 0 and 1."


```

    Task 3a - AG tests
    Task 3a - your answer:
    ([['Beak', 'Beescratch', 'Bumper', 'CCL', 'Cross', 'DN16', 'DN21', 'DN63', 'Double', 'Feather', 'Fish', 'Five', 'Fork', 'Gallatin', 'Grin', 'Haecksel', 'Hook', 'Jet', 'Jonah', 'Knit', 'Kringel', 'MN105', 'MN23', 'MN60', 'MN83', 'Mus', 'Notch', 'Number1', 'Oscar', 'Patchback', 'PL', 'Quasi', 'Ripplefluke', 'Scabs', 'Shmuddel', 'SMN5', 'SN100', 'SN4', 'SN63', 'SN89', 'SN9', 'SN90', 'SN96', 'Stripes', 'Thumper', 'Topless', 'TR120', 'TR77', 'TR82', 'TR88', 'TR99', 'Trigger', 'TSN103', 'TSN83', 'Upbang', 'Vau', 'Wave', 'Web', 'Whitetip', 'Zap', 'Zig', 'Zipfel'], ['Beak', 'Beescratch', 'Bumper', 'CCL', 'DN16', 'DN21', 'DN63', 'Double', 'Feather', 'Fish', 'Gallatin', 'Grin', 'Haecksel', 'Hook', 'Jet', 'Jonah', 'Knit', 'Kringel', 'MN105', 'MN60', 'MN83', 'Mus', 'Notch', 'Number1', 'Oscar', 'Patchback', 'PL', 'Ripplefluke', 'Scabs', 'Shmuddel', 'SN100', 'SN4', 'SN63', 'SN89', 'SN9', 'SN90', 'SN96', 'Stripes', 'Thumper', 'Topless', 'TR120', 'TR77', 'TR88', 'TR99', 'Trigger', 'TSN103', 'TSN83', 'Upbang', 'Vau', 'Wave', 'Web', 'Zap', 'Zipfel'], ['Beak', 'Beescratch', 'Bumper', 'CCL', 'DN16', 'DN21', 'DN63', 'Double', 'Feather', 'Fish', 'Gallatin', 'Grin', 'Haecksel', 'Hook', 'Jet', 'Jonah', 'Knit', 'Kringel', 'MN105', 'MN60', 'MN83', 'Mus', 'Notch', 'Number1', 'Oscar', 'Patchback', 'PL', 'Scabs', 'Shmuddel', 'SN100', 'SN4', 'SN63', 'SN9', 'SN90', 'SN96', 'Stripes', 'Thumper', 'Topless', 'TR77', 'TR99', 'Trigger', 'TSN103', 'Upbang', 'Web', 'Zap'], ['Beak', 'Beescratch', 'DN21', 'DN63', 'Double', 'Feather', 'Fish', 'Gallatin', 'Grin', 'Haecksel', 'Hook', 'Jet', 'Jonah', 'Knit', 'Kringel', 'MN105', 'MN83', 'Oscar', 'Patchback', 'PL', 'Scabs', 'SN100', 'SN4', 'SN63', 'SN9', 'SN90', 'SN96', 'Stripes', 'Topless', 'TR77', 'TR99', 'Trigger', 'TSN103', 'Upbang', 'Web', 'Zap']], [1.0, 0.8548387096774194, 0.7258064516129032, 0.5806451612903226])


# Task 3b: 

In our graph, we have a 'smelliness' attribute for each dolphin represented as an integer. 

We'd like to conduct a study on the "smelliness" factor of dolphins and examine the correlation between the smelliness of dolphins and their ability to last in the community. Compute the average smelliness of dolphins that have survived each day, which will be returned in a list, where the first value is the average smelliness of all dolphins on day 0, the second is the average smelliness of all dolphins on day 1, and so on. Also, compute the correlation coefficient between a dolphin's number of days survived and the smelliness factor and return this as a float. The tutorial provides an example of how to accomplish this.

Does it appear that smelly dolphins last longer?

Disclaimer: The “smelliness” data and the survival mechanisms described in this question are entirely fictional and were created solely for educational purposes. Additionally, no dolphins were harmed in the creation of this exercise.



```python
task_id = '3b'

## Utility Code will go here

```


```python
def task_3b_solution():
    daily_survivors, _ = task_3a_solution()

    daily_avg_smelliness = []
    #daily_avg_smelliness.append(np.mean([G_dolphins.nodes[i]['smelliness'] for i in list(G_dolphins.nodes)]))

    for i in daily_survivors:

        daily_avg_smelliness.append(np.mean([G_dolphins.nodes[k]['smelliness'] for k in i]))
    
    days_survived = {node: 0 for node in G_dolphins.nodes()}

    for day_set in daily_survivors:
        for node in day_set:
            days_survived[node] += 1

    
    survival_days = []
    smelliness_values = []

    for node in G_dolphins.nodes():
        survival_days.append(days_survived[node])
        smelliness_values.append(G_dolphins.nodes[node]['smelliness'])


    correlation_coefficient = np.corrcoef(
        survival_days,
        smelliness_values
    )[0, 1]

    
    return daily_avg_smelliness, correlation_coefficient
```


```python
# Use this cell to explore your solution.

task_3b_solution()
```




    ([31.612903225806452, 36.15094339622642, 40.15555555555556, 44.94444444444444],
     0.8640632859507918)




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_3b_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(stu_ans, tuple), f"Task {task_id}: Your function should return a tuple."
assert len(stu_ans) == 2, f"Task {task_id}: Your returned tuple should have two elements."

daily_avg_smelliness, correlation_coefficient = stu_ans

assert isinstance(
    daily_avg_smelliness, list
), f"Task {task_id}: The first element of the returned tuple should be a list."
assert all(
    isinstance(smell, float) for smell in daily_avg_smelliness
), f"Task {task_id}: Each element in `daily_avg_smelliness` should be a float."

assert isinstance(
    correlation_coefficient, float
), f"Task {task_id}: The second element of the returned tuple should be a float."

assert -1 <= correlation_coefficient <= 1, f"Task {task_id}: The correlation coefficient should be between -1 and 1."


```

    Task 3b - AG tests
    Task 3b - your answer:
    ([31.612903225806452, 36.15094339622642, 40.15555555555556, 44.94444444444444], 0.8640632859507918)



```python

```
