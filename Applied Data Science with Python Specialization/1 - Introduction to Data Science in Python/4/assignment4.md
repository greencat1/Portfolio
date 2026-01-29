# Assignment 4
## Description
In this assignment you must read in a file of metropolitan regions and associated sports teams from [assets/wikipedia_data.html](assets/wikipedia_data.html) and answer some questions about each metropolitan region. Each of these regions may have one or more teams from the "Big 4": NFL (football, in [assets/nfl.csv](assets/nfl.csv)), MLB (baseball, in [assets/mlb.csv](assets/mlb.csv)), NBA (basketball, in [assets/nba.csv](assets/nba.csv) or NHL (hockey, in [assets/nhl.csv](assets/nhl.csv)). Please keep in mind that all questions are from the perspective of the metropolitan region, and that this file is the "source of authority" for the location of a given sports team. Thus teams which are commonly known by a different area (e.g. "Oakland Raiders") need to be mapped into the metropolitan region given (e.g. San Francisco Bay Area). This will require some human data understanding outside of the data you've been given (e.g. you will have to hand-code some names, and might need to google to find out where teams are)!

For each sport I would like you to answer the question: **what is the win/loss ratio's correlation with the population of the city it is in?** Win/Loss ratio refers to the number of wins over the number of wins plus the number of losses. Remember that to calculate the correlation with [`pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html), so you are going to send in two ordered lists of values, the populations from the wikipedia_data.html file and the win/loss ratio for a given sport in the same order. Average the win/loss ratios for those cities which have multiple teams of a single sport. Each sport is worth an equal amount in this assignment (20%\*4=80%) of the grade for this assignment. You should only use data **from year 2018** for your analysis -- this is important!

## Notes

1. Do not include data about the MLS or CFL in any of the work you are doing, we're only interested in the Big 4 in this assignment.
2. I highly suggest that you first tackle the four correlation questions in order, as they are all similar and worth the majority of grades for this assignment. This is by design!
3. It's fair game to talk with peers about high level strategy as well as the relationship between metropolitan areas and sports teams. However, do not post code solving aspects of the assignment (including such as dictionaries mapping areas to teams, or regexes which will clean up names).
4. There may be more teams than the assert statements test, remember to collapse multiple teams in one city into a single value!

As this assignment utilizes global variables in the skeleton code, to avoid having errors in your code you can either:

1. You can place all of your code within the function definitions for all of the questions (other than import statements).
2. You can create copies of all the global variables with the copy() method and proceed as usual.

## Question 1
For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NHL** using **2018** data.


```python
my_dict = {
  "New Jersey Devils": "New York City",
  "New Jersey Devils*": "New York City",
  "Carolina Hurricanes": "Raleigh",
  "Vancouver Canucks": "Vancouver",
  "Florida Panthers": "Miami–Fort Lauderdale",
  "Tampa Bay Lightning": "Tampa Bay Area",
  "Tampa Bay Lightning*": "Tampa Bay Area",
  "Buffalo Sabres": "Buffalo",
  "Arizona Coyotes": "Phoenix",
  "Colorado Avalanche": "Denver",
  "Colorado Avalanche*": "Denver",
  "Montreal Canadiens": "Montreal",
  "Detroit Red Wings": "Detroit",
  "Toronto Maple Leafs": "Toronto",
  "Toronto Maple Leafs*": "Toronto",
  "San Jose Sharks": "San Francisco Bay Area",
  "San Jose Sharks*": "San Francisco Bay Area",
  "Winnipeg Jets": "Winnipeg",
  "Winnipeg Jets*": "Winnipeg",
  "Columbus Blue Jackets": "Columbus",
  "Columbus Blue Jackets*": "Columbus",
  "St. Louis Blues": "St. Louis",
  "Los Angeles Kings": "Los Angeles",
  "Los Angeles Kings*": "Los Angeles",
  "Philadelphia Flyers": "Philadelphia",
  "Philadelphia Flyers*": "Philadelphia",
  "Boston Bruins": "Boston",
  "Boston Bruins*": "Boston",
  "Chicago Blackhawks": "Chicago",
  "Washington Capitals": "Washington, D.C.",
  "Washington Capitals*": "Washington, D.C.",
  "Nashville Predators": "Nashville",
  "Nashville Predators*": "Nashville",
  "Dallas Stars": "Dallas–Fort Worth",
  "New York Islanders": "New York City",
  "Vegas Golden Knights": "Las Vegas",
  "Vegas Golden Knights*": "Las Vegas",
  "New York Rangers": "New York City",
  "Pittsburgh Penguins": "Pittsburgh",
  "Pittsburgh Penguins*": "Pittsburgh",
  "Calgary Flames": "Calgary",
  "Ottawa Senators": "Ottawa",
  "Minnesota Wild": "Minneapolis–Saint Paul",
  "Minnesota Wild*": "Minneapolis–Saint Paul",
  "Anaheim Ducks": "Los Angeles",
  "Anaheim Ducks*": "Los Angeles",
  "Edmonton Oilers": "Edmonton"
}
```


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nhl_correlation(): 
    # YOUR CODE HERE
    nhl_df_new = nhl_df[nhl_df['year']==2018].drop([0, 9, 18, 26])
    nhl_df_new['Metropolia'] = nhl_df_new['team'].agg(lambda x: my_dict[x])
    nhl_df_new['W'] =  nhl_df_new['W'].astype(float)
    nhl_df_new['L'] = nhl_df_new['L'].astype(float)
    nhl_df_new = nhl_df_new.groupby('Metropolia').sum()[['W', 'L']]
    nhl_df_new['W/L'] = nhl_df_new['W']/(nhl_df_new['L']+nhl_df_new['W'])
    metropolia = list(nhl_df_new.index)
    
    population_by_region = [] # pass in metropolitan area population from cities
    for i in metropolia:
        
        population_by_region.append(int(cities[cities['Metropolitan area']==i].iloc[0,]['Population (2016 est.)[8]']))
        
    
    win_loss_by_region = list(nhl_df_new['W/L']) # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
    
    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
```


```python

```

## Question 2
For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NBA** using **2018** data.


```python
my_dict = {
  "Indiana Pacers": "Indianapolis",
  "Indiana Pacers\xa0(9)": "Indianapolis",
  "Indiana Pacers*\xa0(7)": "Indianapolis",
  "Indiana Pacers*\xa0(1)": "Indianapolis",
  "Indiana Pacers*\xa0(5)": "Indianapolis",
  "Portland Trail Blazers": "Portland",
  "Portland Trail Blazers*\xa0(8)": "Portland",
  "Portland Trail Blazers*": "Portland",
  "Portland Trail Blazers*\xa0(4)": "Portland",
  "Portland Trail Blazers*\xa0(3)": "Portland",
  "Portland Trail Blazers*\xa0(5)": "Portland",
  "San Antonio Spurs": "San Antonio",
  "San Antonio Spurs*\xa0(7)": "San Antonio",
  "San Antonio Spurs*": "San Antonio",
  "San Antonio Spurs*\xa0(1)": "San Antonio",
  "San Antonio Spurs*\xa0(5)": "San Antonio",
  "San Antonio Spurs*\xa0(2)": "San Antonio",
  "Detroit Pistons": "Detroit",
  "Detroit Pistons\xa0(10)": "Detroit",
  "Detroit Pistons\xa0(9)": "Detroit",
  "Detroit Pistons*\xa0(8)": "Detroit",
  "Detroit Pistons\xa0(11)": "Detroit",
  "Detroit Pistons\xa0(12)": "Detroit",
  "Milwaukee Bucks": "Milwaukee",
  "Milwaukee Bucks\xa0(12)": "Milwaukee",
  "Milwaukee Bucks*\xa0(6)": "Milwaukee",
  "Milwaukee Bucks\xa0(15)": "Milwaukee",
  "Milwaukee Bucks*\xa0(7)": "Milwaukee",
  "Golden State Warriors": "San Francisco Bay Area",
  "Golden State Warriors*\xa0(1)": "San Francisco Bay Area",
  "Golden State Warriors*": "San Francisco Bay Area",
  "Golden State Warriors*\xa0(2)": "San Francisco Bay Area",
  "Golden State Warriors*\xa0(6)": "San Francisco Bay Area",
  "Boston Celtics": "Boston",
  "Boston Celtics*\xa0(2)": "Boston",
  "Boston Celtics*": "Boston",
  "Boston Celtics\xa0(12)": "Boston",
  "Boston Celtics*\xa0(5)": "Boston",
  "Boston Celtics*\xa0(7)": "Boston",
  "Boston Celtics*\xa0(1)": "Boston",
  "Toronto Raptors": "Toronto",
  "Toronto Raptors*\xa0(2)": "Toronto",
  "Toronto Raptors*": "Toronto",
  "Toronto Raptors*\xa0(1)": "Toronto",
  "Toronto Raptors*\xa0(3)": "Toronto",
  "Toronto Raptors*\xa0(4)": "Toronto",
  "New Orleans Pelicans": "New Orleans",
  "New Orleans Pelicans\xa0(10)": "New Orleans",
  "New Orleans Pelicans*\xa0(8)": "New Orleans",
  "New Orleans Pelicans*\xa0(6)": "New Orleans",
  "New Orleans Pelicans\xa0(12)": "New Orleans",
  "Los Angeles Lakers": "Los Angeles",
  "Los Angeles Lakers\xa0(11)": "Los Angeles",
  "Los Angeles Lakers\xa0(15)": "Los Angeles",
  "Los Angeles Lakers\xa0(14)": "Los Angeles",
  "Utah Jazz": "Salt Lake City",
  "Utah Jazz\xa0(11)": "Salt Lake City",
  "Utah Jazz\xa0(9)": "Salt Lake City",
  "Utah Jazz\xa0(15)": "Salt Lake City",
  "Utah Jazz*\xa0(5)": "Salt Lake City",
  "New York Knicks": "New York City",
  "New York Knicks\xa0(11)": "New York City",
  "New York Knicks\xa0(9)": "New York City",
  "New York Knicks\xa0(12)": "New York City",
  "New York Knicks\xa0(13)": "New York City",
  "New York Knicks\xa0(15)": "New York City",
  "Houston Rockets": "Houston",
  "Houston Rockets*\xa0(3)": "Houston",
  "Houston Rockets*": "Houston",
  "Houston Rockets*\xa0(4)": "Houston",
  "Houston Rockets*\xa0(1)": "Houston",
  "Houston Rockets*\xa0(8)": "Houston",
  "Houston Rockets*\xa0(2)": "Houston",
  "Phoenix Suns": "Phoenix",
  "Phoenix Suns\xa0(14)": "Phoenix",
  "Phoenix Suns\xa0(9)": "Phoenix",
  "Phoenix Suns\xa0(15)": "Phoenix",
  "Phoenix Suns\xa0(10)": "Phoenix",
  "Miami Heat": "Miami–Fort Lauderdale",
  "Miami Heat*\xa0(3)": "Miami–Fort Lauderdale",
  "Miami Heat*": "Miami–Fort Lauderdale",
  "Miami Heat*\xa0(6)": "Miami–Fort Lauderdale",
  "Miami Heat\xa0(9)": "Miami–Fort Lauderdale",
  "Miami Heat*\xa0(2)": "Miami–Fort Lauderdale",
  "Miami Heat\xa0(10)": "Miami–Fort Lauderdale",
  "Brooklyn Nets": "New York City",
  "Brooklyn Nets\xa0(12)": "New York City",
  "Brooklyn Nets\xa0(14)": "New York City",
  "Brooklyn Nets\xa0(15)": "New York City",
  "Brooklyn Nets*\xa0(6)": "New York City",
  "Brooklyn Nets*\xa0(8)": "New York City",
  "Oklahoma City Thunder": "Oklahoma City",
  "Oklahoma City Thunder\xa0(9)": "Oklahoma City",
  "Oklahoma City Thunder*": "Oklahoma City",
  "Oklahoma City Thunder*\xa0(2)": "Oklahoma City",
  "Oklahoma City Thunder*\xa0(4)": "Oklahoma City",
  "Oklahoma City Thunder*\xa0(6)": "Oklahoma City",
  "Oklahoma City Thunder*\xa0(3)": "Oklahoma City",
  "Minnesota Timberwolves": "Minneapolis–Saint Paul",
  "Minnesota Timberwolves\xa0(15)": "Minneapolis–Saint Paul",
  "Minnesota Timberwolves\xa0(13)": "Minneapolis–Saint Paul",
  "Minnesota Timberwolves\xa0(10)": "Minneapolis–Saint Paul",
  "Minnesota Timberwolves*\xa0(8)": "Minneapolis–Saint Paul",
  "Atlanta Hawks": "Atlanta",
  "Atlanta Hawks\xa0(15)": "Atlanta",
  "Atlanta Hawks*": "Atlanta",
  "Atlanta Hawks*\xa0(8)": "Atlanta",
  "Atlanta Hawks*\xa0(1)": "Atlanta",
  "Atlanta Hawks*\xa0(4)": "Atlanta",
  "Atlanta Hawks*\xa0(5)": "Atlanta",
  "Dallas Mavericks": "Dallas–Fort Worth",
  "Dallas Mavericks*": "Dallas–Fort Worth",
  "Dallas Mavericks*\xa0(8)": "Dallas–Fort Worth",
  "Dallas Mavericks*\xa0(7)": "Dallas–Fort Worth",
  "Dallas Mavericks*\xa0(6)": "Dallas–Fort Worth",
  "Dallas Mavericks\xa0(13)": "Dallas–Fort Worth",
  "Dallas Mavericks\xa0(11)": "Dallas–Fort Worth",
  "Washington Wizards": "Washington, D.C.",
  "Washington Wizards*": "Washington, D.C.",
  "Washington Wizards*\xa0(8)": "Washington, D.C.",
  "Washington Wizards*\xa0(5)": "Washington, D.C.",
  "Washington Wizards*\xa0(4)": "Washington, D.C.",
  "Washington Wizards\xa0(10)": "Washington, D.C.",
  "Denver Nuggets": "Denver",
  "Denver Nuggets\xa0(10)": "Denver",
  "Denver Nuggets\xa0(9)": "Denver",
  "Denver Nuggets\xa0(11)": "Denver",
  "Denver Nuggets\xa0(12)": "Denver",
  "Philadelphia 76ers": "Philadelphia",
  "Philadelphia 76ers*\xa0(3)": "Philadelphia",
  "Philadelphia 76ers\xa0(14)": "Philadelphia",
  "Philadelphia 76ers\xa0(15)": "Philadelphia",
  "Memphis Grizzlies": "Memphis",
  "Memphis Grizzlies\xa0(14)": "Memphis",
  "Memphis Grizzlies*\xa0(5)": "Memphis",
  "Memphis Grizzlies*\xa0(7)": "Memphis",
  "Charlotte Bobcats*": "Charlotte",
  "Charlotte Bobcats*\xa0(7)": "Charlotte",
  "Cleveland Cavaliers": "Cleveland",
  "Cleveland Cavaliers*": "Cleveland",
  "Cleveland Cavaliers*\xa0(4)": "Cleveland",
  "Cleveland Cavaliers*\xa0(2)": "Cleveland",
  "Cleveland Cavaliers\xa0(10)": "Cleveland",
  "Cleveland Cavaliers*\xa0(1)": "Cleveland",
  "Chicago Bulls": "Chicago",
  "Chicago Bulls*": "Chicago",
  "Chicago Bulls*\xa0(4)": "Chicago",
  "Chicago Bulls\xa0(13)": "Chicago",
  "Chicago Bulls*\xa0(8)": "Chicago",
  "Chicago Bulls*\xa0(3)": "Chicago",
  "Chicago Bulls\xa0(9)": "Chicago",
  "Sacramento Kings": "Sacramento",
  "Sacramento Kings\xa0(10)": "Sacramento",
  "Sacramento Kings\xa0(12)": "Sacramento",
  "Sacramento Kings\xa0(13)": "Sacramento",
  "Orlando Magic": "Orlando",
  "Orlando Magic\xa0(13)": "Orlando",
  "Orlando Magic\xa0(11)": "Orlando",
  "Orlando Magic\xa0(14)": "Orlando",
  "Los Angeles Clippers": "Los Angeles",
  "Los Angeles Clippers*": "Los Angeles",
  "Los Angeles Clippers*\xa0(4)": "Los Angeles",
  "Los Angeles Clippers\xa0(10)": "Los Angeles",
  "Los Angeles Clippers*\xa0(3)": "Los Angeles",
  "Los Angeles Clippers*\xa0(2)": "Los Angeles",
  "Charlotte Hornets": "Charlotte",
  "Charlotte Hornets\xa0(10)": "Charlotte",
  "Charlotte Hornets*\xa0(6)": "Charlotte",
  "Charlotte Hornets\xa0(11)": "Charlotte"
}
```


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nba_correlation():
    # YOUR CODE HERE
    nhl_df_new = nba_df[nba_df['year']==2018]
    nhl_df_new['Metropolia'] = nhl_df_new['team'].agg(lambda x: my_dict[x])
    nhl_df_new['W'] =  nhl_df_new['W'].astype(float)
    nhl_df_new['L'] = nhl_df_new['L'].astype(float)
    nhl_df_new = nhl_df_new.groupby('Metropolia').sum()[['W', 'L']]
    nhl_df_new['W/L'] = nhl_df_new['W']/(nhl_df_new['L']+nhl_df_new['W'])
    metropolia = list(nhl_df_new.index)
    
    population_by_region = [] # pass in metropolitan area population from cities
    for i in metropolia:
        
        population_by_region.append(int(cities[cities['Metropolitan area']==i].iloc[0,]['Population (2016 est.)[8]']))
        
    
    win_loss_by_region = list(nhl_df_new['W/L']) # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    
    
    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
```


```python

```


```python
nba_correlation()
```




    -0.17657160252844614



## Question 3
For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **MLB** using **2018** data.


```python
my_dict = {
  "Boston Red Sox": "Boston",
  "New York Yankees": "New York City",
  "Tampa Bay Rays": "Tampa Bay Area",
  "Toronto Blue Jays": "Toronto",
  "Baltimore Orioles": "Baltimore",
  "Cleveland Indians": "Cleveland",
  "Minnesota Twins": "Minneapolis–Saint Paul",
  "Detroit Tigers": "Detroit",
  "Chicago White Sox": "Chicago",
  "Kansas City Royals": "Kansas City",
  "Houston Astros": "Houston",
  "Oakland Athletics": "San Francisco Bay Area",
  "Seattle Mariners": "Seattle",
  "Los Angeles Angels": "Los Angeles",
  "Texas Rangers": "Dallas–Fort Worth",
  "Atlanta Braves": "Atlanta",
  "Washington Nationals": "Washington, D.C.",
  "Philadelphia Phillies": "Philadelphia",
  "New York Mets": "New York City",
  "Miami Marlins": "Miami–Fort Lauderdale",
  "Milwaukee Brewers": "Milwaukee",
  "Chicago Cubs": "Chicago",
  "St. Louis Cardinals": "St. Louis",
  "Pittsburgh Pirates": "Pittsburgh",
  "Cincinnati Reds": "Cincinnati",
  "Los Angeles Dodgers": "Los Angeles",
  "Colorado Rockies": "Denver",
  "Arizona Diamondbacks": "Phoenix",
  "San Francisco Giants": "San Francisco Bay Area",
  "San Diego Padres": "San Diego"
}
```


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def mlb_correlation(): 
    # YOUR CODE HERE
    nhl_df_new = mlb_df[mlb_df['year']==2018]
    nhl_df_new['Metropolia'] = nhl_df_new['team'].agg(lambda x: my_dict[x])
    nhl_df_new['W'] =  nhl_df_new['W'].astype(float)
    nhl_df_new['L'] = nhl_df_new['L'].astype(float)
    nhl_df_new = nhl_df_new.groupby('Metropolia').sum()[['W', 'L']]
    nhl_df_new['W/L'] = nhl_df_new['W']/(nhl_df_new['L']+nhl_df_new['W'])
    metropolia = list(nhl_df_new.index)
    
    population_by_region = [] # pass in metropolitan area population from cities
    for i in metropolia:
        
        population_by_region.append(int(cities[cities['Metropolitan area']==i].iloc[0,]['Population (2016 est.)[8]']))
        
    
    win_loss_by_region = list(nhl_df_new['W/L']) # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    
    
    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
```


```python

```

## Question 4
For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NFL** using **2018** data.


```python
my_dict = {
  "New England Patriots": "Boston",
  "New England Patriots*": "Boston",
  "Miami Dolphins": "Miami–Fort Lauderdale",
  "Buffalo Bills": "Buffalo",
  "New York Jets": "New York City",
  "Baltimore Ravens": "Baltimore",
  "Baltimore Ravens*": "Baltimore",
  "Pittsburgh Steelers": "Pittsburgh",
  "Cleveland Browns": "Cleveland",
  "Cincinnati Bengals": "Cincinnati",
  "Houston Texans": "Houston",
  "Houston Texans*": "Houston",
  "Indianapolis Colts": "Indianapolis",
  "Indianapolis Colts+": "Indianapolis",
  "Tennessee Titans": "Nashville",
  "Jacksonville Jaguars": "Jacksonville",
  "Kansas City Chiefs": "Kansas City",
  "Kansas City Chiefs*": "Kansas City",
  "Los Angeles Chargers": "Los Angeles",
  "Los Angeles Chargers+": "Los Angeles",
  "Denver Broncos": "Denver",
  "Oakland Raiders": "San Francisco Bay Area", 
  "Dallas Cowboys": "Dallas–Fort Worth",
  "Dallas Cowboys*": "Dallas–Fort Worth",
  "Philadelphia Eagles": "Philadelphia",
  "Philadelphia Eagles+": "Philadelphia",
  "Washington Redskins": "Washington, D.C.",
  "New York Giants": "New York City",
  "Chicago Bears": "Chicago",
  "Chicago Bears*": "Chicago",
  "Minnesota Vikings": "Minneapolis–Saint Paul",
  "Green Bay Packers": "Green Bay",
  "Detroit Lions": "Detroit",
  "New Orleans Saints": "New Orleans",
  "New Orleans Saints*": "New Orleans",
  "Carolina Panthers": "Charlotte",
  "Atlanta Falcons": "Atlanta",
  "Tampa Bay Buccaneers": "Tampa Bay Area",
  "Los Angeles Rams": "Los Angeles",
  "Los Angeles Rams*": "Los Angeles",
  "Seattle Seahawks": "Seattle",
  "Seattle Seahawks+": "Seattle",
  "San Francisco 49ers": "San Francisco Bay Area",
  "Arizona Cardinals": "Phoenix"
}
```


```python
diav  = [
    'Pacific Division',
    'Central Division',
    'Metropolitan Division',
    'Atlantic Division',
    'Southeast Division',
    'Southwest Division',
    'Northwest Division',
    'NFC West',
    'NFC East',
    'NFC North',
    'NFC South',
    'AFC West',
    'AFC South',
    'AFC North',
    'AFC East'
]
```


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nfl_correlation(): 
    # YOUR CODE HERE
    nhl_df_new = nfl_df[nfl_df['year']==2018]
    nhl_df_new = nhl_df_new[~nhl_df_new['team'].isin(diav)]
    
    nhl_df_new['Metropolia'] = nhl_df_new['team'].agg(lambda x: my_dict[x])
    nhl_df_new['W'] =  nhl_df_new['W'].astype(float)
    nhl_df_new['L'] = nhl_df_new['L'].astype(float)
    nhl_df_new = nhl_df_new.groupby('Metropolia').sum()[['W', 'L']]
    nhl_df_new['W/L'] = nhl_df_new['W']/(nhl_df_new['L']+nhl_df_new['W'])
    metropolia = list(nhl_df_new.index)
    
    population_by_region = [] # pass in metropolitan area population from cities
    for i in metropolia:
        
        population_by_region.append(int(cities[cities['Metropolitan area']==i].iloc[0,]['Population (2016 est.)[8]']))
        
    
    win_loss_by_region = list(nhl_df_new['W/L']) # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    
    
    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
```


```python

```

## Question 5
In this question I would like you to explore the hypothesis that **given that an area has two sports teams in different sports, those teams will perform the same within their respective sports**. How I would like to see this explored is with a series of paired t-tests (so use [`ttest_rel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)) between all pairs of sports. Are there any sports where we can reject the null hypothesis? Again, average values where a sport has multiple teams in one region. Remember, you will only be including, for each sport, cities which have teams engaged in that sport, drop others as appropriate. This question is worth 20% of the grade for this assignment.


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
nhl_df=pd.read_csv("assets/nhl.csv")
nba_df=pd.read_csv("assets/nba.csv")
nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def sports_team_performance():
    # YOUR CODE HERE
    raise NotImplementedError()
    
    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k:np.nan for k in sports}, index=sports)
    
    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values
```


```python

```
