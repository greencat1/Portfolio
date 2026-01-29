# Assignment 1

In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 

Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 

Here is a list of some of the variants you might encounter in this dataset:
* 04/20/2009; 04/20/09; 4/20/09; 4/3/09
* Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
* 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
* Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
* Feb 2009; Sep 2009; Oct 2010
* 6/2008; 12/2009
* 2009; 2010

Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
* Assume all dates in xx/xx/xx format are mm/dd/yy
* Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
* If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
* If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
* Watch out for potential typos as this is a raw, real-life derived dataset.

With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices. **This Series should be sorted by a tie-break sort in the format of ("extracted date", "original row number").**

For example if the original series was this:

    0    1999
    1    2010
    2    1978
    3    2015
    4    1985

Your function should return this:

    0    2
    1    4
    2    0
    3    1
    4    3

Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.

*This function should return a Series of length 500 and dtype int.*


```python
import pandas as pd

doc = []
with open('assets/dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)
```

- 04/20/2009; 04/20/09; 4/20/09; 4/3/09
- Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
- 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
- Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
- Feb 2009; Sep 2009; Oct 2010
- 6/2008; 12/2009
- 2009; 2010


```python
import pandas as pd
import numpy as np

def date_sorter():
    global df
    
    # 1. Регулярные выражения в строгом порядке приоритетности
    # Группы: month, day, year
    p1 = r'(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})'
    p2 = r'(?P<day>\d{1,2}) ?(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,.]+(?P<year>\d{4})'
    p3 = r'(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,.-]+(?P<day>\d{1,2})[a-z\s,.-]+(?P<year>\d{4})'
    p4 = r'(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,.]+(?P<year>\d{4})'
    p5 = r'(?P<month>\d{1,2})/(?P<year>\d{4})'
    p6 = r'(?P<year>[12]\d{3})'

    patterns = [p1, p2, p3, p4, p5, p6]
    
    found_data = pd.DataFrame()
    unprocessed = df.index
    
    # 2. Поэтапное извлечение
    for p in patterns:
        extracts = df.loc[unprocessed].str.extract(p)
        # Находим строки, где извлекся хотя бы год
        valid = extracts[extracts['year'].notna()]
        found_data = pd.concat([found_data, valid])
        unprocessed = df.index.difference(found_data.index)
    
    # 3. Нормализация данных
    # Заполнение пропусков
    found_data['day'] = found_data['day'].fillna('1')
    found_data['month'] = found_data['month'].fillna('1')
    
    # Словарь для текстовых месяцев
    m_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    
    def convert_month(m):
        if str(m).isdigit():
            return int(m)
        # Берем первые 3 буквы, приводим к Title Case
        short_m = str(m)[:3].title()
        return m_map.get(short_m, 1)

    found_data['month'] = found_data['month'].apply(convert_month)
    
    # Исправление года (двузначные в четырехзначные)
    found_data['year'] = found_data['year'].apply(lambda x: '19'+str(x) if len(str(x))==2 else str(x))
    
    # 4. Создание финальной колонки даты для сортировки
    # Принудительно переводим в числа, чтобы избежать ошибок форматирования
    found_data['day'] = pd.to_numeric(found_data['day'], errors='coerce').fillna(1).astype(int)
    found_data['month'] = pd.to_numeric(found_data['month'], errors='coerce').fillna(1).astype(int)
    found_data['year'] = pd.to_numeric(found_data['year'], errors='coerce').fillna(1900).astype(int)
    
    # Создаем datetime
    found_data['dt'] = pd.to_datetime(found_data[['year', 'month', 'day']], errors='coerce')
    
    # 5. Сортировка
    # Нам нужно вернуть индексы исходного df в порядке возрастания дат
    # Используем mergesort для стабильности (важно для тестов)
    result = found_data.sort_values(by=['dt', 'year'], kind='mergesort').index
    
    return pd.Series(result)
```


```python

```


```python

```
