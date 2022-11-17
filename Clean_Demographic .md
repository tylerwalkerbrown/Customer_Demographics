# Python Script Matching Zip to Customer Demographic 


```python
import pandas as pd
import os 
import uszipcode
from uszipcode import SearchEngine
import numpy as np
from uszipcode import SearchEngine, SimpleZipcode, ComprehensiveZipcode
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
      warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')



```python
#Changing the current working directory 
os.chdir("Desktop/OCF_Demo_project")
```


```python
#Reading in data
df = pd.read_csv("master_table.csv")
```


```python
#Orginal dataframe 
df1 = df
```


```python
#Finding nulls in order postal (postal code)
df["order postal"].isnull().sum()
```




    0




```python
#Empyt frame to store product and zip 
prod_zip = pd.DataFrame()
```


```python
#Creating search variable
search = SearchEngine()
```


```python
prod_zip[['zip','product','total','quantity']] = df[[ 'order postal' ,'product_cat','total','quantity']]
```


```python
#Splitting to keep the first simple zip
prod_zip['zip'] = prod_zip.zip.str.split('-', expand = True)[0]
```


```python
#Filtering out all zip codes that do not have proper length 
prod_zip = prod_zip[prod_zip.zip.str.len() == 5]
```


```python
#Storijng the zipcodes in a dictionary
data = { 'zipcodes':prod_zip.zip.astype(str).values.tolist(),
        'quantity': prod_zip.quantity.values.tolist(),
        'total': prod_zip.total.astype(str).values.tolist(),
        'product': prod_zip.astype(str).values.tolist(),
       } 
```


```python
#Creating empty lists to store variables in
df = pd.DataFrame(data)
major_city1 = []
county1 = []
state1 = []
population1 = []
population_density1 = []
land_area_in_sqmi1 = []
housing_units1 = []
median_home_value1 = []
median_household_income1 = []
```


```python
for i in np.arange(0, len(df["zipcodes"])):
    zipcode = search.by_zipcode(df["zipcodes"][i])

    # Checking for non std postal codes
    # Demographic info in std postal codes
    if not zipcode.population:
        # Checking for non std zipcodes like postal boxes
        res = search.by_city_and_state( city=zipcode.major_city,state=zipcode.state)
        if (len(res)) > 0:
            zipcode = res[0]
    major_city1.append(zipcode.major_city)
    county1.append(zipcode.county)
    state1.append(zipcode.state)
    population1.append(zipcode.population)
    population_density1.append(zipcode.population_density)
    land_area_in_sqmi1.append(zipcode.land_area_in_sqmi)
    housing_units1.append(zipcode.housing_units)
    median_household_income1.append(zipcode.median_household_income)
```


```python
#Searching demographics
res = search.by_city_and_state(city=zipcode.major_city, state=zipcode.state)
```


```python
#Storing results in column 
df["county"] = county1

df["state"] = state1

df["population"] = population1

df["population_density"] = population_density1

df["land_area_in_sqmi"] = land_area_in_sqmi1

df["housing_units"] = housing_units1

df["median_household_income"] = median_household_income1
```


```python
#Columns 
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9658 entries, 0 to 9657
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   zipcodes                 9658 non-null   object 
     1   quantity                 9658 non-null   int64  
     2   total                    9658 non-null   object 
     3   product                  9658 non-null   object 
     4   county                   9658 non-null   object 
     5   state                    9658 non-null   object 
     6   population               9643 non-null   float64
     7   population_density       9643 non-null   float64
     8   land_area_in_sqmi        9643 non-null   float64
     9   housing_units            9643 non-null   float64
     10  median_household_income  9643 non-null   float64
    dtypes: float64(5), int64(1), object(5)
    memory usage: 830.1+ KB



```python
#Descriptive stats
df.describe()
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
      <th>quantity</th>
      <th>population</th>
      <th>population_density</th>
      <th>land_area_in_sqmi</th>
      <th>housing_units</th>
      <th>median_household_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9658.000000</td>
      <td>9643.000000</td>
      <td>9643.000000</td>
      <td>9643.000000</td>
      <td>9643.000000</td>
      <td>9643.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.161421</td>
      <td>22226.550036</td>
      <td>1647.559369</td>
      <td>87.578353</td>
      <td>9334.095821</td>
      <td>60570.417298</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.688622</td>
      <td>17274.414441</td>
      <td>4659.494654</td>
      <td>215.454811</td>
      <td>6810.178181</td>
      <td>23271.072403</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>106.000000</td>
      <td>0.000000</td>
      <td>0.170000</td>
      <td>54.000000</td>
      <td>11146.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>7498.500000</td>
      <td>101.000000</td>
      <td>14.250000</td>
      <td>3384.000000</td>
      <td>44167.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>19283.000000</td>
      <td>400.000000</td>
      <td>39.110000</td>
      <td>8417.000000</td>
      <td>55208.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>32799.500000</td>
      <td>1851.000000</td>
      <td>93.920000</td>
      <td>13905.000000</td>
      <td>72091.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>25.000000</td>
      <td>111086.000000</td>
      <td>128441.000000</td>
      <td>4798.530000</td>
      <td>47617.000000</td>
      <td>214219.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Storing in frame
df = pd.DataFrame(df)
```
