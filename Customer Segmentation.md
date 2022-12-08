```python
"""SELECT distinct(`Lineitem name`) FROM Shopify.shopfy_export;
"""
"""with products as (
select *,case when 
`Lineitem name` like 'Oyster Shell%' then 'Oyster Shell'
when `Lineitem name` like '%Winter Rye%' or  '%Grass%' then 'Winter Rye'
when `Lineitem name` like '%Grass%' then 'Winter Rye'
when `Lineitem name` like '%Green Sand%' then 'Greensand'
when `Lineitem name` like '%Alfalfa Pellets%' then 'Alfalfa Pellets'
when `Lineitem name` like '%Aluminum Sulfate%' then 'Aluminum Sulfate' 
when `Lineitem name` like '%Buckwheat%' then 'Buckwheat' end as product
from Shopify.shopfy_export)
select * from products"""

```


```python
import pandas as pd
import os 
import uszipcode
from uszipcode import SearchEngine
import numpy as np
from uszipcode import SearchEngine, SimpleZipcode, ComprehensiveZipcode
#Getting clean zipcode
from geopy.geocoders import Nominatim
from geopy import ArcGIS
import geopandas
import geopy
#Genderize
import gender_guesser.detector as gender

```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
      warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')



```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import plotly.offline as py
```


```python
#Arc object
Arc = ArcGIS()
#Function to get lat and long 
def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']

geolocator = geopy.Nominatim(user_agent='my-application')
```


```python
#Changing directory
os.chdir("Desktop/Shopify")
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [34], in <cell line: 2>()
          1 #Changing directory
    ----> 2 os.chdir("Desktop/Shopify")


    FileNotFoundError: [Errno 2] No such file or directory: 'Desktop/Shopify'



```python
df = pd.read_csv('orders_export_1-3.csv')
```


```python
#Storing orginal frame
df1 = df
```

# Census Data 


```python
#Loading in census data
census = pd.read_csv('ACSST1Y2021.S0101-Data.csv')

```


```python
census['County'] = census['Geographic Area Name'].str.split(',' ,expand = True)[0]
```

# Data Cleansing


```python
#Getting unique values to rename 
np.unique(df['Lineitem name'])
```




    array(['24" Maple Syrup Tapping Kit - 10 pack',
           'Automatic Plant Watering Bulbs Drip Irrigation System - 3pcs Transparent / China',
           'Automatic Plant Watering Bulbs Drip Irrigation System - 5pcs Green / China',
           'Automatic Plant Watering Bulbs Drip Irrigation System - 5pcs Transparent / China',
           'Corn Gluten Weed Prevention by Old Cobblers Farm - 10 lbs',
           'Discount Lawn Care 10LBS Winter Rye Seed Cover Crop,Food Plot Deer,Wildlife',
           'Discount Lawn Care 10LBS Winter Rye Seed Cover Crop,Food Plot Deer,Wildlife - 10 lbs',
           'Drip Irrigation System - Blue 2',
           'Fabric Raised Garden Bed Square Garden - 50x20cm Black',
           'Greensand All-Purpose Fertilizer by Old Cobblers Farm - 5 lbs',
           'Maple Syrup Pre-Filters - 6 Pack of Reemay Filter Cones',
           'Maple Syrup Vacuum Tubing Line 5/16" hose x 500 foot length tubing',
           'Old Cobblers Farm 10 Garden Staple 6" Metal Professional Grade Bulk Wholesale Price Made in America by Americans',
           'Old Cobblers Farm 20-20-20 Fertilizer 2lb (Peters)',
           'Old Cobblers Farm All-Natural Corn Gluten Meal (10lbs)',
           'Old Cobblers Farm All-Natural Corn Gluten Meal (2lb)',
           'Old Cobblers Farm All-Natural Corn Gluten Meal (4lb)',
           'Old Cobblers Farm Aluminum Sulfate 5lbs Pure Aluminum Sulfate',
           'Old Cobblers Farm Buckwheat Seeds, 5lbs',
           'Old Cobblers Farm Empty Maple Syrup Jugs Quart Size 4 Pack with Caps',
           'Old Cobblers Farm Maple Syrup Line 5/16" X 50\' long',
           'Old Cobblers Farm Maple Syrup Taps, 5/16" Pack of 10',
           'Old Cobblers Farm Maple Syrup Vacuum Tubing Line 5/16 hosex50 foot length,Green',
           'Old Cobblers Farm Organic Greensand Fertilizer 5lb',
           'Old Cobblers Farm Rock Phosphate Fertilizer Organic 0-3-0 Fertilizer 10lb',
           'Old Cobblers Farm Rock Phosphate Fertilizer Organic 0-3-0 Fertilizer 5lb',
           'Old Cobblers Farm Rye Grass Seed for Lawn | Fast Growing | Winter Grass Seed | Wildlife - Cover Crop',
           'Old Cobblers Farm Rye Grass Seed for Lawn | Fast Growing | Winter Grass Seed | Wildlife - Cover Crop 10lbs',
           'Old Cobblers Farm Winter Rye Seeds 25lbs, Non-GMO, Cover Crop',
           'Old Cobblers Farm Winter Rye Seeds 5 lbs, Non-GMO, Cover Crop',
           'Old Cobblers Farm Winter Rye Seeds Non-GMO, Cover Crop (10lb)',
           'Old Cobblers Farm Winter Rye Seeds Non-GMO, Cover Crop (20lb)',
           'Old Cobblers Farm Winter Rye Seeds Non-GMO, Cover Crop (25lb)',
           'Old Cobblers Farm Winter Rye Seeds Non-GMO, Cover Crop (5lb)',
           'Oyster Shell (Crushed) 15lb. Bag by Coastal',
           'Oyster Shell (Crushed) 5lb. Bag by Old Cobblers Farm',
           'Oyster Shells by Old Cobblers Farm - 5 lbs',
           'Premium Alfalfa Pellets Animal Feed (Rabbits, Guinea Pig, Etc) 5lbs. by Old Cobblers Farm',
           'Premium Maple Syrup Tapping Kit (20 Sap Line 36" & 20 Spouts Per Kit)',
           'Premium Maple Syrup Tapping Kit (5 Sap Line 30" & 5 Spouts Per Kit)',
           'Premium Maple Syrup Tapping Kit (5 Sap Line 36" & 5 Spouts Per Kit)',
           'Premium Maple Syrup Tapping Kit (Sap Line & Tap 10 Per Kit) 30" Drop Line',
           'Premium Maple Syrup Tapping Kit (Sap Line & Tap 10 Per Kit) 36" Drop Line',
           'Premium Organic Green Sand Fertilizer by Old Cobblers Farm 10lbs.',
           'Premium Organic Green Sand Fertilizer by Old Cobblers Farm 22lb. Bulk Box',
           'Premium Winter Rye Grass Seeds 10 lbs, Non-GMO, Cover Crop',
           'Premium Winter Rye Grass Seeds 1000 Seed Packet, Non-GMO, Cover Crop',
           'Premium Winter Rye Grass Seeds 20 lbs, Non-GMO, Cover Crop',
           'Premium Winter Rye Seeds 5 lbs, Non-GMO, Cover Crop',
           'Self Watering Spike Automatic Drip Irrigation System - China / 6 Pcs Green',
           'Sweet Lovers Couple Chair Figurines Miniatures Fairy Garden Gnome - 1',
           'Winter Rye (Grass Seed) by Old Cobblers Farm - 1000 Seeds',
           'Winter Rye (Grass Seed) by Old Cobblers Farm - 20 lbs',
           'Winter Rye (Grass Seed) by Old Cobblers Farm - 25 lbs',
           'Winter Rye (Grass Seed) by Old Cobblers Farm - 5 lbs'],
          dtype=object)




```python
#Converting name to general product categories 
df.loc[df['Lineitem name'].str.contains('Tapping | Syrup'), 'Lineitem name'] = 'Tapping'
df.loc[df['Lineitem name'].str.contains('Rye | Grass'), 'Lineitem name'] = 'Winter Rye'
df.loc[df['Lineitem name'].str.contains('Corn | Gluten'), 'Lineitem name'] = 'Corn Gluten'
df.loc[df['Lineitem name'].str.contains('Green|Greensand'), 'Lineitem name'] = 'Greensand'
df.loc[df['Lineitem name'].str.contains('Gnome'), 'Lineitem name'] = 'Decoration'
df.loc[df['Lineitem name'].str.contains('Drip | Watering'), 'Lineitem name'] = 'Watering Bulbs'
df.loc[df['Lineitem name'].str.contains('20-20-20'), 'Lineitem name'] = '20-20-20'
df.loc[df['Lineitem name'].str.contains('Rock Phosphate'), 'Lineitem name'] = 'Rock Phosphate'
df.loc[df['Lineitem name'].str.contains('Buckwheat'), 'Lineitem name'] = 'Buckwheat'
df.loc[df['Lineitem name'].str.contains('Oyster Shell'), 'Lineitem name'] = 'Oyster Shell'
df.loc[df['Lineitem name'].str.contains('Aluminum'), 'Lineitem name'] = 'Aluminum Sulfate'
df.loc[df['Lineitem name'].str.contains('Alfalfa'), 'Lineitem name'] = 'Alfalfa'
df.loc[df['Lineitem name'].str.contains('Raised Garden'), 'Lineitem name'] = 'Raised Garden'
df.loc[df['Lineitem name'].str.contains('Garden Staple'), 'Lineitem name'] = 'Garden Staple'

```


```python
#Subsetting useful columns
df = df[['Shipping Name','Name','Total','Shipping Method','Lineitem price','Lineitem quantity','Billing Street','Paid at'
    ,'Subtotal','Accepts Marketing','Taxes','Total','Created at','Lineitem price','Billing Address1','Billing Address1'
    ,'Billing City','Billing Zip','Billing Province','Shipping Street','Payment Method','Payment Reference','Source'
    ,'Tax 1 Name']]
```


```python
#Splitting string to get first name to generali
df['First Name'] = df['Shipping Name'].str.split(' ', expand = True)[0]
```

    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_2580/832190624.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    



```python
name = df['First Name']
```


```python
#Identifying genders
d = gender.Detector()
genders = []
for i in name[0:len(name)]:
    if d.get_gender(i) == 'male':
        genders.append('male')
    elif d.get_gender(i) == 'female':
        genders.append('female')
    else:
        genders.append('unknown')
```


```python
#Storing genders into a column 
df['gender'] = genders
```

    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_2580/3289624674.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    



```python
#Reading data to csv file because were getting error when running straight
df.to_csv('demo.csv')
```


```python
#Reading in 'new' df
df = pd.read_csv('demo.csv')
```


```python
#Dropping unknown genders 
df.drop(df[df['gender'] == 'unknown'].index, inplace = True)

```


```python
#Renaming column to count
df = df.rename(columns={"Unnamed: 0":"count"})
```


```python
#Concating to get full address
df['address'] = df['Billing Street'].str.lower() + ',' + df['Billing City'] + ',' + df['Billing Province']
```


```python
#Splitting the postal codes 
df['zip'] = df['Billing Zip'].str.split('-', expand = True)[0]
```


```python
#Filtering out all zip codes that do not have proper length 
df = df[df['zip'].str.len() == 5]
```


```python
#creating int column for zipcode
df['zipcodes']= df.zip.astype(int)
```


```python
#Saving to csv so zipcodes work
df.to_csv('df.csv')
```


```python
#reading in the csv again
df = pd.read_csv('df.csv')
```


```python
#Dropping column that was created on export
df = df.drop(['Unnamed: 0'], axis=1)
```

# Demographic Information


```python
import uszipcode
from uszipcode import SearchEngine
import numpy as np
from uszipcode import SearchEngine, SimpleZipcode, ComprehensiveZipcode
```


```python
#Creating search variable
search = SearchEngine()
```


```python
major_city1 = []
county1 = []
state1 = []
population1 = []
population_density1 = []
land_area_in_sqmi1 = []
housing_units1 = []
median_home_value1 = []
median_household_income1 = []
population_by_gender =[]
```


```python
for i in np.arange(0, len(df['zipcodes'])):
    zipcode = search.by_zipcode(df['zipcodes'][i])

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

# Calcualting Spending Score


```python
df['Date']= pd.to_datetime(df['Paid at'])
```


```python
df_recency = df.groupby(by='Name',
                        as_index=False)['Date'].max()
df_recency.columns = ['Name', 'LastPurchaseDate']
recent_date = df_recency['LastPurchaseDate'].max()
df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(
    lambda x: (recent_date - x).days)
df_recency.head()
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
      <th>Name</th>
      <th>LastPurchaseDate</th>
      <th>Recency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>2022-08-28 15:24:00</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>2022-10-24 06:41:00</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>2022-10-19 07:19:00</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>2022-10-18 18:59:00</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>2022-10-23 15:23:00</td>
      <td>42.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Frequency of purchase 
frequency_df = df.drop_duplicates().groupby(
    by=['Name'], as_index=False)['Date'].count()
frequency_df.columns = ['Name', 'Frequency']
frequency_df.head()
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
      <th>Name</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Monetary value of customer
monetary_df = df.groupby(by='Name', as_index=False)['Total'].sum()
monetary_df.columns = ['Name', 'Monetary']
monetary_df.head()
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
      <th>Name</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>8.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>29.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>29.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>29.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>29.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
rf_df = df_recency.merge(frequency_df, on='Name')
rfm_df = rf_df.merge(monetary_df, on='Name').drop(
    columns='LastPurchaseDate')
rfm_df.head()
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
      <th>Name</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>98.0</td>
      <td>1</td>
      <td>8.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>41.0</td>
      <td>1</td>
      <td>29.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>46.0</td>
      <td>1</td>
      <td>29.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>47.0</td>
      <td>1</td>
      <td>29.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>42.0</td>
      <td>1</td>
      <td>29.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merging and caluclating
rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)
 
# normalizing the rank of the customers
rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100
 
rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
 
rfm_df.head()
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
      <th>Name</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R_rank_norm</th>
      <th>F_rank_norm</th>
      <th>M_rank_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>98.0</td>
      <td>1</td>
      <td>8.63</td>
      <td>0.100604</td>
      <td>50.074147</td>
      <td>50.049407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>41.0</td>
      <td>1</td>
      <td>29.92</td>
      <td>28.973843</td>
      <td>50.074147</td>
      <td>50.049407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>46.0</td>
      <td>1</td>
      <td>29.20</td>
      <td>13.732394</td>
      <td>50.074147</td>
      <td>50.049407</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>47.0</td>
      <td>1</td>
      <td>29.79</td>
      <td>11.167002</td>
      <td>50.074147</td>
      <td>50.049407</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>42.0</td>
      <td>1</td>
      <td>29.31</td>
      <td>25.251509</td>
      <td>50.074147</td>
      <td>50.049407</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Final RFM score
rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm']+0.28 * \
    rfm_df['F_rank_norm']+0.57*rfm_df['M_rank_norm']
rfm_df['RFM_Score'] *= 0.05
rfm_df = rfm_df.round(2)
rfm_df[['Name', 'RFM_Score']].head(7)
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
      <th>Name</th>
      <th>RFM_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1001</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#111-0082648-6897023</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#111-0085922-4030662</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#111-0128415-1405821</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#111-0143015-6414647</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>#111-0171651-3457077</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>#111-0201289-1031428</td>
      <td>2.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
rfm_df.to_csv('rfm_df.csv')
rfm_df = pd.read_csv('rfm_df.csv')
```


```python
df.drop_duplicates()
```


```python
rfm_df.RFM_Score
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
      <th>Name</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R_rank_norm</th>
      <th>F_rank_norm</th>
      <th>M_rank_norm</th>
      <th>RFM_Score</th>
      <th>Customer_segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>#1001</td>
      <td>98.0</td>
      <td>1</td>
      <td>8.63</td>
      <td>0.10</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.13</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>#111-0082648-6897023</td>
      <td>41.0</td>
      <td>1</td>
      <td>29.92</td>
      <td>28.97</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.34</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>#111-0085922-4030662</td>
      <td>46.0</td>
      <td>1</td>
      <td>29.20</td>
      <td>13.73</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.23</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>#111-0128415-1405821</td>
      <td>47.0</td>
      <td>1</td>
      <td>29.79</td>
      <td>11.17</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.21</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>#111-0143015-6414647</td>
      <td>42.0</td>
      <td>1</td>
      <td>29.31</td>
      <td>25.25</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.32</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>1007</td>
      <td>114-9708004-0948263</td>
      <td>14.0</td>
      <td>1</td>
      <td>29.75</td>
      <td>73.44</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.68</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1008</td>
      <td>114-9771694-3345019</td>
      <td>5.0</td>
      <td>1</td>
      <td>58.40</td>
      <td>89.94</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.80</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>1009</td>
      <td>114-9784925-9644222</td>
      <td>5.0</td>
      <td>1</td>
      <td>29.65</td>
      <td>89.94</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.80</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>1010</td>
      <td>114-9844247-6164233</td>
      <td>15.0</td>
      <td>1</td>
      <td>29.31</td>
      <td>71.53</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.66</td>
      <td>Low Value Customers</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>1011</td>
      <td>114-9979555-4515414</td>
      <td>14.0</td>
      <td>1</td>
      <td>29.65</td>
      <td>73.44</td>
      <td>50.07</td>
      <td>50.05</td>
      <td>2.68</td>
      <td>Low Value Customers</td>
    </tr>
  </tbody>
</table>
<p>1012 rows × 10 columns</p>
</div>




```python
df = rfm_df.merge(df, on='Name')
```

# Merging in Data From Census


```python
census = census.groupby(['County']).mean()
```


```python
df = df.merge(census,left_on = 'county' ,right_on = 'County')
```


```python
df['age'] = np.where( (df['gender'] == 'male') , df['Median Male Age'], df['Median Female Age'])

```

# EDA


```python
#Pairplot to identify
sns.pairplot(df[['age','gender','Lineitem quantity','median_household_income','Total','land_area_in_sqmi','population','Lineitem price','RFM_Score']], hue='gender')
sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth":2.5}) 
plt.show()
```


    
![png](output_53_0.png)
    



```python
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['RFM_Score' ,'age', 'median_household_income' , 'Total']:
    n += 1
    plt.subplot(2 , 2 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    



    
![png](output_54_1.png)
    



```python
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'gender' , data = df)
plt.show()
```


    
![png](output_55_0.png)
    



```python
plt.figure(1 , figsize = (30 , 12))
n = 0 
for x in ['RFM_Score' , 'median_household_income' , 'Total','age']:
    for y in ['RFM_Score' , 'median_household_income' , 'Total','age']:
        n += 1
        plt.subplot(4 , 4 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()
```


    
![png](output_56_0.png)
    



```python
plt.figure(1 , figsize = (15 , 6))
for gender in ['male' , 'female']:
    plt.scatter(x = 'age' , y = 'RFM_Score' , data = df[df['gender'] == gender] ,
                s = 200 , alpha = 0.2 , label = gender)
plt.xlabel('Age'), plt.ylabel('Amount Spent') 
plt.title('Total vs Income')
plt.legend()
plt.show()
```


    
![png](output_57_0.png)
    



```python
#Subsetting random sample
subset = df.sample(frac=0.15, replace=True, random_state=1)

```


```python
plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['age' , 'median_household_income' , 'Total','RFM_Score']:
    n += 1 
    plt.subplot(2 , 2 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'gender' , data = subset , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'gender' , data = subset)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    29.7% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    48.9% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    



    
![png](output_59_1.png)
    


# Clustering


```python
df1 = df.dropna()
```


```python
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()
```


    
![png](output_62_0.png)
    



```python
X1 = df1[['age' , 'RFM_Score']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py:965: RuntimeWarning:
    
    algorithm='elkan' doesn't make sense for a single cluster. Using 'full' instead.
    



```python
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
```


```python
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
```


```python
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'age' ,y = 'RFM_Score' , data = df1 , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()
```


    
![png](output_66_0.png)
    



```python
X3 = df1[['age' ,'median_household_income', 'RFM_Score']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X3)
    inertia.append(algorithm.inertia_)
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py:965: RuntimeWarning:
    
    algorithm='elkan' doesn't make sense for a single cluster. Using 'full' instead.
    



```python
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_
```


```python
#3D Clustering
df2['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['age'],
    y= df['Total'],
    z= df['median_household_income'],
    mode='markers',
     marker=dict(
        color = df2['label3'], 
        size= 20,
        line=dict(
            color= df2['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
```

    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_2580/3948778450.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    



<div>                            <div id="8df648e5-a039-49a0-bd23-56fe2a0f4122" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("8df648e5-a039-49a0-bd23-56fe2a0f4122")) {                    Plotly.newPlot(                        "8df648e5-a039-49a0-bd23-56fe2a0f4122",                        [{"marker":{"color":[5,5,4,3,0,0,4,5,1,4,4,0,1,5,0,3,5,5,1,0,4,5,5,4,5,1,5,1,4,2,5,5,0,5,5,4,1,1,5,0,4,5,0,1,4,0,5,5,0,5,0,4,5,4,0,1,5,4,5,0,4,0,0,3,3,0,0,0,2,5,1,5,5,4,5,5,4,5,5,5,5,4,1,0,0,5,0,5,0,4,2,5,2,5,5,0,0,4,2,5,4,1,0,2,4,0,4,5,2,4,0,4,5,1,5,4,5,0,1,3,1,2,1,5,5,1,2,0,4,0,5,2,2,5,0,5,5,5,0,4,1,5,5,5,1,5,5,5,3,5,4,3,5,4,1,4,2,5,0,0,5,1,4,2,5,0,0,1,0,4,4,0,5,0,5,5,5,0,0,5,1,5,5,5,2,0,2,2,0,2,3,4,5,5,5,5,0,0,5,1,5,2,2,2,4,5,3,2,0,4,2,4,2,1,5,4,5,2,1,2,0,2,5,5,1,2,0,5,4,0,5,0,5,2,0,5,5,0,0,5,3,2,0,4,5,0,4,5,1,5,5,0,5,0,2,2,4,0,0,2,1,2,4,4,2,4,1,4,2,2,5,1,1,0,2,4,4,5,2,5,5,5,4,4,2,1,1,0,0,2,2,0,0,4,4,1,2,4,0,1,0,0,3,5,0,0,5,4,5,1,1,4,5,5,5,0,5,2,1,2,1,1,4,0,5,5,1,4,0,0,0,5,3,0,4,5,2,4,5,5,2,1,2,0,1,5,2,0,0,1,0,1,1,5,1,4,2,0,3,5,4,5,0,4,0,0,2,2,0,5,0,5,5,5,0,2,0,0,5,5,2,2,5,3,2,0,5,2,0,5,0,4,5,1,1,4,4,2,0,5,2,2,4,5,4,5,0,5,4,4,4,4,0,0,4,5,0,0,0,2,1,5,4,5,4,0,1,4,1,5,5,2,1,1,5,5,2,4,2,5,5,0,4,0,0,3,5,5,5,4,4,5,0,5,5,4,2,3,0,2,0,0,4,2,4,0,5,2,5,1,5,4,5,0,5,4,4,4,4,0,0,0,0,2,0,5,1,5,0,3,4,2,1,0,4,5,0,5,0,1,0,5,1,0,0,5,4,1,0,1,0,1,1,0,5,1,3,4,0,0,4,0,2,2,5,5,2,5,2,5,0,4,5,4,0,0,2,5,2,0,4,5,0,4,5,2,5,2,5,0,2,4,1,0,0,5,1,0,1,5,1,0,5,4,4,4,1,5,5,4,0,5,5,5,2,4,0,0,1,2,5,0,1,0,1,5,0,0,4,4,0,5,4,2,1,5,0,1,0,0,2,4,0,5,0,4,5,4,2,4,5,0,5,2,2,5,5,4,5,2,4,5,5,1,2,1,0,1,0,0,2,5,3,5,5,4,2,4,4,2,4,0,0,0,4,0,0,5,4,2,2,0,4,5,1,0,1,1,0,5,0,4,5,1,5,4,5,1,5,2,3,2,0,0,4,2,0,0,4,2,1,4,5,4,5,5,4,5,5,5,5,5,5,1,2,5,0,1,5,4,1,2,5,0,0,4,3,5,2,0,5,1,4,4,4,0,1,5,4,5,0,2,5,2,2,0,2,0,5,4,1,2,2,1,0,5,3,4,0,0,1,0,0,0,2,4,0,0,5,4,2,0,2,4,2,5,5,1,0,5,5,1,0,0,1,2,4,2,5,4,2,0,2,0,0,0,4,2,0,2,5,1,0,5,5,0,5,2,0,4,5,5,0,0,1,2,5,5,0,5,0,0,0,5,5,2,3,2,2,3,5,5,0,4,1,5,0,4,4,2,5,0,0,4,5,2,0,4,5,4,0,0,1,5,5,0,1,4,5,0,4,5,2,4,1,0,5,5,1,5,3,5,4,0,1,0,3,5,0,5,2,4,0,0,2,2,5,5,5,0,0,0,0,5,0,5,1,1,2,4,4,2,4,5,0,5,2,5,1,1,2,5,4,1,4,5,2,5,5,4,4,4,3,5,1,5,2,1,0,0,4,3,0,5,5,5,1,2,1,5,4,2,0,5,5,4,5,0,0,5,3,5,0,0,4,2,5,5,2,1,5,5,4,5,1,0,5,5,1,4,5,2,5,5,4,0,0,4,4,2,5,2,5,0,0,0,5,0,1,0,0,5,4,0,4,1,0,4,0,3,5,0,0,5,5,5,5,0,2,1,4,3,5,4,0,2,5,5,0,1,0,1],"line":{"color":[5,5,4,3,0,0,4,5,1,4,4,0,1,5,0,3,5,5,1,0,4,5,5,4,5,1,5,1,4,2,5,5,0,5,5,4,1,1,5,0,4,5,0,1,4,0,5,5,0,5,0,4,5,4,0,1,5,4,5,0,4,0,0,3,3,0,0,0,2,5,1,5,5,4,5,5,4,5,5,5,5,4,1,0,0,5,0,5,0,4,2,5,2,5,5,0,0,4,2,5,4,1,0,2,4,0,4,5,2,4,0,4,5,1,5,4,5,0,1,3,1,2,1,5,5,1,2,0,4,0,5,2,2,5,0,5,5,5,0,4,1,5,5,5,1,5,5,5,3,5,4,3,5,4,1,4,2,5,0,0,5,1,4,2,5,0,0,1,0,4,4,0,5,0,5,5,5,0,0,5,1,5,5,5,2,0,2,2,0,2,3,4,5,5,5,5,0,0,5,1,5,2,2,2,4,5,3,2,0,4,2,4,2,1,5,4,5,2,1,2,0,2,5,5,1,2,0,5,4,0,5,0,5,2,0,5,5,0,0,5,3,2,0,4,5,0,4,5,1,5,5,0,5,0,2,2,4,0,0,2,1,2,4,4,2,4,1,4,2,2,5,1,1,0,2,4,4,5,2,5,5,5,4,4,2,1,1,0,0,2,2,0,0,4,4,1,2,4,0,1,0,0,3,5,0,0,5,4,5,1,1,4,5,5,5,0,5,2,1,2,1,1,4,0,5,5,1,4,0,0,0,5,3,0,4,5,2,4,5,5,2,1,2,0,1,5,2,0,0,1,0,1,1,5,1,4,2,0,3,5,4,5,0,4,0,0,2,2,0,5,0,5,5,5,0,2,0,0,5,5,2,2,5,3,2,0,5,2,0,5,0,4,5,1,1,4,4,2,0,5,2,2,4,5,4,5,0,5,4,4,4,4,0,0,4,5,0,0,0,2,1,5,4,5,4,0,1,4,1,5,5,2,1,1,5,5,2,4,2,5,5,0,4,0,0,3,5,5,5,4,4,5,0,5,5,4,2,3,0,2,0,0,4,2,4,0,5,2,5,1,5,4,5,0,5,4,4,4,4,0,0,0,0,2,0,5,1,5,0,3,4,2,1,0,4,5,0,5,0,1,0,5,1,0,0,5,4,1,0,1,0,1,1,0,5,1,3,4,0,0,4,0,2,2,5,5,2,5,2,5,0,4,5,4,0,0,2,5,2,0,4,5,0,4,5,2,5,2,5,0,2,4,1,0,0,5,1,0,1,5,1,0,5,4,4,4,1,5,5,4,0,5,5,5,2,4,0,0,1,2,5,0,1,0,1,5,0,0,4,4,0,5,4,2,1,5,0,1,0,0,2,4,0,5,0,4,5,4,2,4,5,0,5,2,2,5,5,4,5,2,4,5,5,1,2,1,0,1,0,0,2,5,3,5,5,4,2,4,4,2,4,0,0,0,4,0,0,5,4,2,2,0,4,5,1,0,1,1,0,5,0,4,5,1,5,4,5,1,5,2,3,2,0,0,4,2,0,0,4,2,1,4,5,4,5,5,4,5,5,5,5,5,5,1,2,5,0,1,5,4,1,2,5,0,0,4,3,5,2,0,5,1,4,4,4,0,1,5,4,5,0,2,5,2,2,0,2,0,5,4,1,2,2,1,0,5,3,4,0,0,1,0,0,0,2,4,0,0,5,4,2,0,2,4,2,5,5,1,0,5,5,1,0,0,1,2,4,2,5,4,2,0,2,0,0,0,4,2,0,2,5,1,0,5,5,0,5,2,0,4,5,5,0,0,1,2,5,5,0,5,0,0,0,5,5,2,3,2,2,3,5,5,0,4,1,5,0,4,4,2,5,0,0,4,5,2,0,4,5,4,0,0,1,5,5,0,1,4,5,0,4,5,2,4,1,0,5,5,1,5,3,5,4,0,1,0,3,5,0,5,2,4,0,0,2,2,5,5,5,0,0,0,0,5,0,5,1,1,2,4,4,2,4,5,0,5,2,5,1,1,2,5,4,1,4,5,2,5,5,4,4,4,3,5,1,5,2,1,0,0,4,3,0,5,5,5,1,2,1,5,4,2,0,5,5,4,5,0,0,5,3,5,0,0,4,2,5,5,2,1,5,5,4,5,1,0,5,5,1,4,5,2,5,5,4,0,0,4,4,2,5,2,5,0,0,0,5,0,1,0,0,5,4,0,4,1,0,4,0,3,5,0,0,5,5,5,5,0,2,1,4,3,5,4,0,2,5,5,0,1,0,1],"width":12},"opacity":0.8,"size":20},"mode":"markers","x":[42.2,36.93333333333334,44.1,41.6,39.8,41.7,32.6,33.9,36.76666666666667,39.65,33.4,32.8,36.6,38.9,38.8,37.5,36.4,40.7,46.0,38.2,37.43333333333334,36.1,40.2,40.0,36.4,38.65,36.06666666666667,49.2,38.2,37.7,36.6,41.0,40.05,40.6,38.4,36.44,38.4,32.8,41.7,41.7,43.86666666666667,42.150000000000006,36.8,39.2,41.2,35.6,37.771428571428565,32.3,41.7,36.06666666666667,54.25,36.4,37.0,43.6,40.2,36.9,39.4,38.4,46.6,35.48571428571428,36.4,37.2,39.65,38.56666666666666,39.2,42.3,38.9,35.9,32.1,39.6,36.9,40.2,35.4,38.2,37.771428571428565,35.0,36.6,43.2,36.6,32.6,36.4,39.3,36.4,38.1,34.15,40.65,38.4,40.6,42.4,42.75,35.833333333333336,48.4,48.1,38.9,39.85,37.771428571428565,37.3,49.4,32.8,40.93333333333334,34.3,43.6,36.55,38.6,32.1,38.599999999999994,37.3,35.4,38.8,36.4,39.0,34.3,40.6,36.4,42.75,40.199999999999996,32.3,39.9,34.3,35.48571428571428,32.8,40.0,40.4,38.0,40.95,40.9,37.771428571428565,57.6,41.0,35.55,37.0,36.06666666666667,37.1,37.1,33.7,41.0,43.1,39.0,60.5,54.4,38.8,33.3,40.9,35.2,30.3,39.8,58.3,44.7,39.85,36.7,41.4,44.6,37.9,40.1375,39.2,33.9,37.5,39.8,41.0,36.5,41.8,38.65,36.6,38.4,32.3,35.6,37.45,34.3,40.1,33.3,35.48571428571428,40.45,37.8,37.3,44.7,37.45,38.3,35.6,32.8,32.4,35.6,38.599999999999994,40.8,40.528571428571425,35.0,40.5,42.75,42.0,36.7,39.2,35.6,36.9,38.599999999999994,39.8,37.6625,34.15,40.1375,33.9,36.4,39.4,38.3,35.8,35.4,38.1,42.599999999999994,34.5,34.3,35.3,41.8,35.9,42.0,38.45,36.8,41.1,41.3,38.96666666666667,43.8,32.8,39.3,36.4,35.2,33.6,38.1,39.0,39.6,42.7,42.6,35.0,39.175,34.1,36.0,37.1,39.3,38.3,38.6,41.63333333333333,38.8,40.45,39.7,37.15,33.900000000000006,35.0,33.6,38.9,40.199999999999996,57.1,35.3,40.8,35.6,42.4,38.18333333333333,36.1,30.6,37.05,35.8,39.36666666666667,40.0,33.3,41.4,45.1,40.8,36.4,35.7,38.3,37.8,36.1,36.4,36.55,35.48571428571428,59.7,40.6,38.7,36.93333333333334,43.1,40.2,40.45,36.4,40.45,41.6,36.95,35.2,40.528571428571425,38.2,38.739999999999995,39.51111111111111,38.4,37.0,32.1,33.7,35.6,41.63333333333333,47.45,38.8,35.48571428571428,32.8,41.2,34.15,38.65,35.3,38.739999999999995,36.5,38.4,36.6,40.300000000000004,45.2,35.55,37.45,55.4,39.0,55.4,39.4,37.4,34.3,32.5,38.6,27.5,39.0,41.0,38.739999999999995,47.0,31.0,36.76666666666667,40.4,38.1,38.0,37.8,32.5,36.9,43.6,35.3,37.3,35.6,37.4,42.6,36.6,36.3,38.1,40.15,48.4,37.099999999999994,38.96666666666667,37.4,38.4,37.8,32.6,41.0,39.51111111111111,41.1,40.916666666666664,35.1,39.65,46.0,40.75,44.8,38.8,37.05,36.9,41.5,40.15,54.25,38.4,32.5,42.75,39.65,32.5,38.8,39.4,38.4,40.75,40.2,37.4,35.4,34.15,45.3,36.8,36.0,38.0,38.15,37.0,39.650000000000006,40.7,34.15,39.175,35.2,36.4,38.9,35.55,40.8,35.3,34.5,42.0,43.6,35.2,38.3,37.771428571428565,43.1,35.8,36.4,39.0,37.4,39.650000000000006,54.1,38.18333333333333,38.1,36.4,37.0,42.8,38.9,38.2,36.1,35.6,41.4,33.7,34.8,38.3,37.8,40.2,43.8,35.9,34.3,35.8,35.7,37.971428571428575,35.8,43.0,30.5,37.5,43.2,38.0,36.8,36.93333333333334,34.3,34.5,44.4,38.8,37.971428571428575,57.1,38.2,34.5,36.6,38.18333333333333,38.7,36.1,32.8,40.075,41.7,35.9,40.300000000000004,38.45,41.6,30.5,38.2,36.0,40.4,35.6,40.4,38.4,35.1,42.5,40.45,37.6625,39.51111111111111,33.7,41.63333333333333,40.199999999999996,26.1,35.1,35.6,33.9,35.48571428571428,46.45,38.739999999999995,48.4,35.8,35.3,36.8,36.4,44.3,35.0,36.4,33.7,36.7,40.9,41.9,40.528571428571425,47.6,36.86666666666667,36.8,38.1,35.8,40.95,37.4,39.4,38.8,38.0,40.2,39.6,35.2,39.1,35.7,39.3,32.8,40.45,38.4,36.44,35.48571428571428,44.6,38.8,42.9,43.1,43.8,36.8,40.15,40.300000000000004,38.8,41.1,38.4,37.971428571428575,38.0,37.5,35.3,33.6,35.0,36.0,39.8,37.760000000000005,33.7,39.9,49.8,37.6625,35.6,36.4,38.1,44.2,38.2,39.3,35.6,47.7,39.4,44.7,36.93333333333334,37.3,39.0,37.3,40.9,40.075,36.8,41.4,40.5,34.8,42.2,43.3,37.5,37.5,40.4,33.6,33.3,50.0,37.771428571428565,37.771428571428565,37.9,37.771428571428565,36.6,38.0,40.400000000000006,44.6,36.0,38.9,32.1,32.4,38.4,35.6,39.4,37.3,48.1,39.0,33.4,36.4,37.2,35.400000000000006,36.9,41.36666666666667,40.8,38.0,31.0,38.6,42.75,41.7,38.3,39.0,32.8,40.6,37.4,39.1,33.7,43.0,40.9,39.4,40.8,37.1,32.8,38.96666666666667,39.9,41.2,34.3,41.7,41.65,35.9,33.3,39.0,41.9,33.3,38.599999999999994,43.1,39.4,41.5,50.0,33.9,36.4,43.1,43.1,43.1,40.6,33.3,40.75,35.6,50.0,47.4,32.7,44.32,35.48571428571428,40.528571428571425,38.8,35.5,39.4,40.2,35.1,38.9,40.9,38.9,32.1,37.8,36.8,36.4,38.0,34.9,50.1,44.6,50.1,37.8,37.1,36.0,33.4,35.5,35.48571428571428,37.1,33.3,29.4,38.9,29.4,36.8,40.9,36.4,55.4,33.3,36.4,35.1,38.2,32.4,35.8,41.4,36.8,35.2,40.2,37.8,35.3,39.36666666666667,33.3,40.0,38.9,35.9,39.4,52.8,41.4,43.1,38.2,43.1,35.8,33.3,48.4,36.8,33.9,41.333333333333336,39.0,43.1,31.0,38.699999999999996,41.4,44.7,36.3,38.199999999999996,57.1,35.8,38.1,36.800000000000004,38.6,56.3,36.8,49.0,34.6,37.0,42.12,37.1,36.4,37.3,40.0,35.6,38.4,35.6,35.55,35.8,36.4,37.1,34.15,35.48571428571428,36.1,35.8,31.3,36.4,38.599999999999994,38.4,39.3,33.9,39.0,42.2,33.3,43.1,38.0,47.7,36.8,33.9,35.8,37.7,38.25,37.2,40.1,45.3,52.8,38.18333333333333,32.5,33.7,35.5,45.2,45.2,39.8,43.6,36.6,28.9,39.6,39.4,39.3,37.1,38.18333333333333,40.1,35.2,38.8,39.8,38.3,36.44,36.6,40.75,35.8,36.8,39.3,43.1,39.8,38.6,33.3,35.8,38.8,33.4,38.2,41.65,40.75,38.2,43.3,37.771428571428565,38.4,38.1,42.0,38.699999999999996,38.1,38.1,38.8,33.3,33.4,38.6,45.0,42.6,35.8,45.3,38.900000000000006,33.4,39.0,36.8,36.800000000000004,38.3,39.25,33.9,40.1,38.1,40.9,43.13333333333333,35.9,35.55,54.5,35.0,59.7,39.8,36.8,40.75,40.2,38.45,38.4,35.9,36.55,33.4,36.4,35.9,37.9,47.7,35.5,38.0,38.4,36.76666666666667,33.3,39.2,42.2,34.3,45.9,43.2,38.96666666666667,36.8,38.599999999999994,50.0,35.6,39.7,39.2,36.8,39.3,32.5,38.1,40.916666666666664,38.599999999999994,38.0,36.8,38.1,34.7,36.76666666666667,35.8,37.6625,34.3,46.35,32.8,36.44,36.44,40.9,36.8,40.660000000000004,38.0,36.8,33.4,41.5,35.8,35.0,31.3,39.4,35.4,40.8,38.0,35.6,38.3,35.0,32.9,40.5,37.8,33.9,36.6,39.0,56.3,36.8,35.6,40.45,32.9,59.7,37.771428571428565,35.5,39.2,40.9,38.599999999999994,38.599999999999994,37.099999999999994,31.0,37.771428571428565,37.3,38.0,42.5,38.96666666666667,36.4,36.4,38.4,43.6,35.0,41.3,48.4,36.6,36.4,33.3,36.8,35.9,42.76666666666667,43.1,36.9,50.1,33.3,39.8,38.9,39.4,38.9,38.3,32.8,40.2,35.8,39.25,35.5,39.85,35.6,37.771428571428565,39.3,37.05,35.0,33.4,38.0,36.0,33.9,32.8,38.9,51.199999999999996,38.3,35.0,34.3,35.9,32.0,34.55,38.2,32.8,37.6625,40.3,42.4,37.8,37.6625,41.0,38.599999999999994,35.4,38.2,38.5,35.3,33.9,55.6,35.9,41.3,37.6,38.5,41.65,41.64,35.8,41.5,33.3,39.175,37.8,35.8,35.1,38.7,32.1,45.3,41.9,39.3,34.3,39.0,38.3,41.5,38.1,38.1,36.4,35.8,40.6,38.0,57.1,38.4,35.3,41.1,35.55,41.7,35.3,37.971428571428575,35.3,41.5,40.3,40.6,36.4,46.2,32.8,43.1,35.0],"y":[8.63,29.92,29.2,29.79,29.31,29.65,29.64,29.65,29.44,29.58,29.51,46.75,46.75,46.43,20.66,29.03,17.38,45.93,18.46,46.65,29.03,18.01,29.92,45.89,47.16,49.09,29.77,22.17,20.66,46.32,19.49,20.26,20.66,29.24,29.59,28.84,46.9,117.52,43.19,20.85,16.99,49.21,47.29,28.84,20.66,21.82,28.84,29.65,29.65,8.75,29.31,29.73,20.9,46.97,46.81,27.39,66.33,46.67,96.86,88.95,29.72,29.65,46.21,29.31,29.92,21.61,29.31,88.95,38.98,36.91,29.65,28.9,29.65,29.31,59.3,29.65,29.1,48.86,21.02,29.31,29.75,47.4,59.22,18.11,29.65,18.35,37.44,46.75,20.71,92.42,22.02,18.01,29.58,36.18,65.28,29.65,21.2,20.85,93.5,20.85,29.65,18.09,20.66,94.8,46.75,29.44,29.72,29.65,19.49,29.12,29.23,58.2,29.65,47.21,20.85,29.65,29.65,47.08,18.39,29.38,88.14,29.59,29.03,29.51,8.53,29.58,20.66,30.13,61.99,29.65,19.49,29.99,47.03,47.03,21.22,27.64,29.65,29.31,29.31,21.82,22.43,58.88,29.31,20.66,37.79,29.58,29.31,18.65,36.91,21.61,37.79,21.41,46.32,17.75,22.08,18.39,21.05,29.31,29.03,29.03,29.58,29.24,45.89,29.53,59.84,46.75,29.75,46.75,29.03,46.75,45.78,46.43,18.55,47.49,20.25,29.75,46.32,46.75,19.49,49.21,21.1,29.44,18.31,8.55,29.65,29.58,36.84,66.64,29.58,65.71,30.16,46.75,29.44,18.59,48.06,19.49,29.31,29.65,45.78,18.01,29.13,29.65,57.77,21.1,18.56,21.15,29.65,16.99,18.35,19.49,29.31,46.21,29.99,49.67,29.1,47.19,37.44,29.38,92.42,59.5,18.18,29.31,29.86,20.9,28.9,29.31,29.31,29.65,27.13,87.3,46.54,29.51,37.09,46.32,17.89,8.55,46.75,41.7,29.12,28.9,29.38,31.79,28.88,21.34,21.39,29.24,29.65,92.0,46.75,58.62,18.01,20.85,29.24,29.58,21.97,46.21,29.17,41.8,38.98,18.09,18.35,29.53,29.44,29.65,29.06,29.51,47.21,28.9,29.38,29.03,29.44,29.31,12.94,19.49,46.21,29.44,46.86,29.44,20.81,46.65,21.23,21.02,37.88,37.09,20.85,29.75,38.01,46.75,18.41,8.65,45.57,29.65,29.31,29.58,29.65,21.07,29.65,284.4,29.65,46.21,29.03,21.98,18.05,18.05,18.01,29.65,21.34,29.24,18.39,18.14,19.49,49.04,29.65,29.78,18.35,0.0,46.95,18.48,20.66,29.31,92.86,91.56,20.81,29.65,29.51,29.91,19.49,93.5,116.68,29.65,29.24,18.54,29.38,36.36,29.1,29.44,27.0,21.87,29.31,18.48,29.53,29.31,21.17,37.79,18.01,26.7,19.49,47.4,20.85,22.07,46.21,20.73,29.58,29.2,47.29,17.26,46.75,46.75,29.69,18.18,18.37,29.85,58.62,49.67,18.41,47.62,20.66,29.91,27.38,29.31,20.78,46.65,29.65,29.17,47.29,19.49,29.31,27.39,36.91,27.39,43.19,29.65,29.31,18.26,29.75,45.78,18.39,43.19,29.65,29.75,19.49,41.52,8.47,20.73,29.99,61.99,29.51,29.12,29.1,18.31,27.39,29.17,46.86,29.65,29.53,46.21,46.75,21.72,21.0,29.27,29.65,48.43,29.68,29.38,20.52,20.66,20.85,29.31,47.69,27.39,46.54,46.21,29.58,37.7,8.7,46.11,29.58,29.58,58.26,38.52,20.85,29.65,29.89,17.92,22.15,21.05,21.82,34.99,21.17,49.78,29.31,50.13,21.05,45.78,20.56,19.49,29.84,20.71,47.19,29.03,29.65,29.1,19.49,21.68,16.99,20.81,18.45,29.65,46.11,19.49,19.49,29.44,18.2,21.59,46.75,29.31,21.77,21.26,18.39,21.1,27.39,45.35,21.24,99.1,16.99,30.2,59.88,28.84,29.58,29.66,29.68,21.23,29.31,49.67,37.79,20.81,20.52,20.95,32.6,101.18,27.39,20.66,21.61,19.49,16.99,196.36,30.09,29.31,29.5,29.44,20.85,29.38,21.92,29.61,21.0,29.99,29.03,47.62,27.13,29.65,20.56,29.58,29.69,20.78,47.62,18.18,22.14,46.65,29.51,29.58,46.75,20.85,29.03,18.48,18.18,19.49,47.17,18.18,27.39,28.9,29.65,20.9,32.69,20.81,20.85,47.16,18.39,46.21,29.58,29.58,59.84,18.05,59.96,28.96,29.31,58.06,46.21,29.31,20.66,46.21,21.05,86.38,29.1,29.03,20.66,20.85,37.53,29.17,29.03,29.65,29.38,29.58,29.65,99.1,19.49,48.29,20.71,29.58,46.32,29.65,28.23,46.75,21.21,29.24,59.16,20.85,16.99,46.67,29.1,29.65,46.75,21.39,29.77,45.99,38.99,43.19,43.19,27.39,27.39,27.39,27.39,27.39,27.39,27.39,27.39,7.99,27.39,61.99,86.38,43.19,109.56,27.39,43.19,29.62,29.65,29.24,29.31,29.65,29.65,29.77,29.13,29.38,58.88,29.65,29.58,29.65,29.17,29.65,59.22,38.99,38.99,38.99,18.0,66.64,46.21,46.43,48.98,29.31,45.89,58.62,18.01,30.06,16.99,34.78,37.09,29.38,29.1,59.02,9.37,49.78,29.65,29.75,29.99,47.16,29.31,29.03,46.65,29.03,45.78,59.52,29.65,29.79,29.31,29.65,46.32,18.01,29.65,18.35,29.51,29.58,30.2,46.21,17.24,46.11,29.65,29.91,29.51,29.65,46.21,46.97,29.65,16.99,49.55,27.99,16.99,49.78,37.09,59.3,45.89,29.17,46.75,27.39,29.1,29.65,29.65,27.39,29.31,58.06,16.99,29.31,29.99,29.65,29.31,47.4,8.65,27.99,49.21,46.75,13.91,46.43,27.39,46.11,29.65,38.03,18.38,8.55,16.99,30.2,29.95,29.03,32.58,29.92,29.65,29.8,46.97,87.93,29.65,59.5,29.65,17.39,46.54,16.99,16.99,27.7,29.65,46.65,29.51,93.72,17.24,29.44,29.91,16.99,49.78,47.91,29.03,29.65,29.31,29.51,29.31,29.31,29.65,16.99,29.98,29.58,48.98,27.39,18.54,61.99,29.24,16.99,29.57,29.24,46.21,46.21,88.74,18.12,16.99,16.99,37.44,18.12,16.99,18.72,56.07,29.2,18.44,50.36,60.04,18.22,29.44,29.65,29.58,29.84,46.21,46.21,187.0,18.26,16.94,29.65,29.51,30.2,59.16,29.31,27.39,29.31,29.1,18.35,29.03,38.0,59.72,27.13,29.31,16.53,17.33,45.99,29.65,58.62,27.39,29.31,29.31,29.58,8.51,29.58,16.99,29.65,30.2,29.31,46.32,18.65,29.65,16.99,29.86,29.68,29.24,29.65,46.75,45.61,29.65,34.99,29.58,16.99,16.99,48.06,29.68,29.03,46.75,35.84,16.99,29.72,29.65,60.24,18.18,46.75,29.51,17.24,29.44,46.75,49.79,17.1,46.75,18.65,29.58,29.92,29.99,29.31,29.17,29.65,49.47,29.96,58.62,87.93,29.57,29.86,29.03,29.44,29.51,29.99,29.86,19.03,29.44,29.79,29.58,16.99,37.09,29.65,29.51,17.39,16.99,29.99,27.39,46.54,29.31,59.02,29.65,18.01,8.65,29.51,29.58,29.31,29.75,29.51,46.75,29.65,29.65,94.98,37.44,46.9,29.86,29.1,29.13,137.34,29.31,88.32,18.69,30.25,18.01,29.38,29.65,29.31,29.31,29.31,29.31,29.24,29.44,8.65,46.97,20.85,58.62,47.19,59.22,29.75,50.04,29.38,29.65,29.03,29.31,46.75,29.72,93.5,30.13,29.24,27.39,29.65,29.65,29.03,46.75,59.16,18.09,45.57,29.51,16.99,29.65,29.92,33.98,18.14,46.75,18.61,29.44,33.98,58.62,17.33,33.98,29.51,29.51,30.25,29.65,59.3,18.35,46.21,45.99,29.65,59.3,16.99,27.39,29.38,18.18,16.99,46.65,45.78,27.0,29.24,46.65,46.05,29.44,29.31,29.51,18.26,29.65,29.65,29.31,29.65,45.78,52.17,18.69,29.31,29.31,8.61,93.5,93.5,24.61,59.54,29.51,29.8,45.99,59.3,29.72,38.03,17.39,29.65,49.78,29.24,29.65,29.03,29.03,17.33,29.51,46.43,29.31,29.24,116.48,17.39,29.31,29.65,27.39,29.65,46.65,29.65,29.65,46.21,18.0,29.75,58.4,29.65,29.31,29.65],"z":[42938.0,43533.0,85301.0,137395.0,61545.0,63361.0,76810.0,53879.0,89419.0,76156.0,70768.0,61226.0,99629.0,45464.0,57143.0,136677.0,44658.0,50972.0,96563.0,60568.0,72580.0,46667.0,53400.0,78750.0,43477.0,92686.0,55508.0,108629.0,82159.0,37061.0,46828.0,53167.0,69948.0,52471.0,49111.0,73876.0,93878.0,95830.0,49088.0,56255.0,83288.0,50314.0,59199.0,108333.0,74730.0,68116.0,51055.0,44775.0,63361.0,44878.0,56133.0,73367.0,41921.0,80243.0,68368.0,91632.0,42396.0,76522.0,43095.0,67616.0,70302.0,59385.0,66724.0,118929.0,151538.0,65344.0,60798.0,57457.0,31721.0,48355.0,104168.0,53621.0,42500.0,75344.0,48136.0,46159.0,77386.0,44535.0,44040.0,52021.0,47200.0,75672.0,101526.0,68667.0,68863.0,49490.0,59181.0,52471.0,67039.0,77815.0,33475.0,47031.0,41108.0,51996.0,47261.0,67616.0,68275.0,73085.0,33711.0,51463.0,79049.0,87878.0,62602.0,41665.0,86026.0,62546.0,81322.0,53945.0,35954.0,76437.0,57969.0,76330.0,50376.0,88938.0,54688.0,80363.0,44775.0,63805.0,89899.0,131564.0,95830.0,null,38834.0,91103.0,43234.0,46500.0,89224.0,39803.0,64891.0,72111.0,66735.0,54806.0,39421.0,20951.0,54258.0,69144.0,54672.0,54967.0,46974.0,59993.0,76082.0,87761.0,48922.0,47753.0,44225.0,87794.0,42885.0,49748.0,54758.0,139508.0,51156.0,79940.0,128218.0,47841.0,81909.0,89258.0,83616.0,37803.0,47867.0,68427.0,65458.0,47781.0,104547.0,76632.0,40851.0,55283.0,65052.0,68538.0,90196.0,62482.0,71923.0,76333.0,56549.0,51060.0,56377.0,47179.0,53976.0,43537.0,61226.0,56964.0,43537.0,93838.0,52205.0,52146.0,53194.0,29401.0,60977.0,35843.0,37286.0,58929.0,28500.0,116344.0,71935.0,50428.0,47841.0,44283.0,44671.0,69139.0,61343.0,50544.0,87736.0,48974.0,27911.0,32447.0,38312.0,71516.0,51123.0,128936.0,30139.0,68538.0,83182.0,37244.0,71153.0,30230.0,91764.0,45035.0,82541.0,42543.0,39861.0,93878.0,39413.0,56551.0,35656.0,45947.0,51000.0,90484.0,36519.0,61980.0,42489.0,85677.0,64991.0,47421.0,69475.0,53976.0,30955.0,64554.0,42627.0,54871.0,55852.0,69850.0,51850.0,159016.0,40167.0,61188.0,73043.0,49029.0,66309.0,80271.0,43537.0,103696.0,46063.0,43131.0,57884.0,47392.0,60929.0,36728.0,36124.0,72987.0,64792.0,63651.0,38819.0,96558.0,39984.0,84732.0,74060.0,28316.0,70700.0,95567.0,72458.0,32649.0,31133.0,46377.0,101497.0,104538.0,65064.0,40000.0,70302.0,81213.0,51268.0,36164.0,54854.0,48790.0,46528.0,80120.0,75692.0,32043.0,94527.0,90495.0,65678.0,69676.0,38098.0,37212.0,59528.0,62260.0,86026.0,78573.0,110445.0,37795.0,76007.0,64328.0,96091.0,56414.0,63416.0,135217.0,51111.0,69114.0,66266.0,44597.0,74986.0,49161.0,111744.0,103516.0,76216.0,47781.0,49141.0,36101.0,52870.0,57375.0,52054.0,36502.0,92809.0,22324.0,112794.0,98524.0,81000.0,64988.0,48115.0,45907.0,96166.0,76007.0,57581.0,67351.0,58366.0,45579.0,123935.0,56388.0,72209.0,45296.0,36673.0,71042.0,47114.0,52121.0,38415.0,93188.0,38930.0,64747.0,100063.0,46399.0,36014.0,68287.0,66724.0,93580.0,67049.0,105943.0,91623.0,47392.0,111726.0,75805.0,39665.0,56133.0,117052.0,55266.0,70341.0,51040.0,65678.0,79140.0,63406.0,61078.0,41607.0,33656.0,56135.0,46433.0,69168.0,44518.0,50132.0,47280.0,64867.0,38601.0,69859.0,55976.0,45120.0,45170.0,34750.0,38793.0,47200.0,125375.0,33830.0,69007.0,43537.0,38675.0,55976.0,51442.0,67018.0,77560.0,46477.0,95328.0,87701.0,76437.0,80363.0,25139.0,55976.0,50716.0,30198.0,40686.0,81710.0,50632.0,83892.0,43163.0,62844.0,53764.0,84907.0,70845.0,75200.0,76566.0,55569.0,63636.0,73078.0,47124.0,56938.0,59394.0,56517.0,39413.0,101759.0,51542.0,76526.0,46996.0,83658.0,57584.0,90725.0,71720.0,99707.0,50739.0,52556.0,38182.0,97796.0,101759.0,44667.0,55212.0,38675.0,80407.0,31806.0,53820.0,49519.0,60176.0,75357.0,62642.0,60590.0,129578.0,43625.0,50283.0,54493.0,75452.0,71875.0,43572.0,66828.0,46260.0,48280.0,85677.0,41067.0,122517.0,63183.0,38016.0,62594.0,56195.0,75375.0,39593.0,75104.0,57929.0,44651.0,34097.0,52766.0,100739.0,53556.0,86406.0,42939.0,67068.0,47237.0,77604.0,75563.0,77704.0,75200.0,61977.0,60312.0,59469.0,59352.0,37951.0,55611.0,43582.0,88906.0,54303.0,61486.0,121484.0,83056.0,28493.0,90960.0,64280.0,79833.0,43432.0,62208.0,45589.0,59449.0,90495.0,58609.0,50455.0,99314.0,57443.0,66798.0,44399.0,72091.0,92560.0,61650.0,87794.0,68358.0,96067.0,109922.0,61233.0,48461.0,90256.0,120106.0,83616.0,59763.0,65651.0,78848.0,56280.0,34356.0,39728.0,48817.0,43644.0,40129.0,48951.0,41721.0,50998.0,63805.0,77316.0,48892.0,79838.0,56151.0,66711.0,31180.0,42005.0,41596.0,63646.0,74683.0,54854.0,60312.0,74684.0,51750.0,41548.0,51303.0,35507.0,51625.0,66927.0,38668.0,70098.0,97649.0,65651.0,67292.0,43522.0,100435.0,57758.0,104665.0,50310.0,111318.0,64026.0,53260.0,76500.0,71875.0,85227.0,90495.0,54784.0,43576.0,76007.0,65428.0,45724.0,43792.0,52204.0,32445.0,72089.0,67088.0,59174.0,111726.0,30925.0,46733.0,64026.0,92809.0,41665.0,77815.0,63361.0,87736.0,57969.0,90495.0,52471.0,58366.0,62208.0,75200.0,76526.0,60312.0,42396.0,80271.0,20951.0,95830.0,47114.0,63805.0,100842.0,64137.0,60860.0,38935.0,83709.0,64741.0,44910.0,60630.0,74621.0,49302.0,82917.0,39630.0,75805.0,48382.0,60782.0,45886.0,92560.0,92560.0,92560.0,52471.0,35515.0,39300.0,46863.0,48382.0,70196.0,51295.0,30120.0,82275.0,52233.0,52011.0,106014.0,26368.0,94368.0,56148.0,91755.0,64821.0,55915.0,31721.0,55102.0,156557.0,43477.0,47331.0,82689.0,35927.0,71124.0,71124.0,39309.0,71432.0,68461.0,66528.0,60726.0,71326.0,67395.0,62482.0,53807.0,77526.0,36181.0,37267.0,68995.0,73367.0,50079.0,98729.0,56321.0,100120.0,98442.0,56964.0,44440.0,64792.0,73151.0,47421.0,104665.0,45224.0,84907.0,44342.0,98669.0,47536.0,31966.0,173507.0,40661.0,58056.0,64792.0,82917.0,37722.0,62474.0,64721.0,83696.0,36673.0,101575.0,86523.0,52204.0,73043.0,54672.0,43047.0,74651.0,52543.0,42271.0,43992.0,50177.0,49161.0,49521.0,100644.0,33139.0,45297.0,59250.0,96054.0,45507.0,76599.0,107477.0,39303.0,49399.0,67132.0,68107.0,76124.0,128936.0,46233.0,33798.0,58895.0,42516.0,93878.0,71432.0,79408.0,78353.0,58802.0,100452.0,52546.0,81710.0,54340.0,68796.0,39205.0,43905.0,36845.0,29522.0,64741.0,23090.0,64026.0,43597.0,81923.0,89393.0,41335.0,26431.0,88837.0,66162.0,46814.0,135259.0,78583.0,60658.0,65678.0,106235.0,63942.0,59604.0,59604.0,37973.0,71316.0,56555.0,63938.0,54725.0,86357.0,39205.0,59524.0,21908.0,72429.0,37873.0,49231.0,50428.0,95816.0,57627.0,46828.0,44741.0,88494.0,59449.0,67840.0,92560.0,32223.0,86352.0,40727.0,54441.0,79140.0,41122.0,65751.0,29058.0,63338.0,60074.0,63974.0,78738.0,34995.0,60954.0,37699.0,47027.0,95648.0,59216.0,44475.0,45224.0,59068.0,54635.0,34283.0,64302.0,71962.0,48544.0,52097.0,58420.0,57787.0,90414.0,33055.0,53976.0,47594.0,63374.0,52543.0,68721.0,62201.0,58523.0,51123.0,52192.0,34680.0,119708.0,32649.0,32741.0,159696.0,47339.0,48958.0,68383.0,76114.0,93031.0,44091.0,58420.0,70302.0,75472.0,19135.0,52584.0,63942.0,64026.0,86426.0,52800.0,35282.0,79313.0,69313.0,74587.0,53671.0,75657.0,66859.0,66268.0,87326.0,48382.0,50930.0,65081.0,105924.0,73218.0,43624.0,60743.0,84665.0,46063.0,32481.0,86406.0,88415.0,59216.0,43033.0,52800.0,88355.0,45224.0,121094.0,49688.0,86026.0,64680.0,105144.0,68160.0,118860.0,46290.0,64044.0,51814.0,37175.0,75805.0,62949.0,68102.0,39345.0,39630.0,43365.0,55102.0,52242.0,69676.0,58255.0,67742.0,64650.0,42407.0,69577.0,44809.0,110294.0,108854.0,39441.0,77658.0,83023.0,32091.0,73424.0,42170.0,67616.0,45224.0,40653.0,54483.0,87326.0,87326.0,34831.0,44722.0,75269.0,88355.0,75578.0,46834.0,41596.0,45886.0,48280.0,75271.0,71713.0,82936.0,126621.0,45583.0,99629.0,49093.0,35282.0,97158.0,68939.0,64642.0,82917.0,116101.0,57665.0,54399.0,null,47151.0,49473.0,97931.0,28806.0,90495.0,55271.0,84905.0,31942.0,55936.0,49011.0,50338.0,75269.0,54264.0,58037.0,62610.0,42421.0,125051.0,44439.0,60782.0,64137.0,70448.0,32809.0,44541.0,44651.0,34574.0,93763.0,47457.0,52611.0,81378.0,47593.0,94504.0,62523.0,47089.0,53166.0,94504.0,84864.0,49315.0,32964.0,45967.0,48276.0,79984.0,59958.0,66035.0,82038.0,84843.0,40037.0,48276.0,39165.0,49167.0,56737.0,59710.0,55936.0,42059.0,69871.0,92531.0,61355.0,56501.0,54126.0,80719.0,62889.0,76124.0,111606.0,61401.0,73417.0,59710.0,163632.0,48594.0,67132.0,64026.0,44579.0,47331.0,44597.0,44469.0,66309.0,34451.0,110445.0,72286.0,118795.0,54949.0,76236.0,59710.0,34451.0,52471.0,42937.0,64240.0,87228.0,64533.0,97692.0],"type":"scatter3d"}],                        {"scene":{"xaxis":{"title":{"text":"Age"}},"yaxis":{"title":{"text":"Spending Score"}},"zaxis":{"title":{"text":"Annual Income"}}},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Clusters"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('8df648e5-a039-49a0-bd23-56fe2a0f4122');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python

```
