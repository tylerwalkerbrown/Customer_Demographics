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
columns.loc[df['Lineitem name'].str.contains("['Geography']"), 'Lineitem name'] = 'Tapping'

```


```python
columns.to_clipboard()
```


```python
#df.loc[df['Lineitem name'].str.contains('Tapping | Syrup'), 'Lineitem name'] = 'Tapping'

```


```python
columns.to_csv('columns.csv')
```


```python
columns = pd.read_csv('columns.csv')
```


```python
columns = columns.replace(', | " | . ',' ', regex=True)

```


```python
columns = columns.replace(" ' | ] ",' ', regex=True)
```


```python
columns.loc[columns['0'].str.contains("Geography"), '0'] = 'Geography'

```


```python
columns.loc[columns['0'].str.contains("Geographic Area Name"), '0'] = 'Geographic Area Name'

```


```python
columns.loc[columns['0'].str.contains("Total population"), '0'] = 'Total population'

```


```python
columns.to_clipboard()
```


```python
census.groupby(['Geographic Area Name']).mean()
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
      <th>Estimate!!Total!!Total population</th>
      <th>Margin of Error!!Total!!Total population!!AGE!!5 to 9 years</th>
      <th>Margin of Error!!Total!!Total population!!AGE!!10 to 14 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!15 to 19 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!25 to 29 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!30 to 34 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!35 to 39 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!45 to 49 years</th>
      <th>Estimate!!Total!!Total population!!AGE!!50 to 54 years</th>
      <th>Annotation of Estimate!!Total!!Total population!!AGE!!50 to 54 years</th>
      <th>...</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!15 to 44 years</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!18 years and over</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!21 years and over</th>
      <th>Margin of Error!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!21 years and over</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!60 years and over</th>
      <th>Margin of Error!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!60 years and over</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!62 years and over</th>
      <th>Margin of Error!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!62 years and over</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!65 years and over</th>
      <th>Estimate!!Percent Female!!Total population!!SELECTED AGE CATEGORIES!!75 years and over</th>
    </tr>
    <tr>
      <th>Geographic Area Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ada County, Idaho</th>
      <td>511931.0</td>
      <td>3226.0</td>
      <td>3206.0</td>
      <td>33845.0</td>
      <td>34743.0</td>
      <td>36695.0</td>
      <td>40542.0</td>
      <td>34128.0</td>
      <td>31079.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.4</td>
      <td>77.7</td>
      <td>74.2</td>
      <td>0.6</td>
      <td>22.2</td>
      <td>0.8</td>
      <td>20.0</td>
      <td>0.6</td>
      <td>16.7</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>Adams County, Colorado</th>
      <td>522140.0</td>
      <td>2681.0</td>
      <td>2932.0</td>
      <td>36142.0</td>
      <td>40324.0</td>
      <td>43459.0</td>
      <td>40696.0</td>
      <td>34114.0</td>
      <td>30871.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.2</td>
      <td>75.0</td>
      <td>71.8</td>
      <td>0.4</td>
      <td>17.7</td>
      <td>0.6</td>
      <td>15.4</td>
      <td>0.5</td>
      <td>12.2</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>Adams County, Pennsylvania</th>
      <td>104127.0</td>
      <td>804.0</td>
      <td>887.0</td>
      <td>7509.0</td>
      <td>5526.0</td>
      <td>5901.0</td>
      <td>6648.0</td>
      <td>5296.0</td>
      <td>7169.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>34.3</td>
      <td>81.1</td>
      <td>77.0</td>
      <td>0.8</td>
      <td>31.5</td>
      <td>1.4</td>
      <td>27.7</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Aiken County, South Carolina</th>
      <td>170776.0</td>
      <td>1381.0</td>
      <td>1337.0</td>
      <td>12349.0</td>
      <td>10449.0</td>
      <td>12124.0</td>
      <td>10359.0</td>
      <td>9712.0</td>
      <td>9402.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>36.3</td>
      <td>80.3</td>
      <td>76.6</td>
      <td>1.3</td>
      <td>30.1</td>
      <td>1.4</td>
      <td>25.6</td>
      <td>1.0</td>
      <td>22.3</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>Alabama</th>
      <td>5039877.0</td>
      <td>7263.0</td>
      <td>8859.0</td>
      <td>338347.0</td>
      <td>314216.0</td>
      <td>319367.0</td>
      <td>311035.0</td>
      <td>301406.0</td>
      <td>316964.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>37.7</td>
      <td>78.8</td>
      <td>74.5</td>
      <td>0.2</td>
      <td>26.1</td>
      <td>0.2</td>
      <td>23.3</td>
      <td>0.2</td>
      <td>19.1</td>
      <td>7.9</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>York County, Pennsylvania</th>
      <td>458696.0</td>
      <td>2310.0</td>
      <td>2347.0</td>
      <td>30289.0</td>
      <td>25916.0</td>
      <td>29428.0</td>
      <td>30855.0</td>
      <td>27928.0</td>
      <td>30977.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>35.9</td>
      <td>78.9</td>
      <td>75.3</td>
      <td>0.5</td>
      <td>26.8</td>
      <td>0.7</td>
      <td>23.4</td>
      <td>0.6</td>
      <td>19.8</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>York County, South Carolina</th>
      <td>288595.0</td>
      <td>2002.0</td>
      <td>1977.0</td>
      <td>20511.0</td>
      <td>15562.0</td>
      <td>18354.0</td>
      <td>20994.0</td>
      <td>19604.0</td>
      <td>20546.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>38.5</td>
      <td>77.2</td>
      <td>72.7</td>
      <td>0.8</td>
      <td>22.6</td>
      <td>0.9</td>
      <td>20.4</td>
      <td>0.8</td>
      <td>16.3</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>York County, Virginia</th>
      <td>70915.0</td>
      <td>857.0</td>
      <td>923.0</td>
      <td>4541.0</td>
      <td>4948.0</td>
      <td>4091.0</td>
      <td>4702.0</td>
      <td>3939.0</td>
      <td>4100.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>36.5</td>
      <td>76.7</td>
      <td>73.3</td>
      <td>2.1</td>
      <td>25.0</td>
      <td>1.9</td>
      <td>21.2</td>
      <td>1.8</td>
      <td>18.9</td>
      <td>9.7</td>
    </tr>
    <tr>
      <th>Yuba County, California</th>
      <td>83421.0</td>
      <td>1167.0</td>
      <td>1431.0</td>
      <td>5562.0</td>
      <td>6005.0</td>
      <td>6066.0</td>
      <td>5609.0</td>
      <td>4494.0</td>
      <td>4578.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.8</td>
      <td>72.5</td>
      <td>68.7</td>
      <td>1.6</td>
      <td>21.3</td>
      <td>1.5</td>
      <td>19.1</td>
      <td>1.6</td>
      <td>14.7</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>Yuma County, Arizona</th>
      <td>206990.0</td>
      <td>1743.0</td>
      <td>1693.0</td>
      <td>14506.0</td>
      <td>15098.0</td>
      <td>14193.0</td>
      <td>13045.0</td>
      <td>10574.0</td>
      <td>10213.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>36.9</td>
      <td>75.0</td>
      <td>70.1</td>
      <td>1.0</td>
      <td>26.2</td>
      <td>0.9</td>
      <td>24.2</td>
      <td>0.8</td>
      <td>21.0</td>
      <td>10.3</td>
    </tr>
  </tbody>
</table>
<p>893 rows × 273 columns</p>
</div>




```python
census.info(census['Margin of Error!!Total!!Total population','Annotation of Margin of Error!!Total!!Total population','Annotation of Estimate!!Total!!Total population'
                   ,'Annotation of Estimate!!Total!!Total population!!AGE!!Under 5 years','Annotation of Margin of Error!!Total!!Total population!!AGE!!Under 5 years'
                   ,'',''
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                   '','',
                  ] = '*****')
```


      Input In [231]
        ,''
        ^
    SyntaxError: invalid syntax



# Data Cleansing


```python
#Getting unique values to rename 
np.unique(df['Lineitem name'])
```


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
      <th>count</th>
      <th>Shipping Name</th>
      <th>Name</th>
      <th>Total</th>
      <th>Shipping Method</th>
      <th>Lineitem price</th>
      <th>Lineitem quantity</th>
      <th>Billing Street</th>
      <th>Paid at</th>
      <th>Subtotal</th>
      <th>...</th>
      <th>Billing City</th>
      <th>Billing Zip</th>
      <th>Billing Province</th>
      <th>Shipping Street</th>
      <th>Payment Method</th>
      <th>Payment Reference</th>
      <th>Source</th>
      <th>Tax 1 Name</th>
      <th>First Name</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Maureen M. Meadows</td>
      <td>112-5051599-3909864</td>
      <td>16.94</td>
      <td>Standard</td>
      <td>7.99</td>
      <td>2</td>
      <td>3276 TIMBER BLUFF DR NE</td>
      <td>12/5/22 0:00</td>
      <td>15.98</td>
      <td>...</td>
      <td>MARIETTA</td>
      <td>30062-4461</td>
      <td>GA</td>
      <td>3276 TIMBER BLUFF DR NE</td>
      <td>amazon</td>
      <td>112-5051599-3909864.1</td>
      <td>amazon</td>
      <td>Item Tax 6.008%</td>
      <td>Maureen</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Jonathon Woody</td>
      <td>111-8853220-9913805</td>
      <td>18.38</td>
      <td>Standard</td>
      <td>18.00</td>
      <td>1</td>
      <td>152 HIGHWAY 270</td>
      <td>12/4/22 22:14</td>
      <td>18.00</td>
      <td>...</td>
      <td>MENA</td>
      <td>71953-8583</td>
      <td>AR</td>
      <td>152 HIGHWAY 270</td>
      <td>amazon</td>
      <td>111-8853220-9913805.1</td>
      <td>amazon</td>
      <td>Item Tax 2.111%</td>
      <td>Jonathon</td>
      <td>male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Kevin Bitterman</td>
      <td>113-0064639-0072216</td>
      <td>48.06</td>
      <td>Standard</td>
      <td>45.99</td>
      <td>1</td>
      <td>48293 N HIAWATHA PL</td>
      <td>12/4/22 22:00</td>
      <td>45.99</td>
      <td>...</td>
      <td>CANTON</td>
      <td>57013-5875</td>
      <td>SD</td>
      <td>48293 N HIAWATHA PL</td>
      <td>amazon</td>
      <td>113-0064639-0072216.1</td>
      <td>amazon</td>
      <td>Item Tax 4.501%</td>
      <td>Kevin</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Sheila Cantando</td>
      <td>114-5038150-2389849</td>
      <td>45.78</td>
      <td>Standard</td>
      <td>43.19</td>
      <td>1</td>
      <td>707 Clover Ridge Dr</td>
      <td>12/4/22 21:53</td>
      <td>43.19</td>
      <td>...</td>
      <td>West Chester</td>
      <td>19380-1857</td>
      <td>PA</td>
      <td>707 Clover Ridge Dr</td>
      <td>amazon</td>
      <td>114-5038150-2389849.1</td>
      <td>amazon</td>
      <td>Item Tax 5.997%</td>
      <td>Sheila</td>
      <td>female</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Jennifer Miller</td>
      <td>111-0558911-8966641</td>
      <td>29.31</td>
      <td>Standard</td>
      <td>27.39</td>
      <td>1</td>
      <td>11324 NW 35th Ave</td>
      <td>12/4/22 21:51</td>
      <td>27.39</td>
      <td>...</td>
      <td>Gainesville</td>
      <td>'32606</td>
      <td>FL</td>
      <td>11324 NW 35th Ave</td>
      <td>amazon</td>
      <td>111-0558911-8966641.1</td>
      <td>amazon</td>
      <td>Item Tax 7.01%</td>
      <td>Jennifer</td>
      <td>female</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>1896</td>
      <td>Christine Reinprecht</td>
      <td>#114-8107137-9936205</td>
      <td>65.71</td>
      <td>Standard (MFN)</td>
      <td>61.99</td>
      <td>1</td>
      <td>131 Griffith Court</td>
      <td>10/14/22 18:58</td>
      <td>61.99</td>
      <td>...</td>
      <td>Perkasie</td>
      <td>'18944</td>
      <td>PA</td>
      <td>131 Griffith Court</td>
      <td>Amazon.com</td>
      <td>#114-8107137-9936205.1</td>
      <td>Amazon</td>
      <td>Shipping Taxes 0%</td>
      <td>Christine</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>1897</td>
      <td>Nancy  Austin</td>
      <td>#111-3087815-3793067</td>
      <td>28.84</td>
      <td>Standard (MFN)</td>
      <td>27.39</td>
      <td>1</td>
      <td>3332 ELK CREEK DR</td>
      <td>10/14/22 18:18</td>
      <td>27.39</td>
      <td>...</td>
      <td>CHRISTIANSBURG</td>
      <td>24073-6608</td>
      <td>VA</td>
      <td>3332 ELK CREEK DR</td>
      <td>Amazon.com</td>
      <td>#111-3087815-3793067.1</td>
      <td>Amazon</td>
      <td>Shipping Taxes 0%</td>
      <td>Nancy</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>1900</td>
      <td>Leilani Brown</td>
      <td>#1006</td>
      <td>19.99</td>
      <td>Standard Shipping</td>
      <td>19.99</td>
      <td>1</td>
      <td>15542 NEBO LN</td>
      <td>10/14/22 16:54</td>
      <td>19.99</td>
      <td>...</td>
      <td>Onancock</td>
      <td>23417-3516</td>
      <td>NaN</td>
      <td>15542 NEBO LN</td>
      <td>Etsy</td>
      <td>#1006.1</td>
      <td>2438651</td>
      <td>NaN</td>
      <td>Leilani</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>1902</td>
      <td>James C. Brooks</td>
      <td>#1004</td>
      <td>8.47</td>
      <td>Standard (MFN)</td>
      <td>7.99</td>
      <td>1</td>
      <td>954 Sheppard Avenue</td>
      <td>10/14/22 15:19</td>
      <td>7.99</td>
      <td>...</td>
      <td>Norfolk</td>
      <td>'23518</td>
      <td>VA</td>
      <td>954 Sheppard Avenue</td>
      <td>Amazon.com</td>
      <td>#1004.1</td>
      <td>Amazon</td>
      <td>Shipping Taxes 0%</td>
      <td>James</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>1905</td>
      <td>Stephen Pfuntner</td>
      <td>#1001</td>
      <td>8.63</td>
      <td>Standard Shipping</td>
      <td>7.99</td>
      <td>1</td>
      <td>208 MAIN ST APT 1</td>
      <td>8/28/22 15:24</td>
      <td>7.99</td>
      <td>...</td>
      <td>Dansville</td>
      <td>14437-1260</td>
      <td>NY</td>
      <td>208 Main Street, Apt 1</td>
      <td>Etsy</td>
      <td>#1001.1</td>
      <td>2438651</td>
      <td>total tax cost 0%</td>
      <td>Stephen</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
<p>1363 rows × 27 columns</p>
</div>




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

# EDA


```python
#Pairplot to identify
sns.pairplot(df[['gender','Lineitem quantity','count','median_household_income','Total','land_area_in_sqmi','population_density','population','Lineitem price']], hue='gender')
sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth":2.5}) 
plt.show()
```


    
![png](output_49_0.png)
    



```python
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Lineitem quantity' , 'median_household_income' , 'Total']:
    n += 1
    plt.subplot(1 , 3 , n)
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
    



    
![png](output_50_1.png)
    



```python
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'gender' , data = df)
plt.show()
```


    
![png](output_51_0.png)
    



```python
plt.figure(1 , figsize = (30 , 12))
n = 0 
for x in ['Lineitem quantity' , 'median_household_income' , 'Total']:
    for y in ['Lineitem quantity' , 'median_household_income' , 'Total']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()
```


    
![png](output_52_0.png)
    



```python
'Lineitem quantity' , 'median_household_income' , 'Total'
```


```python
plt.figure(1 , figsize = (15 , 6))
for gender in ['male' , 'female']:
    plt.scatter(x = 'median_household_income' , y = 'Total' , data = df[df['gender'] == gender] ,
                s = 200 , alpha = 0.2 , label = gender)
plt.xlabel('Income'), plt.ylabel('Total Spent') 
plt.title('Total vs Income')
plt.legend()
plt.show()
```


    
![png](output_54_0.png)
    



```python
plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['Lineitem quantity' , 'median_household_income' , 'Total']:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'gender' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'gender' , data = df)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    85.1% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    89.4% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    17.4% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    31.8% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    64.6% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:1296: UserWarning:
    
    73.4% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    



    
![png](output_55_1.png)
    


# Clustering


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1127 entries, 0 to 1126
    Data columns (total 37 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   count                    1127 non-null   int64  
     1   Shipping Name            1127 non-null   object 
     2   Name                     1127 non-null   object 
     3   Total                    1127 non-null   float64
     4   Shipping Method          1116 non-null   object 
     5   Lineitem price           1127 non-null   float64
     6   Lineitem quantity        1127 non-null   int64  
     7   Billing Street           1127 non-null   object 
     8   Paid at                  1125 non-null   object 
     9   Subtotal                 1127 non-null   float64
     10  Accepts Marketing        1127 non-null   object 
     11  Taxes                    1127 non-null   float64
     12  Total.1                  1127 non-null   float64
     13  Created at               1127 non-null   object 
     14  Lineitem price.1         1127 non-null   float64
     15  Billing Address1         1127 non-null   object 
     16  Billing Address1.1       1127 non-null   object 
     17  Billing City             1127 non-null   object 
     18  Billing Zip              1127 non-null   object 
     19  Billing Province         1126 non-null   object 
     20  Shipping Street          1127 non-null   object 
     21  Payment Method           1126 non-null   object 
     22  Payment Reference        1126 non-null   object 
     23  Source                   1127 non-null   object 
     24  Tax 1 Name               1115 non-null   object 
     25  First Name               1127 non-null   object 
     26  gender                   1127 non-null   object 
     27  zip                      1127 non-null   int64  
     28  address                  1126 non-null   object 
     29  zipcodes                 1127 non-null   int64  
     30  county                   1127 non-null   object 
     31  state                    1127 non-null   object 
     32  population               1124 non-null   float64
     33  population_density       1124 non-null   float64
     34  land_area_in_sqmi        1124 non-null   float64
     35  housing_units            1124 non-null   float64
     36  median_household_income  1124 non-null   float64
    dtypes: float64(11), int64(4), object(22)
    memory usage: 325.9+ KB



```python
a = df.dropna()
```


```python
X1 = a[['Lineitem quantity' , 'median_household_income' , 'Total']].iloc[: , :].values
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

```
