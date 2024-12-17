# importing libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import datetime as dt
from sklearn.feature_selection import SelectKBest,chi2,RFE,SelectFromModel,mutual_info_classif
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
warnings.filterwarnings("ignore")
%matplotlib inline
```


```python
customer = pd.read_csv(r"C:\Users\lakshita\Desktop\datasets\customer retention.csv")
```


```python
customer.shape
```




    (1249, 28)




```python
customer['CUS_DOB']=pd.to_datetime(customer['CUS_DOB'])
```


```python
del customer['CIF']
```


```python
customer.tail(10).T
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
      <th>1239</th>
      <th>1240</th>
      <th>1241</th>
      <th>1242</th>
      <th>1243</th>
      <th>1244</th>
      <th>1245</th>
      <th>1246</th>
      <th>1247</th>
      <th>1248</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CUS_DOB</th>
      <td>1963-05-21 00:00:00</td>
      <td>1985-01-24 00:00:00</td>
      <td>1974-07-30 00:00:00</td>
      <td>1981-01-15 00:00:00</td>
      <td>1981-08-24 00:00:00</td>
      <td>1951-09-10 00:00:00</td>
      <td>1984-03-23 00:00:00</td>
      <td>1985-02-04 00:00:00</td>
      <td>1950-02-03 00:00:00</td>
      <td>1961-02-23 00:00:00</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>56</td>
      <td>34</td>
      <td>45</td>
      <td>38</td>
      <td>38</td>
      <td>68</td>
      <td>35</td>
      <td>34</td>
      <td>69</td>
      <td>58</td>
    </tr>
    <tr>
      <th>CUS_Month_Income</th>
      <td>5000000.0</td>
      <td>500000.0</td>
      <td>9000.0</td>
      <td>4600000.0</td>
      <td>1500000.0</td>
      <td>4500.0</td>
      <td>3500000.0</td>
      <td>1000.0</td>
      <td>2000000.0</td>
      <td>5000000.0</td>
    </tr>
    <tr>
      <th>CUS_Gender</th>
      <td>MALE</td>
      <td>MALE</td>
      <td>FEMALE</td>
      <td>FEMALE</td>
      <td>FEMALE</td>
      <td>MALE</td>
      <td>MALE</td>
      <td>MALE</td>
      <td>MALE</td>
      <td>FEMALE</td>
    </tr>
    <tr>
      <th>CUS_Marital_Status</th>
      <td>MARRIED</td>
      <td>MARRIED</td>
      <td>DIVORCE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
      <td>SINGLE</td>
    </tr>
    <tr>
      <th>CUS_Customer_Since</th>
      <td>21-07-2005</td>
      <td>22-07-2005</td>
      <td>22-07-2005</td>
      <td>22-07-2005</td>
      <td>22-07-2005</td>
      <td>25-07-2005</td>
      <td>27-07-2005</td>
      <td>25-07-2005</td>
      <td>26-07-2005</td>
      <td>26-07-2005</td>
    </tr>
    <tr>
      <th>YEARS_WITH_US</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th># total debit transactions for S1</th>
      <td>6</td>
      <td>6</td>
      <td>58</td>
      <td>152</td>
      <td>125</td>
      <td>13</td>
      <td>55</td>
      <td>415</td>
      <td>147</td>
      <td>76</td>
    </tr>
    <tr>
      <th># total debit transactions for S2</th>
      <td>6</td>
      <td>13</td>
      <td>55</td>
      <td>160</td>
      <td>106</td>
      <td>12</td>
      <td>35</td>
      <td>368</td>
      <td>202</td>
      <td>88</td>
    </tr>
    <tr>
      <th># total debit transactions for S3</th>
      <td>6</td>
      <td>21</td>
      <td>84</td>
      <td>184</td>
      <td>64</td>
      <td>1</td>
      <td>37</td>
      <td>424</td>
      <td>234</td>
      <td>69</td>
    </tr>
    <tr>
      <th>total debit amount for S1</th>
      <td>903.2</td>
      <td>363317.02</td>
      <td>368910.58</td>
      <td>1266889.55</td>
      <td>851430.65</td>
      <td>3069.6</td>
      <td>119297.02</td>
      <td>1068684.72</td>
      <td>397602.73</td>
      <td>59730.64</td>
    </tr>
    <tr>
      <th>total debit amount for S2</th>
      <td>903.08</td>
      <td>495615.5</td>
      <td>1200425.76</td>
      <td>926481.15</td>
      <td>723807.15</td>
      <td>2376.0</td>
      <td>118884.0</td>
      <td>571814.89</td>
      <td>514584.25</td>
      <td>420105.56</td>
    </tr>
    <tr>
      <th>total debit amount for S3</th>
      <td>943.26</td>
      <td>116852.43</td>
      <td>3969005.02</td>
      <td>1198957.54</td>
      <td>459656.41</td>
      <td>200.0</td>
      <td>147554.88</td>
      <td>833122.07</td>
      <td>459665.24</td>
      <td>281991.71</td>
    </tr>
    <tr>
      <th># total credit transactions for S1</th>
      <td>0</td>
      <td>2</td>
      <td>22</td>
      <td>39</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>63</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th># total credit transactions for S2</th>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>23</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>76</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th># total credit transactions for S3</th>
      <td>0</td>
      <td>3</td>
      <td>20</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>78</td>
      <td>23</td>
      <td>19</td>
    </tr>
    <tr>
      <th>total credit amount for S1</th>
      <td>0.0</td>
      <td>375000.0</td>
      <td>3880264.51</td>
      <td>1245324.52</td>
      <td>950108.0</td>
      <td>0.0</td>
      <td>114601.14</td>
      <td>936134.09</td>
      <td>377245.0</td>
      <td>30500.0</td>
    </tr>
    <tr>
      <th>total credit amount for S2</th>
      <td>0.0</td>
      <td>1253300.0</td>
      <td>1762123.05</td>
      <td>944760.79</td>
      <td>450054.0</td>
      <td>0.0</td>
      <td>119137.62</td>
      <td>733658.65</td>
      <td>532708.75</td>
      <td>422000.0</td>
    </tr>
    <tr>
      <th>total credit amount for S3</th>
      <td>0.0</td>
      <td>146735.0</td>
      <td>4099024.64</td>
      <td>1169392.4</td>
      <td>450000.0</td>
      <td>0.0</td>
      <td>440891.65</td>
      <td>971555.27</td>
      <td>400814.6</td>
      <td>214350.0</td>
    </tr>
    <tr>
      <th>total debit amount</th>
      <td>2749.54</td>
      <td>975784.95</td>
      <td>5538341.36</td>
      <td>3392328.24</td>
      <td>2034894.21</td>
      <td>5645.6</td>
      <td>385735.9</td>
      <td>2473621.68</td>
      <td>1371852.22</td>
      <td>761827.91</td>
    </tr>
    <tr>
      <th>total debit transactions</th>
      <td>18</td>
      <td>40</td>
      <td>197</td>
      <td>496</td>
      <td>295</td>
      <td>26</td>
      <td>127</td>
      <td>1207</td>
      <td>583</td>
      <td>233</td>
    </tr>
    <tr>
      <th>total credit amount</th>
      <td>0.0</td>
      <td>1775035.0</td>
      <td>9741412.2</td>
      <td>3359477.71</td>
      <td>1850162.0</td>
      <td>0.0</td>
      <td>674630.41</td>
      <td>2641348.01</td>
      <td>1310768.35</td>
      <td>666850.0</td>
    </tr>
    <tr>
      <th>total credit transactions</th>
      <td>0</td>
      <td>8</td>
      <td>51</td>
      <td>89</td>
      <td>8</td>
      <td>0</td>
      <td>22</td>
      <td>217</td>
      <td>48</td>
      <td>37</td>
    </tr>
    <tr>
      <th>total transactions</th>
      <td>18</td>
      <td>48</td>
      <td>248</td>
      <td>585</td>
      <td>303</td>
      <td>26</td>
      <td>149</td>
      <td>1424</td>
      <td>631</td>
      <td>270</td>
    </tr>
    <tr>
      <th>CUS_Target</th>
      <td>2223</td>
      <td>2222</td>
      <td>2230</td>
      <td>2222</td>
      <td>2222</td>
      <td>2223</td>
      <td>2222</td>
      <td>2232</td>
      <td>2222</td>
      <td>2222</td>
    </tr>
    <tr>
      <th>TAR_Desc</th>
      <td>LOW</td>
      <td>MIDLE</td>
      <td>PLATINUM</td>
      <td>MIDLE</td>
      <td>MIDLE</td>
      <td>LOW</td>
      <td>MIDLE</td>
      <td>MIDLE</td>
      <td>MIDLE</td>
      <td>MIDLE</td>
    </tr>
    <tr>
      <th>Status</th>
      <td>CHURN</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
      <td>CHURN</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
      <td>ACTIVE</td>
    </tr>
  </tbody>
</table>
</div>




```python
customer.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1249 entries, 0 to 1248
    Data columns (total 27 columns):
     #   Column                              Non-Null Count  Dtype         
    ---  ------                              --------------  -----         
     0   CUS_DOB                             1249 non-null   datetime64[ns]
     1   AGE                                 1249 non-null   int64         
     2   CUS_Month_Income                    1238 non-null   float64       
     3   CUS_Gender                          1247 non-null   object        
     4   CUS_Marital_Status                  1249 non-null   object        
     5   CUS_Customer_Since                  1249 non-null   object        
     6   YEARS_WITH_US                       1249 non-null   int64         
     7   # total debit transactions for S1   1249 non-null   int64         
     8   # total debit transactions for S2   1249 non-null   int64         
     9   # total debit transactions for S3   1249 non-null   int64         
     10  total debit amount for S1           1249 non-null   float64       
     11  total debit amount for S2           1249 non-null   float64       
     12  total debit amount for S3           1249 non-null   float64       
     13  # total credit transactions for S1  1249 non-null   int64         
     14  # total credit transactions for S2  1249 non-null   int64         
     15  # total credit transactions for S3  1249 non-null   int64         
     16  total credit amount for S1          1249 non-null   float64       
     17  total credit amount for S2          1249 non-null   float64       
     18  total credit amount for S3          1249 non-null   float64       
     19  total debit amount                  1249 non-null   float64       
     20  total debit transactions            1249 non-null   int64         
     21  total credit amount                 1249 non-null   float64       
     22  total credit transactions           1249 non-null   int64         
     23  total transactions                  1249 non-null   int64         
     24  CUS_Target                          1249 non-null   int64         
     25  TAR_Desc                            1249 non-null   object        
     26  Status                              1249 non-null   object        
    dtypes: datetime64[ns](1), float64(9), int64(12), object(5)
    memory usage: 263.6+ KB
    


```python
# Convert the 'CUS_Customer_Since' column to datetime format
customer['CUS_Customer_Since'] = pd.to_datetime(customer['CUS_Customer_Since'], errors='coerce')
# Now you can access the year attribute
customer['CUS_DOB'] = customer['CUS_Customer_Since'].dt.year

```


```python
customer.describe()
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
      <th>CUS_DOB</th>
      <th>AGE</th>
      <th>CUS_Month_Income</th>
      <th>CUS_Customer_Since</th>
      <th>YEARS_WITH_US</th>
      <th># total debit transactions for S1</th>
      <th># total debit transactions for S2</th>
      <th># total debit transactions for S3</th>
      <th>total debit amount for S1</th>
      <th>total debit amount for S2</th>
      <th>...</th>
      <th># total credit transactions for S3</th>
      <th>total credit amount for S1</th>
      <th>total credit amount for S2</th>
      <th>total credit amount for S3</th>
      <th>total debit amount</th>
      <th>total debit transactions</th>
      <th>total credit amount</th>
      <th>total credit transactions</th>
      <th>total transactions</th>
      <th>CUS_Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1249.000000</td>
      <td>1249.000000</td>
      <td>1.238000e+03</td>
      <td>1249</td>
      <td>1249.000000</td>
      <td>1249.000000</td>
      <td>1249.000000</td>
      <td>1249.000000</td>
      <td>1.249000e+03</td>
      <td>1.249000e+03</td>
      <td>...</td>
      <td>1249.000000</td>
      <td>1.249000e+03</td>
      <td>1.249000e+03</td>
      <td>1.249000e+03</td>
      <td>1.249000e+03</td>
      <td>1249.000000</td>
      <td>1.249000e+03</td>
      <td>1249.000000</td>
      <td>1249.000000</td>
      <td>1249.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2004.883106</td>
      <td>46.746998</td>
      <td>2.764869e+06</td>
      <td>2005-06-16 05:28:34.971977472</td>
      <td>14.116894</td>
      <td>54.262610</td>
      <td>55.680544</td>
      <td>56.966373</td>
      <td>3.147439e+05</td>
      <td>3.090418e+05</td>
      <td>...</td>
      <td>8.675741</td>
      <td>3.089540e+05</td>
      <td>3.173819e+05</td>
      <td>3.020052e+05</td>
      <td>9.299314e+05</td>
      <td>166.909528</td>
      <td>9.283411e+05</td>
      <td>24.559648</td>
      <td>191.469175</td>
      <td>2222.296237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1991.000000</td>
      <td>14.000000</td>
      <td>0.000000e+00</td>
      <td>1991-10-31 00:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2211.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2005.000000</td>
      <td>39.000000</td>
      <td>7.568750e+03</td>
      <td>2005-06-17 00:00:00</td>
      <td>14.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>1.155000e+04</td>
      <td>1.122578e+04</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.642350e+04</td>
      <td>26.000000</td>
      <td>1.511744e+04</td>
      <td>2.000000</td>
      <td>33.000000</td>
      <td>2222.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2005.000000</td>
      <td>46.000000</td>
      <td>1.500000e+06</td>
      <td>2005-08-26 00:00:00</td>
      <td>14.000000</td>
      <td>22.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>5.395632e+04</td>
      <td>6.158130e+04</td>
      <td>...</td>
      <td>6.000000</td>
      <td>4.522084e+04</td>
      <td>4.679082e+04</td>
      <td>4.952000e+04</td>
      <td>2.031400e+05</td>
      <td>74.000000</td>
      <td>1.525319e+05</td>
      <td>18.000000</td>
      <td>90.000000</td>
      <td>2223.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2005.000000</td>
      <td>55.000000</td>
      <td>3.000000e+06</td>
      <td>2005-10-12 00:00:00</td>
      <td>14.000000</td>
      <td>65.000000</td>
      <td>65.000000</td>
      <td>68.000000</td>
      <td>1.892775e+05</td>
      <td>2.096500e+05</td>
      <td>...</td>
      <td>10.000000</td>
      <td>1.720000e+05</td>
      <td>1.872971e+05</td>
      <td>1.950319e+05</td>
      <td>6.613808e+05</td>
      <td>196.000000</td>
      <td>5.853255e+05</td>
      <td>28.000000</td>
      <td>227.000000</td>
      <td>2223.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019.000000</td>
      <td>119.000000</td>
      <td>8.000000e+07</td>
      <td>2019-04-04 00:00:00</td>
      <td>28.000000</td>
      <td>715.000000</td>
      <td>547.000000</td>
      <td>757.000000</td>
      <td>3.573349e+07</td>
      <td>3.723382e+07</td>
      <td>...</td>
      <td>169.000000</td>
      <td>4.920688e+07</td>
      <td>1.753799e+07</td>
      <td>3.764708e+07</td>
      <td>6.997262e+07</td>
      <td>1859.000000</td>
      <td>1.043919e+08</td>
      <td>429.000000</td>
      <td>2174.000000</td>
      <td>2236.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.279882</td>
      <td>12.315109</td>
      <td>5.406761e+06</td>
      <td>NaN</td>
      <td>2.279882</td>
      <td>80.732325</td>
      <td>81.020146</td>
      <td>84.649516</td>
      <td>1.513433e+06</td>
      <td>1.293928e+06</td>
      <td>...</td>
      <td>14.087572</td>
      <td>1.631418e+06</td>
      <td>1.048443e+06</td>
      <td>1.273680e+06</td>
      <td>3.142967e+06</td>
      <td>235.386076</td>
      <td>3.685585e+06</td>
      <td>38.467079</td>
      <td>263.353988</td>
      <td>3.314255</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>




```python
customer.isnull().sum()
```




    CUS_DOB                                0
    AGE                                    0
    CUS_Month_Income                      11
    CUS_Gender                             2
    CUS_Marital_Status                     0
    CUS_Customer_Since                     0
    YEARS_WITH_US                          0
    # total debit transactions for S1      0
    # total debit transactions for S2      0
    # total debit transactions for S3      0
    total debit amount for S1              0
    total debit amount for S2              0
    total debit amount for S3              0
    # total credit transactions for S1     0
    # total credit transactions for S2     0
    # total credit transactions for S3     0
    total credit amount for S1             0
    total credit amount for S2             0
    total credit amount for S3             0
    total debit amount                     0
    total debit transactions               0
    total credit amount                    0
    total credit transactions              0
    total transactions                     0
    CUS_Target                             0
    TAR_Desc                               0
    Status                                 0
    dtype: int64




```python
customer['CUS_Gender']=customer['CUS_Gender'].fillna("MALE")
customer['CUS_Month_Income'].fillna(customer['CUS_Month_Income'].median(),inplace=True)
```


```python
original=sns.distplot(customer['CUS_DOB'])
original.plot()
```




    []




    
![png](output_12_1.png)
    



```python
sns.boxplot(customer['CUS_Month_Income'])
```




    <Axes: ylabel='CUS_Month_Income'>




    
![png](output_13_1.png)
    



```python
customer['CUS_DOB'].skew()
```




    -1.4071175012772106




```python
skewness_log=np.log(customer['CUS_Month_Income'])
skewness_log.skew()
```




    nan




```python
# Assuming skewness_log is a NumPy array or a Pandas Series
print("Max value:", np.max(skewness_log))
print("Min value:", np.min(skewness_log))
print("Any NaN:", np.isnan(skewness_log).any())
print("Any Inf:", np.isinf(skewness_log).any())
```

    Max value: 18.197537192638155
    Min value: -inf
    Any NaN: False
    Any Inf: True
    

## Skew 1 


```python
# Plotting the cleaned data
sns.histplot(skewness_log, kde=True)  # Use histplot instead of distplot
plt.show()
```


    
![png](output_18_0.png)
    


## Skew 2


```python
skew_sqrt = np.sqrt(customer['CUS_Month_Income'])
skew_sqrt.skew()

```




    1.5258467496358188




```python
sns.distplot(skew_sqrt)
plt.show()
```


    
![png](output_21_0.png)
    


# **Handling Skewness**


```python
cat= customer.dtypes[customer.dtypes!='object'].index
```


```python
#skew_feat = customer[cat].skew().sort_values(ascending=False)
skew_feat = customer.select_dtypes(include=['number']).skew().sort_values(ascending=False)

```


```python
skewness=pd.DataFrame({'skew':skew_feat})
```


```python
skewness
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
      <th>skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>total credit amount for S1</th>
      <td>22.980122</td>
    </tr>
    <tr>
      <th>total credit amount for S3</th>
      <td>21.175667</td>
    </tr>
    <tr>
      <th>total debit amount for S2</th>
      <td>20.187508</td>
    </tr>
    <tr>
      <th>total credit amount</th>
      <td>19.095959</td>
    </tr>
    <tr>
      <th>total debit amount for S1</th>
      <td>16.296164</td>
    </tr>
    <tr>
      <th>total debit amount</th>
      <td>12.574292</td>
    </tr>
    <tr>
      <th>total debit amount for S3</th>
      <td>9.573946</td>
    </tr>
    <tr>
      <th>total credit amount for S2</th>
      <td>8.947780</td>
    </tr>
    <tr>
      <th>CUS_Month_Income</th>
      <td>6.342003</td>
    </tr>
    <tr>
      <th># total credit transactions for S1</th>
      <td>6.144899</td>
    </tr>
    <tr>
      <th>total credit transactions</th>
      <td>4.618276</td>
    </tr>
    <tr>
      <th># total credit transactions for S3</th>
      <td>4.479344</td>
    </tr>
    <tr>
      <th># total credit transactions for S2</th>
      <td>4.411057</td>
    </tr>
    <tr>
      <th># total debit transactions for S1</th>
      <td>2.939292</td>
    </tr>
    <tr>
      <th># total debit transactions for S3</th>
      <td>2.815952</td>
    </tr>
    <tr>
      <th>total transactions</th>
      <td>2.687731</td>
    </tr>
    <tr>
      <th>total debit transactions</th>
      <td>2.641702</td>
    </tr>
    <tr>
      <th># total debit transactions for S2</th>
      <td>2.544719</td>
    </tr>
    <tr>
      <th>YEARS_WITH_US</th>
      <td>1.407118</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.343214</td>
    </tr>
    <tr>
      <th>CUS_Target</th>
      <td>-0.951079</td>
    </tr>
    <tr>
      <th>CUS_DOB</th>
      <td>-1.407118</td>
    </tr>
  </tbody>
</table>
</div>




```python
skewed_features=customer[skewness.index]
```


```python
skewed_features.columns
```




    Index(['total credit amount for S1', 'total credit amount for S3',
           'total debit amount for S2', 'total credit amount',
           'total debit amount for S1', 'total debit amount',
           'total debit amount for S3', 'total credit amount for S2',
           'CUS_Month_Income', '# total credit transactions for S1',
           'total credit transactions', '# total credit transactions for S3',
           '# total credit transactions for S2',
           '# total debit transactions for S1',
           '# total debit transactions for S3', 'total transactions',
           'total debit transactions', '# total debit transactions for S2',
           'YEARS_WITH_US', 'AGE', 'CUS_Target', 'CUS_DOB'],
          dtype='object')




```python
##cube rt transformation +ve skew
for i in skewed_features[:-1]:
    customer[i]=customer[i]**(1/3)
```


```python
customer.rename(columns={' Married ': 'MARRIED'}, inplace=True)
if 'MARRIED' not in customer.columns:
    print("Column 'MARRIED' is missing. Adding it.")
    customer['MARRIED'] = None
```

    Column 'MARRIED' is missing. Adding it.
    


```python
print(customer.dtypes)
```

    CUS_DOB                                      float64
    AGE                                          float64
    CUS_Month_Income                             float64
    CUS_Gender                                    object
    CUS_Marital_Status                            object
    CUS_Customer_Since                    datetime64[ns]
    YEARS_WITH_US                                float64
    # total debit transactions for S1            float64
    # total debit transactions for S2            float64
    # total debit transactions for S3            float64
    total debit amount for S1                    float64
    total debit amount for S2                    float64
    total debit amount for S3                    float64
    # total credit transactions for S1           float64
    # total credit transactions for S2           float64
    # total credit transactions for S3           float64
    total credit amount for S1                   float64
    total credit amount for S2                   float64
    total credit amount for S3                   float64
    total debit amount                           float64
    total debit transactions                     float64
    total credit amount                          float64
    total credit transactions                    float64
    total transactions                           float64
    CUS_Target                                   float64
    TAR_Desc                                      object
    Status                                        object
    MARRIED                                       object
    dtype: object
    


```python
# Calculate skewness for numeric columns only
numeric_customer = customer.select_dtypes(include=['number'])
skewness = numeric_customer.skew()
customer['MARRIED'] = customer['MARRIED'].map({'MARRIED': 1, 'SINGLE': 0})
skewness = customer.skew(numeric_only=True)
print(skewness)
```

    CUS_DOB                              -1.428156
    AGE                                  -0.535882
    CUS_Month_Income                      0.447012
    YEARS_WITH_US                        -4.952400
    # total debit transactions for S1     0.340211
    # total debit transactions for S2     0.222557
    # total debit transactions for S3     0.201903
    total debit amount for S1             2.025088
    total debit amount for S2             1.557775
    total debit amount for S3             1.307329
    # total credit transactions for S1    0.044715
    # total credit transactions for S2    0.048543
    # total credit transactions for S3    0.059512
    total credit amount for S1            1.672035
    total credit amount for S2            1.345104
    total credit amount for S3            1.349011
    total debit amount                    1.721168
    total debit transactions              0.554672
    total credit amount                   1.454028
    total credit transactions             0.028287
    total transactions                    0.566934
    CUS_Target                           -0.963957
    MARRIED                                    NaN
    dtype: float64
    


```python
sns.boxplot(customer['YEARS_WITH_US'])
```




    <Axes: ylabel='YEARS_WITH_US'>




    
![png](output_33_1.png)
    


# sns.pairplot(customer=customer)


```python
sns.pairplot(customer.select_dtypes(include=['float64', 'int64']))  # Select only numeric columns
plt.show()
```


    
![png](output_35_0.png)
    



```python
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.barplot(y=customer['AGE'],x=customer['Status'])
plt.subplot(2,3,2)
sns.barplot(y=customer['CUS_Month_Income'],x=customer['Status'])
plt.subplot(2,3,3)
sns.barplot(y=customer['YEARS_WITH_US'],x=customer['Status'])
plt.subplot(2,3,4)
sns.barplot(y=customer['total transactions'],x=customer['Status'])
plt.subplot(2,3,5)
sns.barplot(y=customer['total debit transactions'],x=customer['Status'])
plt.subplot(2,3,6)
sns.barplot(y=customer['total credit transactions'],x=customer['Status'])
plt.tight_layout()
```


    
![png](output_36_0.png)
    



```python
cat1 = customer.dtypes[customer.dtypes == 'object'].index
cat1 = cat1[:-1]
cat1
```




    Index(['CUS_Gender', 'CUS_Marital_Status', 'TAR_Desc'], dtype='object')




```python
list = customer.copy()
```


```python
sns.pairplot(customer)
```




    <seaborn.axisgrid.PairGrid at 0x18b55a498e0>




    
![png](output_39_1.png)
    



```python
customer.head()
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
      <th>CUS_DOB</th>
      <th>AGE</th>
      <th>CUS_Month_Income</th>
      <th>CUS_Gender</th>
      <th>CUS_Marital_Status</th>
      <th>CUS_Customer_Since</th>
      <th>YEARS_WITH_US</th>
      <th># total debit transactions for S1</th>
      <th># total debit transactions for S2</th>
      <th># total debit transactions for S3</th>
      <th>...</th>
      <th>total credit amount for S3</th>
      <th>total debit amount</th>
      <th>total debit transactions</th>
      <th>total credit amount</th>
      <th>total credit transactions</th>
      <th>total transactions</th>
      <th>CUS_Target</th>
      <th>TAR_Desc</th>
      <th>Status</th>
      <th>MARRIED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.586599</td>
      <td>3.659306</td>
      <td>19.234977</td>
      <td>MALE</td>
      <td>MARRIED</td>
      <td>1994-06-30</td>
      <td>2.924018</td>
      <td>6.518684</td>
      <td>6.423158</td>
      <td>7.013579</td>
      <td>...</td>
      <td>133.487323</td>
      <td>168.196761</td>
      <td>9.608182</td>
      <td>178.209127</td>
      <td>4.020726</td>
      <td>9.837369</td>
      <td>13.066718</td>
      <td>EXECUTIVE</td>
      <td>ACTIVE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.609701</td>
      <td>3.583048</td>
      <td>114.471424</td>
      <td>FEMALE</td>
      <td>SINGLE</td>
      <td>2005-05-19</td>
      <td>2.410142</td>
      <td>3.332222</td>
      <td>2.466212</td>
      <td>3.556893</td>
      <td>...</td>
      <td>38.597213</td>
      <td>51.846096</td>
      <td>4.594701</td>
      <td>44.310476</td>
      <td>2.154435</td>
      <td>4.747459</td>
      <td>13.051081</td>
      <td>LOW</td>
      <td>ACTIVE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.609701</td>
      <td>3.756286</td>
      <td>170.997595</td>
      <td>FEMALE</td>
      <td>SINGLE</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>3.141381</td>
      <td>2.410142</td>
      <td>2.000000</td>
      <td>...</td>
      <td>41.397844</td>
      <td>33.434553</td>
      <td>3.756286</td>
      <td>100.691333</td>
      <td>3.000000</td>
      <td>4.308869</td>
      <td>13.049124</td>
      <td>MIDLE</td>
      <td>ACTIVE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.609701</td>
      <td>2.466212</td>
      <td>7.937005</td>
      <td>FEMALE</td>
      <td>SINGLE</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>26.568315</td>
      <td>15.874011</td>
      <td>1.000000</td>
      <td>42.533438</td>
      <td>2.289428</td>
      <td>2.351335</td>
      <td>13.051081</td>
      <td>LOW</td>
      <td>ACTIVE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.628540</td>
      <td>3.634241</td>
      <td>208.008382</td>
      <td>FEMALE</td>
      <td>SINGLE</td>
      <td>2014-06-30</td>
      <td>1.709976</td>
      <td>2.466212</td>
      <td>3.207534</td>
      <td>2.289428</td>
      <td>...</td>
      <td>41.310746</td>
      <td>69.690490</td>
      <td>3.914868</td>
      <td>76.179860</td>
      <td>2.758924</td>
      <td>4.326749</td>
      <td>13.051081</td>
      <td>LOW</td>
      <td>ACTIVE</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
sns.boxplot(list['CUS_Month_Income'])
```




    <Axes: ylabel='CUS_Month_Income'>




    
![png](output_41_1.png)
    



```python
customer['CUS_Month_Income'].skew()
```




    0.4470119464703883




```python
# plt.figure(figsize=(15,10))

plt3 = sns.countplot(x=list['TAR_Desc'],hue=list['Status'])

plt.plot()
plt.tight_layout()

```


    
![png](output_43_0.png)
    



```python
sns.countplot(x=list['CUS_Gender'],hue=list['Status'])
```




    <Axes: xlabel='CUS_Gender', ylabel='count'>




    
![png](output_44_1.png)
    



```python
# plt2 = sns.barplot(x=df['CUS_Marital_Status'],y=df['Status'])
sns.countplot(x=list['CUS_Marital_Status'],hue=list['Status'])
```




    <Axes: xlabel='CUS_Marital_Status', ylabel='count'>




    
![png](output_45_1.png)
    



```python
sns.countplot(customer['CUS_Marital_Status'])
```




    <Axes: xlabel='count', ylabel='CUS_Marital_Status'>




    
![png](output_46_1.png)
    



```python
sns.countplot(customer['Status'])
# 5:1 Ratio
```




    <Axes: xlabel='count', ylabel='Status'>




    
![png](output_47_1.png)
    



```python
customer['Status'].value_counts()
```




    Status
    ACTIVE    1022
    CHURN      227
    Name: count, dtype: int64



# Label Encoder


```python
from sklearn.preprocessing import LabelEncoder
##binary variable
bi_var = [col for col in list.columns if len(list[col].unique()) ==2 ]
cat_col = [col for col in list.select_dtypes(['object']).columns.tolist() if col not in bi_var]

encoder = LabelEncoder()
for i in bi_var:
    list[i] = encoder.fit_transform(list[i]) 
    
list = pd.get_dummies(list,columns= cat_col)

```


```python
# list = list.drop(['CUS_DOB','CUS_Customer_Since'],axis=1)
```


```python
list.head()
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
      <th>CUS_DOB</th>
      <th>AGE</th>
      <th>CUS_Month_Income</th>
      <th>CUS_Gender</th>
      <th>CUS_Customer_Since</th>
      <th>YEARS_WITH_US</th>
      <th># total debit transactions for S1</th>
      <th># total debit transactions for S2</th>
      <th># total debit transactions for S3</th>
      <th>total debit amount for S1</th>
      <th>...</th>
      <th>CUS_Marital_Status_DIVORCE</th>
      <th>CUS_Marital_Status_MARRIED</th>
      <th>CUS_Marital_Status_OTHER</th>
      <th>CUS_Marital_Status_PARTNER</th>
      <th>CUS_Marital_Status_SINGLE</th>
      <th>CUS_Marital_Status_WIDOWED</th>
      <th>TAR_Desc_EXECUTIVE</th>
      <th>TAR_Desc_LOW</th>
      <th>TAR_Desc_MIDLE</th>
      <th>TAR_Desc_PLATINUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.586599</td>
      <td>3.659306</td>
      <td>19.234977</td>
      <td>1</td>
      <td>1994-06-30</td>
      <td>2.924018</td>
      <td>6.518684</td>
      <td>6.423158</td>
      <td>7.013579</td>
      <td>113.422094</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.609701</td>
      <td>3.583048</td>
      <td>114.471424</td>
      <td>0</td>
      <td>2005-05-19</td>
      <td>2.410142</td>
      <td>3.332222</td>
      <td>2.466212</td>
      <td>3.556893</td>
      <td>32.826314</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.609701</td>
      <td>3.756286</td>
      <td>170.997595</td>
      <td>0</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>3.141381</td>
      <td>2.410142</td>
      <td>2.000000</td>
      <td>30.455583</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.609701</td>
      <td>2.466212</td>
      <td>7.937005</td>
      <td>0</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.628540</td>
      <td>3.634241</td>
      <td>208.008382</td>
      <td>0</td>
      <td>2014-06-30</td>
      <td>1.709976</td>
      <td>2.466212</td>
      <td>3.207534</td>
      <td>2.289428</td>
      <td>41.032174</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
X = list.drop('Status',axis=1)
y = list['Status']
```


```python
X.head()
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
      <th>CUS_DOB</th>
      <th>AGE</th>
      <th>CUS_Month_Income</th>
      <th>CUS_Gender</th>
      <th>CUS_Customer_Since</th>
      <th>YEARS_WITH_US</th>
      <th># total debit transactions for S1</th>
      <th># total debit transactions for S2</th>
      <th># total debit transactions for S3</th>
      <th>total debit amount for S1</th>
      <th>...</th>
      <th>CUS_Marital_Status_DIVORCE</th>
      <th>CUS_Marital_Status_MARRIED</th>
      <th>CUS_Marital_Status_OTHER</th>
      <th>CUS_Marital_Status_PARTNER</th>
      <th>CUS_Marital_Status_SINGLE</th>
      <th>CUS_Marital_Status_WIDOWED</th>
      <th>TAR_Desc_EXECUTIVE</th>
      <th>TAR_Desc_LOW</th>
      <th>TAR_Desc_MIDLE</th>
      <th>TAR_Desc_PLATINUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.586599</td>
      <td>3.659306</td>
      <td>19.234977</td>
      <td>1</td>
      <td>1994-06-30</td>
      <td>2.924018</td>
      <td>6.518684</td>
      <td>6.423158</td>
      <td>7.013579</td>
      <td>113.422094</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.609701</td>
      <td>3.583048</td>
      <td>114.471424</td>
      <td>0</td>
      <td>2005-05-19</td>
      <td>2.410142</td>
      <td>3.332222</td>
      <td>2.466212</td>
      <td>3.556893</td>
      <td>32.826314</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.609701</td>
      <td>3.756286</td>
      <td>170.997595</td>
      <td>0</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>3.141381</td>
      <td>2.410142</td>
      <td>2.000000</td>
      <td>30.455583</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.609701</td>
      <td>2.466212</td>
      <td>7.937005</td>
      <td>0</td>
      <td>2005-05-20</td>
      <td>2.410142</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.628540</td>
      <td>3.634241</td>
      <td>208.008382</td>
      <td>0</td>
      <td>2014-06-30</td>
      <td>1.709976</td>
      <td>2.466212</td>
      <td>3.207534</td>
      <td>2.289428</td>
      <td>41.032174</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Adjust test_size
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train.dtypes)
```

    (999, 35)
    (999,)
    (250, 35)
    (250,)
    CUS_DOB                                      float64
    AGE                                          float64
    CUS_Month_Income                             float64
    CUS_Gender                                     int32
    CUS_Customer_Since                    datetime64[ns]
    YEARS_WITH_US                                float64
    # total debit transactions for S1            float64
    # total debit transactions for S2            float64
    # total debit transactions for S3            float64
    total debit amount for S1                    float64
    total debit amount for S2                    float64
    total debit amount for S3                    float64
    # total credit transactions for S1           float64
    # total credit transactions for S2           float64
    # total credit transactions for S3           float64
    total credit amount for S1                   float64
    total credit amount for S2                   float64
    total credit amount for S3                   float64
    total debit amount                           float64
    total debit transactions                     float64
    total credit amount                          float64
    total credit transactions                    float64
    total transactions                           float64
    CUS_Target                                   float64
    MARRIED                                      float64
    CUS_Marital_Status_DIVORCE                      bool
    CUS_Marital_Status_MARRIED                      bool
    CUS_Marital_Status_OTHER                        bool
    CUS_Marital_Status_PARTNER                      bool
    CUS_Marital_Status_SINGLE                       bool
    CUS_Marital_Status_WIDOWED                      bool
    TAR_Desc_EXECUTIVE                              bool
    TAR_Desc_LOW                                    bool
    TAR_Desc_MIDLE                                  bool
    TAR_Desc_PLATINUM                               bool
    dtype: object
    


```python
# Drop datetime columns (or convert them)
if 'datetime_column' in X_train.columns:
    X_train['datetime_column'] = pd.to_numeric(X_train['datetime_column'], errors='coerce')  # Converts to timestamp
    # Or drop if not needed
    # X_train = X_train.drop('datetime_column', axis=1)

# Convert boolean columns
bool_cols = X_train.select_dtypes(include='bool').columns
X_train[bool_cols] = X_train[bool_cols].astype(int)

# Ensure all features are numerical
X_train = X_train.select_dtypes(include=[float, int])

```


```python
print(X_train.isna().sum())
```

    CUS_DOB                                 0
    AGE                                     0
    CUS_Month_Income                        0
    CUS_Gender                              0
    YEARS_WITH_US                           0
    # total debit transactions for S1       0
    # total debit transactions for S2       0
    # total debit transactions for S3       0
    total debit amount for S1               0
    total debit amount for S2               0
    total debit amount for S3               0
    # total credit transactions for S1      0
    # total credit transactions for S2      0
    # total credit transactions for S3      0
    total credit amount for S1              0
    total credit amount for S2              0
    total credit amount for S3              0
    total debit amount                      0
    total debit transactions                0
    total credit amount                     0
    total credit transactions               0
    total transactions                      0
    CUS_Target                              0
    MARRIED                               999
    CUS_Marital_Status_DIVORCE              0
    CUS_Marital_Status_MARRIED              0
    CUS_Marital_Status_OTHER                0
    CUS_Marital_Status_PARTNER              0
    CUS_Marital_Status_SINGLE               0
    CUS_Marital_Status_WIDOWED              0
    TAR_Desc_EXECUTIVE                      0
    TAR_Desc_LOW                            0
    TAR_Desc_MIDLE                          0
    TAR_Desc_PLATINUM                       0
    dtype: int64
    


```python
X_train.fillna(X_train.mean(), inplace=True)  # Replace NaNs with mean
X_train.fillna(X_train.mode().iloc[0], inplace=True)  # Replace NaNs with mode

```


```python
X_train.dropna(inplace=True)

```


```python
print("Original dataset shape:", X.shape)

```

    Original dataset shape: (1249, 35)
    


```python
print(X.isnull().sum())  # Check for missing values in each column
print(X.shape)  # Check the overall shape of the dataset

```

    CUS_DOB                                  0
    AGE                                      0
    CUS_Month_Income                         0
    CUS_Gender                               0
    CUS_Customer_Since                       0
    YEARS_WITH_US                            0
    # total debit transactions for S1        0
    # total debit transactions for S2        0
    # total debit transactions for S3        0
    total debit amount for S1                0
    total debit amount for S2                0
    total debit amount for S3                0
    # total credit transactions for S1       0
    # total credit transactions for S2       0
    # total credit transactions for S3       0
    total credit amount for S1               0
    total credit amount for S2               0
    total credit amount for S3               0
    total debit amount                       0
    total debit transactions                 0
    total credit amount                      0
    total credit transactions                0
    total transactions                       0
    CUS_Target                               0
    MARRIED                               1249
    CUS_Marital_Status_DIVORCE               0
    CUS_Marital_Status_MARRIED               0
    CUS_Marital_Status_OTHER                 0
    CUS_Marital_Status_PARTNER               0
    CUS_Marital_Status_SINGLE                0
    CUS_Marital_Status_WIDOWED               0
    TAR_Desc_EXECUTIVE                       0
    TAR_Desc_LOW                             0
    TAR_Desc_MIDLE                           0
    TAR_Desc_PLATINUM                        0
    dtype: int64
    (1249, 35)
    


```python
# Check the first few rows of the 'MARRIED' column to ensure it is a valid series
print(X['MARRIED'].head())

# Check for null values in 'MARRIED' column specifically
print(X['MARRIED'].isnull().sum())

# Check the mode (most frequent value) of the 'MARRIED' column
most_frequent_value = X['MARRIED'].mode()

# If mode is empty or not found, it could mean the column is non-numeric and doesn't have a valid mode.
if not most_frequent_value.empty:
    X['MARRIED'] = X['MARRIED'].fillna(most_frequent_value[0])
else:
    print("No mode value found for 'MARRIED'")

# Option 2: Drop the column if not important
X = X.drop(columns=['MARRIED'])

```

    0   NaN
    1   NaN
    2   NaN
    3   NaN
    4   NaN
    Name: MARRIED, dtype: float64
    1249
    No mode value found for 'MARRIED'
    


```python
print(X.isnull().sum())  # To check if other columns have missing data

```

    CUS_DOB                               0
    AGE                                   0
    CUS_Month_Income                      0
    CUS_Gender                            0
    CUS_Customer_Since                    0
    YEARS_WITH_US                         0
    # total debit transactions for S1     0
    # total debit transactions for S2     0
    # total debit transactions for S3     0
    total debit amount for S1             0
    total debit amount for S2             0
    total debit amount for S3             0
    # total credit transactions for S1    0
    # total credit transactions for S2    0
    # total credit transactions for S3    0
    total credit amount for S1            0
    total credit amount for S2            0
    total credit amount for S3            0
    total debit amount                    0
    total debit transactions              0
    total credit amount                   0
    total credit transactions             0
    total transactions                    0
    CUS_Target                            0
    CUS_Marital_Status_DIVORCE            0
    CUS_Marital_Status_MARRIED            0
    CUS_Marital_Status_OTHER              0
    CUS_Marital_Status_PARTNER            0
    CUS_Marital_Status_SINGLE             0
    CUS_Marital_Status_WIDOWED            0
    TAR_Desc_EXECUTIVE                    0
    TAR_Desc_LOW                          0
    TAR_Desc_MIDLE                        0
    TAR_Desc_PLATINUM                     0
    dtype: int64
    

# simple imputer


```python
from sklearn.impute import SimpleImputer

# Separate the columns by data type
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
datetime_columns = X.select_dtypes(include=['datetime64']).columns

# Handle missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
X[numerical_columns] = imputer.fit_transform(X[numerical_columns])

# Convert boolean columns to integers (0 or 1)
X[categorical_columns] = X[categorical_columns].apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

# Handle missing values for categorical columns (using mode or constant for imputation)
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

# Optionally, convert back the boolean columns to their original format (if needed)
X[categorical_columns] = X[categorical_columns].astype(bool)

```


```python
default_datetime = pd.to_datetime('2024-01-01')
X[datetime_columns] = X[datetime_columns].fillna(default_datetime)

```


```python
# Check the overall missing values again
print(X.isnull().sum())
# Impute missing values for the remaining columns
numerical_imputer = SimpleImputer(strategy='mean')  # Or use 'median', 'most_frequent', etc.
X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])
```

    CUS_DOB                               0
    AGE                                   0
    CUS_Month_Income                      0
    CUS_Gender                            0
    CUS_Customer_Since                    0
    YEARS_WITH_US                         0
    # total debit transactions for S1     0
    # total debit transactions for S2     0
    # total debit transactions for S3     0
    total debit amount for S1             0
    total debit amount for S2             0
    total debit amount for S3             0
    # total credit transactions for S1    0
    # total credit transactions for S2    0
    # total credit transactions for S3    0
    total credit amount for S1            0
    total credit amount for S2            0
    total credit amount for S3            0
    total debit amount                    0
    total debit transactions              0
    total credit amount                   0
    total credit transactions             0
    total transactions                    0
    CUS_Target                            0
    CUS_Marital_Status_DIVORCE            0
    CUS_Marital_Status_MARRIED            0
    CUS_Marital_Status_OTHER              0
    CUS_Marital_Status_PARTNER            0
    CUS_Marital_Status_SINGLE             0
    CUS_Marital_Status_WIDOWED            0
    TAR_Desc_EXECUTIVE                    0
    TAR_Desc_LOW                          0
    TAR_Desc_MIDLE                        0
    TAR_Desc_PLATINUM                     0
    dtype: int64
    


```python
print(f"Number of rows in X_train: {X_train.shape[0]}")

```

    Number of rows in X_train: 0
    


```python
print(X_train.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 0 entries
    Data columns (total 34 columns):
     #   Column                              Non-Null Count  Dtype  
    ---  ------                              --------------  -----  
     0   CUS_DOB                             0 non-null      float64
     1   AGE                                 0 non-null      float64
     2   CUS_Month_Income                    0 non-null      float64
     3   CUS_Gender                          0 non-null      int32  
     4   YEARS_WITH_US                       0 non-null      float64
     5   # total debit transactions for S1   0 non-null      float64
     6   # total debit transactions for S2   0 non-null      float64
     7   # total debit transactions for S3   0 non-null      float64
     8   total debit amount for S1           0 non-null      float64
     9   total debit amount for S2           0 non-null      float64
     10  total debit amount for S3           0 non-null      float64
     11  # total credit transactions for S1  0 non-null      float64
     12  # total credit transactions for S2  0 non-null      float64
     13  # total credit transactions for S3  0 non-null      float64
     14  total credit amount for S1          0 non-null      float64
     15  total credit amount for S2          0 non-null      float64
     16  total credit amount for S3          0 non-null      float64
     17  total debit amount                  0 non-null      float64
     18  total debit transactions            0 non-null      float64
     19  total credit amount                 0 non-null      float64
     20  total credit transactions           0 non-null      float64
     21  total transactions                  0 non-null      float64
     22  CUS_Target                          0 non-null      float64
     23  MARRIED                             0 non-null      float64
     24  CUS_Marital_Status_DIVORCE          0 non-null      int32  
     25  CUS_Marital_Status_MARRIED          0 non-null      int32  
     26  CUS_Marital_Status_OTHER            0 non-null      int32  
     27  CUS_Marital_Status_PARTNER          0 non-null      int32  
     28  CUS_Marital_Status_SINGLE           0 non-null      int32  
     29  CUS_Marital_Status_WIDOWED          0 non-null      int32  
     30  TAR_Desc_EXECUTIVE                  0 non-null      int32  
     31  TAR_Desc_LOW                        0 non-null      int32  
     32  TAR_Desc_MIDLE                      0 non-null      int32  
     33  TAR_Desc_PLATINUM                   0 non-null      int32  
    dtypes: float64(23), int32(11)
    memory usage: 0.0 bytes
    None
    

#  Ensure Proper Datetime Conversion


```python
# Convert 'CUS_DOB' to datetime, invalid parsing will result in NaT
X_train['CUS_DOB'] = pd.to_datetime(X_train['CUS_DOB'], errors='coerce')

```

# Check for Missing or Invalid Values


```python
print(X_train['CUS_DOB'].isnull().sum())

```

    0
    

# Convert to Days Since Reference Date


```python
reference_date = pd.Timestamp('2000-01-01')
X_train['CUS_DOB'] = (X_train['CUS_DOB'] - reference_date).dt.days
```

# Convert the Entire Dataset to Float


```python
# Force conversion to float with errors='ignore' to prevent issues
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_train = X_train.fillna(0)  # Replace NaN values with 0 or another placeholder

```

# Convert the Data to Float


```python
X_train = X_train.astype(float)

```


```python
# Example of filtering issue
X = X.dropna()  # Dropping NaNs might result in empty data
y = y[X.index]  # Ensure target `y` matches filtered `X`

# Check shape after preprocessing
print(f"X shape after preprocessing: {X.shape}")
print(f"y shape after preprocessing: {y.shape}")

```

    X shape after preprocessing: (1249, 34)
    y shape after preprocessing: (1249,)
    


```python
# Use a smaller dataset (e.g., first 100 rows)
X_small = X.iloc[:100]
y_small = y.iloc[:100]

X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

```

    X_train shape: (80, 34)
    y_train shape: (80,)
    


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Check data types of columns
print(X_train.dtypes)

# Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

```

    CUS_DOB                                      float64
    AGE                                          float64
    CUS_Month_Income                             float64
    CUS_Gender                                     int32
    CUS_Customer_Since                    datetime64[ns]
    YEARS_WITH_US                                float64
    # total debit transactions for S1            float64
    # total debit transactions for S2            float64
    # total debit transactions for S3            float64
    total debit amount for S1                    float64
    total debit amount for S2                    float64
    total debit amount for S3                    float64
    # total credit transactions for S1           float64
    # total credit transactions for S2           float64
    # total credit transactions for S3           float64
    total credit amount for S1                   float64
    total credit amount for S2                   float64
    total credit amount for S3                   float64
    total debit amount                           float64
    total debit transactions                     float64
    total credit amount                          float64
    total credit transactions                    float64
    total transactions                           float64
    CUS_Target                                   float64
    CUS_Marital_Status_DIVORCE                      bool
    CUS_Marital_Status_MARRIED                      bool
    CUS_Marital_Status_OTHER                        bool
    CUS_Marital_Status_PARTNER                      bool
    CUS_Marital_Status_SINGLE                       bool
    CUS_Marital_Status_WIDOWED                      bool
    TAR_Desc_EXECUTIVE                              bool
    TAR_Desc_LOW                                    bool
    TAR_Desc_MIDLE                                  bool
    TAR_Desc_PLATINUM                               bool
    dtype: object
    Non-numeric columns: Index([], dtype='object')
    


```python
# Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(include=['object', 'category']).columns
print(non_numeric_cols)

# Encode non-numeric columns
le = LabelEncoder()
for col in non_numeric_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

```

    Index([], dtype='object')
    


```python
# Convert the column to datetime, invalid parsing will result in NaT (Not a Time)
customer['CUS_Customer_Since'] = pd.to_datetime(customer['CUS_Customer_Since'], errors='coerce')

# Encode non-numeric columns
le = LabelEncoder()
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = le.fit_transform(X_train[col])

# Fill missing values
X_train.fillna(X_train.mean(), inplace=True)

label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    customer[col] = le.fit_transform(customer[col])
    label_encoders[col] = le  # Save encoders if needed later

# Ensure y_train (target) is numeric
y_train = LabelEncoder().fit_transform(customer['CUS_Gender'])

# Drop the target column to create X_train
X_train = customer.drop(columns=['CUS_Gender'])
# Ensure the target variable is categorical
y_train = y_train.astype('int')

# Print the mutual information scores for each feature
print("Mutual Information Scores for Features:")
for col, score in zip(X_train.columns, y_train):
    print(f"{col}: {score:.4f}")

# Convert all boolean columns to integers (if not already done)
bool_cols = X_train.select_dtypes(include=['bool']).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)

# Verify data types
print("\nData types of X_train columns:\n", X_train.dtypes)

```

    Mutual Information Scores for Features:
    CUS_DOB: 1.0000
    AGE: 0.0000
    CUS_Month_Income: 0.0000
    CUS_Marital_Status: 0.0000
    CUS_Customer_Since: 0.0000
    YEARS_WITH_US: 1.0000
    # total debit transactions for S1: 0.0000
    # total debit transactions for S2: 1.0000
    # total debit transactions for S3: 1.0000
    total debit amount for S1: 0.0000
    total debit amount for S2: 1.0000
    total debit amount for S3: 1.0000
    # total credit transactions for S1: 1.0000
    # total credit transactions for S2: 0.0000
    # total credit transactions for S3: 1.0000
    total credit amount for S1: 1.0000
    total credit amount for S2: 1.0000
    total credit amount for S3: 1.0000
    total debit amount: 0.0000
    total debit transactions: 1.0000
    total credit amount: 1.0000
    total credit transactions: 1.0000
    total transactions: 0.0000
    CUS_Target: 1.0000
    TAR_Desc: 0.0000
    Status: 0.0000
    MARRIED: 0.0000
    
    Data types of X_train columns:
     CUS_DOB                                      float64
    AGE                                          float64
    CUS_Month_Income                             float64
    CUS_Marital_Status                            object
    CUS_Customer_Since                    datetime64[ns]
    YEARS_WITH_US                                float64
    # total debit transactions for S1            float64
    # total debit transactions for S2            float64
    # total debit transactions for S3            float64
    total debit amount for S1                    float64
    total debit amount for S2                    float64
    total debit amount for S3                    float64
    # total credit transactions for S1           float64
    # total credit transactions for S2           float64
    # total credit transactions for S3           float64
    total credit amount for S1                   float64
    total credit amount for S2                   float64
    total credit amount for S3                   float64
    total debit amount                           float64
    total debit transactions                     float64
    total credit amount                          float64
    total credit transactions                    float64
    total transactions                           float64
    CUS_Target                                   float64
    TAR_Desc                                      object
    Status                                        object
    MARRIED                                      float64
    dtype: object
    


```python
assert len(X_train) > 0, "X_train is empty!"
assert len(y_train) > 0, "y_train is empty!"
assert len(X_train) == len(y_train), "Mismatch in data sizes!"

```


```python
# Convert CUS_Customer_Since to datetime if it's not already
X_train['CUS_Customer_Since'] = pd.to_datetime(X_train['CUS_Customer_Since'])

# Calculate the number of days since a reference date (e.g., today's date)
reference_date = pd.Timestamp('today')
X_train['CUS_Customer_Since'] = (reference_date - X_train['CUS_Customer_Since']).dt.days.astype(int)

print("\nData types in X_train after conversion:\n", X_train.dtypes)

```

    
    Data types in X_train after conversion:
     CUS_DOB                               float64
    AGE                                   float64
    CUS_Month_Income                      float64
    CUS_Marital_Status                     object
    CUS_Customer_Since                      int32
    YEARS_WITH_US                         float64
    # total debit transactions for S1     float64
    # total debit transactions for S2     float64
    # total debit transactions for S3     float64
    total debit amount for S1             float64
    total debit amount for S2             float64
    total debit amount for S3             float64
    # total credit transactions for S1    float64
    # total credit transactions for S2    float64
    # total credit transactions for S3    float64
    total credit amount for S1            float64
    total credit amount for S2            float64
    total credit amount for S3            float64
    total debit amount                    float64
    total debit transactions              float64
    total credit amount                   float64
    total credit transactions             float64
    total transactions                    float64
    CUS_Target                            float64
    TAR_Desc                               object
    Status                                 object
    MARRIED                               float64
    dtype: object
    


```python
# Separate numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Fill NaN in numeric columns with their mean
X_train[numeric_cols].fillna(X_train[numeric_cols].mean(), inplace=True)

# For categorical columns, fill NaN with the mode (most frequent value)
X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0], inplace=True)

# Drop rows with any remaining NaN values
X_train.dropna(inplace=True)

# Align y_train with X_train after row drop
y_train = y_train[X_train.index]

print(X_train.isnull().sum())  # Confirm no missing values

```

    CUS_DOB                               0
    AGE                                   0
    CUS_Month_Income                      0
    CUS_Marital_Status                    0
    CUS_Customer_Since                    0
    YEARS_WITH_US                         0
    # total debit transactions for S1     0
    # total debit transactions for S2     0
    # total debit transactions for S3     0
    total debit amount for S1             0
    total debit amount for S2             0
    total debit amount for S3             0
    # total credit transactions for S1    0
    # total credit transactions for S2    0
    # total credit transactions for S3    0
    total credit amount for S1            0
    total credit amount for S2            0
    total credit amount for S3            0
    total debit amount                    0
    total debit transactions              0
    total credit amount                   0
    total credit transactions             0
    total transactions                    0
    CUS_Target                            0
    TAR_Desc                              0
    Status                                0
    MARRIED                               0
    dtype: int64
    


```python
# Fill NaN with the mean of each column
X_train.fillna(X_train.mean(), inplace=True)

# Drop rows containing NaN values
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]  # Ensure y_train aligns with X_train after row drop

print(X_train.isnull().sum())  # Confirm no missing values

```

    CUS_DOB                               0
    AGE                                   0
    CUS_Month_Income                      0
    CUS_Marital_Status                    0
    CUS_Customer_Since                    0
    YEARS_WITH_US                         0
    # total debit transactions for S1     0
    # total debit transactions for S2     0
    # total debit transactions for S3     0
    total debit amount for S1             0
    total debit amount for S2             0
    total debit amount for S3             0
    # total credit transactions for S1    0
    # total credit transactions for S2    0
    # total credit transactions for S3    0
    total credit amount for S1            0
    total credit amount for S2            0
    total credit amount for S3            0
    total debit amount                    0
    total debit transactions              0
    total credit amount                   0
    total credit transactions             0
    total transactions                    0
    CUS_Target                            0
    TAR_Desc                              0
    Status                                0
    MARRIED                               0
    dtype: int64
    

# Correlation heatmap


```python
# correlation_matrix
correlation_matrix = X_test.corr()
print(correlation_matrix['CUS_DOB'].sort_values(ascending=False))
```

    CUS_DOB                               1.000000
    CUS_Customer_Since                    0.995944
    CUS_Month_Income                      0.116811
    CUS_Marital_Status_SINGLE             0.096769
    TAR_Desc_LOW                          0.078002
    total debit amount for S2             0.045616
    total credit amount for S2            0.031167
    # total debit transactions for S2     0.027698
    TAR_Desc_EXECUTIVE                    0.025207
    # total credit transactions for S2    0.015844
    total debit amount                    0.010723
    total credit amount                   0.006029
    CUS_Marital_Status_WIDOWED            0.005633
    total credit amount for S3            0.000209
    total debit amount for S1            -0.000309
    total credit amount for S1           -0.002823
    total credit transactions            -0.004335
    # total credit transactions for S3   -0.005162
    # total credit transactions for S1   -0.005321
    total transactions                   -0.006711
    total debit transactions             -0.009923
    total debit amount for S3            -0.010117
    CUS_Marital_Status_MARRIED           -0.014755
    # total debit transactions for S1    -0.017193
    # total debit transactions for S3    -0.021453
    CUS_Target                           -0.024181
    CUS_Gender                           -0.028094
    TAR_Desc_MIDLE                       -0.028669
    TAR_Desc_PLATINUM                    -0.153464
    AGE                                  -0.184417
    CUS_Marital_Status_DIVORCE           -0.274123
    YEARS_WITH_US                        -0.969533
    CUS_Marital_Status_OTHER                   NaN
    CUS_Marital_Status_PARTNER                 NaN
    Name: CUS_DOB, dtype: float64
    


```python
# Plot Heatmap
plt.figure(figsize=(26, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='black')
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_92_0.png)
    



```python

plt.figure(figsize=(10, 5))
sns.boxplot(x=X_test['CUS_DOB'])
plt.title('Box Plot to Detect Outliers')
plt.show()
```


    
![png](output_93_0.png)
    


# Model implementation


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
# Drop rows with NaN
X.dropna(inplace=True)
y = y[X.index]  # Align target with features after dropping rows

```

    (999, 34)
    (999,)
    


```python
# Check data types and missing values
print(X_train.dtypes)
print(X_train.isnull().sum())
print(y_train.unique())  # Check classes in target variable

```

    CUS_DOB                                      float64
    AGE                                          float64
    CUS_Month_Income                             float64
    CUS_Gender                                     int32
    CUS_Customer_Since                    datetime64[ns]
    YEARS_WITH_US                                float64
    # total debit transactions for S1            float64
    # total debit transactions for S2            float64
    # total debit transactions for S3            float64
    total debit amount for S1                    float64
    total debit amount for S2                    float64
    total debit amount for S3                    float64
    # total credit transactions for S1           float64
    # total credit transactions for S2           float64
    # total credit transactions for S3           float64
    total credit amount for S1                   float64
    total credit amount for S2                   float64
    total credit amount for S3                   float64
    total debit amount                           float64
    total debit transactions                     float64
    total credit amount                          float64
    total credit transactions                    float64
    total transactions                           float64
    CUS_Target                                   float64
    CUS_Marital_Status_DIVORCE                      bool
    CUS_Marital_Status_MARRIED                      bool
    CUS_Marital_Status_OTHER                        bool
    CUS_Marital_Status_PARTNER                      bool
    CUS_Marital_Status_SINGLE                       bool
    CUS_Marital_Status_WIDOWED                      bool
    TAR_Desc_EXECUTIVE                              bool
    TAR_Desc_LOW                                    bool
    TAR_Desc_MIDLE                                  bool
    TAR_Desc_PLATINUM                               bool
    dtype: object
    CUS_DOB                               0
    AGE                                   0
    CUS_Month_Income                      0
    CUS_Gender                            0
    CUS_Customer_Since                    0
    YEARS_WITH_US                         0
    # total debit transactions for S1     0
    # total debit transactions for S2     0
    # total debit transactions for S3     0
    total debit amount for S1             0
    total debit amount for S2             0
    total debit amount for S3             0
    # total credit transactions for S1    0
    # total credit transactions for S2    0
    # total credit transactions for S3    0
    total credit amount for S1            0
    total credit amount for S2            0
    total credit amount for S3            0
    total debit amount                    0
    total debit transactions              0
    total credit amount                   0
    total credit transactions             0
    total transactions                    0
    CUS_Target                            0
    CUS_Marital_Status_DIVORCE            0
    CUS_Marital_Status_MARRIED            0
    CUS_Marital_Status_OTHER              0
    CUS_Marital_Status_PARTNER            0
    CUS_Marital_Status_SINGLE             0
    CUS_Marital_Status_WIDOWED            0
    TAR_Desc_EXECUTIVE                    0
    TAR_Desc_LOW                          0
    TAR_Desc_MIDLE                        0
    TAR_Desc_PLATINUM                     0
    dtype: int64
    [0 1]
    


```python
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = le.fit_transform(X_train[col])

X_train.fillna(X_train.mean(), inplace=True)

```


```python
# Calculate correlation coefficients
correlation = X_train.corrwith(y_train)

# Print correlation scores for each column
for col, score in correlation.items():
    print(f"{col}: {score:.4f}")

```

    CUS_DOB: 0.0242
    AGE: -0.0557
    CUS_Month_Income: -0.0135
    CUS_Gender: 0.0936
    CUS_Customer_Since: 0.0212
    YEARS_WITH_US: -0.0277
    # total debit transactions for S1: -0.3284
    # total debit transactions for S2: -0.3977
    # total debit transactions for S3: -0.5032
    total debit amount for S1: -0.2780
    total debit amount for S2: -0.3344
    total debit amount for S3: -0.4104
    # total credit transactions for S1: -0.2938
    # total credit transactions for S2: -0.3672
    # total credit transactions for S3: -0.4273
    total credit amount for S1: -0.2702
    total credit amount for S2: -0.3295
    total credit amount for S3: -0.3749
    total debit amount: -0.3362
    total debit transactions: -0.4077
    total credit amount: -0.3237
    total credit transactions: -0.3700
    total transactions: -0.4148
    CUS_Target: 0.0357
    CUS_Marital_Status_DIVORCE: -0.0414
    CUS_Marital_Status_MARRIED: -0.0590
    CUS_Marital_Status_OTHER: -0.0148
    CUS_Marital_Status_PARTNER: -0.0148
    CUS_Marital_Status_SINGLE: 0.0764
    CUS_Marital_Status_WIDOWED: -0.0169
    TAR_Desc_EXECUTIVE: -0.0915
    TAR_Desc_LOW: 0.1234
    TAR_Desc_MIDLE: -0.0793
    TAR_Desc_PLATINUM: -0.0247
    

# Data Exploring


```python
# Summary statistics for numeric columns
print(X_train.describe())
```

              CUS_DOB         AGE  CUS_Month_Income  CUS_Gender  \
    count  999.000000  999.000000        999.000000  999.000000   
    mean    12.609475    3.576408        100.693176    0.588589   
    min     12.580283    2.410142          0.000000    0.000000   
    25%     12.609701    3.391211         19.129312    0.000000   
    50%     12.609701    3.583048        114.471424    1.000000   
    75%     12.609701    3.802952        144.224957    1.000000   
    max     12.636895    4.431048        368.403150    1.000000   
    std      0.004785    0.323798         70.626049    0.492336   
    
                      CUS_Customer_Since  YEARS_WITH_US  \
    count                            999     999.000000   
    mean   2005-06-19 21:44:30.270270208       2.408071   
    min              1991-10-31 00:00:00       1.000000   
    25%              2005-06-17 00:00:00       2.410142   
    50%              2005-08-29 00:00:00       2.410142   
    75%              2005-10-12 00:00:00       2.410142   
    max              2018-02-23 00:00:00       3.036589   
    std                              NaN       0.145381   
    
           # total debit transactions for S1  # total debit transactions for S2  \
    count                         999.000000                         999.000000   
    mean                            3.024058                           2.961085   
    min                             0.000000                           0.000000   
    25%                             2.000000                           1.912931   
    50%                             2.843867                           2.884499   
    75%                             4.020726                           4.020726   
    max                             8.942014                           8.178289   
    std                             1.689063                           1.774579   
    
           # total debit transactions for S3  total debit amount for S1  ...  \
    count                         999.000000                 999.000000  ...   
    mean                            2.943177                  43.904877  ...   
    min                             0.000000                   0.000000  ...   
    25%                             1.817121                  23.412044  ...   
    50%                             2.884499                  38.075565  ...   
    75%                             4.071602                  57.548781  ...   
    max                             9.113782                 285.072695  ...   
    std                             1.849518                  33.115192  ...   
    
           # total credit transactions for S3  total credit amount for S1  \
    count                          999.000000                  999.000000   
    mean                             1.479716                   39.059027   
    min                              0.000000                    0.000000   
    25%                              0.000000                    0.000000   
    50%                              1.817121                   35.734145   
    75%                              2.154435                   56.301378   
    max                              5.528775                  366.444831   
    std                              1.111237                   37.899616   
    
           total credit amount for S2  total credit amount for S3  \
    count                  999.000000                  999.000000   
    mean                    40.039910                   39.445077   
    min                      0.000000                    0.000000   
    25%                      0.000000                    0.000000   
    50%                     36.034643                   36.840315   
    75%                     56.937364                   57.809681   
    max                    259.812448                  335.153497   
    std                     38.167012                   38.195924   
    
           total debit amount  total debit transactions  total credit amount  \
    count          999.000000                999.000000           999.000000   
    mean            67.935859                  4.520615            61.299551   
    min              0.000000                  0.000000             0.000000   
    25%             38.003646                  2.962496            25.661900   
    50%             58.873294                  4.198336            53.566912   
    75%             87.258708                  5.813717            83.677333   
    max            412.074784                 12.295885           470.856963   
    std             46.260133                  2.252118            52.291555   
    
           total credit transactions  total transactions  CUS_Target  
    count                 999.000000          999.000000  999.000000  
    mean                    2.254535            4.799951   13.049691  
    min                     0.000000            1.000000   13.027555  
    25%                     1.259921            3.207534   13.049124  
    50%                     2.620741            4.497941   13.051081  
    75%                     3.018294            6.077690   13.051081  
    max                     7.541987           12.954476   13.076472  
    std                     1.395802            2.279624    0.006591  
    
    [8 rows x 24 columns]
    

# Splitting the data


```python
from sklearn.model_selection import train_test_split

# Split dataset (80% for training, 20% for testing)
X_train_data, X_test_data = train_test_split(X_train, test_size=0.2, random_state=42)

```

# Model Training


```python
# Drop non-numeric columns before scaling
X_train_data_clean = X_train_data.drop(['CUS_Customer_Since'], axis=1)  # Replace 'DateColumn' with actual column name

```


```python
X_train_data['CUS_Customer_Since'] = pd.to_numeric(X_train_data['CUS_Customer_Since'], errors='coerce')
X_train_data.dropna(inplace=True)  # Remove any rows where conversion fails

```


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_data_scaled = scaler.fit_transform(X_train_data.drop('AGE', axis=1))

```

# LinearRegression


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_data.drop('AGE', axis=1), X_train_data['AGE'])

# Predict AGE for the training data
train_predictions = model.predict(X_train_data.drop('AGE', axis=1))

# Evaluate the model on the training data
mae = mean_absolute_error(X_train_data['AGE'], train_predictions)
mse = mean_squared_error(X_train_data['AGE'], train_predictions)
r2 = r2_score(X_train_data['AGE'], train_predictions)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


```

    Mean Absolute Error: 0.2497459952606198
    Mean Squared Error: 0.1048636194260272
    R^2 Score: 0.020225039918721044
    


```python
# Align the indices
predictions =train_predictions[:len(X_test_data['AGE'])]

plt.scatter(X_test_data['AGE'], predictions)
plt.xlabel('Actual AGE')
plt.ylabel('Predicted AGE')
plt.title('Actual vs Predicted AGE')
plt.show()

```


    
![png](output_109_0.png)
    


# RandomForestRegressor


```python
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)

grid_search.fit(X_train_data.drop('AGE', axis=1), X_train_data['AGE'])
print(f"Best Parameters: {grid_search.best_params_}")

```

    Best Parameters: {'max_depth': 10, 'n_estimators': 100}
    


```python
print(f"Lengths: X_train={len(X_train)}, X_test={len(X_test)}, y_train={len(y_train)}, y_test={len(y_test)}")

```

    Lengths: X_train=999, X_test=250, y_train=999, y_test=250
    


```python
print(X_test.index.equals(y_test.index))  # Should return True

```

    True
    


```python
# Handle datetime and bool columns
for col in X_train.columns:
    if np.issubdtype(X_train[col].dtype, np.datetime64):
        # Convert datetime to numerical days from minimum date
        X_train[col] = (X_train[col] - X_train[col].min()).dt.days
        X_test[col] = (X_test[col] - X_test[col].min()).dt.days
    elif np.issubdtype(X_train[col].dtype, np.bool_):
        # Convert bool to int
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

# Drop irrelevant columns (if needed)
irrelevant_columns = ['id', 'description']  # Replace with your actual column names
X_train = X_train.drop(columns=irrelevant_columns, errors='ignore')
X_test = X_test.drop(columns=irrelevant_columns, errors='ignore')

# Fit the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

```


```python
# Check lengths of y_test and predictions
print(f"Length of y_test: {len(y_test)}")
print(f"Length of predictions: {len(predictions)}")

# If lengths differ, adjust predictions to match y_test
if len(predictions) > len(y_test):
    predictions = predictions[:len(y_test)]  # Trim predictions if they are longer

# Plot the distribution of residuals (errors)
residuals = y_test - predictions
sns.distplot(residuals, bins=50, kde=True, color='blue')
plt.xlabel('Residuals (y_test - Predictions)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
```

    Length of y_test: 250
    Length of predictions: 250
    


    
![png](output_115_1.png)
    



```python
# Calculate residuals
residuals = y_test - predictions
# Create a DataFrame
customer = pd.DataFrame({'Residuals': residuals})
# Plot residuals using sns.histplot (since sns.distplot is deprecated)
sns.histplot(customer['Residuals'], kde=True, color="purple", bins=50)
plt.title('Distribution of Residuals')
plt.show()
customer.head()
```


    
![png](output_116_0.png)
    





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
      <th>Residuals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>679</th>
      <td>-0.70</td>
    </tr>
    <tr>
      <th>1050</th>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>901</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>243</th>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>328</th>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>



# Cross validation


```python
from sklearn.model_selection import cross_val_score

cross_val = cross_val_score(model, X_train_data.drop('AGE', axis=1), X_train_data['AGE'], cv=5)
print(f"Cross-Validation Scores: {cross_val}")
print(f"Mean Cross-Validation Score: {cross_val.mean()}")

```

    Cross-Validation Scores: [0.36605929 0.35319356 0.31590054 0.33957495 0.24106094]
    Mean Cross-Validation Score: 0.32315785732092994
    

# Residual Analysis


```python
# Ensure both Series have the same number of elements
min_length = min(len(X_test_data['CUS_DOB']), len(predictions))
residuals = X_test_data['CUS_DOB'][:min_length] - predictions[:min_length]

# Plot the residuals histogram
plt.hist(residuals, bins=50, color='g')
plt.title('Residual Histogram')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

```


    
![png](output_120_0.png)
    



```python
# Ensure both Series have the same length
min_length = min(len(X_test_data['AGE']), len(predictions))
residuals = X_test_data['AGE'][:min_length] - predictions[:min_length]

# Plot the residuals
plt.hist(residuals, bins=50, color='b', alpha=0.7)
plt.title('Residuals Histogram')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

```


    
![png](output_121_0.png)
    



```python
# Align predictions to the length of X_test_data['YEARS_WITH_US']
predictions_aligned = predictions[:len(X_test_data['YEARS_WITH_US'])]

# Calculate residuals
residuals = X_test_data['YEARS_WITH_US'] - predictions_aligned

# Plot the histogram
plt.hist(residuals, bins=50, color='b', alpha=0.7)
plt.title('Residuals Histogram')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

```


    
![png](output_122_0.png)
    



```python
# Convert predictions to binary classes (0 or 1)
predictions_classes = np.round(predictions_aligned).astype(int)

# Compare actual and predicted gender classifications
print(confusion_matrix(X_test_data['CUS_Gender'], predictions_classes))
print("Accuracy:", accuracy_score(X_test_data['CUS_Gender'], predictions_classes))


plt.hist(residuals, bins=50, color='g', alpha=0.7)
plt.title('Residuals Histogram for CUS_Gender')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

    [[78 15]
     [89 18]]
    Accuracy: 0.48
    


    
![png](output_123_1.png)
    



```python
# Align predictions to match residuals
predictions_aligned = predictions[:len(residuals)]

# Plot Residuals vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_aligned, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

```


    
![png](output_124_0.png)
    



```python
# Ensure predictions have the same length as X_test_data['CUS_DOB']
predictions_aligned = predictions[:len(X_test_data['CUS_DOB'])]

# Calculate error metrics
mae = mean_absolute_error(X_test_data['CUS_DOB'], predictions_aligned)
mse = mean_squared_error(X_test_data['CUS_DOB'], predictions_aligned)
rmse = mse**0.5
r2 = r2_score(X_test_data['CUS_DOB'], predictions_aligned)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

```

    Mean Absolute Error (MAE): 12.417850827894599
    Mean Squared Error (MSE): 154.2651750266588
    Root Mean Squared Error (RMSE): 12.420353256919016
    R² Score: -6266925.0198471695
    

# GridSearchCV


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train_data.drop('CUS_DOB', axis=1), X_train_data['CUS_DOB'])
print(grid_search.best_params_)

```

    {'max_depth': 20, 'n_estimators': 100}
    

## (i)Convert 'CUS_DOB' column to datetime


```python
# Convert 'CUS_DOB' column to datetime
X_train_data['CUS_DOB'] = pd.to_datetime(X_train_data['CUS_DOB'], errors='coerce')
X_test_data['CUS_DOB'] = pd.to_datetime(X_test_data['CUS_DOB'], errors='coerce')

```


```python
# Fill NaT values with placeholder reference date
X_train_data['CUS_DOB'] = X_train_data['CUS_DOB'].fillna(pd.Timestamp('2000-01-01'))
X_test_data['CUS_DOB'] = X_test_data['CUS_DOB'].fillna(pd.Timestamp('2000-01-01'))

```


```python
# Convert 'CUS_DOB' to days since '2000-01-01'
X_train_data['CUS_DOB'] = (X_train_data['CUS_DOB'] - pd.Timestamp('2000-01-01')).dt.days
X_test_data['CUS_DOB'] = (X_test_data['CUS_DOB'] - pd.Timestamp('2000-01-01')).dt.days

```

## (ii)Select only numerical columns


```python
# Select only numerical columns (int64 and float64 types)
numerical_columns = X_train_data.select_dtypes(include=['int64', 'float64']).columns

```


```python
# Convert any datetime columns to Unix timestamps
for col in X_train_data.select_dtypes(include=['datetime64']).columns:
    X_train_data[col] = X_train_data[col].astype('int64') // 1_000_000_000

for col in X_test_data.select_dtypes(include=['datetime64']).columns:
    X_test_data[col] = X_test_data[col].astype('int64') // 1_000_000_000

```


```python
# Exclude datetime columns
numerical_columns = [col for col in numerical_columns if col not in X_train_data.select_dtypes(include=['datetime64']).columns]

# Convert to float
X_train_data[numerical_columns] = X_train_data[numerical_columns].astype(float)
X_test_data[numerical_columns] = X_test_data[numerical_columns].astype(float)

```


```python
print(X_train_data.dtypes)
print(X_test_data.dtypes)

```

    CUS_DOB                               float64
    AGE                                   float64
    CUS_Month_Income                      float64
    CUS_Gender                              int32
    CUS_Customer_Since                    float64
    YEARS_WITH_US                         float64
    # total debit transactions for S1     float64
    # total debit transactions for S2     float64
    # total debit transactions for S3     float64
    total debit amount for S1             float64
    total debit amount for S2             float64
    total debit amount for S3             float64
    # total credit transactions for S1    float64
    # total credit transactions for S2    float64
    # total credit transactions for S3    float64
    total credit amount for S1            float64
    total credit amount for S2            float64
    total credit amount for S3            float64
    total debit amount                    float64
    total debit transactions              float64
    total credit amount                   float64
    total credit transactions             float64
    total transactions                    float64
    CUS_Target                            float64
    CUS_Marital_Status_DIVORCE               bool
    CUS_Marital_Status_MARRIED               bool
    CUS_Marital_Status_OTHER                 bool
    CUS_Marital_Status_PARTNER               bool
    CUS_Marital_Status_SINGLE                bool
    CUS_Marital_Status_WIDOWED               bool
    TAR_Desc_EXECUTIVE                       bool
    TAR_Desc_LOW                             bool
    TAR_Desc_MIDLE                           bool
    TAR_Desc_PLATINUM                        bool
    dtype: object
    CUS_DOB                               float64
    AGE                                   float64
    CUS_Month_Income                      float64
    CUS_Gender                              int32
    CUS_Customer_Since                    float64
    YEARS_WITH_US                         float64
    # total debit transactions for S1     float64
    # total debit transactions for S2     float64
    # total debit transactions for S3     float64
    total debit amount for S1             float64
    total debit amount for S2             float64
    total debit amount for S3             float64
    # total credit transactions for S1    float64
    # total credit transactions for S2    float64
    # total credit transactions for S3    float64
    total credit amount for S1            float64
    total credit amount for S2            float64
    total credit amount for S3            float64
    total debit amount                    float64
    total debit transactions              float64
    total credit amount                   float64
    total credit transactions             float64
    total transactions                    float64
    CUS_Target                            float64
    CUS_Marital_Status_DIVORCE               bool
    CUS_Marital_Status_MARRIED               bool
    CUS_Marital_Status_OTHER                 bool
    CUS_Marital_Status_PARTNER               bool
    CUS_Marital_Status_SINGLE                bool
    CUS_Marital_Status_WIDOWED               bool
    TAR_Desc_EXECUTIVE                       bool
    TAR_Desc_LOW                             bool
    TAR_Desc_MIDLE                           bool
    TAR_Desc_PLATINUM                        bool
    dtype: object
    


```python
# Select only numeric columns
numeric_data = customer.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_data.corr()

```

# feature importance


```python
# Get feature importances and column names
importance = model.feature_importances_
feature_names = X_train_data.drop('CUS_DOB', axis=1).columns

# Check lengths to ensure they match
print(f"Length of importance array: {len(importance)}")
print(f"Number of features: {len(feature_names)}")

```

    Length of importance array: 34
    Number of features: 33
    


```python
# Check if the model has been fit
if hasattr(model, 'feature_importances_'):
    print("The model has been trained successfully.")
else:
    print("Train the model first using the `fit` method.")

```

    The model has been trained successfully.
    


```python
# Filter non-zero importances
non_zero_indices = importance > 0
importance = importance[non_zero_indices]
feature_names = np.array(X_train_data.drop('CUS_DOB', axis=1).columns)
# Plot Feature Importances
plt.figure(figsize=(10, 10))
sns.barplot(x=importance, y=feature_names)
plt.title("Filtered Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
print(len(feature_names), len(importance))

```


    
![png](output_141_0.png)
    


    33 33
    


```python
print("Residuals Statistics:")
print(residuals.describe())

```

    Residuals Statistics:
    count    200.000000
    mean       2.221311
    std        0.290794
    min        1.200084
    25%        2.077642
    50%        2.350142
    75%        2.400142
    max        2.924018
    Name: YEARS_WITH_US, dtype: float64
    


```python
print("Missing values in y_test:", y_test.isnull().sum())
print("Missing values in predictions:", pd.Series(predictions).isnull().sum())

```

    Missing values in y_test: 0
    Missing values in predictions: 0
    

# GradientBoostingRegressor


```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=200)
model.fit(X_train_data.drop('CUS_DOB', axis=1), X_train_data['CUS_DOB'])
predictions = model.predict(X_test_data.drop('CUS_DOB', axis=1))
```


```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train_data.drop('CUS_DOB', axis=1), X_train_data['CUS_DOB'])

# Get the best hyperparameters
print(f"Best Parameters: {grid_search.best_params_}")

```

    Best Parameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
    


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(X_test_data['CUS_DOB'], predictions)
mse = mean_squared_error(X_test_data['CUS_DOB'], predictions)
rmse = mse**0.5
r2 = r2_score(X_test_data['CUS_DOB'], predictions)

print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")

```

    MAE: 0.0, RMSE: 0.0, R2: 1.0
    


```python
plt.figure(figsize=(8, 5))
sns.distplot(X_test_data['CUS_DOB'], kde=True, color="purple")
plt.title('Target Variable Distribution')
plt.xlabel('CUS_DOB')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_148_0.png)
    

