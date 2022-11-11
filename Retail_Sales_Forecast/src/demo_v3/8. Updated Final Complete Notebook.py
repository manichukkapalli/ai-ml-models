#!/usr/bin/env python
# coding: utf-8

# # Sales Forecasting of Major Retail Clothing Product Categories using Machine Learning Techniques

# **Problem Description:** Building a framework that provides monthly forecasts of the 12 months (ie., from Jan 2016 to Dec 2106) for "Womenclothing" product category with influencing factors of the sales such as holiday events, weather changes, Macroeconomic factors etc using the Regression model.

# **How this Business problem is converted into a Machine Learning problem and how it will help the business?**
# 
# A leading retailer in USA, wants to forecast sales for their product categories in their store based on the sales history of each category. Sales forecast has very high influence on the performance of the company’s business and hence these sales forecasts can be used to estimate company’s success or performance in the coming year. Accurate forecasts may lead to better decisions in business. Sales or revenues forecasting is very important for retail operations. Forecasting of retail sales helps retailer to take necessary measures to plan their budgets or investments in a period (monthly, yearly) among different product categories like women clothing, men clothing and other clothing and at the same time they can plan to minimize revenue loss from unavailability of products by investing accordingly.The good news is that powerful Machine Learning (ML) algorithm can help to forecast sales of different category clothng product for next few years.

# **Data**: There are 4 datasets 
# 1. Macro Economic Dataset 
# 2. Events and Holidays Dataset 
# 3. Weather Data Set 
# 4. Train Data (Sales and the Year/Month)
# 

# #####                                                                                 Sales Data Description
#     AttributeName                  Type               Description
#     Year                           temporal            Year
#     Month                          temporal            Month
#     ProductCategory                categorical         Clothing Products category - WomenClothing, MenClothing,OtherClothing
#     Sales(In ThousandDollars)      numeric             Value of the sales or revenue in thousand dollars
# 

# #### Weather Data Description
#     
#     AttributeName                             Description                                                 ActualType
#     Year                                       Year                                                        Temporal
#     Month                                      Month                                                       Temporal
#     Day                                        Day                                                         Temporal
#     Temp high (°C)                             Temperature HighInF                                         numeric
#     Temp avg (°C)                              Temperature AvgInF                                          numeric
#     Temp low (°C)                              Temperature LowinF                                          numeric
#     Dew Point high (°C)	                       DewPointHighInF	                                           numeric
#     Dew Point avg (°C)	                       DewPointAvginF	                                           numeric
#     Dew Point low (°C)	                       DewPointLowinF	                                           numeric
#     Humidity (%) high	                       Humidity HighPercent	                                       numeric
#     Humidity (%) avg	                       Humidity AvgPercent	                                       numeric
#     Humidity (%) low	                       Humidity LowPercent	                                       numeric
#     Sea Level Press.(hPa) high	               Sea Level Pressure High_hPa	                               numeric
#     Sea Level Press.(hPa) avg	               Sea Level Pressure Avg_hPa	                               numeric
#     Sea Level Press.(hPa) low	               Sea Level Pressure Low_hPa	                               numeric
#     Visibility (km) high	                   Visibility HighInKM	                                       numeric
#     Visibility (km) avg	                       Visibility AvgInKM	                                       numeric
#     Visibility (km) low	                       Visibility LowInKM	                                       numeric
#     Wind (km/h) low	Wind                       LowInKmperhour	                                           numeric
#     Wind (km/h) avg	Wind                       AvgInKmperhour	                                           numeric
#     Wind (km/h) high Wind                      HighInKmperhour	                                           numeric
#     Precip. (mm) sum	                       Precipitation sum in mm	                           numeric values and character
#     WeatherEvent	                           Details of weather like snow, rain, fog etc	               categorical
# 

# ##### Events Holiday Data Description
# 
#      AttributeName                    Description	                                   ActualType
#            Year	                        Year	                                        Temporal
#            MonthDate	                Month and date combination	                    Temporal
#            Event	                    Details of special event or holiday	            categorical
#            DayCategory	                Whether federal holiday or event	            categorical
# 

# ##### MacroEconomic Data Description
# 
#     AttributeName	                                 Description	                                         ActualType
#     1.Year-Month	                            Combination of Year and month	                               Temporal
#     2.MonthlyNominal GDP Index 	                Monthly NominalGDPIndex In Million Dollars	                   numeric
#     3.Monthly Real GDP Index 	                Monthly RealGDPIndex In Million Dollars	                       numeric
#     4.CPI	                                    CPI	                                                           numeric
#     5.PartyInPower	                            Political party which is in power	                           categorical
#     6.unemployment rate	                        unemployment rate	                                           numeric
#     7.CommercialBankInterestRateon
#     8.CreditCardPlans Commercial Bank           Interest Rate on Credit Card Plans	                           numeric
#     9.Finance Rate on Personal Loans  
#     at Commercial Banks, 24 Month Loan	        Finance Rate on Personal Loans 
#                                                 at CommercialBanks_24MonthLoan	                               numeric                                                                                
#     10.Earnings or wages in dollars per hour	Earnings or wages in dollars per hour	                       numeric
#     11.AdvertisingExpenses                      Expenses for ads in thousand dollars	                       numeric
#     12.Cotton Monthly Price                     Cotton Monthly Price_US cents per Pound_lbs	                   numeric
#     13.Change Percentage                        Change In Mly Cotton Price	                                   numeric
#     14.Average upland planted	                Average upland Cotton planted In Million Acres	               numeric
#     15.Average upland harvested             	Average upland Cotton harvested In Million Acres	           numeric
#     yieldperharvested acre	                    Cotton yield per harvested acre( in pounds ie lbs)	           numeric        
#     16.Production	                            Cotton Production In480_l bnetweight in Million Bales          numeric
#     17.Mill use	                                Cotton Mill Use In480_lb netweight in Million Bales	           numeric
#     18.Exports Cotton                           Explorts In480_lb netweight in Million Bales	               numeric
# 

# ### Importing Required Pacakges



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from IPython.display import Image
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from statsmodels.formula.api import ols
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import re
import time
import joblib


# ### Loading the Files and Reading the Data

# Reading all the given files Train.csv, Events_HolidaysData.xlsx, MacroEconomicData.xlsx & WeatherDataNew.xlsx into pandas using pd.read() function.


os.getcwd()




os.chdir('C:\\Users\\Riyaz Mehendi\\OneDrive - CriticalRiver Technologies Pvt. Ltd\\Desktop\\Retail_Sales_Forecast\\Data\\raw')




Train_Data = pd.read_csv("Train.csv")
EventHolidays_Data = pd.read_excel("Events_HolidaysData.xlsx")
Macroeconomic_Data = pd.read_excel("MacroEconomicData.xlsx")
Weather_Data_xlsx = pd.ExcelFile("WeatherData.xlsx")


# ### Understanding the Data

# ##### 1) Train Sales Data




#Checking top 5 rows of Train Data
Train_Data.head(5)





#Checking bottom 5 rows of Train Data
Train_Data.tail(5)





#Checking the Dimension
print("Shape of Train data is:")
Train_Data.shape





#Checking the datatypes
Train_Data.dtypes


#As ProdcutCategory is categorical so checking the count value of its 3 category
Train_Data['ProductCategory'].value_counts()


# #### 2) Event_Holidays Data


EventHolidays_Data = EventHolidays_Data[EventHolidays_Data['Year'] < 2016]


# In[11]:


EventHolidays_Data.head()


# In[12]:


EventHolidays_Data.tail()


# In[13]:


#Checking the datatypes
print("Datatypes of each columns:")
print(EventHolidays_Data.dtypes)


# In[14]:


#Checking for  Daycategory column and their counnt values
print("The categories and their count in 'DayCategory' column:")
print(EventHolidays_Data.DayCategory.value_counts())


# In[15]:


#Checking all the event columns count value
print(EventHolidays_Data.Event.value_counts())


# #### 3) MacroEconomic Data

# In[16]:


Macroeconomic_Data


# In[17]:


Macroeconomic_Data.tail(15)


# In[18]:


Macroeconomic_Data = Macroeconomic_Data.iloc[:84]


# In[19]:


Macroeconomic_Data.head()


# In[20]:


Macroeconomic_Data.tail()


# In[21]:


#Checking the datatypes of all the coulmns in data
print("Datatypes of each columns :")
print(Macroeconomic_Data.dtypes)


# In[22]:


# Printing the statistics of the macro economic table
Macroeconomic_Data.describe(include = 'all')


# #### 4) Weather Data

# As we see that weather dataset has different sheets with specific years data so Creating a single dataframe from the given tables

# In[23]:


## Adding each sheet from the excel file to a list of dataframes.
Weather_Data_list = []
for i in range(0,len(Weather_Data_xlsx.sheet_names)):
    Weather_Data_list.append(Weather_Data_xlsx.parse(Weather_Data_xlsx.sheet_names[i])) 
    
    # Setting the Year column with proper values
    Weather_Data_list[i].Year = Weather_Data_xlsx.sheet_names[i]

# Shifting each row one step upwards in the dataframe.
Weather_Data_list[5].loc[:,"Temp high (°C)":"WeatherEvent"] = Weather_Data_list[5].loc[:,"Temp high (°C)":"WeatherEvent"].shift(-1)
Weather_Data_list[5] = Weather_Data_list[5][:-1].copy()

# Combining list of weather datas into a single dataframe
Weather_Data = pd.DataFrame()
for df in Weather_Data_list:
    Weather_Data = pd.concat([Weather_Data,df])


# In[24]:


#checking the top 5 rows
Weather_Data.head()


# In[25]:


#checking the  bottom 5 rows
Weather_Data.tail()


# In[26]:


Weather_Data


# In[27]:


#checking the datatypes
Weather_Data.dtypes


# ##### Indexing all the dataframes with Date as index

# Pandas set_index() is a method to set a List or Series as index of a Data Frame. So as observed month date and year column in every data so here in all the four DataFrames we made new column 'Date' and set it as index.

# #### 1) Train Sales Data

# In[28]:


Train_Data['Date'] = Train_Data.Year.astype(str).str.cat(Train_Data.Month.astype(str), sep='-')
Train_Data['Date'] = pd.to_datetime(Train_Data.Date.astype(str) + "-1")
Train_Data.set_index('Date', inplace = True)
Train_Data.drop(["Year","Month"],axis =1, inplace=True)


# In[29]:


Train_Data.head(5)


# #### 2) Holidays Data

# In[30]:


# Extracting the date and month from the 'MonthDate' column.
EventHolidays_Data.MonthDate = EventHolidays_Data.MonthDate.astype(str).str[2:7]


# In[31]:


## Joining two columns year and  monthdate as new single column 'Date'.
EventHolidays_Data['Date'] = pd.to_datetime(EventHolidays_Data.Year.astype(str).str.cat(EventHolidays_Data.MonthDate.astype(str), sep='-'), format='%Y-%d-%m')
EventHolidays_Data.drop(["Year","MonthDate"],axis =1, inplace=True)
EventHolidays_Data.set_index('Date', inplace = True)


# In[32]:


EventHolidays_Data.head(5)


# In[33]:


EventHolidays_Data.tail(5)


# #### 3) MacroEconomic Data

# In[34]:


### Making column year-month as new single column 'Date'.
Macroeconomic_Data.rename(columns = {'Year-Month':'Date'}, inplace = True)
Macroeconomic_Data['Date'] = pd.to_datetime(Macroeconomic_Data.Date.astype(str) + "-1")
Macroeconomic_Data.set_index('Date', inplace = True)


# In[35]:


Macroeconomic_Data.head(5)


# In[36]:


Macroeconomic_Data.tail(5)


# #### 4) Weather Data

# In[37]:


### Making column year-month as new single column 'Date'.
Weather_Data['Date'] = pd.to_datetime(Weather_Data.Year.astype(str).str.cat(Weather_Data.Month.astype(str), sep='-')                                      .str.cat(Weather_Data.Day.astype(str), sep='/'))
Weather_Data.drop(["Year","Month","Day"],axis =1, inplace=True)
Weather_Data.set_index('Date', inplace = True)


# In[38]:


Weather_Data.head(5)


# In[39]:


Weather_Data.tail(5)


# ### Exploratory Data Analysis (EDA)  and Data Preprocessing 

# #### 1. Train Sales Data

# Checking for Null values

# In[40]:


Train_Data.isnull().sum()


# As it is numeric data so we will impute them using Mean value

# In[41]:


Train_Data.isnull().mean()


# In[42]:


trainmean=Train_Data["Sales(In ThousandDollars)"].mean()
trainmean


# In[43]:


Train_Data["Sales(In ThousandDollars)"]=Train_Data["Sales(In ThousandDollars)"].fillna(trainmean)


# In[44]:


Train_Data.isnull().sum()


# Our missing values got imputed with mean value

# Now Lets visualize Countplot on three categories 

# In[45]:


plt.title('Count value across all product category')
plt.xlabel('values')
plt.ylabel(' Product categories')
Train_Data["ProductCategory"].value_counts().plot(kind='barh')


# From above plot we observed that all the 3 categories of 'ProductCategory' has same count value of 84.

# In[46]:


g = sns.FacetGrid(data=Train_Data,col='ProductCategory')
g.map(plt.hist,'Sales(In ThousandDollars)')
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle('Count of sales value in each category', fontsize=14)


# From above plot we observed that Sales values are more for women clothing compared to other product category.

# In[47]:


# boxplot of a variable across various product categories
sns.boxplot(x='ProductCategory', y='Sales(In ThousandDollars)', data=Train_Data)
plt.yscale('log')
plt.title('Boxplot of variable across various product categories')
plt.xlabel('Product Categories')
plt.ylabel('Values')
plt.show()


# So above plot dipicts that the sales of MenClothing are on an average, lower than the other two categories, whereas the sales of other clothing looks much better.Overall, sales of WomenClothing is very much consistent and very high compared to other two product categories.

# In[48]:


Train_Data


# Checking the plot for growth of all 3 Category sales on yearwise.

# In[49]:


Women_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "WomenClothing" ,:]


# In[50]:


Women_Sales_Data


# In[51]:


Women_Sales_Data.plot()
plt.title("Sales for WomenClothing")
plt.legend().remove()
plt.show()


# In[52]:


Men_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "MenClothing" ,:]


# In[53]:


Men_Sales_Data


# In[54]:


Men_Sales_Data.plot()
plt.title("Sales for Men Clothing")
plt.legend().remove()
plt.show()


# In[55]:


Other_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "OtherClothing" ,:]


# In[56]:


Other_Sales_Data


# In[57]:


Other_Sales_Data.plot()
plt.title("Sales for Other Clothing")
plt.legend().remove()
plt.show()


# #### 2) Event_Holidays Data

# In[58]:


EventHolidays_Data


# Checking the Null values

# In[59]:


EventHolidays_Data.isnull().sum()


# In[60]:


plt.title('Count Values of Event and Federal Holiday Category')
plt.xlabel('values')
plt.ylabel('Categroy')
EventHolidays_Data["DayCategory"].value_counts().plot(kind='barh')


# From above plot we visualize that Federal Holiday having count value more than 85 is Higher than count value of Event presiding at 62.

# In[61]:


plt.title('Count Values of different categories in Event column')
plt.xlabel(' Count values')
plt.ylabel('Categroy')
EventHolidays_Data["Event"].value_counts().plot(kind='barh')


# From above plot we observed that 17 out of 23 categories of event columns has same count value 8.

# **Resampling the data**
# 
# Resampling involves changing the frequency of your Data observations. Two types of resampling are: Upsampling and Downsampling. There are perhaps two main reasons why I am interested in resampling the data: 
# 
#     1.Problem Framing: Resampling may be required if data is available at the same frequency that we want to make predictions.
#     2.Feature Engineering: Resampling can also be used to provide additional structure or insight into the learning problem for supervised learning models. So, Here we Resampled Data with 'M' which point outs to be Month End Frequency.

# In[62]:


# Converting 'Events' as 1 and 'Federal Holiday' as 4
#EventHolidays_Data['DayCategory'] = EventHolidays_Data['DayCategory'].map({'Event':1, 'Federal Holiday':4})
dummy_Data=pd.get_dummies(EventHolidays_Data['DayCategory'])


# In[63]:


dummy_Data


# In[64]:


EventHolidays_Data=dummy_Data


# In[65]:


EventHolidays_Data = EventHolidays_Data.resample('M').sum()


# In[66]:


EventHolidays_Data.head(10)


# In[67]:


EventHolidays_Data.tail(10)


# #### 3) Macroeconomic Data

# Checking for null values using Heatmap

# In[68]:


ax = plt.axes()
ax.set_title('Checking null values using HeatMap')
sns.heatmap(Macroeconomic_Data.isnull(), cbar=False, ax=ax)


# Now We have to look for categorical and numerical variables in the data

# In[69]:


Macroeconomic_Data.dtypes


# **Categorical variables:**

# In[70]:


print(Macroeconomic_Data["PartyInPower"].value_counts(), "\n")


# The attribute 'PartyInPower' has only one category for all the instances and hence is not useful for the analysis.

# In[71]:


# Droping the column 'PartyInPower'
Macroeconomic_Data.drop('PartyInPower', axis=1,inplace=True)


# In[72]:


print(Macroeconomic_Data["AdvertisingExpenses (in Thousand Dollars)"].value_counts(), "\n")


# The attribute 'AdvertisingExpenses (in Thousand Dollars)' has 88% of missing values which are denoted as '?'so will drop from our data.

# In[73]:


# Droping the column 'AdvertisingExpenses (in Thousand Dollars)'
Macroeconomic_Data.drop('AdvertisingExpenses (in Thousand Dollars)', axis=1,inplace=True)


# In[74]:


#Checking whether columns are dropped or not 
Macroeconomic_Data.info()


# **Numerical Variables**:

# In[75]:


# Calculating the correlation of each variables
correlation = Macroeconomic_Data.corr()

### Ploting the correlation 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize =(13,13))
sns.heatmap(data= correlation,cmap="YlGnBu", annot =True, square= True)
plt.title('The Correlation of each features in Macroeconomic dataset', fontsize= '20')
plt.show()


# From above correlation graph we see that the dark blue boxes are the most correlated, we observed that most correlated pairs
# 
#     1.Exports-Cotton Monthly Price - US cents per Pound(lbs) (0.72) 
#     2.Exports-Production (in 480-lb netweright in million bales)(0.64) 
#     3.Also, Exports-Average upland harvested(million acres) (0.57) are quite correlated. 
# 
# We also note that this data is from a specific time period only. Thus, from a risk-minimisation point of view,we should not invest in these pairs , since if one goes down, the other is likely to go down as well (if one goes up, it will be benefited).

# #### 4) Weather Data

# Checking Null values and specific datatypes

# In[76]:


Weather_Data


# In[77]:


Weather_Data.dtypes


# In[78]:


Weather_Data.describe()


# In[79]:


#Counting WeatherEvent columns count value
Weather_Data['WeatherEvent'].value_counts()


# In[80]:


plt.title('Count Values of different categories in WeatherEvent column')
plt.xlabel(' Count values')
plt.ylabel('WeatherEvent')
Weather_Data['WeatherEvent'].value_counts().plot(kind='barh')


# From above plot we observed that Rain has more count values and so it has most significant role in WeatherEvent

# Checking for non-numeric characters in numeric variables

# NOTE:- We find value T in 'Precip. (mm) sum' column so "T" stands for Trace. This is a small amount of precipitation that will wet a raingage but is less than the 0.01 inch measuring limit. Hence converting all the "T" values to 0.01

# In[81]:


# Converting the cells that have the value "T" to O.01
Weather_Data[Weather_Data.columns[18]] = Weather_Data[Weather_Data.columns[18]].apply(lambda x: 0.01 if x == 'T' else x)


# In[82]:


#Checking data having values '-'.
print((Weather_Data.iloc[:,:] =='-').sum())


# In[83]:


# Converting all cells that have the value "-" to NaN
Weather_Data = Weather_Data.applymap(lambda x: np.nan if x == '-' else x)


# In[84]:


Weather_Data.head(5)


# In[85]:


Weather_Data.tail(5)


# In[86]:


# Replacing the missing values in 'WeatherEvent' column to "NA"
Weather_Data.WeatherEvent.fillna(value="NotApplicable", inplace=True)

# Droping the rows having all columns as NaN
Weather_Data = Weather_Data[~Weather_Data["Temp high (°C)"].isnull()]


# In[87]:


Weather_Data.describe(include='all')


# In[88]:


Weather_Data.dtypes


# In[89]:


print("Size before dropping NaN rows",Weather_Data.shape,"\n")
Weather_Data = Weather_Data.dropna()
print("\nSize after dropping NaN rows",Weather_Data.shape)


# In[90]:


Weather_Data.isnull().sum()


# In[91]:


Weather_Data['WeatherEvent'].value_counts().plot(kind='barh')


# In[92]:


#count value  of WeatherEvent column
Weather_Data['WeatherEvent'].value_counts()


# Around 95% are Not Applicable values in WeatherEvent so drop Column 'WeatherEvent"

# In[93]:


Weather_Data.drop(['WeatherEvent'],axis=1,inplace=True)


# In[94]:


#checking whether all datatypes are correct
Weather_Data.dtypes


# Plotting correlation plot for Weather Data

# In[95]:


correlation_weather = Weather_Data.corr()

### Ploting the correlation 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize =(12,12))
sns.heatmap(data= correlation_weather, annot =True, square= True)
plt.title('The Correlation of each features in Weather dataset', fontsize= '20')
plt.show()


# Checking for Outliers in the data

# In[96]:


plt.figure(figsize=(40,50))
sns.boxplot(data=Weather_Data,orient='h',palette="Set2")
plt.show()


# From above plot we observed that 10 columns has outliers in the data.

# In[97]:


Weather_Data.info()


# In[98]:


##Resampling the weather data with the mean of each month
Weather_Data_mean = Weather_Data.resample('M').mean()


# In[99]:


#Rename columns so that they have no spaces
Weather_Data_mean.columns = ['Temphigh', 'Tempavg', 'Templow', 'DewPointhigh','DewPointavg','DewPointlow','Humidityhigh',
                        'Humidityavg','Humiditylow','SeaLevelPresshigh','SeaLevelPressavg','SeaLevelPresslow','Visibilityhigh',
                         'Visibilityavg', 'Visibilitylow','Windlow','Windavg','Windhigh','Precipsum']
Weather_Data_mean.head()


# In[100]:


Weather_Data_mean.tail()


# Removing Columns with High and Low Values as we have Average values which will make more weightage in prediction and No. of features get reduce. 

# In[101]:


Weather_Data_mean.columns


# In[102]:


Weather_Data_mean=Weather_Data_mean.drop( ['Temphigh','Templow','DewPointhigh','DewPointlow', 'Humidityhigh','Humiditylow', 'SeaLevelPresshigh', 'SeaLevelPresslow','Visibilityhigh','Visibilitylow', 'Windlow','Windhigh'],axis = 1)


# In[103]:


Weather_Data_mean.info()


# In[104]:


Weather_Data_mean.shape


# In[105]:


EventHolidays_Data.shape


# In[106]:


Macroeconomic_Data.shape


# #### Merging the DataFrames

# Now,we had same shape(Number of rows) in all 3 Dataframes except Train Data. We will merge them before that we have to synchronise the Indexes of the dataframes
# 
# 

# In[107]:


# Synchronising the index before merging
EventHolidays_Data.index = Macroeconomic_Data.index
Weather_Data_mean.index = Macroeconomic_Data.index


# In[108]:


#Merging the data
Merged_Data = Macroeconomic_Data.join(Weather_Data_mean)
Merged_Data = Merged_Data.join(EventHolidays_Data)


# In[109]:


Merged_Data.shape


# In[110]:


Merged_Data.head(5)


# Correlation plot of Merged_Data

# In[111]:


correlation_merge = Merged_Data.corr()

### Ploting the correlation 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize =(25,25))
sns.heatmap(data= correlation_merge, annot =True, square= True)
plt.title('The Correlation of each features in Merged dataset', fontsize= '25')
plt.show()


# Checking the top coorelated features having highest values

# In[112]:


def get_redundant_pairs(Merged_Data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = Merged_Data.columns
    for i in range(0, Merged_Data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(Merged_Data, n=43):
    au_corr = Merged_Data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(Merged_Data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(Merged_Data, 30))


# In[113]:


Merged_Data.to_csv("C:\\Users\\Riyaz Mehendi\\OneDrive - CriticalRiver Technologies Pvt. Ltd\\Desktop\\Retail_Sales_Forecast\\data\\processed\\Preprocessed_Merged_Data.csv")


# In[114]:


Merged_Data


# In[115]:


Merged_Data= Merged_Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# In[116]:


Merged_Data


# In[117]:


Train_Data = Train_Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# In[118]:


Train_Data


# In[119]:


Train_Data.isnull().sum()


# As we are predicitng sales for all three categories so we will be dividing into three different dataframe.

# In[120]:


Women_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "WomenClothing" ,:]


# In[121]:


Women_Sales_Data


# In[122]:


Women_Sales_Data.info()


# In[123]:


Men_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "MenClothing" ,:]


# In[124]:


Men_Sales_Data


# In[125]:


Men_Sales_Data.info()


# In[126]:


Other_Sales_Data = Train_Data.loc[Train_Data.ProductCategory == "OtherClothing" ,:]


# In[127]:


Other_Sales_Data


# In[128]:


Other_Sales_Data.info()


# Let's merge our Target in Sales data with merged data with category wise

# In[129]:


Womens_Merged_data = Women_Sales_Data.join(Merged_Data)


# In[130]:


Womens_Merged_data


# In[131]:


Womens_Merged_data.info()


# In[132]:


Mens_Merged_data = Men_Sales_Data.join(Merged_Data)


# In[133]:


Mens_Merged_data


# In[134]:


Mens_Merged_data.info()


# In[135]:


Others_Merged_data = Other_Sales_Data.join(Merged_Data)


# In[136]:


Others_Merged_data


# Now merging all catgeory merged data

# In[137]:


All_Merged= pd.concat([Mens_Merged_data,Womens_Merged_data,Others_Merged_data],ignore_index=True)


# In[138]:


All_Merged.to_csv('C:\\Users\\Riyaz Mehendi\\OneDrive - CriticalRiver Technologies Pvt. Ltd\\Desktop\\Retail_Sales_Forecast\\data\\processed\\All_Combined_Data.csv')


# In[139]:


All_Merged.info()


# In[140]:


All_Merged


# ##### Correlation with the Target label - All Categories

# In[141]:


plt.figure(figsize=(40,40))
cor = All_Merged.corr()
sns.heatmap(cor, annot=True)
plt.show()


# Creating Dataframe with only numeric columns

# In[142]:


All_Merged_Numeric_Cols = All_Merged.drop(['ProductCategory'], axis = 1)


# In[143]:


All_Merged_Numeric_Cols


# Top Absolute Correlations from Numeric data

# In[144]:


def get_redundant_pairs(All_Merged_Numeric_Cols):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = All_Merged_Numeric_Cols.columns
    for i in range(0, All_Merged_Numeric_Cols.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(All_Merged_Numeric_Cols, n=43):
    au_corr = All_Merged_Numeric_Cols.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(All_Merged_Numeric_Cols)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(All_Merged_Numeric_Cols, 50))


# Creating separate dataframe for Indpendent features and Target feature

# In[145]:


Indpendent_features= All_Merged[['MonthlyNominalGDPIndexinMillion', 'MonthlyRealGDPIndexinMillion',
       'CPI', 'unemploymentrate',
       'CommercialBankInterestRateonCreditCardPlans',
       'FinanceRateonPersonalLoansatCommercialBanks24MonthLoan',
       'Earningsorwagesindollarsperhour',
       'CottonMonthlyPriceUScentsperPoundlbs', 'Changein',
       'Averageuplandplantedmillionacres',
       'Averageuplandharvestedmillionacres', 'yieldperharvestedacre',
       'Productionin480lbnetwerightinmillionbales',
       'Millusein480lbnetwerightinmillionbales', 'Exports', 'Tempavg',
       'DewPointavg', 'Humidityavg', 'SeaLevelPressavg', 'Visibilityavg',
       'Windavg', 'Precipsum', 'Event', 'FederalHoliday']]


# In[146]:


Target_features=All_Merged[['SalesInThousandDollars']]


# ##### Normalising all the features

# In[147]:


#defining a normalisation function 
def normalize (x): 
    return ( (x-np.min(x))/ (max(x) - min(x)))


# In[148]:


# applying normalize ( ) to all columns 
Indpendent_features = Indpendent_features.apply(normalize)


#  ##### Checking for Multicolinearity using Variable Inflation Factor
# In regression, "multicollinearity" refers to predictors that are correlated with other predictors. Multicollinearity occurs when your model includes multiple factors that are correlated not just to your response variable, but also to each other. So we use VIF to remove highly correlated predictors from the model.

# In[149]:


x_var = Indpendent_features
y_var = Target_features
x_var_names = Indpendent_features.columns
for i in range(0, len(x_var_names)):
    y = x_var[x_var_names[i]]
    x = x_var[x_var_names.drop(x_var_names[i])]
    lr1 = LinearRegression()
    rsq = lr1.fit(x,y).score(x,y)
    vif = round(1/(1-rsq),2)
    print (x_var_names[i], " VIF = " , vif)


# As observed above VIF value is too high because of Macroeconomic data and Weather data which has less amount of data

# ##### Feature Importance using ANOVA

# In[150]:


Sel = SelectKBest(k =5, score_func = f_regression)
Sel.fit(Indpendent_features, Target_features)


# In[151]:


#creating imporatnce features for Merged_Train data
Important_features = []
importances = Sel.scores_
indices = np.argsort(importances)[::-1]
for f in range(Indpendent_features.shape[1]):    
    Important_features = np.append(Important_features, Indpendent_features.columns.values[indices[f]])


# In[152]:


Important_features


# In[153]:


#plotting bar plot of feature importance
plt.figure(figsize =(15,5))
plt.title("Feature importance")
plt.bar(range(Indpendent_features.shape[1]), importances[indices],
       color="#138D75", align="center")
plt.xticks(range(Indpendent_features.shape[1]), Important_features, rotation = 90)
plt.xlim([-1, Indpendent_features.shape[1]])
plt.ylabel('F - score')
plt.xlabel('Features according to their importance')
plt.show()


# Dropping Columns which are not important for predictions

# In[154]:


All_Sales_df= All_Merged.drop(['Averageuplandharvestedmillionacres','Productionin480lbnetwerightinmillionbales','Tempavg','DewPointavg','Humidityavg' ], axis = 1)


# In[155]:


All_Sales_df.info()


# Product category need to be converted into numeric

# In[156]:


# label_encoder
#label_encoder = preprocessing.LabelEncoder()


# In[157]:


#All_Sales_df['ProductCategory']= label_encoder.fit_transform(All_Sales_df['ProductCategory'])


# In[158]:


All_Sales_df.replace({'MenClothing': 0, 'WomenClothing': 1,'OtherClothing':2},inplace=True)


# In[159]:


All_Sales_df


# Shuffling the dataframe rows as our product category column has values in sequential format so which will not give us desired ouput and may lead to poor performance on model

# In[160]:


# shuffle the DataFrame rows
All_Sales_df = All_Sales_df.sample(frac = 1)


# In[161]:


All_Sales_df


# In[162]:


All_Sales_df.reset_index(drop=True, inplace=True)


# In[163]:


All_Sales_df


# In[164]:


All_Sales_df.to_csv("C:\\Users\\Riyaz Mehendi\\OneDrive - CriticalRiver Technologies Pvt. Ltd\\Desktop\\Retail_Sales_Forecast\\data\\processed\\Final_Data.csv")


# In[165]:


All_Sales_df


# ### Data Splitting

# In[166]:


X= All_Sales_df.drop(['SalesInThousandDollars'],axis=1)


# In[167]:


X


# In[168]:


y= All_Sales_df['SalesInThousandDollars']


# In[169]:


y


# Splitting data intp train-test with 75% and 25% ratio

# In[170]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle = False)


# In[171]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Model Building & Evaluation
# To undertake the model selection step, we first need to create a dictionary containing the name of each model we want to test, and the name of the model class,

# In[172]:


regressors = {
    "LinearRegression": LinearRegression(),
    "Ridge":Ridge(),
    "XGBRegressor": XGBRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "SVR": SVR()
}


# Next we’ll create a Pandas dataframe into which we’ll store the data. Then we’ll loop over each of the models, fit it using the X_train and y_train data, then generate predictions from X_test and calculate the mean MAPE from 10 rounds of cross-validation. That will give us the MAPE for the X_test data, plus the average MAPE for the training data set.

# In[173]:


df_models = pd.DataFrame(columns=['model', 'run_time', 'mape', 'mape_cv'])

for key in regressors:

    print('*',key)

    start_time = time.time()

    regressor = regressors[key]
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = cross_val_score(model, 
                             X_train, 
                             y_train,
                             scoring="neg_mean_absolute_percentage_error", 
                             cv=10)

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60,2)),
           'mape': (mean_absolute_percentage_error(y_pred, y_test)*100),
           'mape_cv': (np.mean(np.sqrt(-scores)))
    }

    df_models = df_models.append(row, ignore_index=True)


# By sorting the results of the df_models dataframe, you can see the performance of each model.

# In[174]:


df_models.head(6).sort_values(by='mape_cv', ascending=True)


# ##### Assess the top performing model
# Next, we’ll fit the the XGBRegressor() model to the data using its default parameters and plot the performance of the predictions against the actual values. As you can see, this is already looking pretty good. The tuning step might shave a bit more off the MAPE.

# In[175]:


regressor = XGBRegressor(random_state=42)
model = regressor.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[176]:


test = pd.DataFrame({'Predicted value':y_pred, 'Actual value':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual value','Predicted value'])


# ##### Tune the model’s hyperparameters
# To tune the XGBRegressor() model (or any Scikit-Learn compatible model) the first step is to determine which hyperparameters are available for tuning. You can view these by printing model.get_params(), however, you’ll likely need to check the documentation for the selected model to determine how they can be tuned.

# In[177]:


model.get_params()


# Next, taking a small selection of the hyperparameters and add them to a dict() and assign this to the param_grid. We’ll then define the model and configure the GridSearchCV function to test each unique combination of hyperparameters and record the neg_root_mean_squared_error on each iteration. After going through the whole batch, the optimum model parameters can be printed.

# In[178]:


param_grid = dict(
    n_jobs=[16],
    learning_rate=[0.1, 0.5],
    objective=['reg:squarederror'],
    max_depth=[5, 10, 15], 
    n_estimators=[100, 200, 500],
    subsample=[0.2, 0.8, 1.0],
    gamma=[0.05, 0.5],
    scale_pos_weight=[0, 1],
    reg_alpha=[0, 0.5],
    reg_lambda=[1, 0],
)


# In[179]:


model = XGBRegressor(random_state=42, verbosity=1)

grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='neg_mean_absolute_percentage_error',
                           )


# In[180]:


get_ipython().run_cell_magic('time', '', "best_model = grid_search.fit(X_train, y_train)\nprint('Optimum parameters', best_model.best_params_)")


# Finally, take the optimum parameters identified by GridSearchCV and add them to the XGBRegressor() model and re-fit it to your training data, generating new predictions from the test data and assess the mape value.

# In[200]:


ModelXGBregressor = XGBRegressor(gamma=0.05, learning_rate= 0.1, max_depth= 10, n_estimators= 100,
                         n_jobs=16, objective='reg:squarederror', reg_alpha= 0, reg_lambda= 1, 
                         scale_pos_weight=0, subsample=1.0,random_state=42)


# In[201]:


TunedXGBmodel = ModelXGBregressor.fit(X_train, y_train)
xgboost_pred = TunedXGBmodel.predict(X_test)


# In[202]:


print(mean_absolute_percentage_error(y_test,xgboost_pred)*100)


# In[203]:


test_1 = pd.DataFrame({'Predicted value':xgboost_pred, 'Actual value':y_test})
fig= plt.figure(figsize=(16,8))
test_1 = test_1.reset_index()
test_1 = test_1.drop(['index'],axis=1)
plt.plot(test_1[:50])
plt.legend(['Actual value','Predicted value'])


# Changing working directory to models folder to save our model

# In[204]:


os.chdir("C:\\Users\\Riyaz Mehendi\\OneDrive - CriticalRiver Technologies Pvt. Ltd\\Desktop\\Retail_Sales_Forecast\\models")


# Let's try with Hyperparameter tuning Random forest model and check Model performance

# In[186]:


# RandomForest with cross validation
parameter_value = [
  {'n_estimators': [75], 'max_features': [9, 20, 25],'oob_score' : [True], 'max_depth': [60, 75, 80], 'min_samples_leaf':[1]}
]


# In[187]:


rf_grid = GridSearchCV(estimator= RandomForestRegressor(),param_grid=parameter_value, n_jobs=-1)


# In[188]:


rf_grid = rf_grid.fit(X_train,y_train)


# In[189]:


rf_grid_predicted= rf_grid.predict(X_test)
rf_grid_predicted


# In[190]:


print(mean_absolute_percentage_error(y_test,rf_grid_predicted)*100)


# In[191]:


rf_grid.best_estimator_


# In[192]:


# Random Forest
RF_model = RandomForestRegressor(max_depth=80, max_features=9,
                      n_estimators=75, oob_score=True,random_state=42)


# In[193]:


RF_model.fit(X_train,y_train)


# In[194]:


RF_Predicted= RF_model.predict(X_test)
RF_Predicted


# In[195]:


print(mean_absolute_percentage_error(y_test,RF_Predicted)*100)


# In[196]:


test_2 = pd.DataFrame({'Predicted value':RF_Predicted, 'Actual value':y_test})
fig= plt.figure(figsize=(16,8))
test_2 = test_2.reset_index()
test_2 = test_2.drop(['index'],axis=1)
plt.plot(test_2[:50])
plt.legend(['Actual value','Predicted value'])


# Plotting feature importance using RF

# In[197]:


important_features = []
importances = RF_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("*************************************")
print("*                                   *")
print("*      Ranking of the features      *")
print("*                                   *")
print("*************************************")
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f+1, X_train.columns.values[indices[f]], importances[indices[f]]))
    important_features = np.append(important_features, X_train.columns.values[indices[f]])


# In[198]:


std = np.std([tree.feature_importances_ for tree in RF_model.estimators_], axis=0)
plt.figure(figsize =(15,5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="#138D75", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), important_features, rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel('Importance with their inter-trees variability')
plt.xlabel('Features according to their importance')
plt.show()


# In[199]:


# Save the model as a pickle in a file
joblib.dump(RF_model, 'TunedRF_Model.pkl')
 
# Load the model from the file
RFModel_from_joblib = joblib.load('TunedRF_Model.pkl')
 
# Use the loaded model to make predictions
RFModel_from_joblib.predict(X_test)


# In[ ]:




