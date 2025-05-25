# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
df = pd.read_csv('Algerian_forest_fires_dataset.csv')
df.head()

# %%
df.info()

# %%
## Data Cleaning
df.isnull().sum()

# %%
df[df.isnull().any(axis=1)]

# %%
### The data set is converted into two sets based on the Region from 122th index , we can make a new column based on Region:
### 1. Bejaia Region Dataset
### 2. Sidi-Bel Abbes Region Dataset

## Add new column with region

# %%
df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1

# %%
df.head()


# %%
df.tail()

# %%
df[['Region']] = df[['Region']].astype(int)

# %%
df.info()

# %%
df.isnull().sum()

# %%
## Removing the null values 
df=df.dropna().reset_index(drop=True)

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.iloc[[122]]

# %%
df = df.drop(122).reset_index(drop=True)

# %%
df.iloc[[122]]

# %%
df.columns

# %%
## Fixing the spaces in the column names 

df.columns = df.columns.str.strip()

# %%
df.columns

# %%
## Changing the required columns to integer datatype
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

# %%
df.info()

# %%
df.head()


# %%
## Changing the other columns to float datatype 
objects = [features for features in df.columns if df[features].dtypes=='O']

# %%
for i in objects:
    if i!='Classes':
        df[i] = df[i].astype(float)

# %%
df.info()

# %%
df.describe()

# %%
## Let's saved the cleaned dataset
df.to_csv('Algerian_Forest_Fires_Cleaned_Dataset',index=False)

# %%
## exploratory data analysis
dfcopy = df

# %%
dfcopy.head()

# %%
dfcopy=dfcopy.drop(['day','month','year'],axis=1)

# %%
dfcopy.head()

# %%
## Encoding of categorical classes
dfcopy['Classes'] = np.where(dfcopy['Classes'].str.contains('not fire'),0,1)

# %%
dfcopy.head()


# %%
dfcopy.tail()

# %%
dfcopy['Classes'].value_counts()

# %%
import matplotlib.pyplot as plt

# Use updated style
plt.style.use('seaborn-v0_8')

dfcopy.hist(bins=50, figsize=(20,15))
plt.show()


# %%
## Percentage for pie chart
percentage = dfcopy['Classes'].value_counts(normalize=True)*100

# %%
## Plotting pie chart 
classlabels = ['Fire','Not Fire']
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
plt.title('Pie Chart of Classes')
plt.show()

# %%
dfcopy.corr()

# %%
sns.heatmap(dfcopy.corr())

# %%
## Box plot to check for outliers 
sns.boxplot(dfcopy['FWI'])

# %%
## Monthly Fire Analysis
## Encoding of categorical classes

dftemp = df.loc[df['Region']==1]
plt.subplots(figsize=(10,10))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)

# %%



