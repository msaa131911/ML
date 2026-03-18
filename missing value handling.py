# missing value handling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# load the Titanic dataset
data = sns.load_dataset('titanic')
data.head()

# check the number of missing values in each column
data.isnull().sum().sort_values(ascending=False)

# drop deck column
# data = data.drop('deck', axis=1)
data.drop('deck', axis=1, inplace=True)

# impute missing values with mean
data['age'] = data['age'].fillna(data['age'].mean()) # mode,median,mean can be used

# check the number of missing values in each column
data.isnull().sum().sort_values(ascending=False)