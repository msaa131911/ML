#Iterative Imputer (Random Forest)

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer  # important!
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

 
# 2. Load Dataset
 
df = sns.load_dataset('titanic')

# Check missing values
print("Before Imputation:\n", df.isnull().sum().sort_values(ascending=False))

 
# 3. Drop High Missing Column

df.drop('deck', axis=1, inplace=True)


# 4. Encode Categorical Columns

columns_to_encode = ['sex', 'embarked', 'who', 'class', 'embark_town', 'alive']

label_encoders = {}

for col in columns_to_encode:
    le = LabelEncoder()
    df[col] = df[col].astype(str)   # avoid error with NaN
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# 5. Select Columns for Imputation

cols_to_impute = [
    'age', 'fare', 'sibsp', 'parch',
    'sex', 'embarked', 'pclass'
]


# 6. Apply Iterative Imputer (Random Forest)

imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=0),max_iter=10,random_state=0)

df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])


# 7. Final Check

print("\nAfter Imputation:\n", df.isnull().sum().sort_values(ascending=False))


# 8. Show Data

print("\nSample Data:\n", df.head())