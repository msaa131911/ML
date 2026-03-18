#K-Nearest Neighbors (KNN)

# load the dataset
df = sns.load_dataset('titanic')

# check the number of missing values in each column
df.isnull().sum().sort_values(ascending=False)

# impute missing values with KNN imputer
from sklearn.impute import KNNImputer

# call the KNN class with number of neighbors = 4
imputer = KNNImputer(n_neighbors=4)

#impute missing values with KNN imputer
df['age'] = imputer.fit_transform(df[['age']])

# check the number of missing values in each column
df.isnull().sum().sort_values(ascending=False)