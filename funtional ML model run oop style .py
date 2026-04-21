#funtional ML model run oop style 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#kaj holo missing value gulo ke handle kora, jemon mean diye fill kora numeric data er jonno, most_frequent diye fill kora categorical data er jonno.
from sklearn.impute import SimpleImputer
import pandas as pd

# load data
df = pd.read_csv(r'E:\ALL_IN_ONE\DATA\Teen_Mental_Health_Dataset.csv')


class ML_Model:
    def __init__(self, df):
        self.df = df.copy()

        #categorical → numeric
        self.df["gender"] = self.df["gender"].map({
            "Male": 0,
            "Female": 1,
            "Other": 2
        })

        #all string → numeric (auto)
        self.df = pd.get_dummies(self.df, drop_first=True)

        #handle NaN (important for ML)
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)

    def split_data(self, target_variable, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_variable])
        y = self.df[target_variable]

        # extra safety (imputer)
        #ay ta numeric data er jonno mean diye fill korbe, categorical data er jonno most_frequent diye fill korbe
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return r2, mse



ml = ML_Model(df)

X_train, X_test, y_train, y_test = ml.split_data("stress_level")

model = ml.train_model(X_train, y_train)

r2, mse = ml.evaluate_model(model, X_test, y_test)

predictions = model.predict(X_test)
print(f"stress_level Prediction: {predictions[0]:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"MSE: {mse:.2f}")
