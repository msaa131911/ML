#LDA and QDA
"""LDA ধরে নেয়:
সব class-এর data same covariance (spread/variance structure) follow করে
Decision boundary হবে linear (straight line / plane)"""

"""QDA ধরে নেয়:
প্রতিটি class-এর covariance different হতে পারে"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import seaborn as sns

data = sns.load_dataset("titanic")

# missing value drop
data = data[["age", "fare", "survived"]].dropna()

X = data[["age", "fare"]]
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train, y_train)
qda.fit(X_train, y_train)

print("LDA Accuracy:", lda.score(X_test, y_test))
print("QDA Accuracy:", qda.score(X_test, y_test))